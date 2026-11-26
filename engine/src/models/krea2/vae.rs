//! Qwen-Image (v1) VAE decoder, the one Krea 2 uses: Wan-2.1 shaped, 8x, 16 latent channels, RGB.
//! diffusers AutoencoderKLQwenImage on a single frame: every causal 3D conv sees zero padding in front of the one
//! frame, so it equals a 2D conv with the LAST temporal slice of its kernel; upsample3d skips its time_conv on the
//! first frame ("Rep"), leaving nearest 2x + conv. bf16 like ai-toolkit's VAE dtype, fp32 channel RMS norm.
use crate::core::util::blend;
use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor};

use crate::core::act_dtype;
use crate::core::weights::Weights;

pub const LATENTS_MEAN: [f32; 16] =
    [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921];
pub const LATENTS_STD: [f32; 16] =
    [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916];

struct Conv {
    w: Tensor,
    b: Tensor,
    pad: usize,
}

impl Conv {
    /// 5D causal kernels [o, i, kt, kh, kw] -> last temporal slice; 4D kernels as they are.
    fn load(wt: &Weights, p: &str, dev: &Device) -> Result<Self> {
        let mut w = wt.tensor_act(&format!("{p}.weight"), dev)?;
        if w.rank() == 5 {
            let kt = w.dim(2)?;
            w = w.narrow(2, kt - 1, 1)?.squeeze(2)?;
        }
        let pad = w.dim(3)? / 2;
        Ok(Self { w: w.contiguous()?, b: wt.tensor_act(&format!("{p}.bias"), dev)?, pad })
    }
}

impl Module for Conv {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        x.conv2d(&self.w, self.pad, 1, 1, 1)?.broadcast_add(&self.b.reshape((1, (), 1, 1))?)
    }
}

/// F.normalize over channels in fp32 -> bf16, * sqrt(C) * gamma.
struct Rms {
    gamma: Tensor,
    scale: f64,
}

impl Rms {
    fn load(wt: &Weights, p: &str, dev: &Device) -> Result<Self> {
        let g = wt.tensor_act(&format!("{p}.gamma"), dev)?;
        let c = g.dim(0)?;
        Ok(Self { gamma: g.reshape((1, c, 1, 1))?, scale: (c as f64).sqrt() })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let xf = x.to_dtype(DType::F32)?;
        let n = xf.sqr()?.sum_keepdim(1)?.sqrt()?.maximum(1e-12)?;
        Ok((xf.broadcast_div(&n)?.to_dtype(act_dtype())? * self.scale)?.broadcast_mul(&self.gamma)?)
    }
}

struct Res {
    n1: Rms,
    c1: Conv,
    n2: Rms,
    c2: Conv,
    short: Option<Conv>,
}

impl Res {
    fn load(wt: &Weights, p: &str, dev: &Device) -> Result<Self> {
        Ok(Self {
            n1: Rms::load(wt, &format!("{p}.norm1"), dev)?,
            c1: Conv::load(wt, &format!("{p}.conv1"), dev)?,
            n2: Rms::load(wt, &format!("{p}.norm2"), dev)?,
            c2: Conv::load(wt, &format!("{p}.conv2"), dev)?,
            short: if wt.has(&format!("{p}.conv_shortcut.weight")) { Some(Conv::load(wt, &format!("{p}.conv_shortcut"), dev)?) } else { None },
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = match &self.short {
            Some(c) => c.forward(x)?,
            None => x.clone(),
        };
        let y = self.c1.forward(&self.n1.forward(x)?.silu()?)?;
        let y = self.c2.forward(&self.n2.forward(&y)?.silu()?)?;
        Ok((y + h)?)
    }
}

struct Attn {
    norm: Rms,
    qkv: Conv,
    proj: Conv,
}

impl Attn {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = x.dims4()?;
        let qkv = self.qkv.forward(&self.norm.forward(x)?)?; // [b, 3c, h, w]
        let qkv = qkv.reshape((b, 3 * c, h * w))?.transpose(1, 2)?; // [b, hw, 3c]
        let q = qkv.narrow(2, 0, c)?.unsqueeze(1)?.contiguous()?;
        let k = qkv.narrow(2, c, c)?.unsqueeze(1)?.contiguous()?;
        let v = qkv.narrow(2, 2 * c, c)?.unsqueeze(1)?.contiguous()?;
        let o = crate::core::attention::sdpa(&q, &k, &v, None, false)?; // [b, 1, hw, c]
        let o = o.squeeze(1)?.transpose(1, 2)?.reshape((b, c, h, w))?;
        Ok((self.proj.forward(&o)? + x)?)
    }
}

struct Up {
    res: Vec<Res>,
    up: Option<Conv>, // nearest 2x then this 3x3 conv
}

pub struct VaeQi {
    post_quant: Conv,
    conv_in: Conv,
    mid0: Res,
    attn: Attn,
    mid1: Res,
    ups: Vec<Up>,
    norm_out: Rms,
    conv_out: Conv,
    mean: Tensor,
    std: Tensor,
}

impl VaeQi {
    pub fn load(path: &str, dev: &Device) -> Result<Self> {
        let wt = Weights::open(path)?;
        let d = "decoder";
        let mut ups = vec![];
        for i in 0..4 {
            let p = format!("{d}.up_blocks.{i}");
            let res = (0..3).map(|j| Res::load(&wt, &format!("{p}.resnets.{j}"), dev)).collect::<Result<Vec<_>>>()?;
            let up = if wt.has(&format!("{p}.upsamplers.0.resample.1.weight")) {
                Some(Conv::load(&wt, &format!("{p}.upsamplers.0.resample.1"), dev)?)
            } else {
                None
            };
            ups.push(Up { res, up });
        }
        let a = format!("{d}.mid_block.attentions.0");
        Ok(Self {
            post_quant: Conv::load(&wt, "post_quant_conv", dev)?,
            conv_in: Conv::load(&wt, &format!("{d}.conv_in"), dev)?,
            mid0: Res::load(&wt, &format!("{d}.mid_block.resnets.0"), dev)?,
            attn: Attn {
                norm: Rms::load(&wt, &format!("{a}.norm"), dev)?,
                qkv: Conv::load(&wt, &format!("{a}.to_qkv"), dev)?,
                proj: Conv::load(&wt, &format!("{a}.proj"), dev)?,
            },
            mid1: Res::load(&wt, &format!("{d}.mid_block.resnets.1"), dev)?,
            ups,
            norm_out: Rms::load(&wt, &format!("{d}.norm_out"), dev)?,
            conv_out: Conv::load(&wt, &format!("{d}.conv_out"), dev)?,
            mean: Tensor::from_slice(&LATENTS_MEAN, (1, 16, 1, 1), dev)?.to_dtype(act_dtype())?,
            std: Tensor::from_slice(&LATENTS_STD, (1, 16, 1, 1), dev)?.to_dtype(act_dtype())?,
        })
    }

    /// Normalized latents [b, 16, h, w] (any float dtype) -> RGB [b, 3, 8h, 8w] f32 clamped to [-1, 1].
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        self.decode_cancel(z, None)
    }

    /// Like `decode`; canvases over 512 x 512 are decoded in 512 px tiles (384 px stride, 128 px linear blends; the
    /// diffusers tiled_decode). Bigger tiles next to the 13 GB DiT overflowed 24 GB and Windows spilled VRAM to RAM
    /// (packs at 1536 px stalled). `cancel` is checked per tile.
    pub fn decode_cancel(&self, z: &Tensor, cancel: Option<&std::sync::atomic::AtomicBool>) -> Result<Tensor> {
        // ai-toolkit decode_latents: cast to the VAE dtype (bf16) first, then latents * std + mean in bf16
        let z = z.to_dtype(act_dtype())?.broadcast_mul(&self.std)?.broadcast_add(&self.mean)?;
        let (_, _, h, w) = z.dims4()?;
        if h * w * 64 <= 512 * 512 {
            return Ok(self.decode_raw(&z)?.to_dtype(DType::F32)?.clamp(-1f32, 1f32)?);
        }
        // latent tile / stride, pixel blend extent: 512 px tiles, 25 % overlap. 1024 px tiles needed ~7 GB of conv
        // buffers on top of the DiT, which with a big LoRA and other apps open overflowed 24 GB and stalled
        let (tile, stride, extent) = (64usize, 48usize, 128usize);
        let mut rows: Vec<Vec<Tensor>> = vec![];
        for i in (0..h).step_by(stride) {
            let mut row = vec![];
            for j in (0..w).step_by(stride) {
                if cancel.is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed)) {
                    anyhow::bail!("cancelled");
                }
                let t = z.narrow(2, i, tile.min(h - i))?.narrow(3, j, tile.min(w - j))?;
                row.push(self.decode_raw(&t)?.to_dtype(DType::F32)?);
            }
            rows.push(row);
        }
        let out_stride = stride * 8;
        let mut result_rows = vec![];
        for i in 0..rows.len() {
            for j in 0..rows[i].len() {
                let mut t = rows[i][j].clone();
                if i > 0 {
                    t = blend(&rows[i - 1][j], &t, extent, 2)?;
                }
                if j > 0 {
                    t = blend(&rows[i][j - 1], &t, extent, 3)?;
                }
                rows[i][j] = t;
            }
            let parts: Vec<Tensor> = rows[i]
                .iter()
                .map(|t| {
                    let (th, tw) = (t.dim(2).unwrap(), t.dim(3).unwrap());
                    t.narrow(2, 0, out_stride.min(th)).and_then(|t| t.narrow(3, 0, out_stride.min(tw)))
                })
                .collect::<candle_core::Result<_>>()?;
            result_rows.push(Tensor::cat(&parts, 3)?);
        }
        let dec = Tensor::cat(&result_rows, 2)?.narrow(2, 0, h * 8)?.narrow(3, 0, w * 8)?;
        Ok(dec.clamp(-1f32, 1f32)?)
    }

    /// post_quant_conv + decoder on denormalized bf16 latents, no clamp.
    fn decode_raw(&self, z: &Tensor) -> Result<Tensor> {
        let mut x = self.conv_in.forward(&self.post_quant.forward(z)?)?;
        x = self.mid1.forward(&self.attn.forward(&self.mid0.forward(&x)?)?)?;
        for u in &self.ups {
            for r in &u.res {
                crate::core::check_abort()?;
                x = r.forward(&x)?;
            }
            if let Some(c) = &u.up {
                let (_, _, h, w) = x.dims4()?;
                x = c.forward(&x.upsample_nearest2d(2 * h, 2 * w)?)?;
            }
        }
        Ok(self.conv_out.forward(&self.norm_out.forward(&x)?.silu()?)?)
    }
}

/// Stride-2 downsample of the encoder (diffusers QwenImageResample "downsample2d/3d" on one frame): ZeroPad2d
/// (left 0, right 1, top 0, bottom 1), then a 3x3 conv with stride 2 and no padding. The 3D variant's time_conv is
/// skipped on the first frame, so both are this.
struct Down {
    w: Tensor,
    b: Tensor,
}

impl Down {
    fn load(wt: &Weights, p: &str, dev: &Device) -> Result<Self> {
        Ok(Self { w: wt.tensor_act(&format!("{p}.weight"), dev)?.contiguous()?, b: wt.tensor_act(&format!("{p}.bias"), dev)? })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.pad_with_zeros(3, 0, 1)?.pad_with_zeros(2, 0, 1)?;
        Ok(x.conv2d(&self.w, 0, 2, 1, 1)?.broadcast_add(&self.b.reshape((1, (), 1, 1))?)?)
    }
}

enum EncStep {
    Res(Res),
    Down(Down),
}

/// Qwen-Image (v1) VAE encoder on a single frame, for img2img: RGB [-1, 1] -> normalized latents (the posterior mean,
/// (mu - mean) / std), the inverse of VaeQi::decode. Same reductions as the decoder: causal 3D convs keep their last
/// temporal slice, channel RMS norms in fp32, bf16 compute.
pub struct VaeEnc {
    conv_in: Conv,
    steps: Vec<EncStep>,
    mid0: Res,
    attn: Attn,
    mid1: Res,
    norm_out: Rms,
    conv_out: Conv,
    quant: Conv,
    mean: Tensor,
    std: Tensor,
}

impl VaeEnc {
    pub fn load(path: &str, dev: &Device) -> Result<Self> {
        let wt = Weights::open(path)?;
        let e = "encoder";
        let mut steps = vec![];
        let mut i = 0;
        loop {
            let p = format!("{e}.down_blocks.{i}");
            if wt.has(&format!("{p}.resample.1.weight")) {
                steps.push(EncStep::Down(Down::load(&wt, &format!("{p}.resample.1"), dev)?));
            } else if wt.has(&format!("{p}.conv1.weight")) {
                steps.push(EncStep::Res(Res::load(&wt, &p, dev)?));
            } else {
                break;
            }
            i += 1;
        }
        let a = format!("{e}.mid_block.attentions.0");
        Ok(Self {
            conv_in: Conv::load(&wt, &format!("{e}.conv_in"), dev)?,
            steps,
            mid0: Res::load(&wt, &format!("{e}.mid_block.resnets.0"), dev)?,
            attn: Attn {
                norm: Rms::load(&wt, &format!("{a}.norm"), dev)?,
                qkv: Conv::load(&wt, &format!("{a}.to_qkv"), dev)?,
                proj: Conv::load(&wt, &format!("{a}.proj"), dev)?,
            },
            mid1: Res::load(&wt, &format!("{e}.mid_block.resnets.1"), dev)?,
            norm_out: Rms::load(&wt, &format!("{e}.norm_out"), dev)?,
            conv_out: Conv::load(&wt, &format!("{e}.conv_out"), dev)?,
            quant: Conv::load(&wt, "quant_conv", dev)?,
            mean: Tensor::from_slice(&LATENTS_MEAN, (1, 16, 1, 1), dev)?,
            std: Tensor::from_slice(&LATENTS_STD, (1, 16, 1, 1), dev)?,
        })
    }

    /// RGB [b, 3, H, W] in [-1, 1] (H, W multiples of 8) -> normalized latents [b, 16, H/8, W/8] f32; tiled above 512 px.
    pub fn encode(&self, img: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = img.dims4()?;
        if h * w <= 512 * 512 {
            return self.encode_whole(img);
        }
        crate::core::util::tiled_encode(img, 8, |t| self.encode_whole(t))
    }

    fn encode_whole(&self, img: &Tensor) -> Result<Tensor> {
        let mut x = self.conv_in.forward(&img.to_dtype(act_dtype())?)?;
        for s in &self.steps {
            x = match s {
                EncStep::Res(r) => r.forward(&x)?,
                EncStep::Down(d) => d.forward(&x)?,
            };
        }
        x = self.mid1.forward(&self.attn.forward(&self.mid0.forward(&x)?)?)?;
        let x = self.conv_out.forward(&self.norm_out.forward(&x)?.silu()?)?;
        let moments = self.quant.forward(&x)?; // [b, 32, h, w]: mean, logvar
        let mu = moments.narrow(1, 0, 16)?.to_dtype(DType::F32)?;
        Ok(mu.broadcast_sub(&self.mean)?.broadcast_div(&self.std)?)
    }
}
