//! Qwen-Image 2.1 VAE (Wan-2.2 shaped, RGBA, 16x, 64 latent channels) for still images: every "3D" op reduces to
//! 2D on a single frame. Two single-frame quirks are kept exactly: the decoder's DupUp3D keeps the second temporal
//! slice (first_chunk), and the encoder's AvgDown3D averages over a zero frame padded in front.
use crate::core::util::blend;
use anyhow::Result;
use candle_core::{D, DType, Device, Module, Tensor};

use crate::core::weights::Weights;

#[allow(clippy::approx_constant, reason = "measured latent statistics, not pi / 6")]
const LATENTS_MEAN: [f32; 64] = [
    0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722, 1.8873, -1.7177, -0.651, 0.2732, 0.7562, -0.6163, -1.0277, 3.8363, 2.021, 0.0472,
    0.932, 2.0087, 2.4954, -0.1391, -1.4249, 1.8464, -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955, -0.0342, -1.0265, -0.7636, 3.0555,
    0.0746, -3.0751, -0.1076, 1.7376, -1.0914, -1.9435, -0.2784, -1.368, 0.4809, -0.4433, 0.3764, 0.5729, -2.0595, 1.096, -1.326, -2.0211, -5.0179, 0.5275,
    4.0162, 1.8505, 0.3026, 1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1562, 0.0304,
];
const LATENTS_STD: [f32; 64] = [
    3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.791, 3.0785, 3.65, 3.9308, 3.0904, 2.8778, 3.7675, 3.732, 5.0756, 3.2864, 4.0397, 3.1317, 4.0443,
    2.9249, 3.9454, 3.0988, 4.2489, 3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.283, 3.5694, 4.2467, 3.9037, 3.2947, 5.077, 3.5075, 3.27, 3.4767, 2.8063, 5.1125,
    3.5327, 4.7833, 3.1286, 4.1819, 3.8527, 3.8312, 3.5605, 4.3875, 3.9624, 4.0168, 3.5643, 4.055, 5.5614, 4.2963, 4.408, 3.4959, 3.8747, 3.7608, 3.5735,
    3.149, 3.7662, 3.6746, 3.4563, 3.8161,
];

struct Conv {
    w: Tensor,
    b: Tensor,
    pad: usize,
    stride: usize,
}

impl Conv {
    fn load(wt: &Weights, p: &str, dev: &Device, pad: usize, stride: usize) -> Result<Self> {
        let mut w = wt.tensor(&format!("{p}.weight"), dev)?.to_dtype(DType::BF16)?;
        if w.rank() == 5 {
            w = w.squeeze(2)?; // [out, in, 1, kh, kw] -> 2D
        }
        let b = wt.tensor(&format!("{p}.bias"), dev)?.to_dtype(DType::BF16)?;
        Ok(Self { w, b, pad, stride })
    }
}

impl Module for Conv {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let y = x.conv2d(&self.w, self.pad, self.stride, 1, 1)?;
        y.broadcast_add(&self.b.reshape((1, (), 1, 1))?)
    }
}

/// Channel RMS norm: F.normalize over C in fp32 (to bf16), * sqrt(C) * gamma.
struct Rms {
    gamma: Tensor,
    scale: f64,
}

impl Rms {
    fn load(wt: &Weights, p: &str, dev: &Device) -> Result<Self> {
        let g = wt.tensor(&format!("{p}.gamma"), dev)?.to_dtype(DType::BF16)?;
        let c = g.dim(0)?;
        Ok(Self { gamma: g.reshape((1, c, 1, 1))?, scale: (c as f64).sqrt() })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let xf = x.to_dtype(DType::F32)?;
        let n = xf.sqr()?.sum_keepdim(1)?.sqrt()?.maximum(1e-12)?;
        let normed = xf.broadcast_div(&n)?.to_dtype(DType::BF16)?;
        Ok((normed * self.scale)?.broadcast_mul(&self.gamma)?)
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
            n1: Rms::load(wt, &format!("{p}.residual.0"), dev)?,
            c1: Conv::load(wt, &format!("{p}.residual.2"), dev, 1, 1)?,
            n2: Rms::load(wt, &format!("{p}.residual.3"), dev)?,
            c2: Conv::load(wt, &format!("{p}.residual.6"), dev, 1, 1)?,
            short: if wt.has(&format!("{p}.shortcut.weight")) { Some(Conv::load(wt, &format!("{p}.shortcut"), dev, 0, 1)?) } else { None },
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = match &self.short {
            Some(s) => s.forward(x)?,
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
    fn load(wt: &Weights, p: &str, dev: &Device) -> Result<Self> {
        Ok(Self {
            norm: Rms::load(wt, &format!("{p}.norm"), dev)?,
            qkv: Conv::load(wt, &format!("{p}.to_qkv"), dev, 0, 1)?,
            proj: Conv::load(wt, &format!("{p}.proj"), dev, 0, 1)?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = x.dims4()?;
        let qkv = self.qkv.forward(&self.norm.forward(x)?)?.reshape((b, 3 * c, h * w))?.transpose(1, 2)?; // [b, hw, 3c]
        let q = qkv.narrow(2, 0, c)?.unsqueeze(1)?;
        let k = qkv.narrow(2, c, c)?.unsqueeze(1)?;
        let v = qkv.narrow(2, 2 * c, c)?.unsqueeze(1)?;
        let o = crate::core::attention::sdpa(&q, &k, &v, None, false)?; // one head of width c
        let o = o.squeeze(1)?.transpose(1, 2)?.reshape((b, c, h, w))?;
        Ok((self.proj.forward(&o)? + x)?)
    }
}

struct Mid {
    r0: Res,
    a: Attn,
    r1: Res,
}

impl Mid {
    fn load(wt: &Weights, p: &str, dev: &Device) -> Result<Self> {
        Ok(Self { r0: Res::load(wt, &format!("{p}.0"), dev)?, a: Attn::load(wt, &format!("{p}.1"), dev)?, r1: Res::load(wt, &format!("{p}.2"), dev)? })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.r1.forward(&self.a.forward(&self.r0.forward(x)?)?)
    }
}

struct Up {
    res: Vec<Res>,
    up: Option<Conv>, // nearest 2x + conv
    ft: usize,        // DupUp3D temporal factor (0 = no shortcut)
    out: usize,
}

struct Down {
    res: Vec<Res>,
    down: Option<Conv>, // pad (0,1,0,1) + conv stride 2
    ft: usize,
    fs: usize,
    out: usize,
}

pub struct Vae {
    post_quant: Conv,
    dec_in: Conv,
    dec_mid: Mid,
    ups: Vec<Up>,
    dec_norm: Rms,
    dec_out: Conv,
    quant: Conv,
    enc_in: Conv,
    downs: Vec<Down>,
    enc_mid: Mid,
    enc_norm: Rms,
    enc_out: Conv,
    mean: Tensor,
    std: Tensor,
}

/// DupUp3D(in -> out, factor_t ft, factor_s 2) on one frame with first_chunk: [b, cin, h, w] -> [b, out, 2h, 2w].
fn dup_up(x: &Tensor, out: usize, ft: usize) -> Result<Tensor> {
    let (b, cin, h, w) = x.dims4()?;
    let rep = out * ft * 4 / cin;
    let x = x.unsqueeze(2)?.repeat((1, 1, rep, 1, 1))?.reshape(vec![b, out, ft, 2, 2, h, w])?; // repeat_interleave on C
    let x = x.narrow(2, ft - 1, 1)?.squeeze(2)?; // [b, out, 2, 2, h, w], the kept temporal slice
    Ok(x.permute((0, 1, 4, 2, 5, 3))?.reshape((b, out, 2 * h, 2 * w))?)
}

/// AvgDown3D(in -> out, factor_t ft, factor_s fs) on one frame (zero frame padded in front when ft = 2).
fn avg_down(x: &Tensor, out: usize, ft: usize, fs: usize) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    let frames = if ft == 2 { Tensor::stack(&[&x.zeros_like()?, x], 2)? } else { x.unsqueeze(2)? }; // [b, c, ft, h, w]
    let (h2, w2) = (h / fs, w / fs);
    let y = frames.reshape(vec![b, c, ft, h2, fs, w2, fs])?.permute(vec![0usize, 1, 2, 4, 6, 3, 5])?;
    let factor = ft * fs * fs;
    let y = y.reshape((b, c * factor, h2, w2))?.reshape((b, out, c * factor / out, h2, w2))?;
    Ok(y.to_dtype(DType::F32)?.mean(2)?.to_dtype(x.dtype())?)
}

impl Vae {
    pub fn load(path: &str, dev: &Device) -> Result<Self> {
        let wt = Weights::open(path)?;
        let mut ups = vec![];
        let up_dims = [(1152, 1152, 2), (1152, 1152, 2), (1152, 576, 2), (576, 288, 1), (288, 144, 0)];
        for (i, &(_, out, ft)) in up_dims.iter().enumerate() {
            let p = format!("decoder.upsamples.{i}.upsamples");
            let res = (0..3).map(|j| Res::load(&wt, &format!("{p}.{j}"), dev)).collect::<Result<Vec<_>>>()?;
            let up = if i < 4 { Some(Conv::load(&wt, &format!("{p}.3.resample.1"), dev, 1, 1)?) } else { None };
            ups.push(Up { res, up, ft, out });
        }
        let mut downs = vec![];
        let down_dims = [(96, 96, 1, 2), (96, 192, 2, 2), (192, 384, 2, 2), (384, 768, 2, 2), (768, 768, 1, 1)];
        for (i, &(_, out, ft, fs)) in down_dims.iter().enumerate() {
            let p = format!("encoder.downsamples.{i}.downsamples");
            let res = (0..2).map(|j| Res::load(&wt, &format!("{p}.{j}"), dev)).collect::<Result<Vec<_>>>()?;
            let down = if i < 4 { Some(Conv::load(&wt, &format!("{p}.2.resample.1"), dev, 0, 2)?) } else { None };
            downs.push(Down { res, down, ft, fs, out });
        }
        Ok(Self {
            post_quant: Conv::load(&wt, "conv2", dev, 0, 1)?,
            dec_in: Conv::load(&wt, "decoder.conv1", dev, 1, 1)?,
            dec_mid: Mid::load(&wt, "decoder.middle", dev)?,
            ups,
            dec_norm: Rms::load(&wt, "decoder.head.0", dev)?,
            dec_out: Conv::load(&wt, "decoder.head.2", dev, 1, 1)?,
            quant: Conv::load(&wt, "conv1", dev, 0, 1)?,
            enc_in: Conv::load(&wt, "encoder.conv1", dev, 1, 1)?,
            downs,
            enc_mid: Mid::load(&wt, "encoder.middle", dev)?,
            enc_norm: Rms::load(&wt, "encoder.head.0", dev)?,
            enc_out: Conv::load(&wt, "encoder.head.2", dev, 1, 1)?,
            mean: Tensor::from_slice(&LATENTS_MEAN, (1, 64, 1, 1), dev)?,
            std: Tensor::from_slice(&LATENTS_STD, (1, 64, 1, 1), dev)?,
        })
    }

    /// Normalised latents [b, 64, h, w] -> RGBA image [b, 4, 16h, 16w] in [-1, 1] (f32).
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        self.decode_cancel(z, None)
    }

    /// Like `decode`; canvases over 1 MP go through the tiled path the PyTorch studio used (1024 px tiles, 768 px
    /// stride, 256 px linear blends; diffusers' Wan tiled_decode), which keeps the conv buffers small. Low VRAM mode
    /// tiles everything over 512 x 512 in 512 px tiles (same 25 % overlap): a whole 1024 px decode needs ~8 GB.
    /// `cancel` is checked between tiles.
    pub fn decode_cancel(&self, z: &Tensor, cancel: Option<&std::sync::atomic::AtomicBool>) -> Result<Tensor> {
        let z = z.to_dtype(DType::F32)?.broadcast_mul(&self.std)?.broadcast_add(&self.mean)?.to_dtype(DType::BF16)?;
        let (_, _, h, w) = z.dims4()?;
        let low = crate::core::low_vram();
        if h * w * 256 <= if low { 512 * 512 } else { 1024 * 1024 } {
            return Ok(self.decode_raw(&z)?.to_dtype(DType::F32)?.clamp(-1f32, 1f32)?);
        }
        // latent tile / stride, pixel blend extent
        let (tile, stride, extent) = if low { (32usize, 24usize, 128usize) } else { (64usize, 48usize, 256usize) };
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
        // blend in place, in the same order as diffusers (later tiles see already blended neighbours)
        let out_stride = stride * 16;
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
        let dec = Tensor::cat(&result_rows, 2)?.narrow(2, 0, h * 16)?.narrow(3, 0, w * 16)?;
        Ok(dec.clamp(-1f32, 1f32)?)
    }

    /// post_quant_conv + decoder on denormalised bf16 latents, no clamp.
    fn decode_raw(&self, z: &Tensor) -> Result<Tensor> {
        let mut x = self.dec_in.forward(&self.post_quant.forward(z)?)?;
        x = self.dec_mid.forward(&x)?;
        for up in &self.ups {
            let copy = x.clone();
            for r in &up.res {
                crate::core::check_abort()?;
                x = r.forward(&x)?;
            }
            if let Some(conv) = &up.up {
                let (_, _, h, w) = x.dims4()?;
                x = conv.forward(&x.upsample_nearest2d(h * 2, w * 2)?)?;
            }
            if up.ft > 0 {
                x = (x + dup_up(&copy, up.out, up.ft)?)?;
            }
        }
        Ok(self.dec_out.forward(&self.dec_norm.forward(&x)?.silu()?)?)
    }

    /// RGBA image [b, 4, H, W] in [-1, 1] -> normalised posterior mean [b, 64, H/16, W/16] (f32).
    pub fn encode(&self, img: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = img.dims4()?;
        if h * w <= 512 * 512 {
            return self.encode_whole(img);
        }
        crate::core::util::tiled_encode(img, 16, |t| self.encode_whole(t))
    }

    fn encode_whole(&self, img: &Tensor) -> Result<Tensor> {
        let mut x = self.enc_in.forward(&img.to_dtype(DType::BF16)?)?;
        for d in &self.downs {
            let copy = x.clone();
            for r in &d.res {
                x = r.forward(&x)?;
            }
            if let Some(conv) = &d.down {
                x = conv.forward(&x.pad_with_zeros(D::Minus1, 0, 1)?.pad_with_zeros(D::Minus2, 0, 1)?)?;
            }
            x = (x + avg_down(&copy, d.out, d.ft, d.fs)?)?;
        }
        let x = self.enc_mid.forward(&x)?;
        let x = self.enc_out.forward(&self.enc_norm.forward(&x)?.silu()?)?;
        let mu = self.quant.forward(&x)?.narrow(1, 0, 64)?;
        Ok(mu.to_dtype(DType::F32)?.broadcast_sub(&self.mean)?.broadcast_div(&self.std)?)
    }
}
