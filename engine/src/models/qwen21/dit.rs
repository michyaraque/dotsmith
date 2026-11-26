//! Qwen-Image 2.1 single-stream DiT (32 blocks, width 4096, 32 heads x 128), ComfyUI convrot int8 checkpoint.
//!
//! Text-to-image sequence = [prompt tokens (block-causal: text sees only earlier text), target image tokens (see
//! everything)]. Text tokens modulate from t = 0 ("causal_condition"), so their per-layer K/V never depend on the
//! timestep or on the image: `prefill` computes them once per prompt and every denoising step only runs the image
//! tokens against that cache (exact, and the text's share of every step is gone).
use anyhow::{Context, Result};
use candle_core::{D, DType, Device, Tensor};

use crate::core::attention::sdpa;
use crate::core::fused;
use crate::core::int8::{AnyLinear, Linear, Lora};
use crate::core::memory::{Stack, StackPlan};
use crate::core::stream::StreamedStack;
use crate::core::weights::Weights;

pub const DIM: usize = 4096;
const HEADS: usize = 32;
const HEAD_DIM: usize = 128;
const AXES: [usize; 3] = [16, 56, 56];
const EPS: f64 = 1e-6;
/// The int8 linears of one block, as packed by the weight streamer.
const BLOCK_LINEARS: [&str; 6] = ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "img_mlp.gate_up", "img_mlp.out"];

struct Block {
    q: AnyLinear,
    k: AnyLinear,
    v: AnyLinear,
    o: AnyLinear,
    norm_q: Tensor,
    norm_k: Tensor,
    gate_up: AnyLinear,
    out: AnyLinear,
}

pub struct Transformer {
    img_in: Linear,
    txt_norm: Tensor,
    txt_in1: Linear,
    txt_in2: Linear,
    t_lin1: Linear,
    t_lin2: Linear,
    modulation: Linear,
    blocks: Vec<Block>,
    norm_out: Linear,
    proj_out: Linear,
    dev: Device,
    /// blocks placed by the memory plan: `first..` stream from host RAM (None = all on the GPU)
    stream: Option<StreamedStack>,
}

/// Per-prompt cache: per layer the text keys/values after norm + RoPE, [b, h, l, d].
pub struct TextCache {
    k: Vec<Tensor>,
    v: Vec<Tensor>,
    valid: Tensor, // [b, l] u8
    pub len: usize,
}

/// RoPE tables for the given (frame, height, width) positions per token: (cos, sin) [s, 64] f32.
fn rope_tables(pos: &[[i64; 3]], dev: &Device) -> Result<(Tensor, Tensor)> {
    let mut inv: Vec<Vec<f32>> = vec![];
    for &d in &AXES {
        inv.push((0..d).step_by(2).map(|i| 1.0f32 / 10000f32.powf(i as f32 / d as f32)).collect());
    }
    let s = pos.len();
    let mut cos = Vec::with_capacity(s * 64);
    let mut sin = Vec::with_capacity(s * 64);
    for p in pos {
        for (axis, freqs) in inv.iter().enumerate() {
            for f in freqs {
                let a = p[axis] as f32 * f;
                cos.push(a.cos());
                sin.push(a.sin());
            }
        }
    }
    Ok((Tensor::from_vec(cos, (s, 64), dev)?, Tensor::from_vec(sin, (s, 64), dev)?))
}

impl Transformer {
    /// For core::memory::plan: the block stack (None if not int8) and the bytes that always stay on the GPU.
    pub fn memory(path: &str) -> Result<(Option<Stack>, u64)> {
        let wt = Weights::open(path)?;
        let stack = wt.streamable_layer_bytes("transformer_blocks.0", &BLOCK_LINEARS).map(|b| Stack { name: "DiT", layers: 32, layer_bytes: b, hot: true });
        let in_stack = |k: &str| k.starts_with("transformer_blocks.") && BLOCK_LINEARS.iter().any(|l| k.ends_with(&format!(".{l}.weight")));
        let stack_bytes = if stack.is_some() { wt.bytes_where(in_stack) } else { 0 };
        Ok((stack, wt.bytes_where(|_| true) - stack_bytes))
    }

    /// Fully on the GPU.
    pub fn load(path: &str, dev: &Device) -> Result<Self> {
        Self::load_planned(path, dev, StackPlan::GPU)
    }

    pub fn load_planned(path: &str, dev: &Device, plan: StackPlan) -> Result<Self> {
        let wt = Weights::open(path)?;
        let lin = |p: &str| Linear::load(&wt, p, dev, false);
        let plan = if wt.streamable_layer_bytes("transformer_blocks.0", &BLOCK_LINEARS).is_some() { plan } else { StackPlan::GPU };
        let ids: Vec<String> = (0..32).map(|i| format!("transformer_blocks.{i}")).collect();
        let stream = StreamedStack::build(&wt, std::path::Path::new(path), dev, &ids, &BLOCK_LINEARS, plan, true)?;
        crate::core::memory::set_report(match &stream {
            Some(s) => s.placed("dit", 32),
            None => crate::core::memory::Placed::gpu("dit", 32, wt.layer_format("transformer_blocks.0", &BLOCK_LINEARS)),
        });
        let mut blocks = vec![];
        for i in 0..32 {
            let p = format!("transformer_blocks.{i}");
            let l = |n: &str| -> Result<AnyLinear> {
                match &stream {
                    Some(s) if s.covers(i) => s.linear(&wt, i, &p, n, dev),
                    _ => AnyLinear::load(&wt, &format!("{p}.{n}"), dev),
                }
            };
            blocks.push(Block {
                q: l("attn.to_q")?,
                k: l("attn.to_k")?,
                v: l("attn.to_v")?,
                o: l("attn.to_out.0")?,
                norm_q: wt.tensor(&format!("{p}.attn.norm_q.weight"), dev)?.to_dtype(DType::BF16)?,
                norm_k: wt.tensor(&format!("{p}.attn.norm_k.weight"), dev)?.to_dtype(DType::BF16)?,
                gate_up: l("img_mlp.gate_up")?,
                out: l("img_mlp.out")?,
            });
        }
        Ok(Self {
            img_in: lin("img_in")?,
            txt_norm: wt.tensor("txt_in.text_norm.weight", dev)?.to_dtype(DType::F32)?,
            txt_in1: lin("txt_in.in_layer")?,
            txt_in2: lin("txt_in.out_layer")?,
            t_lin1: lin("time_text_embed.timestep_embedder.linear_1")?,
            t_lin2: lin("time_text_embed.timestep_embedder.linear_2")?,
            modulation: lin("modulation.1")?,
            blocks,
            norm_out: lin("norm_out.linear")?,
            proj_out: lin("proj_out")?,
            dev: dev.clone(),
            stream,
        })
    }

    pub fn streaming(&self) -> bool {
        self.stream.is_some()
    }

    /// Attach an ai-toolkit LoRA (`diffusion_model.<module>.lora_A/B`), replacing any previous one. Cheap: only the
    /// small A/B matrices move, the base model stays as it is.
    pub fn set_lora(&mut self, path: Option<&str>, multiplier: f64) -> Result<usize> {
        for b in &mut self.blocks {
            for l in [&mut b.q, &mut b.k, &mut b.v, &mut b.o, &mut b.gate_up, &mut b.out] {
                l.loras_mut().clear();
            }
        }
        let Some(path) = path else { return Ok(0) };
        let wt = Weights::open(path)?;
        let mut n = 0;
        for (i, b) in self.blocks.iter_mut().enumerate() {
            let mods: [(&str, &mut AnyLinear); 6] = [
                ("attn.to_q", &mut b.q),
                ("attn.to_k", &mut b.k),
                ("attn.to_v", &mut b.v),
                ("attn.to_out.0", &mut b.o),
                ("img_mlp.gate_up", &mut b.gate_up),
                ("img_mlp.out", &mut b.out),
            ];
            for (name, l) in mods {
                let p = format!("diffusion_model.transformer_blocks.{i}.{name}");
                if !wt.has(&format!("{p}.lora_A.weight")) {
                    continue;
                }
                let down = wt.tensor(&format!("{p}.lora_A.weight"), &self.dev)?.to_dtype(DType::BF16)?;
                let up = wt.tensor(&format!("{p}.lora_B.weight"), &self.dev)?.to_dtype(DType::BF16)?;
                let rank = down.dim(0)? as f64;
                let alpha = match wt.tensor(&format!("{p}.alpha"), &Device::Cpu) {
                    Ok(a) => a.to_dtype(DType::F64)?.flatten_all()?.to_vec1::<f64>()?[0],
                    Err(_) => rank,
                };
                l.loras_mut().push(Lora { down, up, scale: multiplier * alpha / rank });
                n += 1;
            }
        }
        Ok(n)
    }

    /// temb for the given timesteps (already bf16-rounded, in [0, 1]) -> [n, 4096] bf16.
    fn temb(&self, t: &[f32]) -> Result<Tensor> {
        let half = 128;
        let freqs: Vec<f32> = (0..half).map(|i| (-(10000f32).ln() * i as f32 / half as f32).exp()).collect();
        let mut v = Vec::with_capacity(t.len() * 256);
        for &ti in t {
            let ts = 1000.0 * ti;
            let args: Vec<f32> = freqs.iter().map(|f| ts * f).collect();
            v.extend(args.iter().map(|a| a.cos()));
            v.extend(args.iter().map(|a| a.sin()));
        }
        let proj = Tensor::from_vec(v, (t.len(), 256), &self.dev)?.to_dtype(DType::BF16)?;
        self.t_lin2.forward(&self.t_lin1.forward(&proj)?.silu()?)
    }

    fn txt_in(&self, e: &Tensor) -> Result<Tensor> {
        // zero-centred RMSNorm (scale = w + 1) in fp32, GELU(tanh) MLP
        let xf = e.to_dtype(DType::F32)?;
        let r = (xf.sqr()?.mean_keepdim(D::Minus1)? + EPS)?.sqrt()?.recip()?;
        let n = xf.broadcast_mul(&r)?.broadcast_mul(&(&self.txt_norm + 1.0)?)?.to_dtype(DType::BF16)?;
        self.txt_in2.forward(&self.txt_in1.forward(&n)?.gelu()?)
    }

    /// q, k [b, h, s, d] after head RMSNorm + RoPE (fused), v [b, h, s, d].
    fn qkv(blk: &Block, x: &Tensor, cs: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, s, _) = x.dims3()?;
        let q = fused::qk_norm_rope(&blk.q.forward(x)?.reshape((b, s, HEADS, HEAD_DIM))?, &blk.norm_q, cs)?;
        let k = fused::qk_norm_rope(&blk.k.forward(x)?.reshape((b, s, HEADS, HEAD_DIM))?, &blk.norm_k, cs)?;
        let v = blk.v.forward(x)?.reshape((b, s, HEADS, HEAD_DIM))?.transpose(1, 2)?.contiguous()?;
        Ok((q, k, v))
    }

    /// One block on tokens x with modulation rows [b|1, 1, 16384]; `attend` gives the attention output [b, s, 4096].
    fn block(blk: &Block, x: &Tensor, m: &Tensor, attend: impl FnOnce(&Tensor) -> Result<Tensor>) -> Result<Tensor> {
        let b = x.dim(0)?;
        let ch = |i: usize| -> Result<Tensor> {
            let t = m.narrow(D::Minus1, i * DIM, DIM)?;
            Ok(if t.dim(0)? == 1 && b > 1 { t.repeat((b, 1, 1))? } else { t.contiguous()? })
        };
        let (s1, g1, s2, g2) = (ch(0)?, ch(1)?, ch(2)?, ch(3)?);
        let h = fused::ln_mod(x, &s1)?;
        let a = attend(&h)?;
        let x = fused::gated_residual(x, &a, &g1)?;
        let h2 = fused::ln_mod(&x, &s2)?;
        let mlp = blk.out.forward(&fused::swiglu(&blk.gate_up.forward(&h2)?)?)?;
        fused::gated_residual(&x, &mlp, &g2)
    }

    /// Run the prompt tokens once: embeds [b, l, 4096] (right-padded with zeros), valid [b, l] u8.
    pub fn prefill(&self, embeds: &Tensor, valid: &Tensor) -> Result<TextCache> {
        let (b, l, _) = embeds.dims3()?;
        let pos: Vec<[i64; 3]> = (0..l as i64).map(|p| [p, p, p]).collect();
        let (cos, sin) = rope_tables(&pos, &self.dev)?;
        let cs = Tensor::cat(&[&cos, &sin], 1)?.contiguous()?;
        let m0 = self.modulation.forward(&self.temb(&[0.0])?.silu()?)?.reshape((1, 1, 4 * DIM))?;
        let mut x = self.txt_in(embeds)?;
        let (mut ks, mut vs) = (vec![], vec![]);
        for (i, blk) in self.blocks.iter().enumerate() {
            if let Some(s) = &self.stream {
                s.prepare(i)?;
            }
            let mut kv = None;
            x = Self::block(blk, &x, &m0, |h| {
                let (q, k, v) = Self::qkv(blk, h, &cs)?;
                let o = sdpa(&q, &k, &v, Some(valid), true)?;
                kv = Some((k, v));
                blk.o.forward(&o.transpose(1, 2)?.reshape((b, l, DIM))?)
            })?;
            let (k, v) = kv.context("no kv")?;
            ks.push(k.contiguous()?);
            vs.push(v.contiguous()?);
            if let Some(s) = &self.stream {
                s.done(i)?;
            }
        }
        Ok(TextCache { k: ks, v: vs, valid: valid.clone(), len: l })
    }

    /// Velocity for the target latents [b, 64, h, w] at timesteps t (bf16-rounded, one per sample).
    pub fn step(&self, cache: &TextCache, latents: &Tensor, t: &[f32]) -> Result<Tensor> {
        let (b, c, lh, lw) = latents.dims4()?;
        let n = lh * lw;
        let x0 = latents.to_dtype(DType::BF16)?.reshape((b, c, n))?.transpose(1, 2)?.contiguous()?;
        let mut x = self.img_in.forward(&x0)?;
        // image block: frame frozen at the text length, height/width centred on zero
        let f = cache.len as i64;
        let (hh, ww) = (lh as i64, lw as i64);
        let mut pos = Vec::with_capacity(n);
        for i in -(hh - hh / 2)..hh / 2 {
            for j in -(ww - ww / 2)..ww / 2 {
                pos.push([f, i, j]);
            }
        }
        let (cos, sin) = rope_tables(&pos, &self.dev)?;
        let cs = Tensor::cat(&[&cos, &sin], 1)?.contiguous()?;
        let temb = self.temb(t)?;
        let m = self.modulation.forward(&temb.silu()?)?.reshape((b, 1, 4 * DIM))?;
        let img_valid = Tensor::ones((b, n), DType::U8, &self.dev)?;
        let valid = Tensor::cat(&[&cache.valid, &img_valid], 1)?;
        for (i, blk) in self.blocks.iter().enumerate() {
            crate::core::check_abort()?;
            if let Some(s) = &self.stream {
                s.prepare(i)?;
            }
            x = Self::block(blk, &x, &m, |h| {
                let (q, k, v) = Self::qkv(blk, h, &cs)?;
                let k = Tensor::cat(&[&cache.k[i], &k], 2)?;
                let v = Tensor::cat(&[&cache.v[i], &v], 2)?;
                let o = sdpa(&q, &k, &v, Some(&valid), false)?;
                blk.o.forward(&o.transpose(1, 2)?.reshape((b, n, DIM))?)
            })?;
            if let Some(s) = &self.stream {
                s.done(i)?;
            }
        }
        let scale = self.norm_out.forward(&temb.silu()?)?.reshape((b, 1, DIM))?;
        let y = self.proj_out.forward(&fused::ln_mod(&x, &scale)?)?;
        Ok(y.transpose(1, 2)?.reshape((b, c, lh, lw))?)
    }
}
