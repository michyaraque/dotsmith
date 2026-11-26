//! Krea 2 (Raw / Turbo) single-stream MMDiT, ported from ai-toolkit's extensions_built_in/diffusion_models/krea2/src
//! (mmdit.py + pipeline.py). 28 blocks, width 6144, 48 query / 12 KV heads x 128 (GQA), SwiGLU 16384, patch 2 over the
//! 16-channel Qwen-Image latent. Weights: Comfy-Org/Krea-2's int8 ConvRot export.
//!
//! Text: 12 stacked Qwen3-VL-4B hidden states per token -> TextFusionTransformer (2 blocks across the layer axis,
//! projector 12 -> 1, 2 refiner blocks across tokens) -> txtmlp -> width 6144, then [text, image] run as ONE sequence
//! through every block with the same timestep modulation (unlike Qwen 2.1, no text K/V cache across steps).
//! Numerics follow the reference: RMSNorm in fp32 with weight (scale + 1), eps 1e-5; RoPE in fp32 on interleaved
//! pairs, theta 1e3, axes [32, 48, 48] over (0, row, col) for image tokens and (0, 0, 0) for text; attention output
//! times sigmoid(gate(x)); everything else bf16. Padding is masked on keys only (padded queries never feed a valid one).
use anyhow::Result;
use candle_core::{D, DType, Device, Tensor};

use crate::core::act_dtype;
use crate::core::attention::sdpa;
use crate::core::int8::{AnyLinear, Linear, Lora};
use crate::core::memory::{Stack, StackPlan};
use crate::core::stream::StreamedStack;
use crate::core::weights::Weights;

const HEAD_DIM: usize = 128;
const THETA: f64 = 1e3;
const EPS: f64 = 1e-5;
/// The int8 linears of one DiT block, as packed by the weight streamer.
const BLOCK_LINEARS: [&str; 8] = ["attn.wq", "attn.wk", "attn.wv", "attn.gate", "attn.wo", "mlp.gate", "mlp.up", "mlp.down"];

/// Where one block's linears live: on the GPU, in RAM (GGUF swap) or streamed from pinned RAM into a slot.
#[derive(Clone, Copy)]
enum Place<'a> {
    Gpu,
    /// in RAM, uploaded per call (layers that run once per step or per prompt, when streaming)
    Host,
    Swap,
    Stream(&'a StreamedStack, usize),
}

impl Place<'_> {
    fn load(&self, wt: &Weights, p: &str, n: &str, dev: &Device) -> Result<AnyLinear> {
        let full = format!("{p}.{n}");
        match self {
            Place::Gpu => AnyLinear::load(wt, &full, dev),
            Place::Host => AnyLinear::load(wt, &full, &Device::Cpu),
            Place::Swap => AnyLinear::load_swapped(wt, &full, dev),
            Place::Stream(s, i) => {
                let block = p.rsplit_once('.').map_or(p, |(b, _)| b); // "blocks.3.attn" -> "blocks.3"
                s.linear(wt, *i, block, &format!("{}.{n}", p.rsplit('.').next().unwrap()), dev)
            }
        }
    }
}

/// RMSNorm with weight (scale + 1), computed in fp32, back to bf16.
struct Rms {
    w: Tensor, // f32, already + 1
}

impl Rms {
    fn load(wt: &Weights, p: &str, dev: &Device) -> Result<Self> {
        Ok(Self { w: (wt.tensor(&format!("{p}.scale"), dev)?.to_dtype(DType::F32)? + 1.0)? })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let xf = x.to_dtype(DType::F32)?;
        let r = (xf.sqr()?.mean_keepdim(D::Minus1)? + EPS)?.sqrt()?.recip()?;
        Ok(xf.broadcast_mul(&r)?.broadcast_mul(&self.w)?.to_dtype(act_dtype())?)
    }
}

struct Attn {
    wq: AnyLinear,
    wk: AnyLinear,
    wv: AnyLinear,
    gate: AnyLinear,
    wo: AnyLinear,
    qn: Rms,
    kn: Rms,
    heads: usize,
    kvheads: usize,
}

impl Attn {
    /// `place`: GPU, GGUF swap (weights stay in RAM, uploaded per call) or int8 streaming (core::stream)
    fn load(wt: &Weights, p: &str, dev: &Device, place: Place) -> Result<Self> {
        let lin = |n: &str| place.load(wt, p, n, dev);
        let wq = lin("wq")?;
        let wk = lin("wk")?;
        let heads = wq.out_features() / HEAD_DIM;
        let kvheads = wk.out_features() / HEAD_DIM;
        Ok(Self {
            wq,
            wk,
            wv: lin("wv")?,
            gate: lin("gate")?,
            wo: lin("wo")?,
            qn: Rms::load(wt, &format!("{p}.qknorm.qnorm"), dev)?,
            kn: Rms::load(wt, &format!("{p}.qknorm.knorm"), dev)?,
            heads,
            kvheads,
        })
    }

    fn linears(&mut self) -> [(&'static str, &mut AnyLinear); 5] {
        [("wq", &mut self.wq), ("wk", &mut self.wk), ("wv", &mut self.wv), ("gate", &mut self.gate), ("wo", &mut self.wo)]
    }

    /// x [b, s, dim]; `rope` (cos, sin) [s, 64] f32 or None; `valid` [b, s] u8 key mask or None.
    fn forward(&self, x: &Tensor, rope: Option<&(Tensor, Tensor)>, valid: Option<&Tensor>) -> Result<Tensor> {
        let (b, s, dim) = x.dims3()?;
        let heads = |t: Tensor, h: usize| -> Result<Tensor> { Ok(t.reshape((b, s, h, HEAD_DIM))?.transpose(1, 2)?) };
        let mut q = self.qn.forward(&heads(self.wq.forward(x)?, self.heads)?)?;
        let mut k = self.kn.forward(&heads(self.wk.forward(x)?, self.kvheads)?)?;
        let v = heads(self.wv.forward(x)?, self.kvheads)?;
        if let Some((cos, sin)) = rope {
            q = rope_apply(&q, cos, sin)?;
            k = rope_apply(&k, cos, sin)?;
        }
        let (k, v) = if self.kvheads != self.heads {
            // enable_gqa: query head h reads kv head h / (heads / kvheads)
            let rep = self.heads / self.kvheads;
            let r = |t: &Tensor| -> Result<Tensor> { Ok(t.unsqueeze(2)?.repeat((1, 1, rep, 1, 1))?.reshape((b, self.heads, s, HEAD_DIM))?) };
            (r(&k)?, r(&v)?)
        } else {
            (k, v)
        };
        let o = sdpa(&q.contiguous()?, &k.contiguous()?, &v.contiguous()?, valid, false)?;
        let o = o.transpose(1, 2)?.reshape((b, s, self.heads * HEAD_DIM))?;
        let g = candle_nn::ops::sigmoid(&self.gate.forward(x)?)?;
        let _ = dim;
        self.wo.forward(&(o * g)?)
    }
}

/// Interleaved-pair RoPE in fp32 (ropeapply): (x0, x1) -> (cos x0 - sin x1, sin x0 + cos x1); cos/sin [s, 64].
fn rope_apply(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b, h, s, d) = x.dims4()?;
    let xf = x.to_dtype(DType::F32)?.reshape((b, h, s, d / 2, 2))?;
    let x0 = xf.narrow(4, 0, 1)?.squeeze(4)?;
    let x1 = xf.narrow(4, 1, 1)?.squeeze(4)?;
    let o0 = (x0.broadcast_mul(cos)? - x1.broadcast_mul(sin)?)?;
    let o1 = (x0.broadcast_mul(sin)? + x1.broadcast_mul(cos)?)?;
    Ok(Tensor::stack(&[o0, o1], 4)?.reshape((b, h, s, d))?.to_dtype(x.dtype())?)
}

/// RoPE tables for (axis0, row, col) positions: axes [32, 48, 48] of the 128 head dims, theta 1e3 -> (cos, sin) [s, 64].
fn rope_tables(pos: &[[f64; 3]], dev: &Device) -> Result<(Tensor, Tensor)> {
    let hd = HEAD_DIM;
    let axes = [hd - 12 * (hd / 16), 6 * (hd / 16), 6 * (hd / 16)];
    let omegas: Vec<Vec<f64>> = axes.iter().map(|&d| (0..d).step_by(2).map(|i| 1.0 / THETA.powf(i as f64 / d as f64)).collect()).collect();
    let (mut c, mut s) = (Vec::with_capacity(pos.len() * hd / 2), Vec::with_capacity(pos.len() * hd / 2));
    for p in pos {
        for (a, om) in omegas.iter().enumerate() {
            for w in om {
                let ang = p[a] * w; // float64 like the reference, then .float()
                c.push(ang.cos() as f32);
                s.push(ang.sin() as f32);
            }
        }
    }
    let n = pos.len();
    Ok((Tensor::from_vec(c, (n, hd / 2), dev)?, Tensor::from_vec(s, (n, hd / 2), dev)?))
}

struct SwiGlu {
    gate: AnyLinear,
    up: AnyLinear,
    down: AnyLinear,
}

impl SwiGlu {
    fn load(wt: &Weights, p: &str, dev: &Device, place: Place) -> Result<Self> {
        let lin = |n: &str| place.load(wt, p, n, dev);
        Ok(Self { gate: lin("gate")?, up: lin("up")?, down: lin("down")? })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let s = x.dim(1)?;
        const CHUNK: usize = 2048; // low VRAM: the [s, 16384] SwiGLU intermediates in token chunks
        if !crate::core::low_vram() || s <= CHUNK {
            return self.down.forward(&(self.gate.forward(x)?.silu()? * self.up.forward(x)?)?);
        }
        let mut outs = vec![];
        for s0 in (0..s).step_by(CHUNK) {
            let xc = x.narrow(1, s0, CHUNK.min(s - s0))?;
            outs.push(self.down.forward(&(self.gate.forward(&xc)?.silu()? * self.up.forward(&xc)?)?)?);
        }
        Ok(Tensor::cat(&outs, 1)?)
    }
    fn linears(&mut self) -> [(&'static str, &mut AnyLinear); 3] {
        [("gate", &mut self.gate), ("up", &mut self.up), ("down", &mut self.down)]
    }
}

struct FusionBlock {
    pre: Rms,
    post: Rms,
    attn: Attn,
    mlp: SwiGlu,
}

impl FusionBlock {
    fn load(wt: &Weights, p: &str, dev: &Device, place: Place) -> Result<Self> {
        Ok(Self {
            pre: Rms::load(wt, &format!("{p}.prenorm"), dev)?,
            post: Rms::load(wt, &format!("{p}.postnorm"), dev)?,
            attn: Attn::load(wt, &format!("{p}.attn"), dev, place)?,
            mlp: SwiGlu::load(wt, &format!("{p}.mlp"), dev, place)?,
        })
    }
    fn forward(&self, x: &Tensor, valid: Option<&Tensor>) -> Result<Tensor> {
        let x = (x + self.attn.forward(&self.pre.forward(x)?, None, valid)?)?;
        Ok((&x + self.mlp.forward(&self.post.forward(&x)?)?)?)
    }
}

struct Block {
    lin: Tensor, // DoubleSharedModulation [6 * dim] bf16
    pre: Rms,
    post: Rms,
    attn: Attn,
    mlp: SwiGlu,
}

pub struct Krea {
    first: Linear,
    blocks: Vec<Block>,
    t1: Linear,
    t2: Linear,
    tproj: Linear,
    layerwise: Vec<FusionBlock>,
    projector: Tensor, // [1, 12] bf16 (with a LoRA delta merged in when the LoRA has one)
    projector_base: Tensor,
    refiner: Vec<FusionBlock>,
    txt_norm: Rms,
    txt1: Linear,
    txt2: Linear,
    last_norm: Rms,
    last_lin: Tensor, // SimpleModulation [2, dim]
    last: Linear,
    pub dim: usize,
    dev: Device,
    /// blocks placed by the memory plan: `first..` stream from host RAM (None = all on the GPU)
    stream: Option<StreamedStack>,
}

/// Prepared text for one batch: fused features [b, l, dim] and the key mask [b, l] u8 (shared by every step).
pub struct KreaText {
    pub feats: Tensor,
    pub valid: Tensor,
}

fn temb(t: &[f32], dev: &Device) -> Result<Tensor> {
    // t is the bf16-rounded timestep in [0, 1]; args = (t * 1000) * exp(-ln(1e4) * i / 128); cat(cos, sin)
    let half = 128;
    let freqs: Vec<f32> = (0..half).map(|i| (-(10000f32).ln() * i as f32 / half as f32).exp()).collect();
    let mut v = Vec::with_capacity(t.len() * 2 * half);
    for &ti in t {
        let args: Vec<f32> = freqs.iter().map(|f| ti * 1000.0 * f).collect();
        v.extend(args.iter().map(|a| a.cos()));
        v.extend(args.iter().map(|a| a.sin()));
    }
    Ok(Tensor::from_vec(v, (t.len(), 1, 2 * half), dev)?.to_dtype(act_dtype())?)
}

fn gelu_tanh(x: &Tensor) -> Result<Tensor> {
    Ok(x.gelu()?) // candle's gelu is the tanh approximation
}

impl Krea {
    /// True when the int8 blocks stream from pinned RAM (the DiT then holds ~1 GB of VRAM, not ~13).
    pub fn streaming(&self) -> bool {
        self.stream.is_some()
    }

    /// Bytes uploaded by the weight streamer so far.
    pub fn streamed_bytes(&self) -> usize {
        self.stream.as_ref().map_or(0, |s| s.stream.uploaded.load(std::sync::atomic::Ordering::Relaxed))
    }

    /// For core::memory::plan: the block stack (None if its linears are not int8, e.g. GGUF) and (fixed, cold) bytes:
    /// fixed = always on the GPU (norms, modulation, first/last), cold = moved to RAM while streaming.
    pub fn memory(path: &str) -> Result<(Option<Stack>, u64, u64)> {
        let wt = Weights::open(path)?;
        let n = (0..).take_while(|i| wt.has(&format!("blocks.{i}.mod.lin"))).count();
        let stack = wt.streamable_layer_bytes("blocks.0", &BLOCK_LINEARS).map(|b| Stack { name: "DiT", layers: n, layer_bytes: b, hot: true });
        let in_stack = |k: &str| k.starts_with("blocks.") && BLOCK_LINEARS.iter().any(|l| k.ends_with(&format!(".{l}.weight")));
        let cold = wt.bytes_where(|k| ["tproj.", "tmlp.", "txtmlp.", "txtfusion."].iter().any(|c| k.starts_with(c)));
        let total = wt.bytes_where(|_| true);
        let stack_bytes = if stack.is_some() { wt.bytes_where(in_stack) } else { 0 };
        Ok((stack, total - stack_bytes - cold, cold))
    }

    /// Fully on the GPU.
    pub fn load(path: &str, dev: &Device) -> Result<Self> {
        Self::load_planned(path, dev, StackPlan::GPU)
    }

    pub fn load_planned(path: &str, dev: &Device, plan: StackPlan) -> Result<Self> {
        let wt = Weights::open(path)?;
        let lin = |p: &str, bias: bool| Linear::load(&wt, p, dev, bias);
        let mut blocks = vec![];
        let max_blocks: usize = std::env::var("KREA_BLOCKS").ok().and_then(|v| v.parse().ok()).unwrap_or(usize::MAX);
        let total = (0..).take_while(|i| wt.has(&format!("blocks.{i}.mod.lin"))).count();
        // low-VRAM block swap (GGUF only): the last N blocks keep their quantized weights in RAM
        let swap_n: usize = std::env::var("DOTSMITH_SWAP_BLOCKS").ok().and_then(|v| v.parse().ok()).unwrap_or(if crate::core::low_vram() { 8 } else { 0 });
        let n_blocks = total.min(max_blocks);
        // memory plan: blocks plan.resident.. stream from host RAM (int8 checkpoints only)
        let plan = if wt.streamable_layer_bytes("blocks.0", &BLOCK_LINEARS).is_some() { plan } else { StackPlan::GPU };
        let ids: Vec<String> = (0..n_blocks).map(|i| format!("blocks.{i}")).collect();
        let stream = StreamedStack::build(&wt, std::path::Path::new(path), dev, &ids, &BLOCK_LINEARS, plan, true)?;
        crate::core::memory::set_report(match &stream {
            Some(s) => s.placed("dit", n_blocks),
            None => crate::core::memory::Placed::gpu("dit", n_blocks, wt.layer_format("blocks.0", &BLOCK_LINEARS)),
        });
        let mut i = 0;
        while i < max_blocks && wt.has(&format!("blocks.{i}.mod.lin")) {
            let swap = i + swap_n >= total;
            let place = match &stream {
                Some(s) if s.covers(i) => Place::Stream(s, i),
                _ if swap => Place::Swap,
                _ => Place::Gpu,
            };
            let p = format!("blocks.{i}");
            blocks.push(Block {
                lin: wt.tensor_act(&format!("{p}.mod.lin"), dev)?,
                pre: Rms::load(&wt, &format!("{p}.prenorm"), dev)?,
                post: Rms::load(&wt, &format!("{p}.postnorm"), dev)?,
                attn: Attn::load(&wt, &format!("{p}.attn"), dev, place)?,
                mlp: SwiGlu::load(&wt, &format!("{p}.mlp"), dev, place)?,
            });
            i += 1;
        }
        // streaming: the cold layers (timestep MLPs, once per step; text fusion, once per prompt; ~1 GB) stay in RAM
        // and are uploaded per call; `first` / `last` touch every image token, so they stay on the GPU
        let (cold, cold_place) = if stream.is_some() { (Device::Cpu, Place::Host) } else { (dev.clone(), Place::Gpu) };
        let cold_lin = |p: &str| Linear::load(&wt, p, &cold, true);
        let fb =
            |name: &str| -> Result<Vec<FusionBlock>> { (0..2).map(|j| FusionBlock::load(&wt, &format!("txtfusion.{name}.{j}"), dev, cold_place)).collect() };
        let first = lin("first", true)?;
        let dim = first.w.dim(0)?;
        Ok(Self {
            first,
            blocks,
            t1: cold_lin("tmlp.0")?,
            t2: cold_lin("tmlp.2")?,
            tproj: cold_lin("tproj.1")?,
            layerwise: fb("layerwise_blocks")?,
            projector: wt.tensor_act("txtfusion.projector.weight", dev)?,
            projector_base: wt.tensor_act("txtfusion.projector.weight", dev)?,
            refiner: fb("refiner_blocks")?,
            txt_norm: Rms::load(&wt, "txtmlp.0", dev)?,
            txt1: cold_lin("txtmlp.1")?,
            txt2: cold_lin("txtmlp.3")?,
            last_norm: Rms::load(&wt, "last.norm", dev)?,
            last_lin: wt.tensor_act("last.modulation.lin", dev)?,
            last: lin("last.linear", true)?,
            dim,
            dev: dev.clone(),
            stream,
        })
    }

    /// Attach an ai-toolkit Krea LoRA (`diffusion_model.blocks.N.{attn,mlp}.*` and `diffusion_model.txtfusion.*`).
    /// Loads a LoRA in either key format: ai-toolkit / diffusers (`diffusion_model.<name>.lora_A|B.weight`, ours) or
    /// kohya / sd-scripts (`lora_unet_<name with _>.lora_down|up.weight` + `.alpha`, e.g. Civitai Krea 2 LoRAs). Every
    /// linear the reference applies LoRAs to is covered, the single-stream blocks, the text fusion blocks and the
    /// small embedders (first, tmlp, tproj, txtmlp, last.linear); the 12 -> 1 text projector is a plain tensor, so its
    /// delta is merged into the weight (and undone on the next call). Returns the number of layers patched.
    pub fn set_lora(&mut self, path: Option<&str>, multiplier: f64) -> Result<usize> {
        let mut all: Vec<(String, &mut Vec<Lora>)> = vec![];
        for (i, b) in self.blocks.iter_mut().enumerate() {
            for (n, l) in b.attn.linears() {
                all.push((format!("blocks.{i}.attn.{n}"), l.loras_mut()));
            }
            for (n, l) in b.mlp.linears() {
                all.push((format!("blocks.{i}.mlp.{n}"), l.loras_mut()));
            }
        }
        for (name, list) in [("layerwise_blocks", &mut self.layerwise), ("refiner_blocks", &mut self.refiner)] {
            for (j, b) in list.iter_mut().enumerate() {
                for (n, l) in b.attn.linears() {
                    all.push((format!("txtfusion.{name}.{j}.attn.{n}"), l.loras_mut()));
                }
                for (n, l) in b.mlp.linears() {
                    all.push((format!("txtfusion.{name}.{j}.mlp.{n}"), l.loras_mut()));
                }
            }
        }
        for (n, l) in [
            ("first", &mut self.first),
            ("tmlp.0", &mut self.t1),
            ("tmlp.2", &mut self.t2),
            ("tproj.1", &mut self.tproj),
            ("txtmlp.1", &mut self.txt1),
            ("txtmlp.3", &mut self.txt2),
            ("last.linear", &mut self.last),
        ] {
            all.push((n.to_string(), &mut l.loras));
        }
        for (_, l) in all.iter_mut() {
            l.clear();
        }
        self.projector = self.projector_base.clone();
        let Some(path) = path else { return Ok(0) };
        let wt = Weights::open(path)?;
        let dev = self.dev.clone();
        // (down, up, scale) of one layer in whichever format the file uses
        let pair = |name: &str| -> Result<Option<(Tensor, Tensor, f64)>> {
            let ai = format!("diffusion_model.{name}");
            let ko = format!("lora_unet_{}", name.replace('.', "_"));
            let (p, d, u) = if wt.has(&format!("{ai}.lora_A.weight")) {
                (ai, "lora_A", "lora_B")
            } else if wt.has(&format!("{ko}.lora_down.weight")) {
                (ko, "lora_down", "lora_up")
            } else {
                return Ok(None);
            };
            let down = wt.tensor_act(&format!("{p}.{d}.weight"), &dev)?;
            let up = wt.tensor_act(&format!("{p}.{u}.weight"), &dev)?;
            let rank = down.dim(0)? as f64;
            let alpha = match wt.tensor(&format!("{p}.alpha"), &Device::Cpu) {
                Ok(a) => a.to_dtype(DType::F64)?.flatten_all()?.to_vec1::<f64>()?[0],
                Err(_) => rank,
            };
            Ok(Some((down, up, multiplier * alpha / rank)))
        };
        let mut n = 0;
        for (name, l) in all {
            if let Some((down, up, scale)) = pair(&name)? {
                l.push(Lora { down, up, scale });
                n += 1;
            }
        }
        if let Some((down, up, scale)) = pair("txtfusion.projector")? {
            let delta = (up.to_dtype(DType::F32)?.matmul(&down.to_dtype(DType::F32)?)? * scale)?; // [1, 12]
            self.projector = (self.projector_base.to_dtype(DType::F32)? + delta)?.to_dtype(act_dtype())?;
            n += 1;
        }
        Ok(n)
    }

    /// Stacked text features [l, 12, 2560] per prompt (right-padded to one length) -> fused [b, l, dim] + key mask.
    pub fn prepare_text(&self, prompts: &[Tensor]) -> Result<KreaText> {
        let l = prompts.iter().map(|p| p.dim(0).unwrap()).max().unwrap_or(1).max(1);
        let (mut rows, mut valid) = (vec![], vec![]);
        for p in prompts {
            let n = p.dim(0)?;
            rows.push(p.to_dtype(act_dtype())?.pad_with_zeros(0, 0, l - n)?);
            valid.extend((0..l).map(|i| (i < n) as u8));
        }
        let b = prompts.len();
        let x = Tensor::stack(&rows, 0)?; // [b, l, 12, d]
        let (_, _, nl, d) = x.dims4()?;
        let valid = Tensor::from_vec(valid, (b, l), &self.dev)?;
        // layer-axis blocks: every token's 12 layer states attend to each other, no mask, no rope
        let mut h = x.reshape((b * l, nl, d))?;
        for blk in &self.layerwise {
            h = blk.forward(&h, None)?;
        }
        // projector over the layer axis: [b*l, nl, d] -> [b*l, d] (Linear(nl -> 1) on the transposed view)
        let h = self.projector.unsqueeze(0)?.repeat((b * l, 1, 1))?.matmul(&h)?.reshape((b, l, d))?;
        let mut h = h;
        for blk in &self.refiner {
            h = blk.forward(&h, Some(&valid))?;
        }
        let h = self.txt2.forward(&gelu_tanh(&self.txt1.forward(&self.txt_norm.forward(&h)?)?)?)?;
        Ok(KreaText { feats: h, valid })
    }

    /// Velocity for latents [b, 16, h, w] (f32 or bf16) at timesteps t (bf16-rounded, one per sample).
    pub fn step(&self, text: &KreaText, latents: &Tensor, t: &[f32]) -> Result<Tensor> {
        let (b, c, h, w) = latents.dims4()?;
        let (hp, wp) = (h / 2, w / 2);
        let n = hp * wp;
        // patchify: b c (h ph) (w pw) -> b (h w) (c ph pw)
        let img = latents.to_dtype(act_dtype())?.reshape((b, c, hp, 2, wp, 2))?.permute((0, 2, 4, 1, 3, 5))?.reshape((b, n, c * 4))?;
        let img = self.first.forward(&img.contiguous()?)?;
        let tv = self.t2.forward(&gelu_tanh(&self.t1.forward(&temb(t, &self.dev)?)?)?)?; // [b, 1, dim]
        let tvec = self.tproj.forward(&gelu_tanh(&tv)?)?; // [b, 1, 6 dim]
        let tl = text.feats.dim(1)?;
        // lockstep batches: b = n images x the prepared prompt rows, rows in [cond x n, uncond x n] order
        let bt = text.feats.dim(0)?;
        let (feats, tvalid) = if b != bt {
            anyhow::ensure!(b % bt == 0, "batch {b} is not a multiple of the {bt} prompt rows");
            let n = b / bt;
            let d = text.feats.dim(2)?;
            (text.feats.unsqueeze(1)?.repeat((1, n, 1, 1))?.reshape((b, tl, d))?, text.valid.unsqueeze(1)?.repeat((1, n, 1))?.reshape((b, tl))?)
        } else {
            (text.feats.clone(), text.valid.clone())
        };
        let mut x = Tensor::cat(&[&feats, &img], 1)?;
        let mut pos = vec![[0f64; 3]; tl];
        for i in 0..hp {
            for j in 0..wp {
                pos.push([0.0, i as f64, j as f64]);
            }
        }
        let rope = rope_tables(&pos, &self.dev)?;
        let valid = Tensor::cat(&[&tvalid, &Tensor::ones((b, n), DType::U8, &self.dev)?], 1)?;
        let d = self.dim;
        for (bi, blk) in self.blocks.iter().enumerate() {
            crate::core::check_abort()?;
            if let Some(s) = &self.stream {
                s.prepare(bi)?;
            }
            let m = tvec.broadcast_add(&blk.lin)?; // [b, 1, 6d]
            let ch = |i: usize| m.narrow(D::Minus1, i * d, d);
            let (s1, sh1, g1, s2, sh2, g2) = (ch(0)?, ch(1)?, ch(2)?, ch(3)?, ch(4)?, ch(5)?);
            let hpre = blk.pre.forward(&x)?.broadcast_mul(&(s1 + 1.0)?)?.broadcast_add(&sh1)?;
            let a = blk.attn.forward(&hpre, Some(&rope), Some(&valid))?;
            x = (&x + a.broadcast_mul(&g1)?)?;
            let hpost = blk.post.forward(&x)?.broadcast_mul(&(s2 + 1.0)?)?.broadcast_add(&sh2)?;
            x = (&x + blk.mlp.forward(&hpost)?.broadcast_mul(&g2)?)?;
            if let Some(s) = &self.stream {
                s.done(bi)?;
            }
        }
        if std::env::var("DOTSMITH_STREAM_STATS").is_ok() {
            // wall time per step, GPU included (synchronizes, debug only)
            use std::sync::atomic::Ordering::Relaxed;
            static LAST: std::sync::Mutex<Option<std::time::Instant>> = std::sync::Mutex::new(None);
            self.dev.synchronize()?;
            let now = std::time::Instant::now();
            let dt = LAST.lock().unwrap().replace(now).map_or(0.0, |t| (now - t).as_secs_f64());
            match &self.stream {
                Some(s) => eprintln!(
                    "  step {dt:.2} s | stream: {:.1} GB uploaded, staging CPU {:.2} s (cumulative)",
                    s.stream.uploaded.load(Relaxed) as f64 / 1e9,
                    s.stream.stage_ns.load(Relaxed) as f64 / 1e9
                ),
                None => eprintln!("  step {dt:.2} s"),
            }
        }
        // LastLayer: SimpleModulation on the tmlp output (not tvec): (t + lin) -> scale, shift
        let lm = tv.broadcast_add(&self.last_lin.unsqueeze(0)?)?; // [b, 2, dim]
        let (scale, shift) = (lm.narrow(1, 0, 1)?, lm.narrow(1, 1, 1)?);
        let y = self.last_norm.forward(&x)?.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(&shift)?;
        let y = self.last.forward(&y)?.narrow(1, tl, n)?; // [b, n, c*4]
        // unpatchify: b (h w) (c ph pw) -> b c (h ph) (w pw)
        Ok(y.reshape((b, hp, wp, c, 2, 2))?.permute((0, 3, 1, 4, 2, 5))?.reshape((b, c, h, w))?)
    }
}
