//! Qwen3-VL language tower (36 layers, 32 query / 8 KV heads x 128, RoPE theta 5e6; 8B = 4096 wide for Qwen-Image 2.1,
//! 4B = 2560 wide for Krea 2), ComfyUI convrot int8. The width comes from the weights.
//! Text-only prompts: Qwen3-VL's interleaved MRoPE gets equal (t, h, w) positions and reduces to plain RoPE.
//! Output = the last decoder layer BEFORE the final norm (what Qwen-Image 2.1 was trained on), system turn dropped.
//! The vision tower and lm_head are never loaded; the int8 embedding table stays in the memory map (rows on demand).
use anyhow::{Context, Result};
use candle_core::{D, DType, Device, Tensor};
use tokenizers::Tokenizer;

use crate::core::attention::sdpa;
use crate::core::int8::{AnyLinear, rotate};
use crate::core::weights::Weights;

const HEADS: usize = 32;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const THETA: f32 = 5_000_000.0;
const EPS: f64 = 1e-6;

pub const SYSTEM_PROMPT: &str = "Comprehend and analyze the provided prompt.";

struct Layer {
    in_norm: Tensor,
    post_norm: Tensor,
    q: AnyLinear,
    k: AnyLinear,
    v: AnyLinear,
    o: AnyLinear,
    q_norm: Tensor,
    k_norm: Tensor,
    gate: AnyLinear,
    up: AnyLinear,
    down: AnyLinear,
}

pub struct TextEncoder {
    wt: Weights, // kept open: embedding rows are read from the map
    layers: Vec<Layer>,
    /// layers placed by the memory plan: some (or all) stream from host RAM through two GPU slots (core::stream)
    stream: Option<crate::core::stream::StreamedStack>,
    tok: Tokenizer,
    drop: usize,
    dev: Device,
    /// "model" (text-only exports), "model.language_model" (full Qwen3-VL checkpoints such as ComfyUI's) or "" (bare
    /// language-model tensors, as RD Flow 64 ships them).
    prefix: &'static str,
}

fn prefix_of(wt: &Weights) -> &'static str {
    if wt.has("model.language_model.embed_tokens.weight") {
        "model.language_model"
    } else if wt.has("model.embed_tokens.weight") {
        "model"
    } else {
        ""
    }
}

fn join(prefix: &str, rest: &str) -> String {
    if prefix.is_empty() { rest.to_string() } else { format!("{prefix}.{rest}") }
}

/// Decoder layers present in the checkpoint (36 for the full model, fewer for a cut one).
fn layer_count(wt: &Weights, prefix: &str) -> usize {
    (0..).take_while(|i| wt.has(&join(prefix, &format!("layers.{i}.input_layernorm.weight")))).count()
}

fn rms(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let xf = x.to_dtype(DType::F32)?;
    let r = (xf.sqr()?.mean_keepdim(D::Minus1)? + EPS)?.sqrt()?.recip()?;
    Ok(xf.broadcast_mul(&r)?.to_dtype(DType::BF16)?.broadcast_mul(w)?)
}

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let d = x.dim(D::Minus1)? / 2;
    Ok(Tensor::cat(&[&x.narrow(D::Minus1, d, d)?.neg()?, &x.narrow(D::Minus1, 0, d)?], D::Minus1)?)
}

fn template(prompt: &str) -> String {
    let p = if prompt.is_empty() { " " } else { prompt }; // Qwen has no bos: an empty prompt leaves nothing to read
    format!("<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n")
}

const LINEARS: [&str; 7] = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"];

impl TextEncoder {
    /// Fully on the GPU.
    pub fn load(weights: &str, tokenizer_json: &str, dev: &Device) -> Result<Self> {
        Self::load_planned(weights, tokenizer_json, dev, crate::core::memory::StackPlan::GPU)
    }

    /// The layer stack this encoder streams (for core::memory::plan): None when its linears are not int8.
    pub fn stack(weights: &str) -> Result<Option<crate::core::memory::Stack>> {
        let wt = Weights::open(weights)?;
        let prefix = prefix_of(&wt);
        let n = layer_count(&wt, prefix);
        Ok(wt.streamable_layer_bytes(&join(prefix, "layers.0"), &LINEARS).map(|b| crate::core::memory::Stack {
            name: "text encoder",
            layers: n,
            layer_bytes: b,
            hot: false,
        }))
    }

    /// Placed by a memory plan: layers `plan.resident..` stream from host RAM when `plan.stream`.
    pub fn load_planned(weights: &str, tokenizer_json: &str, dev: &Device, plan: crate::core::memory::StackPlan) -> Result<Self> {
        let wt = Weights::open(weights)?;
        let prefix = prefix_of(&wt);
        let n = layer_count(&wt, prefix);
        let ids: Vec<String> = (0..n).map(|i| join(prefix, &format!("layers.{i}"))).collect();
        let stream = crate::core::stream::StreamedStack::build(&wt, std::path::Path::new(weights), dev, &ids, &LINEARS, plan, false)?;
        crate::core::memory::set_report(match &stream {
            Some(s) => s.placed("text_encoder", n),
            None => crate::core::memory::Placed::gpu("text_encoder", n, wt.layer_format(&join(prefix, "layers.0"), &LINEARS)),
        });
        let mut layers = vec![];
        for i in 0..n {
            let p = join(prefix, &format!("layers.{i}"));
            let t = |n: &str| -> Result<Tensor> { Ok(wt.tensor(&format!("{p}.{n}"), dev)?.to_dtype(DType::BF16)?) };
            let l = |n: &str| -> Result<AnyLinear> {
                let full = format!("{p}.{n}");
                match &stream {
                    Some(s) if s.covers(i) => s.linear(&wt, i, &p, n, dev),
                    _ => AnyLinear::load(&wt, &full, dev),
                }
            };
            layers.push(Layer {
                in_norm: t("input_layernorm.weight")?,
                post_norm: t("post_attention_layernorm.weight")?,
                q: l("self_attn.q_proj")?,
                k: l("self_attn.k_proj")?,
                v: l("self_attn.v_proj")?,
                o: l("self_attn.o_proj")?,
                q_norm: t("self_attn.q_norm.weight")?,
                k_norm: t("self_attn.k_norm.weight")?,
                gate: l("mlp.gate_proj")?,
                up: l("mlp.up_proj")?,
                down: l("mlp.down_proj")?,
            });
        }
        let tok = Tokenizer::from_file(tokenizer_json).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        let sys = format!("<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n");
        let drop = tok.encode(sys.as_str(), false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().len();
        Ok(Self { wt, layers, stream, tok, drop, dev: dev.clone(), prefix })
    }

    /// True when the layers stream from host RAM (the encoder then holds ~0.3 GB of VRAM).
    pub fn streaming(&self) -> bool {
        self.stream.is_some()
    }

    fn tokens(&self, prompt: &str) -> Result<Vec<u32>> {
        Ok(self.tok.encode(template(prompt).as_str(), false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec())
    }

    /// Embedding rows, bf16 [1, l, width]: int8 codes * row scale rotated back to the plain basis, or the plain rows.
    fn embed(&self, ids: &[u32]) -> Result<Tensor> {
        let embed = join(self.prefix, "embed_tokens");
        let (q, e) = self.wt.bytes(&format!("{embed}.weight"))?;
        if !self.wt.has(&format!("{embed}.weight_scale")) {
            let k = e.shape[1];
            let dt = match e.dtype.as_str() {
                "BF16" => DType::BF16,
                "F16" => DType::F16,
                _ => DType::F32,
            };
            let w = dt.size_in_bytes();
            let mut rows = Vec::with_capacity(ids.len() * k * w);
            for &id in ids {
                rows.extend_from_slice(&q[id as usize * k * w..(id as usize + 1) * k * w]);
            }
            return Ok(Tensor::from_raw_buffer(&rows, dt, &[ids.len(), k], &self.dev)?.to_dtype(DType::BF16)?.unsqueeze(0)?);
        }
        let (s, _) = self.wt.bytes(&format!("{embed}.weight_scale"))?;
        let k = e.shape[1];
        let mut v = Vec::with_capacity(ids.len() * k);
        for &id in ids {
            let id = id as usize;
            let scale = f32::from_le_bytes(s[id * 4..id * 4 + 4].try_into()?);
            v.extend(q[id * k..(id + 1) * k].iter().map(|&b| (b as i8) as f32 * scale));
        }
        let conf = self.wt.json(&format!("{embed}.comfy_quant")).context("embed quant")?;
        let rot = if conf["convrot"].as_bool().unwrap_or(false) {
            (conf["convrot_groupsize"].as_u64().unwrap_or(256) as usize).min(crate::core::int8::largest_pow4_divisor(k))
        } else {
            1
        };
        let rows = Tensor::from_vec(v, (ids.len(), k), &self.dev)?;
        Ok(rotate(&rows, rot)?.to_dtype(DType::BF16)?.unsqueeze(0)?)
    }

    /// Prompt -> conditioning [l - drop, 4096] bf16 (the transformer's text stream).
    pub fn encode(&self, prompt: &str) -> Result<Tensor> {
        let (x, _) = self.run(prompt, &[])?;
        let l = x.dim(0)?;
        Ok(x.narrow(0, self.drop, l - self.drop)?)
    }

    /// RoPE tables in fp32, cast to bf16 like HF: [1, 1, l, 128].
    fn rope(&self, l: usize) -> Result<(Tensor, Tensor)> {
        let inv: Vec<f32> = (0..HEAD_DIM).step_by(2).map(|i| 1.0 / THETA.powf(i as f32 / HEAD_DIM as f32)).collect();
        let mut ang = Vec::with_capacity(l * HEAD_DIM);
        for p in 0..l {
            for _ in 0..2 {
                ang.extend(inv.iter().map(|f| p as f32 * f));
            }
        }
        let ang = Tensor::from_vec(ang, (1, 1, l, HEAD_DIM), &self.dev)?;
        Ok((ang.cos()?.to_dtype(DType::BF16)?, ang.sin()?.to_dtype(DType::BF16)?))
    }

    /// RoPE + causal GQA attention: q [1, 32, l, d], k/v [1, 8, l, d] -> [1, l, 4096].
    fn attn_core(&self, q: Tensor, k: Tensor, v: Tensor, cos: &Tensor, sin: &Tensor, l: usize) -> Result<Tensor> {
        let q = (q.broadcast_mul(cos)? + rotate_half(&q)?.broadcast_mul(sin)?)?;
        let k = (k.broadcast_mul(cos)? + rotate_half(&k)?.broadcast_mul(sin)?)?;
        let rep = HEADS / KV_HEADS; // repeat_kv
        let k = k.unsqueeze(2)?.repeat((1, 1, rep, 1, 1))?.reshape((1, HEADS, l, HEAD_DIM))?;
        let v = v.unsqueeze(2)?.repeat((1, 1, rep, 1, 1))?.reshape((1, HEADS, l, HEAD_DIM))?;
        Ok(sdpa(&q, &k, &v, None, true)?.transpose(1, 2)?.reshape((1, l, HEADS * HEAD_DIM))?)
    }

    /// Stacked hidden states for a model that conditions on several layers (Krea 2): `prefix + prompt` and `suffix`
    /// tokenized separately (as the reference does), truncated to `max_len + drop` before the suffix, hidden states
    /// after each of `layers` stacked -> [l - drop, layers.len(), width] bf16 (the first `drop` tokens sliced off).
    pub fn encode_layers(&self, prefix: &str, prompt: &str, suffix: &str, drop: usize, max_len: usize, layers: &[usize]) -> Result<Tensor> {
        let enc = |s: &str| -> Result<Vec<u32>> { Ok(self.tok.encode(s, false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec()) };
        let mut ids = enc(&format!("{prefix}{prompt}"))?;
        ids.truncate(max_len + drop);
        ids.extend(enc(suffix)?);
        let (_, kept) = self.run_ids(&ids, layers)?;
        let st = Tensor::stack(&kept, 1)?;
        let l = st.dim(0)?;
        Ok(st.narrow(0, drop, l - drop)?)
    }

    /// The raw prompt (no chat template, at most `max_len` tokens) through every layer and the final norm:
    /// [l, width] bf16, what RD Flow 64 reads.
    pub fn encode_plain(&self, prompt: &str, max_len: usize) -> Result<Tensor> {
        let p = if prompt.trim().is_empty() { " " } else { prompt };
        let mut ids = self.tok.encode(p, false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec();
        ids.truncate(max_len);
        let (x, _) = self.run_ids(&ids, &[])?;
        let norm = self.wt.tensor(&join(self.prefix, "norm.weight"), &self.dev)?.to_dtype(DType::BF16)?;
        rms(&x, &norm)
    }

    fn run(&self, prompt: &str, keep: &[usize]) -> Result<(Tensor, Vec<Tensor>)> {
        let ids = self.tokens(prompt)?;
        self.run_ids(&ids, keep)
    }

    fn run_ids(&self, ids: &[u32], keep: &[usize]) -> Result<(Tensor, Vec<Tensor>)> {
        let mut kept = vec![];
        let l = ids.len();
        let mut x = self.embed(ids)?;
        let (cos, sin) = self.rope(l)?;
        if keep.contains(&0) {
            kept.push(x.squeeze(0)?);
        }
        for (li, ly) in self.layers.iter().enumerate() {
            crate::core::check_abort()?;
            if let Some(s) = &self.stream {
                s.prepare(li)?;
            }
            let h = rms(&x, &ly.in_norm)?;
            let q = rms(&ly.q.forward(&h)?.reshape((1, l, HEADS, HEAD_DIM))?, &ly.q_norm)?.transpose(1, 2)?;
            let k = rms(&ly.k.forward(&h)?.reshape((1, l, KV_HEADS, HEAD_DIM))?, &ly.k_norm)?.transpose(1, 2)?;
            let v = ly.v.forward(&h)?.reshape((1, l, KV_HEADS, HEAD_DIM))?.transpose(1, 2)?;
            let o = self.attn_core(q, k, v, &cos, &sin, l)?;
            x = (x + ly.o.forward(&o)?)?;
            let h = rms(&x, &ly.post_norm)?;
            let m = ly.down.forward(&(ly.gate.forward(&h)?.silu()? * ly.up.forward(&h)?)?)?;
            x = (x + m)?;
            if let Some(s) = &self.stream {
                s.done(li)?;
            }
            if keep.contains(&(li + 1)) {
                kept.push(x.squeeze(0)?);
            }
        }
        Ok((x.squeeze(0)?, kept))
    }
}
