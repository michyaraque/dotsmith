//! MongoDB/voyage-4-nano, LiliPixel's text encoder: a 12-layer Qwen3 (1024 wide, 16 query / 8 KV heads x 128, RoPE
//! theta 1e6) with bidirectional attention over the real tokens, its final norm, then a 1024 -> 2048 linear. f32; the
//! embedding table stays bf16 and rows are widened as they are read.
use anyhow::{Context, Result};
use candle_core::{D, DType, Device, Tensor};
use tokenizers::Tokenizer;

use crate::core::weights::Weights;

const HEADS: usize = 16;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const THETA: f64 = 1_000_000.0;
const EPS: f64 = 1e-6;
const PAD: u32 = 151643; // <|endoftext|>

struct Layer {
    in_norm: Tensor,
    post_norm: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    gate: Tensor,
    up: Tensor,
    down: Tensor,
}

pub struct Voyage {
    embed: Tensor,
    layers: Vec<Layer>,
    norm: Tensor,
    linear: Tensor,
    tok: Tokenizer,
    dev: Device,
}

pub(super) fn rms(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let r = (x.sqr()?.mean_keepdim(D::Minus1)? + EPS)?.sqrt()?.recip()?;
    Ok(x.broadcast_mul(&r)?.broadcast_mul(w)?)
}

pub(super) fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let d = x.dim(D::Minus1)? / 2;
    Ok(Tensor::cat(&[&x.narrow(D::Minus1, d, d)?.neg()?, &x.narrow(D::Minus1, 0, d)?], D::Minus1)?)
}

/// x [.., in] @ w [out, in]^T.
pub(super) fn linear(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    Ok(x.broadcast_matmul(&w.t()?)?)
}

impl Voyage {
    pub fn load(weights: &str, tokenizer_json: &str, dev: &Device) -> Result<Self> {
        let wt = Weights::open(weights)?;
        let f = |n: &str| -> Result<Tensor> { Ok(wt.tensor(n, dev)?.to_dtype(DType::F32)?) };
        let n_layers = (0..).take_while(|i| wt.has(&format!("model.layers.{i}.input_layernorm.weight"))).count();
        let layers = (0..n_layers)
            .map(|i| {
                let p = format!("model.layers.{i}");
                Ok(Layer {
                    in_norm: f(&format!("{p}.input_layernorm.weight"))?,
                    post_norm: f(&format!("{p}.post_attention_layernorm.weight"))?,
                    q: f(&format!("{p}.self_attn.q_proj.weight"))?,
                    k: f(&format!("{p}.self_attn.k_proj.weight"))?,
                    v: f(&format!("{p}.self_attn.v_proj.weight"))?,
                    o: f(&format!("{p}.self_attn.o_proj.weight"))?,
                    q_norm: f(&format!("{p}.self_attn.q_norm.weight"))?,
                    k_norm: f(&format!("{p}.self_attn.k_norm.weight"))?,
                    gate: f(&format!("{p}.mlp.gate_proj.weight"))?,
                    up: f(&format!("{p}.mlp.up_proj.weight"))?,
                    down: f(&format!("{p}.mlp.down_proj.weight"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let tok = Tokenizer::from_file(tokenizer_json).map_err(anyhow::Error::msg).context("voyage tokenizer")?;
        Ok(Self {
            embed: wt.tensor("model.embed_tokens.weight", dev)?,
            layers,
            norm: f("model.norm.weight")?,
            linear: f("linear.weight")?,
            tok,
            dev: dev.clone(),
        })
    }

    /// The prompt as `len` token ids (right-padded) and how many are real; an empty prompt reads as ".".
    pub fn tokens(&self, prompt: &str, len: usize) -> Result<(Vec<u32>, usize)> {
        let p = if prompt.trim().is_empty() { "." } else { prompt };
        let mut ids = self.tok.encode(p, true).map_err(anyhow::Error::msg)?.get_ids().to_vec();
        ids.truncate(len);
        let n = ids.len();
        ids.resize(len, PAD);
        Ok((ids, n))
    }

    /// Prompt -> (features [len, 2048] f32, real-token mask [len] f32).
    pub fn encode(&self, prompt: &str, len: usize) -> Result<(Tensor, Tensor)> {
        let (ids, n) = self.tokens(prompt, len)?;
        let idx = Tensor::new(ids.as_slice(), &self.dev)?;
        let mut x = self.embed.index_select(&idx, 0)?.to_dtype(DType::F32)?;
        let (cos, sin) = rope(len, &self.dev)?;
        // keys past the real tokens are masked for every query
        let bias: Vec<f32> = (0..len).map(|j| if j < n { 0.0 } else { f32::NEG_INFINITY }).collect();
        let bias = Tensor::from_vec(bias, (1, 1, 1, len), &self.dev)?;
        for ly in &self.layers {
            let h = rms(&x, &ly.in_norm)?;
            let heads = |w: &Tensor, nh: usize, norm: &Tensor| -> Result<Tensor> {
                let t = linear(&h, w)?.reshape((1, len, nh, HEAD_DIM))?.transpose(1, 2)?.contiguous()?;
                let t = rms(&t, norm)?;
                Ok((t.broadcast_mul(&cos)? + rotate_half(&t)?.broadcast_mul(&sin)?)?)
            };
            let q = heads(&ly.q, HEADS, &ly.q_norm)?;
            let k = heads(&ly.k, KV_HEADS, &ly.k_norm)?;
            let v = linear(&h, &ly.v)?.reshape((1, len, KV_HEADS, HEAD_DIM))?.transpose(1, 2)?.contiguous()?;
            let rep = HEADS / KV_HEADS;
            let (k, v) = (repeat_kv(&k, rep)?, repeat_kv(&v, rep)?);
            let s = (q.matmul(&k.t()?)? / (HEAD_DIM as f64).sqrt())?.broadcast_add(&bias)?;
            let a = candle_nn::ops::softmax_last_dim(&s)?.matmul(&v)?;
            let a = a.transpose(1, 2)?.reshape((len, HEADS * HEAD_DIM))?;
            x = (x + linear(&a, &ly.o)?)?;
            let h = rms(&x, &ly.post_norm)?;
            let m = (candle_nn::ops::silu(&linear(&h, &ly.gate)?)? * linear(&h, &ly.up)?)?;
            x = (x + linear(&m, &ly.down)?)?;
        }
        let feats = linear(&rms(&x, &self.norm)?, &self.linear)?;
        let mask: Vec<f32> = (0..len).map(|j| if j < n { 1.0 } else { 0.0 }).collect();
        Ok((feats, Tensor::from_vec(mask, len, &self.dev)?))
    }
}

fn repeat_kv(t: &Tensor, rep: usize) -> Result<Tensor> {
    let (b, h, l, d) = t.dims4()?;
    Ok(t.unsqueeze(2)?.expand((b, h, rep, l, d))?.reshape((b, h * rep, l, d))?)
}

/// Plain Qwen3 RoPE over positions 0..len: (cos, sin) [1, 1, len, 128].
fn rope(len: usize, dev: &Device) -> Result<(Tensor, Tensor)> {
    let half = HEAD_DIM / 2;
    let mut cos = Vec::with_capacity(len * HEAD_DIM);
    let mut sin = Vec::with_capacity(len * HEAD_DIM);
    for p in 0..len {
        let row: Vec<f64> = (0..half).map(|i| p as f64 / THETA.powf(2.0 * i as f64 / HEAD_DIM as f64)).collect();
        for _ in 0..2 {
            cos.extend(row.iter().map(|a| a.cos() as f32));
            sin.extend(row.iter().map(|a| a.sin() as f32));
        }
    }
    Ok((Tensor::from_vec(cos, (1, 1, len, HEAD_DIM), dev)?, Tensor::from_vec(sin, (1, 1, len, HEAD_DIM), dev)?))
}
