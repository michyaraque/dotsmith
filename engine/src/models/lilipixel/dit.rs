//! LiliPixel V1's PixelArtDiT (Pinkstack/LiliPixel-V1-190M): 24 blocks of 768 (16 heads x 48) over [text | pixels],
//! RGBA in pixel space, patch 1 below 64 px and 2 from 64 px, 2D RoPE on pixel coordinates (text on a negative y
//! axis), one shared adaLN modulation per step plus a per-block table. f32.
use anyhow::{Result, bail};
use candle_core::{DType, Device, Tensor};

use super::text::{linear, rms, rotate_half};
use crate::core::weights::Weights;

pub const TEXT_LEN: usize = 96;
pub const SIZES: [usize; 3] = [16, 32, 64];
const DIM: usize = 768;
const HEADS: usize = 16;
const HEAD_DIM: usize = DIM / HEADS;
const CHANNELS: usize = 4;

struct Block {
    norm1: Tensor,
    qkv: Tensor,
    proj_out: Tensor,
    norm2: Tensor,
    fc1: Tensor,
    fc2: Tensor,
    fc3: Tensor,
    /// [6, 768]: added to the shared modulation (shift, scale, gate) x (attention, mlp)
    table: Tensor,
}

pub struct Dit {
    x_proj: [Tensor; 2],
    out: [(Tensor, Tensor); 2],
    res_emb: Tensor,
    txt_norm: Tensor,
    txt_proj: Tensor,
    pad_emb: Tensor,
    null_txt: Tensor,
    time: (Tensor, Tensor, Tensor, Tensor),
    modw: (Tensor, Tensor),
    blocks: Vec<Block>,
    norm_out: Tensor,
    dev: Device,
}

/// One prepared caption: tokens projected to the model width [96, 768] and their mean over the real ones [768].
#[derive(Clone)]
pub struct Caption {
    pub txt: Tensor,
    pub pooled: Tensor,
}

impl Dit {
    pub fn load(path: &str, dev: &Device) -> Result<Self> {
        let wt = Weights::open(path)?;
        let f = |n: &str| -> Result<Tensor> {
            let name = if wt.has(&format!("inner.{n}")) { format!("inner.{n}") } else { n.to_string() };
            Ok(wt.tensor(&name, dev)?.to_dtype(DType::F32)?)
        };
        let n_blocks = (0..).take_while(|i| wt.has(&format!("inner.blocks.{i}.qkv.weight")) || wt.has(&format!("blocks.{i}.qkv.weight"))).count();
        let blocks = (0..n_blocks)
            .map(|i| {
                let b = format!("blocks.{i}");
                Ok(Block {
                    norm1: f(&format!("{b}.norm1.weight"))?,
                    qkv: f(&format!("{b}.qkv.weight"))?,
                    proj_out: f(&format!("{b}.proj_out.weight"))?,
                    norm2: f(&format!("{b}.norm2.weight"))?,
                    fc1: f(&format!("{b}.fc1.weight"))?,
                    fc2: f(&format!("{b}.fc2.weight"))?,
                    fc3: f(&format!("{b}.fc3.weight"))?,
                    table: f(&format!("{b}.table"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            x_proj: [f("x_projs.1.weight")?, f("x_projs.2.weight")?],
            out: [(f("outs.1.weight")?, f("outs.1.bias")?), (f("outs.2.weight")?, f("outs.2.bias")?)],
            res_emb: f("res_emb.weight")?,
            txt_norm: f("txt_norm.weight")?,
            txt_proj: f("txt_proj.weight")?,
            pad_emb: f("pad_emb")?,
            null_txt: f("null_txt")?,
            time: (f("time_mlp.0.weight")?, f("time_mlp.0.bias")?, f("time_mlp.2.weight")?, f("time_mlp.2.bias")?),
            modw: (f("mod.weight")?, f("mod.bias")?),
            blocks,
            norm_out: f("norm_out.weight")?,
            dev: dev.clone(),
        })
    }

    /// Encoder features [96, 2048] + mask [96] -> a caption; `None` = the learned "no caption" (the CFG row).
    pub fn caption(&self, feats: Option<(&Tensor, &Tensor)>) -> Result<Caption> {
        let (txt, keep) = match feats {
            Some((f, m)) => {
                let t = linear(&rms(f, &self.txt_norm)?, &self.txt_proj)?;
                let m = m.unsqueeze(1)?;
                let pad = self.pad_emb.unsqueeze(0)?.broadcast_as(t.shape())?;
                let t = (t.broadcast_mul(&m)? + pad.broadcast_mul(&(1.0 - &m)?)?)?;
                (t, m)
            }
            None => (self.null_txt.clone(), Tensor::ones((TEXT_LEN, 1), DType::F32, &self.dev)?),
        };
        let pooled = (txt.broadcast_mul(&keep)?.sum(0)? / keep.sum_all()?.to_scalar::<f32>()?.max(1.0) as f64)?;
        Ok(Caption { txt, pooled })
    }

    /// Velocity for x [b, 4, s, s] at flow time `t` (1 = noise), one caption per row.
    pub fn forward(&self, x: &Tensor, t: f32, caps: &[&Caption]) -> Result<Tensor> {
        let (b, c, h, w) = x.dims4()?;
        if h != w || !SIZES.contains(&h) {
            bail!("LiliPixel draws squares of 16, 32 or 64 px, not {w}x{h}");
        }
        let p = if h >= 64 { 2 } else { 1 };
        let nh = h / p;
        let img = x.reshape((b, c, nh, p, nh, p))?.permute([0, 2, 4, 1, 3, 5])?.reshape((b, nh * nh, c * p * p))?;
        let img = linear(&img, &self.x_proj[p - 1])?;
        let txt = Tensor::stack(&caps.iter().map(|c| &c.txt).collect::<Vec<_>>(), 0)?;
        let pooled = Tensor::stack(&caps.iter().map(|c| &c.pooled).collect::<Vec<_>>(), 0)?;
        let seq = Tensor::cat(&[&txt, &img], 1)?;

        let half = DIM / 2;
        let tf = t as f64 * 1000.0;
        let emb: Vec<f32> = (0..half)
            .map(|i| (tf * (-(10000f64.ln()) * i as f64 / half as f64).exp()).cos() as f32)
            .chain((0..half).map(|i| (tf * (-(10000f64.ln()) * i as f64 / half as f64).exp()).sin() as f32))
            .collect();
        let temb = Tensor::from_vec(emb, (1, DIM), &self.dev)?;
        let (w0, b0, w2, b2) = &self.time;
        let temb = linear(&candle_nn::ops::silu(&linear(&temb, w0)?.broadcast_add(b0)?)?, w2)?.broadcast_add(b2)?;
        let res = self.res_emb.get(SIZES.iter().position(|&s| s == h).unwrap())?;
        let cond = pooled.broadcast_add(&temb)?.broadcast_add(&res)?;
        let m = linear(&candle_nn::ops::silu(&cond)?, &self.modw.0)?.broadcast_add(&self.modw.1)?.reshape((b, 6, 1, DIM))?;

        let (cos, sin) = self.rope(nh, p)?;
        let s = seq.dim(1)?;
        let mut hs = seq;
        for blk in &self.blocks {
            let md = m.broadcast_add(&blk.table.unsqueeze(1)?)?;
            let g = |i: usize| md.narrow(1, i, 1).and_then(|t| t.squeeze(1));
            let (s1, sc1, g1, s2, sc2, g2) = (g(0)?, g(1)?, g(2)?, g(3)?, g(4)?, g(5)?);
            let a = rms(&hs, &blk.norm1)?.broadcast_mul(&(sc1 + 1.0)?)?.broadcast_add(&s1)?;
            let qkv = linear(&a, &blk.qkv)?.reshape((b, s, 3, HEADS, HEAD_DIM))?.permute([2, 0, 3, 1, 4])?;
            let (q, k, v) = (qkv.get(0)?.contiguous()?, qkv.get(1)?.contiguous()?, qkv.get(2)?.contiguous()?);
            let q = (q.broadcast_mul(&cos)? + rotate_half(&q)?.broadcast_mul(&sin)?)?;
            let k = (k.broadcast_mul(&cos)? + rotate_half(&k)?.broadcast_mul(&sin)?)?;
            let sc = (q.matmul(&k.t()?)? / (HEAD_DIM as f64).sqrt())?;
            let o = candle_nn::ops::softmax_last_dim(&sc)?.matmul(&v)?.transpose(1, 2)?.reshape((b, s, DIM))?;
            hs = (hs + linear(&o, &blk.proj_out)?.broadcast_mul(&g1)?)?;
            let a = rms(&hs, &blk.norm2)?.broadcast_mul(&(sc2 + 1.0)?)?.broadcast_add(&s2)?;
            let mlp = (candle_nn::ops::silu(&linear(&a, &blk.fc1)?)? * linear(&a, &blk.fc3)?)?;
            hs = (hs + linear(&mlp, &blk.fc2)?.broadcast_mul(&g2)?)?;
        }
        let hs = rms(&hs, &self.norm_out)?.narrow(1, TEXT_LEN, nh * nh)?;
        let (ow, ob) = &self.out[p - 1];
        let o = linear(&hs, ow)?.broadcast_add(ob)?;
        Ok(o.reshape((b, nh, nh, CHANNELS, p, p))?.permute([0, 3, 1, 4, 2, 5])?.reshape((b, CHANNELS, h, w))?)
    }

    /// (cos, sin) [1, 1, 96 + n*n, 48]: text at y = -1, -2, ... and x = 0; pixels at their patch centres.
    fn rope(&self, n: usize, p: usize) -> Result<(Tensor, Tensor)> {
        let nf = HEAD_DIM / 4;
        let inv: Vec<f64> = (0..nf).map(|i| 1.0 / 10000f64.powf(i as f64 / nf as f64)).collect();
        let mut pos: Vec<(f64, f64)> = (0..TEXT_LEN).map(|i| (-(i as f64 + 1.0), 0.0)).collect();
        let centre = |i: usize| i as f64 * p as f64 + (p as f64 - 1.0) / 2.0;
        for y in 0..n {
            for x in 0..n {
                pos.push((centre(y), centre(x)));
            }
        }
        let (mut cos, mut sin) = (vec![], vec![]);
        for (py, px) in &pos {
            let ang: Vec<f64> = inv.iter().map(|f| py * f).chain(inv.iter().map(|f| px * f)).collect();
            for _ in 0..2 {
                cos.extend(ang.iter().map(|a| a.cos() as f32));
                sin.extend(ang.iter().map(|a| a.sin() as f32));
            }
        }
        let l = pos.len();
        Ok((Tensor::from_vec(cos, (1, 1, l, HEAD_DIM), &self.dev)?, Tensor::from_vec(sin, (1, 1, l, HEAD_DIM), &self.dev)?))
    }
}
