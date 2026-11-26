//! RD Flow 64's network (rdflow/model.py, layers.py): an hourglass over the 64 x 64 pixel grid (8 x 8 window
//! attention at 64 and 32, shifted on odd blocks), a 16 x 16 core attending jointly to registers, text, palette tokens
//! and the image, an x head (clean image in OKLab + alpha) and a pointer head scoring every pixel against the palette.
//! f32 throughout (the weights ship in f16).
use anyhow::Result;
use candle_core::{D, DType, Device, Tensor};

use super::palette;
use crate::core::weights::Weights;

const CH: usize = 4;
const HEAD_DIM: usize = 64;
const THETA: f64 = 100.0;
const WINDOW: usize = 8;
const N0: usize = 64;
const N1: usize = 32;
const NC: usize = 16;
const REGISTERS: usize = 4;
const DK: usize = 128;
const COLOUR_FREQS: usize = 8;
/// The source tag the model is sampled with (its cleanest training source).
const SOURCE: u32 = 6;

fn rms(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let r = (x.sqr()?.mean_keepdim(D::Minus1)? + 1e-6)?.sqrt()?.recip()?;
    Ok(x.broadcast_mul(&r)?.broadcast_mul(w)?)
}

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let d = x.dim(D::Minus1)? / 2;
    Ok(Tensor::cat(&[&x.narrow(D::Minus1, d, d)?.neg()?, &x.narrow(D::Minus1, 0, d)?], D::Minus1)?)
}

struct Lin {
    w: Tensor,
    b: Option<Tensor>,
}

impl Lin {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.broadcast_matmul(&self.w.t()?)?;
        Ok(match &self.b {
            Some(b) => y.broadcast_add(b)?,
            None => y,
        })
    }
}

struct Attn {
    qkv: Lin,
    out: Lin,
    qn: Tensor,
    kn: Tensor,
}

struct Mlp {
    w1: Lin,
    w2: Lin,
    /// depthwise 3x3 on the hidden, grid levels only: (weight [h, 1, 3, 3], bias [h])
    conv: Option<(Tensor, Tensor)>,
}

struct Block {
    n1: Tensor,
    n2: Tensor,
    attn: Attn,
    mlp: Mlp,
    table: Tensor,
}

struct Level {
    blocks: Vec<Block>,
    ada: Lin,
    n: usize,
}

pub struct Net {
    inp: Lin,
    t_mlp: (Lin, Lin),
    style: Tensor,
    source: Tensor,
    text_pool: Lin,
    text_proj: Lin,
    null_text: Tensor,
    registers: Tensor,
    modality: Tensor,
    attr_emb: Vec<Tensor>,
    bg_emb: Tensor,
    adapter: Vec<Block>,
    down0: Level,
    merge0: Lin,
    down1: Level,
    merge1: Lin,
    core: Vec<Block>,
    core_ada: Lin,
    split1: Lin,
    skip1: Lin,
    up1: Level,
    split0: Lin,
    skip0: Lin,
    up0: Level,
    out_norm: Tensor,
    x_head: Lin,
    colour: (Lin, Lin),
    ramp_emb: Tensor,
    level_proj: Lin,
    pal_tok: Lin,
    q: Lin,
    k: (Lin, Lin),
    transparent: Lin,
    dev: Device,
}

/// One row of a batch: its text and whether the caption is dropped (the CFG row, also style 0 like the cond row).
pub struct Row<'a> {
    pub text: &'a Tensor,
    pub dropped: bool,
}

/// A palette as the model takes it: exact colours, their shading ramp ids and levels.
#[derive(Clone)]
pub struct Palette {
    pub rgb: Vec<[u8; 3]>,
    pub ramp: Vec<i64>,
    pub level: Vec<f32>,
}

impl Palette {
    pub fn new(rgb: Vec<[u8; 3]>) -> Self {
        let rgb: Vec<[u8; 3]> = rgb.into_iter().take(256).collect();
        let (ramp, level) = palette::ramps(&rgb);
        Self { rgb, ramp, level }
    }
}

impl Net {
    pub fn load(path: &str, dev: &Device) -> Result<Self> {
        let wt = Weights::open(path)?;
        let t = |n: &str| -> Result<Tensor> { Ok(wt.tensor(n, dev)?.to_dtype(DType::F32)?) };
        let lin = |n: &str| -> Result<Lin> {
            Ok(Lin { w: t(&format!("{n}.weight"))?, b: if wt.has(&format!("{n}.bias")) { Some(t(&format!("{n}.bias"))?) } else { None } })
        };
        let block = |p: &str| -> Result<Block> {
            let conv =
                if wt.has(&format!("{p}.mlp.conv.weight")) { Some((t(&format!("{p}.mlp.conv.weight"))?, t(&format!("{p}.mlp.conv.bias"))?)) } else { None };
            Ok(Block {
                n1: t(&format!("{p}.n1.w"))?,
                n2: t(&format!("{p}.n2.w"))?,
                attn: Attn {
                    qkv: lin(&format!("{p}.attn.qkv"))?,
                    out: lin(&format!("{p}.attn.out"))?,
                    qn: t(&format!("{p}.attn.qn.w"))?,
                    kn: t(&format!("{p}.attn.kn.w"))?,
                },
                mlp: Mlp { w1: lin(&format!("{p}.mlp.w1"))?, w2: lin(&format!("{p}.mlp.w2"))?, conv },
                table: t(&format!("{p}.table"))?,
            })
        };
        let count = |p: &str| (0..).take_while(|i| wt.has(&format!("{p}.{i}.table"))).count();
        let blocks = |p: &str| -> Result<Vec<Block>> { (0..count(p)).map(|i| block(&format!("{p}.{i}"))).collect() };
        let level = |p: &str, n: usize| -> Result<Level> { Ok(Level { blocks: blocks(&format!("{p}.blocks"))?, ada: lin(&format!("{p}.ada"))?, n }) };
        Ok(Self {
            inp: lin("inp")?,
            t_mlp: (lin("t_mlp.0")?, lin("t_mlp.2")?),
            style: t("style.weight")?,
            source: t("source.weight")?,
            text_pool: lin("text_pool")?,
            text_proj: lin("text_proj")?,
            null_text: t("null_text")?,
            registers: t("registers")?,
            modality: t("modality")?,
            attr_emb: (0..4).map(|j| t(&format!("attr_emb.{j}.weight"))).collect::<Result<_>>()?,
            bg_emb: t("bg_emb.weight")?,
            adapter: blocks("adapter")?,
            down0: level("down0", N0)?,
            merge0: lin("merge0")?,
            down1: level("down1", N1)?,
            merge1: lin("merge1")?,
            core: blocks("core")?,
            core_ada: lin("core_ada")?,
            split1: lin("split1")?,
            skip1: lin("skip1")?,
            up1: level("up1", N1)?,
            split0: lin("split0")?,
            skip0: lin("skip0")?,
            up0: level("up0", N0)?,
            out_norm: t("out_norm.w")?,
            x_head: lin("x_head")?,
            colour: (lin("colour.mlp.0")?, lin("colour.mlp.2")?),
            ramp_emb: t("ramp_emb.weight")?,
            level_proj: lin("level_proj")?,
            pal_tok: lin("pal_tok")?,
            q: lin("q")?,
            k: (lin("k.0")?, lin("k.2")?),
            transparent: lin("transparent")?,
            dev: dev.clone(),
        })
    }

    /// (cos, sin) [n, 64] for positions (x, y), None = no position (identity).
    fn rope(&self, pos: &[Option<(f64, f64)>]) -> Result<(Tensor, Tensor)> {
        let quarter = HEAD_DIM / 4;
        let freqs: Vec<f64> = (0..quarter).map(|i| THETA.powf(-(i as f64) / quarter as f64)).collect();
        let (mut cos, mut sin) = (Vec::with_capacity(pos.len() * HEAD_DIM), Vec::with_capacity(pos.len() * HEAD_DIM));
        for p in pos {
            let ang: Vec<f64> = match p {
                Some((x, y)) => freqs.iter().map(|f| x * f).chain(freqs.iter().map(|f| y * f)).collect(),
                None => vec![0.0; HEAD_DIM / 2],
            };
            for _ in 0..2 {
                cos.extend(ang.iter().map(|a| a.cos() as f32));
                sin.extend(ang.iter().map(|a| a.sin() as f32));
            }
        }
        let n = pos.len();
        Ok((Tensor::from_vec(cos, (n, HEAD_DIM), &self.dev)?, Tensor::from_vec(sin, (n, HEAD_DIM), &self.dev)?))
    }

    /// x [B, N, d]; cos/sin broadcastable to [B, h, N, 64]; `bias` additive key mask [B, 1, 1, N].
    fn attention(&self, a: &Attn, x: &Tensor, cos: &Tensor, sin: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
        let (b, n, d) = x.dims3()?;
        let h = d / HEAD_DIM;
        let qkv = a.qkv.forward(x)?.reshape((b, n, 3, h, HEAD_DIM))?.permute([2, 0, 3, 1, 4])?;
        let rope = |t: Tensor, norm: &Tensor| -> Result<Tensor> {
            let t = rms(&t.contiguous()?, norm)?;
            Ok((t.broadcast_mul(cos)? + rotate_half(&t)?.broadcast_mul(sin)?)?)
        };
        let q = rope(qkv.get(0)?, &a.qn)?;
        let k = rope(qkv.get(1)?, &a.kn)?;
        let v = qkv.get(2)?.contiguous()?;
        let mut s = (q.matmul(&k.t()?)? / (HEAD_DIM as f64).sqrt())?;
        if let Some(bias) = bias {
            s = s.broadcast_add(bias)?;
        }
        let o = candle_nn::ops::softmax_last_dim(&s)?.matmul(&v)?.transpose(1, 2)?.reshape((b, n, d))?;
        a.out.forward(&o)
    }

    /// SwiGLU, with the depthwise 3x3 conv on its hidden when the block runs on an `hw` grid.
    fn mlp(&self, m: &Mlp, x: &Tensor, hw: Option<usize>) -> Result<Tensor> {
        let y = m.w1.forward(x)?;
        let half = y.dim(D::Minus1)? / 2;
        let (mut a, g) = (y.narrow(D::Minus1, 0, half)?, y.narrow(D::Minus1, half, half)?);
        if let (Some((w, bias)), Some(n)) = (&m.conv, hw) {
            let (b, _, c) = a.dims3()?;
            let grid = a.transpose(1, 2)?.reshape((b, c, n, n))?.pad_with_zeros(2, 1, 1)?.pad_with_zeros(3, 1, 1)?;
            let mut acc = bias.reshape((1, c, 1, 1))?.broadcast_as((b, c, n, n))?.contiguous()?;
            for ky in 0..3 {
                for kx in 0..3 {
                    let k = w.narrow(2, ky, 1)?.narrow(3, kx, 1)?.reshape((1, c, 1, 1))?;
                    acc = (acc + grid.narrow(2, ky, n)?.narrow(3, kx, n)?.broadcast_mul(&k)?)?;
                }
            }
            a = acc.reshape((b, c, n * n))?.transpose(1, 2)?;
        }
        m.w2.forward(&(candle_nn::ops::silu(&g)? * a)?)
    }

    /// Pre-norm adaLN block: `mods` [B, 6, d] (shared) + the block's table.
    fn block(&self, blk: &Block, x: &Tensor, mods: &Tensor, attn: impl Fn(&Attn, &Tensor) -> Result<Tensor>, hw: Option<usize>) -> Result<Tensor> {
        let m = mods.broadcast_add(&blk.table.unsqueeze(0)?)?;
        let g = |i: usize| m.narrow(1, i, 1);
        let (sh1, sc1, g1, sh2, sc2, g2) = (g(0)?, g(1)?, g(2)?, g(3)?, g(4)?, g(5)?);
        let h = rms(x, &blk.n1)?.broadcast_mul(&(sc1 + 1.0)?)?.broadcast_add(&sh1)?;
        let x = (x + attn(&blk.attn, &h)?.broadcast_mul(&g1)?)?;
        let h = rms(&x, &blk.n2)?.broadcast_mul(&(sc2 + 1.0)?)?.broadcast_add(&sh2)?;
        Ok((&x + self.mlp(&blk.mlp, &h, hw)?.broadcast_mul(&g2)?)?)
    }

    /// Window attention blocks on an n x n grid; odd blocks shift the windows by half a window (with wrap-around).
    fn level(&self, lv: &Level, x: &Tensor, c: &Tensor) -> Result<Tensor> {
        let (b, _, d) = x.dims3()?;
        let n = lv.n;
        let per = n / WINDOW;
        let mods = lv.ada.forward(&candle_nn::ops::silu(c)?)?.reshape((b, 6, d))?;
        let mut x = x.clone();
        for (i, blk) in lv.blocks.iter().enumerate() {
            let shift = if i % 2 == 1 { WINDOW / 2 } else { 0 };
            // positions travel with their tokens through the roll
            let mut pos = vec![];
            for wy in 0..per {
                for wx in 0..per {
                    for iy in 0..WINDOW {
                        for ix in 0..WINDOW {
                            let (y, xx) = ((wy * WINDOW + iy + shift) % n, (wx * WINDOW + ix + shift) % n);
                            pos.push(Some((xx as f64, y as f64)));
                        }
                    }
                }
            }
            let (cos, sin) = self.rope(&pos)?;
            let cos = cos.reshape((per * per, 1, WINDOW * WINDOW, HEAD_DIM))?.repeat((b, 1, 1, 1))?;
            let sin = sin.reshape((per * per, 1, WINDOW * WINDOW, HEAD_DIM))?.repeat((b, 1, 1, 1))?;
            let attn = |a: &Attn, h: &Tensor| -> Result<Tensor> {
                let w = windows(h, n, shift)?;
                let o = self.attention(a, &w, &cos, &sin, None)?;
                unwindows(&o, n, shift, b)
            };
            x = self.block(blk, &x, &mods, attn, Some(n))?;
        }
        Ok(x)
    }

    /// Palette colour embeddings [k, 256]: Fourier(OKLab) MLP + ramp id + level.
    fn palette_embed(&self, pal: &Palette) -> Result<Tensor> {
        let k = pal.rgb.len();
        let mut feats = Vec::with_capacity(k * (3 + 6 * COLOUR_FREQS));
        for c in &pal.rgb {
            // the model's ColourEmbed.oklab runs in float32
            let lab = palette::srgb_to_oklab(*c).map(|v| v as f32);
            let ang: Vec<f32> = lab.iter().flat_map(|l| (0..COLOUR_FREQS).map(move |f| l * 2f32.powi(f as i32) * std::f32::consts::PI)).collect();
            feats.extend_from_slice(&lab);
            feats.extend(ang.iter().map(|a| a.sin()));
            feats.extend(ang.iter().map(|a| a.cos()));
        }
        let x = Tensor::from_vec(feats, (k, 3 + 6 * COLOUR_FREQS), &self.dev)?;
        let e = self.colour.1.forward(&candle_nn::ops::silu(&self.colour.0.forward(&x)?)?)?;
        let ramps = Tensor::new(pal.ramp.iter().map(|&r| r.clamp(0, 15) as u32).collect::<Vec<_>>().as_slice(), &self.dev)?;
        let levels = Tensor::from_vec(pal.level.clone(), (k, 1), &self.dev)?;
        Ok(((e + self.ramp_emb.index_select(&ramps, 0)?)? + self.level_proj.forward(&levels)?)?)
    }

    /// x_t [B, 4096, 4] (t: 0 noise .. 1 clean), self-conditioning `sc` [B, 4096, 4] -> (clean guess [B, 4096, 4],
    /// final pixel features [B, 4096, 128]). `opaque`: painted background (else transparent).
    pub fn forward(&self, x_t: &Tensor, t: f32, rows: &[Row], sc: Option<&Tensor>, pal: Option<&Palette>, opaque: bool) -> Result<(Tensor, Tensor)> {
        let b = rows.len();
        let dev = &self.dev;
        let l = rows[0].text.dim(0)?;
        // [null token | text], the null token valid only for a dropped caption, a dropped caption's text zeroed
        let mut texts = vec![];
        let mut masks = vec![];
        for r in rows {
            let text = if r.dropped { r.text.zeros_like()? } else { r.text.clone() };
            texts.push(Tensor::cat(&[&self.null_text, &text], 0)?);
            masks.push(std::iter::once(if r.dropped { 1f32 } else { 0.0 }).chain((0..l).map(|_| if r.dropped { 0.0 } else { 1.0 })).collect::<Vec<f32>>());
        }
        let text = Tensor::stack(&texts, 0)?;
        let tmask = Tensor::from_vec(masks.concat(), (b, l + 1), dev)?;
        let pooled = text.broadcast_mul(&tmask.unsqueeze(2)?)?.sum(1)?.broadcast_div(&tmask.sum_keepdim(1)?.clamp(1f32, f32::MAX)?)?;

        let half = 128;
        let te: Vec<f32> = (0..half)
            .map(|i| (t as f64 * 1000.0 * (-(10000f64.ln()) * i as f64 / half as f64).exp()).sin() as f32)
            .chain((0..half).map(|i| (t as f64 * 1000.0 * (-(10000f64.ln()) * i as f64 / half as f64).exp()).cos() as f32))
            .collect();
        let te = Tensor::from_vec(te, (1, 256), dev)?;
        let mut c = self.t_mlp.1.forward(&candle_nn::ops::silu(&self.t_mlp.0.forward(&te)?)?)?;
        c = c.broadcast_add(&self.style.get(0)?)?.broadcast_add(&self.source.get(SOURCE as usize)?)?;
        c = c.broadcast_add(&self.text_pool.forward(&pooled)?)?;
        for emb in &self.attr_emb {
            c = c.broadcast_add(&emb.get(0)?)?;
        }
        c = c.broadcast_add(&self.bg_emb.get(if opaque { 1 } else { 2 })?)?;

        let prev = match sc {
            Some(s) => s.clone(),
            None => x_t.zeros_like()?,
        };
        let xin = Tensor::cat(&[x_t, &prev], 2)?;
        let h0 = self.level(&self.down0, &self.inp.forward(&xin)?, &c)?;
        let h1 = self.level(&self.down1, &self.merge0.forward(&merge(&h0, N0)?)?, &c)?;

        let key_bias = |m: &Tensor| -> Result<Tensor> { Ok(((m - 1.0)? * 1e9)?.unsqueeze(1)?.unsqueeze(1)?) };
        let mut ctx = self.text_proj.forward(&text)?;
        if !self.adapter.is_empty() {
            let (cos, sin) = self.rope(&vec![None; l + 1])?;
            let zmod = Tensor::zeros((b, 6, ctx.dim(2)?), DType::F32, dev)?;
            let bias = key_bias(&tmask)?;
            for blk in &self.adapter {
                ctx = self.block(blk, &ctx, &zmod, |a, h| self.attention(a, h, &cos, &sin, Some(&bias)), None)?;
            }
        }
        let hc = self.merge1.forward(&merge(&h1, N1)?)?;
        let modality = |i: usize| self.modality.get(i);
        let mut parts = vec![self.registers.broadcast_add(&modality(0)?)?.unsqueeze(0)?.repeat((b, 1, 1))?, ctx.broadcast_add(&modality(1)?)?];
        let mut masks = vec![Tensor::ones((b, REGISTERS), DType::F32, dev)?, tmask.clone()];
        let pn = pal.map_or(0, |p| p.rgb.len());
        if let Some(p) = pal.filter(|p| !p.rgb.is_empty()) {
            let pt = self.pal_tok.forward(&self.palette_embed(p)?)?.broadcast_add(&modality(3)?)?;
            parts.push(pt.unsqueeze(0)?.repeat((b, 1, 1))?);
            masks.push(Tensor::ones((b, pn), DType::F32, dev)?);
        }
        let s = REGISTERS + (l + 1) + pn;
        parts.push(hc.broadcast_add(&modality(2)?)?);
        masks.push(Tensor::ones((b, NC * NC), DType::F32, dev)?);
        let mut tok = Tensor::cat(&parts, 1)?;
        let bias = key_bias(&Tensor::cat(&masks, 1)?)?;
        let pos: Vec<Option<(f64, f64)>> = (0..s).map(|_| None).chain((0..NC * NC).map(|i| Some(((i % NC) as f64, (i / NC) as f64)))).collect();
        let (cos, sin) = self.rope(&pos)?;
        let mods = self.core_ada.forward(&candle_nn::ops::silu(&c)?)?.reshape((b, 6, tok.dim(2)?))?;
        for blk in &self.core {
            tok = self.block(blk, &tok, &mods, |a, h| self.attention(a, h, &cos, &sin, Some(&bias)), None)?;
        }
        let hc = tok.narrow(1, s, NC * NC)?;
        let u1 = self.level(&self.up1, &self.skip1.forward(&Tensor::cat(&[&split(&self.split1.forward(&hc)?, N1)?, &h1], 2)?)?, &c)?;
        let u0 = self.level(&self.up0, &self.skip0.forward(&Tensor::cat(&[&split(&self.split0.forward(&u1)?, N0)?, &h0], 2)?)?, &c)?;
        let u0 = rms(&u0, &self.out_norm)?;
        Ok((self.x_head.forward(&u0)?, u0))
    }

    /// Pointer logits [B, N, 1 + k]: 0 = transparent, 1 + s = palette slot s.
    pub fn pointer_logits(&self, feats: &Tensor, pal: &Palette) -> Result<Tensor> {
        let tr = self.transparent.forward(feats)?;
        if pal.rgb.is_empty() {
            return Ok(tr);
        }
        let keys = self.k.1.forward(&candle_nn::ops::silu(&self.k.0.forward(&self.palette_embed(pal)?)?)?)?;
        let logits = (self.q.forward(feats)?.broadcast_matmul(&keys.t()?)? / (DK as f64).sqrt())?;
        Ok(Tensor::cat(&[&tr, &logits], 2)?)
    }
}

/// [B, n*n, D] -> [B*(n/w)^2, w*w, D], rolled by -shift first.
fn windows(x: &Tensor, n: usize, shift: usize) -> Result<Tensor> {
    let (b, _, d) = x.dims3()?;
    let mut g = x.reshape((b, n, n, d))?;
    if shift > 0 {
        g = g.roll(-(shift as i32), 1)?.roll(-(shift as i32), 2)?;
    }
    let per = n / WINDOW;
    Ok(g.reshape((b, per, WINDOW, per, WINDOW, d))?.permute([0, 1, 3, 2, 4, 5])?.reshape((b * per * per, WINDOW * WINDOW, d))?)
}

fn unwindows(x: &Tensor, n: usize, shift: usize, b: usize) -> Result<Tensor> {
    let d = x.dim(2)?;
    let per = n / WINDOW;
    let mut g = x.reshape((b, per, per, WINDOW, WINDOW, d))?.permute([0, 1, 3, 2, 4, 5])?.reshape((b, n, n, d))?;
    if shift > 0 {
        g = g.roll(shift as i32, 1)?.roll(shift as i32, 2)?;
    }
    Ok(g.reshape((b, n * n, d))?)
}

/// [B, n*n, D] -> [B, (n/2)^2, 4D] (2x2 patches).
fn merge(x: &Tensor, n: usize) -> Result<Tensor> {
    let (b, _, d) = x.dims3()?;
    Ok(x.reshape((b, n / 2, 2, n / 2, 2, d))?.permute([0, 1, 3, 2, 4, 5])?.reshape((b, (n / 2) * (n / 2), 4 * d))?)
}

/// [B, (n/2)^2, 4D] -> [B, n*n, D].
fn split(x: &Tensor, n: usize) -> Result<Tensor> {
    let (b, _, d4) = x.dims3()?;
    let d = d4 / 4;
    Ok(x.reshape((b, n / 2, n / 2, 2, 2, d))?.permute([0, 1, 3, 2, 4, 5])?.reshape((b, n * n, d))?)
}

pub const PIXELS: usize = N0 * N0;
pub const CHANNELS: usize = CH;
