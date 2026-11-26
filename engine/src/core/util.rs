//! Shared helpers.
use anyhow::Result;
use candle_core::Tensor;

pub fn hello() -> &'static str {
    "dotsmith-engine"
}

/// b with its first `extent` rows (dim 2) or columns (dim 3) faded in from the end of a (diffusers blend_v / blend_h).
pub fn blend(a: &Tensor, b: &Tensor, extent: usize, dim: usize) -> Result<Tensor> {
    let e = extent.min(a.dim(dim)?).min(b.dim(dim)?);
    if e == 0 {
        return Ok(b.clone());
    }
    let wv: Vec<f32> = (0..e).map(|y| y as f32 / e as f32).collect();
    let shape: Vec<usize> = (0..4).map(|d| if d == dim { e } else { 1 }).collect();
    let wt = Tensor::from_vec(wv, shape, b.device())?.to_dtype(b.dtype())?;
    let tail = a.narrow(dim, a.dim(dim)? - e, e)?;
    let head = b.narrow(dim, 0, e)?;
    let mixed = (tail.broadcast_mul(&(1.0 - &wt)?)? + head.broadcast_mul(&wt)?)?;
    let rest = b.dim(dim)? - e;
    Ok(if rest > 0 { Tensor::cat(&[&mixed, &b.narrow(dim, e, rest)?], dim)? } else { mixed })
}

/// VAE encode of an image [b, c, H, W] in 512 px tiles 384 px apart, the latents (`scale` x smaller) cross-faded over
/// the 128 px overlap and stitched (diffusers tiled_encode). Encoding a 1024 px canvas whole took ~7 GB next to the DiT.
pub fn tiled_encode(img: &Tensor, scale: usize, encode: impl Fn(&Tensor) -> Result<Tensor>) -> Result<Tensor> {
    let (tile, stride) = (512usize, 384usize);
    let (_, _, h, w) = img.dims4()?;
    let (ls, extent) = (stride / scale, (tile - stride) / scale);
    let mut rows: Vec<Vec<Tensor>> = vec![];
    for i in (0..h).step_by(stride) {
        let mut row = vec![];
        for j in (0..w).step_by(stride) {
            crate::core::check_abort()?;
            row.push(encode(&img.narrow(2, i, tile.min(h - i))?.narrow(3, j, tile.min(w - j))?)?);
        }
        rows.push(row);
    }
    let mut out_rows = vec![];
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
        let parts = rows[i].iter().map(|t| Ok(t.narrow(2, 0, ls.min(t.dim(2)?))?.narrow(3, 0, ls.min(t.dim(3)?))?)).collect::<Result<Vec<_>>>()?;
        out_rows.push(Tensor::cat(&parts, 3)?);
    }
    Ok(Tensor::cat(&out_rows, 2)?.narrow(2, 0, h / scale)?.narrow(3, 0, w / scale)?)
}

/// Load phase timer, printed with DOTSMITH_STREAM_STATS: elapsed time and bytes read (core::weights) per phase.
pub struct Phases {
    what: &'static str,
    on: bool,
    last: std::cell::Cell<(std::time::Instant, u64)>,
}

impl Phases {
    pub fn start(what: &'static str) -> Self {
        let on = std::env::var("DOTSMITH_STREAM_STATS").is_ok();
        Self { what, on, last: std::cell::Cell::new((std::time::Instant::now(), crate::core::weights::bytes_read())) }
    }

    pub fn mark(&self, phase: &str) {
        if !self.on {
            return;
        }
        let (t, b) = self.last.get();
        let now = (std::time::Instant::now(), crate::core::weights::bytes_read());
        let (s, mb) = (t.elapsed().as_secs_f64(), (now.1 - b) >> 20);
        eprintln!("{}: {phase} {s:.1} s, {mb} MB read ({:.0} MB/s)", self.what, mb as f64 / s.max(1e-3));
        self.last.set(now);
    }
}
