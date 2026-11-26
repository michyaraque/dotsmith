//! The live preview while it streams in: a soft mesh gradient of the preview's own colours. Each step's estimate is
//! reduced to a small grid of colours, every cell becomes a soft blob drifting on its own slow orbit, a new step blends
//! in over a moment, and once the sprite starts taking shape the field gives way to the estimate itself.
use gpui::RenderImage;
use image::RgbaImage;
use std::sync::Arc;
use std::time::Instant;

const GRID: usize = 6;
const FIELD: f32 = 48.;
const BLEND_S: f32 = 0.7;
const FRAME_MS: u128 = 33;
const IMAGE: u32 = 128;
/// Progress where the estimate starts showing through and where it has fully replaced the field.
const REVEAL: (f32, f32) = (0.15, 0.5);

/// One step's colours (GRID x GRID cells, row by row), the image's width / height and the estimate itself, small.
#[derive(Clone, Default)]
pub struct Colors {
    cells: Vec<[f32; 3]>,
    aspect: f32,
    image: Option<Arc<RgbaImage>>,
}

/// Mean colour of each cell (see-through pixels count less), a little more saturated and lifted when the whole is
/// dark: early estimates are greyish and dim. `key`: the flat backdrop a LoRA paints on (removed after), made
/// see-through here and coloured like the rest so it does not flood the field.
pub fn colors(img: &RgbaImage, key: Option<[u8; 3]>) -> Colors {
    let keyed = key.map(|k| keyed(img, k));
    let img = keyed.as_ref().unwrap_or(img);
    let (w, h) = img.dimensions();
    let mut sum = vec![[0f32; 4]; GRID * GRID];
    for (x, y, p) in img.enumerate_pixels() {
        let cell = (y as usize * GRID / h as usize) * GRID + x as usize * GRID / w as usize;
        let a = 0.15 + 0.85 * p[3] as f32 / 255.;
        for c in 0..3 {
            sum[cell][c] += p[c] as f32 / 255. * a;
        }
        sum[cell][3] += a;
    }
    let cells: Vec<[f32; 3]> = sum
        .iter()
        .map(|s| {
            let c = [s[0] / s[3].max(1e-6), s[1] / s[3].max(1e-6), s[2] / s[3].max(1e-6)];
            let l = (c[0] + c[1] + c[2]) / 3.;
            c.map(|v| (l + (v - l) * 1.35).clamp(0., 1.))
        })
        .collect();
    let mean = cells.iter().map(|c| (c[0] + c[1] + c[2]) / 3.).sum::<f32>() / cells.len() as f32;
    let lift = (0.45 / mean.max(0.05)).clamp(1., 2.2);
    let cells = cells.into_iter().map(|c| c.map(|v| 1. - (1. - v).powf(lift))).collect();
    let k = (IMAGE as f32 / w.max(h) as f32).min(1.);
    let small =
        image::imageops::resize(img, ((w as f32 * k).round() as u32).max(1), ((h as f32 * k).round() as u32).max(1), image::imageops::FilterType::Triangle);
    Colors { cells, aspect: w as f32 / h.max(1) as f32, image: Some(Arc::new(small)) }
}

/// `img` with pixels near `key` faded out (softly: early estimates blur the edge) and tinted with the mean of the rest.
fn keyed(img: &RgbaImage, key: [u8; 3]) -> RgbaImage {
    let near = |p: &image::Rgba<u8>| {
        let d = (0..3).map(|c| (p[c] as f32 - key[c] as f32).powi(2)).sum::<f32>().sqrt();
        ((d - 60.) / 90.).clamp(0., 1.)
    };
    let (mut mean, mut n) = ([0f32; 3], 0f32);
    for p in img.pixels() {
        let a = near(p);
        for c in 0..3 {
            mean[c] += p[c] as f32 * a;
        }
        n += a;
    }
    let mean = mean.map(|m| m / n.max(1.));
    let mut out = img.clone();
    for p in out.pixels_mut() {
        let a = near(p);
        for c in 0..3 {
            p[c] = (mean[c] + (p[c] as f32 - mean[c]) * a).round() as u8;
        }
        p[3] = (p[3] as f32 * a).round() as u8;
    }
    out
}

#[derive(Default)]
pub struct Aurora {
    from: Colors,
    to: Colors,
    since: Option<Instant>,
    tex: Option<Arc<RenderImage>>,
    made: Option<Instant>,
}

impl Aurora {
    pub fn push(&mut self, next: Colors) {
        self.from = self.current();
        self.to = next;
        self.since = Some(Instant::now());
    }

    pub fn is_live(&self) -> bool {
        !self.to.cells.is_empty()
    }

    /// Back to nothing; the last texture comes back to be dropped.
    pub fn clear(&mut self) -> Option<Arc<RenderImage>> {
        std::mem::take(self).tex
    }

    /// How far the blend into the newest step is (0..1, eased).
    fn blend(&self) -> f32 {
        let k = self.since.map_or(1., |s| (s.elapsed().as_secs_f32() / BLEND_S).min(1.));
        k * k * (3. - 2. * k)
    }

    fn current(&self) -> Colors {
        if self.from.cells.len() != self.to.cells.len() {
            return self.to.clone();
        }
        let k = self.blend();
        let cells = self.from.cells.iter().zip(&self.to.cells).map(|(a, b)| [0, 1, 2].map(|c| a[c] + (b[c] - a[c]) * k)).collect();
        let image = match (&self.from.image, &self.to.image) {
            (Some(a), Some(b)) if k < 1. && a.dimensions() == b.dimensions() => {
                let mut m = (**b).clone();
                for (p, q) in m.pixels_mut().zip(a.pixels()) {
                    for c in 0..4 {
                        p[c] = (q[c] as f32 + (p[c] as f32 - q[c] as f32) * k).round() as u8;
                    }
                }
                Some(Arc::new(m))
            }
            _ => self.to.image.clone(),
        };
        Colors { cells, aspect: self.to.aspect, image }
    }

    /// The field at time `t` (s) and progress `settled` (0..1), rebuilt at most every 33 ms. Returns the texture to
    /// paint and the one it replaced (to drop).
    pub fn frame(&mut self, t: f32, settled: f32) -> (Option<Arc<RenderImage>>, Option<Arc<RenderImage>>) {
        if !self.is_live() || self.made.is_some_and(|m| m.elapsed().as_millis() < FRAME_MS) {
            return (self.tex.clone(), None);
        }
        let now = self.current();
        let (fw, fh) = if now.aspect >= 1. { (FIELD, FIELD / now.aspect) } else { (FIELD * now.aspect, FIELD) };
        let (fw, fh) = (fw.round().max(2.) as u32, fh.round().max(2.) as u32);
        let (sx, sy) = (fw as f32 / FIELD.max(fw as f32).max(fh as f32), fh as f32 / FIELD.max(fw as f32).max(fh as f32));
        let calm = settled.clamp(0., 1.);
        let sigma = 0.2 - 0.07 * calm;
        let orbit = 0.05 + 0.06 * (1. - calm);
        let centres: Vec<(f32, f32)> = (0..GRID * GRID)
            .map(|i| {
                let h = |k: u32| ((i as u32).wrapping_mul(2_654_435_761).wrapping_add(k.wrapping_mul(40_503)) % 1000) as f32 / 1000.;
                let speed = 0.25 + 0.3 * h(1);
                let (gx, gy) = ((i % GRID) as f32 + 0.5, (i / GRID) as f32 + 0.5);
                ((gx / GRID as f32 + orbit * (t * speed + h(2) * 6.3).sin()) * sx, (gy / GRID as f32 + orbit * (t * speed * 0.8 + h(3) * 6.3).cos()) * sy)
            })
            .collect();
        let inv = -1. / (2. * sigma * sigma);
        let img = RgbaImage::from_fn(fw, fh, |x, y| {
            let (u, v) = ((x as f32 + 0.5) / fw as f32 * sx, (y as f32 + 0.5) / fh as f32 * sy);
            let (mut acc, mut total) = ([0f32; 3], 0f32);
            for (c, col) in centres.iter().zip(&now.cells) {
                let w = (((u - c.0).powi(2) + (v - c.1).powi(2)) * inv).exp();
                for k in 0..3 {
                    acc[k] += col[k] * w;
                }
                total += w;
            }
            let px = acc.map(|a| (a / total.max(1e-6) * 255.).round().clamp(0., 255.) as u8);
            image::Rgba([px[0], px[1], px[2], 255])
        });
        let s = ((calm - REVEAL.0) / (REVEAL.1 - REVEAL.0)).clamp(0., 1.);
        let reveal = s * s * (3. - 2. * s);
        let img = match now.image.as_deref() {
            Some(est) if reveal > 0. => {
                let (w, h) = est.dimensions();
                let mut out = image::imageops::resize(&img, w, h, image::imageops::FilterType::Triangle);
                for (p, e) in out.pixels_mut().zip(est.pixels()) {
                    let k = reveal * e[3] as f32 / 255.; // see-through parts of the estimate stay field
                    for c in 0..3 {
                        p[c] = (p[c] as f32 + (e[c] as f32 - p[c] as f32) * k).round() as u8;
                    }
                }
                out
            }
            _ => img,
        };
        self.made = Some(Instant::now());
        let old = self.tex.replace(crate::pixels::render(&img, 1));
        (self.tex.clone(), old)
    }
}
