//! High-resolution image -> pixel art, a Rust port of PixelOE's legacy "contrast" pipeline (KohakuBlueleaf/PixelOE,
//! Apache-2.0: outline expansion, colour matching, contrast-based downscale) with the palette step done as k-means in
//! OKLab (the way xprite-pixelate snaps palettes, MIT). Pure CPU, no OpenCV: LAB uses OpenCV's 8-bit convention
//! (L * 255 / 100, a + 128, b + 128), morphology clamps at the borders, blurs reflect (BORDER_REFLECT_101).
#![allow(clippy::excessive_precision, reason = "colour-space constants kept exactly as in the reference implementation")]
use image::{RgbImage, imageops};

type Px = [f32; 3];

// ------------------------------------------------------------------ colour spaces
fn srgb_to_lin(c: f32) -> f32 {
    let c = c / 255.0;
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}
fn lin_to_srgb(c: f32) -> f32 {
    let v = if c <= 0.0031308 { c * 12.92 } else { 1.055 * c.max(0.0).powf(1.0 / 2.4) - 0.055 };
    (v * 255.0).clamp(0.0, 255.0)
}

/// sRGB 0..255 -> CIELAB D65 in OpenCV 8-bit scale (L 0..255, a/b + 128).
fn rgb_to_lab(p: Px) -> Px {
    let (r, g, b) = (srgb_to_lin(p[0]), srgb_to_lin(p[1]), srgb_to_lin(p[2]));
    let x = (0.412453 * r + 0.357580 * g + 0.180423 * b) / 0.950456;
    let y = 0.212671 * r + 0.715160 * g + 0.072169 * b;
    let z = (0.019334 * r + 0.119193 * g + 0.950227 * b) / 1.088754;
    let f = |t: f32| if t > 0.008856 { t.cbrt() } else { 7.787 * t + 16.0 / 116.0 };
    let l = if y > 0.008856 { 116.0 * y.cbrt() - 16.0 } else { 903.3 * y };
    [l * 255.0 / 100.0, 500.0 * (f(x) - f(y)) + 128.0, 200.0 * (f(y) - f(z)) + 128.0]
}

fn lab_to_rgb(p: Px) -> Px {
    let l = p[0] * 100.0 / 255.0;
    let (a, b) = (p[1] - 128.0, p[2] - 128.0);
    let fy = (l + 16.0) / 116.0;
    let (fx, fz) = (fy + a / 500.0, fy - b / 200.0);
    let inv = |f: f32| if f > 0.206893 { f * f * f } else { (f - 16.0 / 116.0) / 7.787 };
    let y = if l > 7.9996 { fy * fy * fy } else { l / 903.3 };
    let (x, z) = (inv(fx) * 0.950456, inv(fz) * 1.088754);
    let r = 3.240479 * x - 1.537150 * y - 0.498535 * z;
    let g = -0.969256 * x + 1.875992 * y + 0.041556 * z;
    let bb = 0.055648 * x - 0.204043 * y + 1.057311 * z;
    [lin_to_srgb(r), lin_to_srgb(g), lin_to_srgb(bb)]
}

fn rgb_to_oklab(p: Px) -> Px {
    let (r, g, b) = (srgb_to_lin(p[0]), srgb_to_lin(p[1]), srgb_to_lin(p[2]));
    let l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b).cbrt();
    let m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b).cbrt();
    let s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b).cbrt();
    [
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    ]
}

fn oklab_to_rgb(p: Px) -> Px {
    let l = p[0] + 0.3963377774 * p[1] + 0.2158037573 * p[2];
    let m = p[0] - 0.1055613458 * p[1] - 0.0638541728 * p[2];
    let s = p[0] - 0.0894841775 * p[1] - 1.2914855480 * p[2];
    let (l, m, s) = (l * l * l, m * m * m, s * s * s);
    [
        lin_to_srgb(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s),
        lin_to_srgb(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s),
        lin_to_srgb(-0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s),
    ]
}

// ------------------------------------------------------------------ plain float images
#[derive(Clone)]
struct Img {
    w: usize,
    h: usize,
    px: Vec<Px>,
}

impl Img {
    fn from_rgb(im: &RgbImage) -> Self {
        Img { w: im.width() as usize, h: im.height() as usize, px: im.pixels().map(|p| [p.0[0] as f32, p.0[1] as f32, p.0[2] as f32]).collect() }
    }
    fn to_rgb(&self) -> RgbImage {
        RgbImage::from_fn(self.w as u32, self.h as u32, |x, y| {
            let p = self.px[y as usize * self.w + x as usize];
            image::Rgb([p[0].clamp(0.0, 255.0) as u8, p[1].clamp(0.0, 255.0) as u8, p[2].clamp(0.0, 255.0) as u8])
        })
    }
    fn map(&self, f: impl Fn(Px) -> Px) -> Self {
        Img { w: self.w, h: self.h, px: self.px.iter().map(|&p| f(p)).collect() }
    }
    /// uint8 truncation, as OpenCV's astype(np.uint8) after the float blend.
    fn trunc(&self) -> Self {
        self.map(|p| [p[0].clamp(0.0, 255.0).floor(), p[1].clamp(0.0, 255.0).floor(), p[2].clamp(0.0, 255.0).floor()])
    }
}

/// 3x3 square (`cross` = false) or cross morphology, per channel, `n` iterations; `max` = dilate.
fn morph(im: &Img, n: usize, max: bool, cross: bool) -> Img {
    let (w, h) = (im.w as i64, im.h as i64);
    let mut cur = im.clone();
    for _ in 0..n {
        let src = cur.px.clone();
        for y in 0..h {
            for x in 0..w {
                let mut acc = src[(y * w + x) as usize];
                for dy in -1..=1i64 {
                    for dx in -1..=1i64 {
                        if cross && dx != 0 && dy != 0 {
                            continue;
                        }
                        let (xx, yy) = ((x + dx).clamp(0, w - 1), (y + dy).clamp(0, h - 1));
                        let q = src[(yy * w + xx) as usize];
                        for c in 0..3 {
                            acc[c] = if max { acc[c].max(q[c]) } else { acc[c].min(q[c]) };
                        }
                    }
                }
                cur.px[(y * w + x) as usize] = acc;
            }
        }
    }
    cur
}

fn reflect101(i: i64, n: i64) -> usize {
    if n == 1 {
        return 0;
    }
    let period = 2 * (n - 1);
    let mut j = i.rem_euclid(period);
    if j >= n {
        j = period - j;
    }
    j as usize
}

/// cv2.GaussianBlur(ksize = 2r + 1, sigma 0 -> 0.3 ((k - 1) / 2 - 1) + 0.8) on one channel, BORDER_REFLECT_101.
fn gauss(ch: &[f32], w: usize, h: usize, radius: usize) -> Vec<f32> {
    let k = 2 * radius + 1;
    let sigma = 0.3 * ((k as f32 - 1.0) * 0.5 - 1.0) + 0.8;
    let mut kern: Vec<f32> = (0..k)
        .map(|i| {
            let d = i as f32 - radius as f32;
            (-(d * d) / (2.0 * sigma * sigma)).exp()
        })
        .collect();
    let s: f32 = kern.iter().sum();
    kern.iter_mut().for_each(|v| *v /= s);
    let r = radius as i64;
    let mut tmp = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            tmp[y * w + x] = (0..k).map(|i| kern[i] * ch[y * w + reflect101(x as i64 + i as i64 - r, w as i64)]).sum();
        }
    }
    let mut out = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            out[y * w + x] = (0..k).map(|i| kern[i] * tmp[reflect101(y as i64 + i as i64 - r, h as i64) * w + x]).sum();
        }
    }
    out
}

/// Wavelet decomposition: (sum of the high bands, the residual low band) over radii 2, 4, .. 2^levels.
fn wavelet(ch: &[f32], w: usize, h: usize, levels: u32) -> (Vec<f32>, Vec<f32>) {
    let mut high = vec![0f32; w * h];
    let mut inp = ch.to_vec();
    for i in 1..=levels {
        let low = gauss(&inp, w, h, 1 << i);
        for j in 0..w * h {
            high[j] += inp[j] - low[j];
        }
        inp = low;
    }
    (high, inp)
}

/// PixelOE match_color: global LAB mean/std transfer, then each RGB channel keeps the source's high bands over the
/// target's low band.
fn match_color(source: &Img, target: &Img, level: u32) -> Img {
    let sl: Vec<Px> = source.px.iter().map(|&p| rgb_to_lab(p)).collect();
    let tl: Vec<Px> = target.px.iter().map(|&p| rgb_to_lab(p)).collect();
    let stats = |v: &[Px]| {
        let n = (v.len() * 3) as f64;
        let m = v.iter().flat_map(|p| p.iter()).map(|&x| x as f64 / 255.0).sum::<f64>() / n;
        let s = (v.iter().flat_map(|p| p.iter()).map(|&x| (x as f64 / 255.0 - m).powi(2)).sum::<f64>() / n).sqrt();
        (m as f32, s.max(1e-6) as f32)
    };
    let ((ms, ss), (mt, st)) = (stats(&sl), stats(&tl));
    let moved = Img {
        w: source.w,
        h: source.h,
        px: sl
            .iter()
            .map(|p| {
                let q = p.map(|x| (((x / 255.0 - ms) / ss * st + mt) * 255.0).clamp(0.0, 255.0).floor());
                lab_to_rgb(q).map(|x| x.round())
            })
            .collect(),
    };
    let (w, h) = (source.w, source.h);
    let mut out = moved.clone();
    for c in 0..3 {
        let s: Vec<f32> = moved.px.iter().map(|p| p[c]).collect();
        let t: Vec<f32> = target.px.iter().map(|p| p[c]).collect();
        let (hi, _) = wavelet(&s, w, h, level);
        let (_, lo) = wavelet(&t, w, h, level);
        for j in 0..w * h {
            out.px[j][c] = (hi[j] + lo[j]).clamp(0.0, 255.0).floor();
        }
    }
    out
}

/// PixelOE apply_chunk: each stride x stride block gets f() of the kernel x kernel window around it (edge padded).
fn chunk_grid(ch: &[f32], w: usize, h: usize, kernel: usize, stride: usize, f: impl Fn(&mut [f32]) -> f32) -> (Vec<f32>, usize, usize) {
    let shift = kernel.saturating_sub(stride) / 2;
    let (gw, gh) = (w.div_ceil(stride), h.div_ceil(stride));
    let mut g = vec![0f32; gw * gh];
    let mut buf = Vec::with_capacity(kernel * kernel);
    for by in 0..gh {
        for bx in 0..gw {
            buf.clear();
            for ky in 0..kernel {
                for kx in 0..kernel {
                    let x = (bx * stride + kx) as i64 - shift as i64;
                    let y = (by * stride + ky) as i64 - shift as i64;
                    buf.push(ch[y.clamp(0, h as i64 - 1) as usize * w + x.clamp(0, w as i64 - 1) as usize]);
                }
            }
            g[by * gw + bx] = f(&mut buf);
        }
    }
    (g, gw, gh)
}

fn median_lower(v: &mut [f32]) -> f32 {
    let k = (v.len() - 1) / 2;
    *v.select_nth_unstable_by(k, |a, b| a.total_cmp(b)).1
}

/// Bilinear (half-pixel centres, cv2 INTER_LINEAR) upsample of a gw x gh grid to w x h.
fn upsample(g: &[f32], gw: usize, gh: usize, w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0f32; w * h];
    for y in 0..h {
        let fy = ((y as f32 + 0.5) * gh as f32 / h as f32 - 0.5).clamp(0.0, (gh - 1) as f32);
        let (y0, ty) = (fy.floor() as usize, fy.fract());
        let y1 = (y0 + 1).min(gh - 1);
        for x in 0..w {
            let fx = ((x as f32 + 0.5) * gw as f32 / w as f32 - 0.5).clamp(0.0, (gw - 1) as f32);
            let (x0, tx) = (fx.floor() as usize, fx.fract());
            let x1 = (x0 + 1).min(gw - 1);
            let a = g[y0 * gw + x0] * (1.0 - tx) + g[y0 * gw + x1] * tx;
            let b = g[y1 * gw + x0] * (1.0 - tx) + g[y1 * gw + x1] * tx;
            out[y * w + x] = a * (1.0 - ty) + b * ty;
        }
    }
    out
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Where to erode (bright surroundings) vs dilate (dark ones): PixelOE expansion_weight, normalised to ~0..1.
fn expansion_weight(im: &Img, k: usize, stride: usize, avg_scale: f32, dist_scale: f32) -> Vec<f32> {
    let l: Vec<f32> = im.px.iter().map(|&p| rgb_to_lab(p)[0].round() / 255.0).collect();
    let (avg, gw, gh) = chunk_grid(&l, im.w, im.h, k * 2, stride, median_lower);
    let (mx, _, _) = chunk_grid(&l, im.w, im.h, k, stride, |v| v.iter().cloned().fold(f32::MIN, f32::max));
    let (mn, _, _) = chunk_grid(&l, im.w, im.h, k, stride, |v| v.iter().cloned().fold(f32::MAX, f32::min));
    let g: Vec<f32> = (0..gw * gh)
        .map(|i| {
            let (bright, dark) = (mx[i] - avg[i], avg[i] - mn[i]);
            sigmoid((avg[i] - 0.5) * avg_scale - (bright - dark) * dist_scale)
        })
        .collect();
    let o = upsample(&g, gw, gh, im.w, im.h);
    let lo = o.iter().cloned().fold(f32::MAX, f32::min);
    let hi = o.iter().cloned().fold(f32::MIN, f32::max).max(1e-6);
    o.iter().map(|v| (v - lo) / hi).collect()
}

fn outline_expansion(im: &Img, thickness: usize, k: usize) -> Img {
    let wgt = expansion_weight(im, k, ((k / 4) * 2).max(1), 9.0, 4.0); // patch 2-3: stride 0 divided by zero
    let er = morph(im, thickness, false, false);
    let di = morph(im, thickness, true, false);
    let mut out = im.clone();
    for i in 0..im.px.len() {
        let w = wgt[i];
        let ow = sigmoid((w - 0.5) * 5.0) * 0.25;
        for c in 0..3 {
            let v = er.px[i][c] * w + di.px[i][c] * (1.0 - w);
            out.px[i][c] = v * (1.0 - ow) + im.px[i][c] * ow;
        }
    }
    let out = out.trunc();
    let out = morph(&out, thickness, false, true);
    let out = morph(&out, thickness * 2, true, true);
    morph(&out, thickness, false, true)
}

/// One pixel per patch: L picks the centre, the minimum or the maximum by the patch's skew; a and b the median.
fn contrast_downscale(im: &Img, patch: usize, ow: usize, oh: usize) -> Img {
    let lab: Vec<Px> = im.px.iter().map(|&p| rgb_to_lab(p).map(|x| x.round())).collect();
    let mut out = Img { w: ow, h: oh, px: vec![[0.0; 3]; ow * oh] };
    let mut ls = Vec::with_capacity(patch * patch);
    let (mut as_, mut bs) = (Vec::with_capacity(patch * patch), Vec::with_capacity(patch * patch));
    for by in 0..oh {
        for bx in 0..ow {
            ls.clear();
            as_.clear();
            bs.clear();
            for y in by * patch..(by + 1) * patch {
                for x in bx * patch..(bx + 1) * patch {
                    let p = lab[y.min(im.h - 1) * im.w + x.min(im.w - 1)];
                    ls.push(p[0]);
                    as_.push(p[1]);
                    bs.push(p[2]);
                }
            }
            let mid = ls[ls.len() / 2];
            let mu = ls.iter().sum::<f32>() / ls.len() as f32;
            let mx = ls.iter().cloned().fold(f32::MIN, f32::max);
            let mn = ls.iter().cloned().fold(f32::MAX, f32::min);
            let med = median_lower(&mut ls.clone());
            let l = if med < mu && mx - med > med - mn {
                mn
            } else if med > mu && mx - med < med - mn {
                mx
            } else {
                mid
            };
            out.px[by * ow + bx] = lab_to_rgb([l, median_lower(&mut as_), median_lower(&mut bs)]).map(|x| x.round());
        }
    }
    out
}

/// k-means in OKLab (k-means++ seeding with a fixed LCG, 24 rounds); every pixel takes its centroid colour.
fn kmeans_oklab(im: &Img, k: usize) -> Img {
    let pts: Vec<Px> = im.px.iter().map(|&p| rgb_to_oklab(p)).collect();
    let n = pts.len();
    let k = k.min(n).max(1);
    let d2 = |a: &Px, b: &Px| (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2);
    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut rnd = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut cent = vec![pts[n / 2]];
    let mut best: Vec<f32> = pts.iter().map(|p| d2(p, &cent[0])).collect();
    while cent.len() < k {
        let total: f64 = best.iter().map(|&v| v as f64).sum();
        if total <= 0.0 {
            break;
        }
        let mut t = rnd() * total;
        let mut pick = n - 1;
        for (i, &v) in best.iter().enumerate() {
            t -= v as f64;
            if t <= 0.0 {
                pick = i;
                break;
            }
        }
        cent.push(pts[pick]);
        for (i, p) in pts.iter().enumerate() {
            best[i] = best[i].min(d2(p, &cent[cent.len() - 1]));
        }
    }
    let mut assign = vec![0usize; n];
    for _ in 0..24 {
        for (i, p) in pts.iter().enumerate() {
            assign[i] = (0..cent.len()).min_by(|&a, &b| d2(p, &cent[a]).total_cmp(&d2(p, &cent[b]))).unwrap();
        }
        let mut sum = vec![[0f32; 3]; cent.len()];
        let mut cnt = vec![0usize; cent.len()];
        for (i, p) in pts.iter().enumerate() {
            for c in 0..3 {
                sum[assign[i]][c] += p[c];
            }
            cnt[assign[i]] += 1;
        }
        for j in 0..cent.len() {
            if cnt[j] > 0 {
                cent[j] = sum[j].map(|v| v / cnt[j] as f32);
            }
        }
    }
    let cols: Vec<Px> = cent.iter().map(|&c| oklab_to_rgb(c).map(|x| x.round())).collect();
    Img { w: im.w, h: im.h, px: assign.iter().map(|&a| cols[a]).collect() }
}

/// Image (any size) -> out_w x out_h pixel art. `patch` source pixels per output pixel are analysed (the image is
/// first resized to out x patch); `colors` = palette size (None keeps every colour).
pub fn pixelize(src: &RgbImage, out_w: u32, out_h: u32, patch: u32, thickness: u32, colors: Option<usize>) -> RgbImage {
    let patch = patch.max(2);
    let (ow, oh) = (out_w.max(1), out_h.max(1));
    let big = imageops::resize(src, ow * patch, oh * patch, imageops::FilterType::Triangle);
    let org = Img::from_rgb(&big);
    let mut im = org.clone();
    if thickness > 0 {
        im = outline_expansion(&im, thickness as usize, patch as usize);
    }
    im = match_color(&im, &org, 5);
    let small = contrast_downscale(&im, patch as usize, ow as usize, oh as usize);
    let small = match colors {
        Some(k) => match_color(&kmeans_oklab(&small, k), &small, 3),
        None => small,
    };
    small.to_rgb()
}
