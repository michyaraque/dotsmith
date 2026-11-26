//! Sprite post-processing: model image -> game sprite.
//! depixel (grid-phase search, block median, alpha snap, shared palette), despeckle, cutting, fitting, mirror,
//! matte / pixelate for photos, the angle-gizmo warp. Pure CPU image code.
use image::{Rgba, RgbaImage, imageops};
use std::collections::VecDeque;

pub const COLORS: usize = 192;
const SPECK: usize = 6;
const MIN_PX: usize = 6;

fn round_even(x: f32) -> f32 {
    // numpy .round(): half to even
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 { 2.0 * (x / 2.0).round() } else { r }
}

/// Magenta-tinted pixel: green far below red and blue, red and blue both high and close. Catches the background
/// blended into dark outlines ((166, 0, 146), (227, 3, 213)) that a distance to #FF00FF misses, while purples
/// (blue well above red) and pinks (low blue) stay.
pub fn is_magentaish(p: &Rgba<u8>) -> bool {
    let (r, g, b) = (p.0[0] as i32, p.0[1] as i32, p.0[2] as i32);
    let (lo, hi) = (r.min(b), r.max(b));
    lo >= 110 && g * 100 <= lo * 45 && (r - b).abs() * 100 <= hi * 35
}

/// Leftover background in a native magenta-keyed sprite. OTClient removes the exact #FF00FF anywhere, with no
/// neighbourhood test; a generator never paints it exactly, so:
///  1. near-pure magenta (|dr| + g + |db| < 30) anywhere, enclosed or isolated;
///  2. magenta-tinted groups of 1-2 px (8-connected) with no purple neighbour: stray background dots. A real purple
///     highlight sits in its own ramp, so it has purple neighbours and stays.
///
/// Checked on 2421 hand-drawn game items (real alpha): 0.013 % of their pixels would be touched; both generated
/// creature sheets came out with no magenta-tinted pixel left.
pub fn drop_magenta(im: &RgbaImage) -> RgbaImage {
    let (w, h) = im.dimensions();
    let mut out = im.clone();
    let idx = |x: u32, y: u32| (y * w + x) as usize;
    let opaque = |p: &Rgba<u8>| p.0[3] > 0;
    let purple = |p: &Rgba<u8>| p.0[3] > 0 && p.0[0] as i32 > p.0[1] as i32 + 30 && p.0[2] as i32 > p.0[1] as i32 + 30;
    for p in out.pixels_mut() {
        let d = (255 - p.0[0] as i32) + p.0[1] as i32 + (255 - p.0[2] as i32);
        if opaque(p) && d < 30 {
            p.0[3] = 0;
        }
    }
    let tint: Vec<bool> = out.pixels().map(|p| opaque(p) && is_magentaish(p)).collect();
    let mut seen = vec![false; tint.len()];
    let nb = |x: u32, y: u32| {
        let mut v = Vec::with_capacity(8);
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                if (dx, dy) != (0, 0) && nx >= 0 && ny >= 0 && (nx as u32) < w && (ny as u32) < h {
                    v.push((nx as u32, ny as u32));
                }
            }
        }
        v
    };
    for y in 0..h {
        for x in 0..w {
            if !tint[idx(x, y)] || seen[idx(x, y)] {
                continue;
            }
            let mut group = vec![(x, y)];
            seen[idx(x, y)] = true;
            let mut i = 0;
            while i < group.len() && group.len() <= 2 {
                let (cx, cy) = group[i];
                for (nx, ny) in nb(cx, cy) {
                    if tint[idx(nx, ny)] && !seen[idx(nx, ny)] {
                        seen[idx(nx, ny)] = true;
                        group.push((nx, ny));
                    }
                }
                i += 1;
            }
            if group.len() > 2 {
                continue;
            }
            let purple_ring = group.iter().flat_map(|&(gx, gy)| nb(gx, gy)).any(|(nx, ny)| !group.contains(&(nx, ny)) && purple(im.get_pixel(nx, ny)));
            if !purple_ring {
                for &(gx, gy) in &group {
                    out.get_pixel_mut(gx, gy).0[3] = 0;
                }
            }
        }
    }
    out
}

/// The background colour of a sheet on a flat backdrop: the most common colour along the border (5-bit bins),
/// averaged inside its bin, and the share of border pixels it covers.
pub fn border_color(im: &RgbaImage) -> Option<([u8; 3], f32)> {
    let (w, h) = im.dimensions();
    if w < 3 || h < 3 {
        return None;
    }
    let mut bins: std::collections::HashMap<(u8, u8, u8), (u32, [u64; 3])> = Default::default();
    // the share is over every border pixel, transparent ones included: a sheet whose backdrop is already keyed out
    // has a mostly transparent border and must not have an item colour mistaken for its "backdrop"
    let mut n = 0u32;
    let mut add = |p: &Rgba<u8>| {
        n += 1;
        if p.0[3] < 128 {
            return;
        }
        let e = bins.entry((p.0[0] >> 3, p.0[1] >> 3, p.0[2] >> 3)).or_insert((0, [0; 3]));
        e.0 += 1;
        for c in 0..3 {
            e.1[c] += p.0[c] as u64;
        }
    };
    for x in 0..w {
        add(im.get_pixel(x, 0));
        add(im.get_pixel(x, h - 1));
    }
    for y in 1..h - 1 {
        add(im.get_pixel(0, y));
        add(im.get_pixel(w - 1, y));
    }
    // neighbouring bins belong to the same noisy backdrop: score each bin with its 3x3x3 neighbourhood
    let score = |k: &(u8, u8, u8)| -> u32 {
        let mut s = 0;
        for dr in -1i32..=1 {
            for dg in -1i32..=1 {
                for db in -1i32..=1 {
                    let kk = ((k.0 as i32 + dr) as u8, (k.1 as i32 + dg) as u8, (k.2 as i32 + db) as u8);
                    s += bins.get(&kk).map_or(0, |e| e.0);
                }
            }
        }
        s
    };
    let best = bins.keys().max_by_key(|k| score(k))?;
    let (cnt, sum) = bins[best];
    let col = [(sum[0] / cnt as u64) as u8, (sum[1] / cnt as u64) as u8, (sum[2] / cnt as u64) as u8];
    Some((col, score(best) as f32 / n.max(1) as f32))
}

/// Remove a measured flat backdrop that is not the exact key colour (third-party LoRAs paint "magenta" as a dull
/// pink like (183, 81, 143) with noise, which the #FF00FF key misses):
///  1. flood fill from the border over pixels within `tol` (|dr|+|dg|+|db|) of the measured colour;
///  2. enclosed holes when within tol / 2 (gaps between leaves, the inside of a ring);
///  3. one ring of edge pixels still close to it (< 1.6 tol) that touch the removed area: the blend of backdrop and
///     outline the generator leaves around every item.
///
/// Returns the cleaned image and the backdrop colour that was measured.
pub fn remove_backdrop(im: &RgbaImage, tol: i32) -> Option<(RgbaImage, [u8; 3])> {
    let (key, share) = border_color(im)?;
    if share < 0.15 {
        return None; // no dominant border colour (packs whose items fill the edge still show ~20 %): not a backdrop
    }
    let (w, h) = im.dimensions();
    let d = |p: &Rgba<u8>| (0..3).map(|c| (p.0[c] as i32 - key[c] as i32).abs()).sum::<i32>();
    let mut out = key_color(im, key, tol, tol / 2);
    let gone: Vec<bool> = out.pixels().map(|p| p.0[3] == 0).collect();
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            if gone[i] || d(im.get_pixel(x, y)) >= tol * 16 / 10 {
                continue;
            }
            let touches = [(x.wrapping_sub(1), y), (x + 1, y), (x, y.wrapping_sub(1)), (x, y + 1)]
                .iter()
                .any(|&(nx, ny)| nx < w && ny < h && gone[(ny * w + nx) as usize]);
            if touches {
                out.get_pixel_mut(x, y).0[3] = 0;
            }
        }
    }
    Some((out, key))
}

/// `remove_backdrop` only when the measured backdrop is magenta-hued (red and blue well above green): what magenta
/// LoRAs paint, dull or not. Dark or opaque sheets with no backdrop (their most common border colour is an outline)
/// are left alone.
pub fn remove_magenta_backdrop(im: &RgbaImage, tol: i32) -> Option<(RgbaImage, [u8; 3])> {
    let (c, _) = border_color(im)?;
    let hue = c[0] as i32 > c[1] as i32 + 40 && c[2] as i32 > c[1] as i32 + 20;
    if hue { remove_backdrop(im, tol) } else { None }
}

/// Flat-background key (port of lib/prx_data.key_out): pixels near `key` connected to the border become transparent
/// (flood fill, |dr|+|dg|+|db| < tol), enclosed holes (a ring, an open chest) only when almost exact (< hole_tol).
/// For families with no alpha VAE (Krea 2 generates on #FF00FF).
pub fn key_color(im: &RgbaImage, key: [u8; 3], tol: i32, hole_tol: i32) -> RgbaImage {
    let (w, h) = im.dimensions();
    let d = |p: &Rgba<u8>| (0..3).map(|c| (p.0[c] as i32 - key[c] as i32).abs()).sum::<i32>();
    let near: Vec<bool> = im.pixels().map(|p| d(p) < tol).collect();
    let mut seen = vec![false; (w * h) as usize];
    let mut q = VecDeque::new();
    for y in 0..h {
        for x in [0, w - 1] {
            q.push_back((x, y));
        }
    }
    for x in 0..w {
        for y in [0, h - 1] {
            q.push_back((x, y));
        }
    }
    q.retain(|&(x, y)| near[(y * w + x) as usize]);
    for &(x, y) in &q {
        seen[(y * w + x) as usize] = true;
    }
    while let Some((x, y)) = q.pop_front() {
        let nb = [(x.wrapping_sub(1), y), (x + 1, y), (x, y.wrapping_sub(1)), (x, y + 1)];
        for (nx, ny) in nb {
            if nx < w && ny < h {
                let i = (ny * w + nx) as usize;
                if near[i] && !seen[i] {
                    seen[i] = true;
                    q.push_back((nx, ny));
                }
            }
        }
    }
    let mut out = im.clone();
    for (i, p) in out.pixels_mut().enumerate() {
        if seen[i] || d(p) < hole_tol {
            p.0[3] = 0;
        }
    }
    out
}

/// Scale saturation by k around each pixel's max channel (hue and value unchanged).
pub fn desaturate(im: &RgbaImage, k: f32) -> RgbaImage {
    if k >= 1.0 {
        return im.clone();
    }
    let mut out = im.clone();
    for p in out.pixels_mut() {
        let mx = p.0[0].max(p.0[1]).max(p.0[2]) as f32;
        for c in 0..3 {
            p.0[c] = round_even(mx - (mx - p.0[c] as f32) * k).clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// Each scale x scale block -> one pixel (median), at the grid phase where blocks are flattest; alpha snapped.
pub fn depixel(im: &RgbaImage, colors: usize, scale: usize) -> RgbaImage {
    let (w0, h0) = (im.width() as usize, im.height() as usize);
    let (hh, ww) = (h0 / scale, w0 / scale);
    // edge-padded float copy
    let (pw, ph) = (w0 + scale, h0 + scale);
    let mut a = vec![0f32; pw * ph * 4];
    for y in 0..ph {
        for x in 0..pw {
            let p = im.get_pixel(x.min(w0 - 1) as u32, y.min(h0 - 1) as u32).0;
            for c in 0..4 {
                a[(y * pw + x) * 4 + c] = p[c] as f32;
            }
        }
    }
    let step = (scale / 4).max(1);
    let mut best = (f64::MAX, 0, 0);
    for dy in (0..scale).step_by(step) {
        for dx in (0..scale).step_by(step) {
            let mut total = 0f64;
            for by in 0..hh {
                for bx in 0..ww {
                    for c in 0..4 {
                        let (mut s, mut s2) = (0f64, 0f64);
                        for y in 0..scale {
                            for x in 0..scale {
                                let v = a[((dy + by * scale + y) * pw + dx + bx * scale + x) * 4 + c] as f64;
                                s += v;
                                s2 += v * v;
                            }
                        }
                        let n = (scale * scale) as f64;
                        let m = s / n;
                        total += (s2 / n - m * m).max(0.0).sqrt();
                    }
                }
            }
            let score = total / (hh * ww * 4) as f64;
            if score < best.0 {
                best = (score, dy, dx);
            }
        }
    }
    let (_, dy, dx) = best;
    let mut out = RgbaImage::new(ww as u32, hh as u32);
    let mut vals = vec![0f32; scale * scale];
    for by in 0..hh {
        for bx in 0..ww {
            let mut px = [0f32; 4];
            for c in 0..4 {
                let mut i = 0;
                for y in 0..scale {
                    for x in 0..scale {
                        vals[i] = a[((dy + by * scale + y) * pw + dx + bx * scale + x) * 4 + c];
                        i += 1;
                    }
                }
                vals.sort_by(|p, q| p.partial_cmp(q).unwrap());
                let n = vals.len();
                px[c] = if n % 2 == 1 { vals[n / 2] } else { (vals[n / 2 - 1] + vals[n / 2]) / 2.0 };
            }
            px[3] = if px[3] >= 128.0 { 255.0 } else { 0.0 };
            if px[3] == 0.0 {
                px = [0.0; 4];
            }
            out.put_pixel(bx as u32, by as u32, Rgba(px.map(|v| round_even(v).clamp(0.0, 255.0) as u8)));
        }
    }
    if colors > 0 {
        quantize_rgb(&mut out, colors);
    }
    out
}

/// Median-cut palette of `n` colours over the RGB of every pixel (transparent ones count as black, like PIL's
/// RGB conversion), each pixel mapped to its nearest palette entry; alpha untouched.
pub fn quantize_rgb(im: &mut RgbaImage, n: usize) {
    let px: Vec<[u8; 3]> = im.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();
    let mut uniq = px.clone();
    uniq.sort();
    uniq.dedup();
    if uniq.len() <= n {
        return; // already within the budget: exact
    }
    let mut boxes: Vec<Vec<[u8; 3]>> = vec![px.clone()];
    while boxes.len() < n {
        // split the box with the widest channel range at its median
        let (bi, axis, range) = boxes
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let (mut lo, mut hi) = ([255u8; 3], [0u8; 3]);
                for p in b {
                    for c in 0..3 {
                        lo[c] = lo[c].min(p[c]);
                        hi[c] = hi[c].max(p[c]);
                    }
                }
                let axis = (0..3).max_by_key(|&c| hi[c] - lo[c]).unwrap();
                (i, axis, hi[axis] - lo[axis])
            })
            .max_by_key(|x| (x.2, std::cmp::Reverse(x.0)))
            .unwrap();
        if range == 0 {
            break;
        }
        let mut b = boxes.swap_remove(bi);
        b.sort_by_key(|p| p[axis]);
        let mid = b.len() / 2;
        let mut cut = mid;
        while cut < b.len() && cut > 0 && b[cut][axis] == b[cut - 1][axis] {
            cut += 1;
        }
        if cut == b.len() {
            cut = mid;
            while cut > 0 && b[cut][axis] == b[cut - 1][axis] {
                cut -= 1;
            }
        }
        let right = b.split_off(cut.max(1));
        boxes.push(b);
        boxes.push(right);
    }
    let pal: Vec<[f32; 3]> = boxes
        .iter()
        .map(|b| {
            let mut s = [0f64; 3];
            for p in b {
                for c in 0..3 {
                    s[c] += p[c] as f64;
                }
            }
            s.map(|v| (v / b.len() as f64) as f32)
        })
        .collect();
    for p in im.pixels_mut() {
        let q = [p.0[0] as f32, p.0[1] as f32, p.0[2] as f32];
        let best = pal
            .iter()
            .min_by(|a, b| {
                let d = |x: &[f32; 3]| (0..3).map(|c| (x[c] - q[c]).powi(2)).sum::<f32>();
                d(a).partial_cmp(&d(b)).unwrap()
            })
            .unwrap();
        for c in 0..3 {
            p.0[c] = best[c].round().clamp(0.0, 255.0) as u8;
        }
    }
}

/// Connected components of `mask` (8- or 4-connected): label per pixel (0 = background) and the count.
fn label(mask: &[bool], w: usize, h: usize, eight: bool) -> (Vec<u32>, u32) {
    let mut lab = vec![0u32; w * h];
    let mut n = 0;
    let mut q = VecDeque::new();
    for s in 0..w * h {
        if !mask[s] || lab[s] != 0 {
            continue;
        }
        n += 1;
        lab[s] = n;
        q.push_back(s);
        while let Some(i) = q.pop_front() {
            let (x, y) = ((i % w) as i64, (i / w) as i64);
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    if (dx == 0 && dy == 0) || (!eight && dx != 0 && dy != 0) {
                        continue;
                    }
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                        continue;
                    }
                    let j = ny as usize * w + nx as usize;
                    if mask[j] && lab[j] == 0 {
                        lab[j] = n;
                        q.push_back(j);
                    }
                }
            }
        }
    }
    (lab, n)
}

/// Drop opaque 8-connected islands smaller than SPECK pixels.
pub fn despeckle(im: &RgbaImage) -> RgbaImage {
    let (w, h) = (im.width() as usize, im.height() as usize);
    let mask: Vec<bool> = im.pixels().map(|p| p.0[3] > 0).collect();
    let (lab, n) = label(&mask, w, h, true);
    let mut size = vec![0usize; n as usize + 1];
    for &l in &lab {
        size[l as usize] += 1;
    }
    let mut out = im.clone();
    for (i, p) in out.pixels_mut().enumerate() {
        if lab[i] > 0 && size[lab[i] as usize] < SPECK {
            *p = Rgba([0, 0, 0, 0]);
        }
    }
    out
}

/// The colour of a flat but noisy backdrop (a resampled screenshot's magenta, a studio photo's white): the per-channel
/// median of the opaque border pixels, when at least `share` of the whole border lies within `tol` (|dr|+|dg|+|db|).
pub fn flat_backdrop(im: &RgbaImage, tol: i32, share: f32) -> Option<[u8; 3]> {
    let (w, h) = im.dimensions();
    if w < 3 || h < 3 {
        return None;
    }
    let edge = (0..w).flat_map(|x| [(x, 0), (x, h - 1)]).chain((1..h - 1).flat_map(|y| [(0, y), (w - 1, y)]));
    let total = 2 * w + 2 * (h - 2);
    let opaque: Vec<[u8; 3]> = edge.map(|(x, y)| im.get_pixel(x, y).0).filter(|p| p[3] >= 128).map(|p| [p[0], p[1], p[2]]).collect();
    if opaque.is_empty() {
        return None;
    }
    let median = |c: usize| {
        let mut v: Vec<u8> = opaque.iter().map(|p| p[c]).collect();
        v.sort_unstable();
        v[v.len() / 2]
    };
    let key = [median(0), median(1), median(2)];
    let near = opaque.iter().filter(|p| (0..3).map(|c| (p[c] as i32 - key[c] as i32).abs()).sum::<i32>() < tol).count();
    (near as f32 >= share * total as f32).then_some(key)
}

/// Up to `rings` rings of edge pixels (opaque, touching a transparent one) tinted by the keyed backdrop made
/// transparent: the blend of outline and backdrop a resampled picture leaves around the object. Tinted = close to
/// `key`, or for a saturated key a clearly saturated pixel whose chroma points the same way (a dark magenta rim).
pub fn defringe(im: &RgbaImage, key: [u8; 3], tol: i32, rings: usize) -> RgbaImage {
    let chroma = |p: [u8; 3]| {
        let m = (p[0] as f32 + p[1] as f32 + p[2] as f32) / 3.0;
        [p[0] as f32 - m, p[1] as f32 - m, p[2] as f32 - m]
    };
    let norm = |v: [f32; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let kc = chroma(key);
    let tinted = |p: [u8; 3]| {
        if (0..3).map(|c| (p[c] as i32 - key[c] as i32).abs()).sum::<i32>() < tol {
            return true;
        }
        let pc = chroma(p);
        let (nk, np) = (norm(kc), norm(pc));
        nk > 60.0 && np > 25.0 && (pc[0] * kc[0] + pc[1] * kc[1] + pc[2] * kc[2]) / (nk * np) > 0.85
    };
    let (w, h) = im.dimensions();
    let mut out = im.clone();
    for _ in 0..rings {
        let clear: Vec<bool> = out.pixels().map(|p| p.0[3] < 128).collect();
        let mut changed = false;
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) as usize;
                let p = out.get_pixel(x, y).0;
                if clear[i] || !tinted([p[0], p[1], p[2]]) {
                    continue;
                }
                let touches = [(x.wrapping_sub(1), y), (x + 1, y), (x, y.wrapping_sub(1)), (x, y + 1)]
                    .iter()
                    .any(|&(nx, ny)| nx < w && ny < h && clear[(ny * w + nx) as usize]);
                if touches {
                    out.get_pixel_mut(x, y).0[3] = 0;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    out
}

/// Opaque islands touching the image edge that are smaller than `frac` of the largest island made transparent: what
/// is left of a screenshot frame or a backdrop's rim after keying.
pub fn drop_edge_islands(im: &RgbaImage, frac: f32) -> RgbaImage {
    let (w, h) = (im.width() as usize, im.height() as usize);
    let mask: Vec<bool> = im.pixels().map(|p| p.0[3] >= 128).collect();
    let (lab, n) = label(&mask, w, h, true);
    let mut area = vec![0usize; n as usize + 1];
    for &l in &lab {
        area[l as usize] += 1;
    }
    let biggest = area.iter().skip(1).copied().max().unwrap_or(0);
    let mut edge = std::collections::HashSet::new();
    for x in 0..w {
        edge.insert(lab[x]);
        edge.insert(lab[(h - 1) * w + x]);
    }
    for y in 0..h {
        edge.insert(lab[y * w]);
        edge.insert(lab[y * w + w - 1]);
    }
    let mut out = im.clone();
    for (i, p) in out.pixels_mut().enumerate() {
        let l = lab[i];
        if l != 0 && edge.contains(&l) && (area[l as usize] as f32) < frac * biggest as f32 {
            p.0[3] = 0;
        }
    }
    out
}

/// Bounding box of non-zero alpha (PIL getbbox on RGBA).
pub fn bbox(im: &RgbaImage) -> Option<(u32, u32, u32, u32)> {
    let (mut x0, mut y0, mut x1, mut y1) = (u32::MAX, u32::MAX, 0, 0);
    for (x, y, p) in im.enumerate_pixels() {
        if p.0[3] > 0 {
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x + 1);
            y1 = y1.max(y + 1);
        }
    }
    (x0 != u32::MAX).then_some((x0, y0, x1, y1))
}

fn crop(im: &RgbaImage, b: (u32, u32, u32, u32)) -> RgbaImage {
    imageops::crop_imm(im, b.0, b.1, b.2 - b.0, b.3 - b.1).to_image()
}

/// Trimmed piece placed in a w x h tile: centred in x; resting on the bottom (map props) or centred in y too.
pub fn fit_wh(im: &RgbaImage, w: u32, h: u32, center: bool) -> RgbaImage {
    let im = match bbox(im) {
        Some(b) => crop(im, b),
        None => im.clone(),
    };
    let im = crop(&im, (0, 0, im.width().min(w), im.height().min(h)));
    let mut tile = RgbaImage::new(w, h);
    let x = (w - im.width()) / 2;
    let y = if center { (h - im.height()) / 2 } else { h - im.height() };
    imageops::overlay(&mut tile, &im, x as i64, y as i64);
    tile
}

/// Into a 32 x 32 tile (64 if it does not fit, or `size`), bottom-centred, never resized.
pub fn fit(im: &RgbaImage, size: u32) -> RgbaImage {
    let t = match bbox(im) {
        Some(b) => crop(im, b),
        None => im.clone(),
    };
    let size = if size > 0 {
        size
    } else if t.width() <= 32 && t.height() <= 32 {
        32
    } else {
        64
    };
    fit_wh(&t, size, size, false)
}

/// Grid cells of a pack (w x w with w % cell == 0) with at least MIN_PX opaque pixels; anything else is one item.
pub fn pieces_of(px: &RgbaImage, cell: u32) -> Vec<RgbaImage> {
    let (w, h) = px.dimensions();
    if w == h && w > 64 && w % cell == 0 {
        let k = w / cell;
        let mut out = vec![];
        for r in 0..k {
            for c in 0..k {
                let piece = crop(px, (c * cell, r * cell, c * cell + cell, r * cell + cell));
                if piece.pixels().filter(|p| p.0[3] > 0).count() >= MIN_PX {
                    out.push(piece);
                }
            }
        }
        return out;
    }
    vec![px.clone()]
}

/// Right half reflected onto the left (fixes the oblique projection's edge in the 45° view; symmetric props only).
pub fn mirror_right(tile: &RgbaImage) -> RgbaImage {
    let Some(b) = bbox(tile) else { return tile.clone() };
    let cx = (b.0 + b.2) / 2;
    let mut out = RgbaImage::new(tile.width(), tile.height());
    for y in 0..tile.height() {
        for x in cx..tile.width() {
            let p = *tile.get_pixel(x, y);
            out.put_pixel(x, y, p);
            let mx = 2 * cx as i64 - 1 - x as i64;
            if mx >= 0 {
                out.put_pixel(mx as u32, y, p);
            }
        }
    }
    out
}

/// Flat background connected to the border made transparent (white studio backgrounds, letterbox bars).
pub fn matte(im: &RgbaImage, tol: i32) -> RgbaImage {
    let (w, h) = (im.width() as usize, im.height() as usize);
    let px = |x: usize, y: usize| im.get_pixel(x as u32, y as u32).0;
    let mut border = vec![];
    for x in 0..w {
        border.push(px(x, 0));
        border.push(px(x, h - 1));
    }
    for y in 0..h {
        border.push(px(0, y));
        border.push(px(w - 1, y));
    }
    let mut keys: std::collections::HashMap<[u8; 3], (usize, [f64; 3])> = Default::default();
    for p in &border {
        let k = [p[0] / 16, p[1] / 16, p[2] / 16];
        let e = keys.entry(k).or_insert((0, [0.0; 3]));
        e.0 += 1;
        for c in 0..3 {
            e.1[c] += p[c] as f64;
        }
    }
    let mut bg = vec![false; w * h];
    for (cnt, sum) in keys.values() {
        if (*cnt as f64) <= border.len() as f64 * 0.05 {
            continue;
        }
        let r = sum.map(|s| s / *cnt as f64);
        for (i, p) in im.pixels().enumerate() {
            if (0..3).all(|c| ((p.0[c] as f64) - r[c]).abs() < tol as f64) {
                bg[i] = true;
            }
        }
    }
    let (lab, _) = label(&bg, w, h, false);
    let mut edge = std::collections::HashSet::new();
    for x in 0..w {
        edge.insert(lab[x]);
        edge.insert(lab[(h - 1) * w + x]);
    }
    for y in 0..h {
        edge.insert(lab[y * w]);
        edge.insert(lab[y * w + w - 1]);
    }
    edge.remove(&0);
    let mut out = im.clone();
    for (i, p) in out.pixels_mut().enumerate() {
        if edge.contains(&lab[i]) || p.0[3] < 128 {
            p.0[3] = 0;
        }
    }
    out
}

/// RGB of a matted image where every transparent pixel takes the colour of the nearest opaque one (multi-source
/// BFS), so outline expansion and the contrast downscale (crate::pixeloe) never pull the background into the edges.
pub fn bleed_into_background(c: &RgbaImage) -> image::RgbImage {
    let (w, h) = (c.width() as usize, c.height() as usize);
    let mut col: Vec<Option<[u8; 3]>> = c.pixels().map(|p| (p.0[3] >= 128).then(|| [p.0[0], p.0[1], p.0[2]])).collect();
    let mut q: std::collections::VecDeque<usize> = (0..w * h).filter(|&i| col[i].is_some()).collect();
    while let Some(i) = q.pop_front() {
        let (x, y) = (i % w, i / w);
        let v = col[i];
        for (dx, dy) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
            let (nx, ny) = (x as i64 + dx, y as i64 + dy);
            if nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h {
                let j = ny as usize * w + nx as usize;
                if col[j].is_none() {
                    col[j] = v;
                    q.push_back(j);
                }
            }
        }
    }
    image::RgbImage::from_fn(w as u32, h as u32, |x, y| image::Rgb(col[y as usize * w + x as usize].unwrap_or([128, 128, 128])))
}

/// Photo -> matted, cropped 32 x 32 sprite with a 32-colour palette, resting on the bottom. (Tried crate::pixeloe here:
/// at 30 px it drops small colour details and the Krea img2img that follows erases any gain; it serves the direct
/// "convert without AI" path instead.)
pub fn pixelate(im: &RgbaImage, size: u32) -> Option<RgbaImage> {
    let m = matte(im, 40);
    let b = bbox(&m)?;
    let c = crop(&m, b);
    let s = size as f32 / c.width().max(c.height()) as f32;
    let (nw, nh) = (((c.width() as f32 * s).round() as u32).max(1), ((c.height() as f32 * s).round() as u32).max(1));
    let mut t = if s < 1.0 { imageops::resize(&c, nw, nh, imageops::FilterType::Lanczos3) } else { c };
    for p in t.pixels_mut() {
        p.0[3] = if p.0[3] >= 128 { 255 } else { 0 };
    }
    quantize_rgb(&mut t, 32);
    let mut out = RgbaImage::new(32, 32);
    imageops::overlay(&mut out, &t, ((32 - t.width()) / 2) as i64, (32i64 - t.height() as i64 - 1).max(0));
    Some(out)
}

/// Sprite scaled with NEAREST and centred on a transparent side x side canvas, aspect kept.
pub fn fit_canvas(im: &RgbaImage, side: u32) -> RgbaImage {
    let f = side as f32 / im.width().max(im.height()) as f32;
    let (nw, nh) = (((im.width() as f32 * f).round() as u32).max(1), ((im.height() as f32 * f).round() as u32).max(1));
    let s = imageops::resize(im, nw, nh, imageops::FilterType::Nearest);
    let mut c = RgbaImage::new(side, side);
    imageops::overlay(&mut c, &s, ((side - nw) / 2) as i64, ((side - nh) / 2) as i64);
    c
}

/// Rough geometric hint for the angle mode (front-view source): foreshorten by pitch, narrow and shear by yaw,
/// mirror when seen from behind. NEAREST, anchored at the bottom centre of the sprite.
pub fn warp_to_angle(canvas: &RgbaImage, pitch: f32, yaw: f32) -> RgbaImage {
    let yaw = (yaw + 180.0).rem_euclid(360.0) - 180.0;
    let mut im = canvas.clone();
    if yaw.abs() > 90.0 {
        im = imageops::flip_horizontal(&im);
    }
    let Some(b) = bbox(&im) else { return canvas.clone() };
    let sy = (pitch.to_radians().cos() * 0.75 + 0.25).max(0.4);
    let sx = (yaw.to_radians().cos().abs() * 0.7 + 0.3).max(0.45);
    let sh = yaw.to_radians().sin() * 0.18 * if yaw.abs() <= 90.0 { 1.0 } else { -1.0 };
    let (cx, by) = ((b.0 + b.2) as f32 / 2.0, b.3 as f32);
    let mut out = RgbaImage::new(im.width(), im.height());
    for y in 0..im.height() {
        for x in 0..im.width() {
            let (xf, yf) = (x as f32, y as f32);
            let xi = (xf - cx - sh * (yf - by)) / sx + cx;
            let yi = (yf - by) / sy + by;
            let (xi, yi) = (xi.floor() as i64, yi.floor() as i64);
            if xi >= 0 && yi >= 0 && (xi as u32) < im.width() && (yi as u32) < im.height() {
                out.put_pixel(x, y, *im.get_pixel(xi as u32, yi as u32));
            }
        }
    }
    out
}
