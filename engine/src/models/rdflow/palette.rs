//! RD Flow 64's colour side (rdflow/flow.py, palette.py, ramps.py): sRGB <-> OKLab, the model's pixel space (OKLab
//! scaled to about [-1, 1] + alpha), the palette the model picks from its own guess, and each palette colour's shading
//! ramp (which colours shade together and where each sits from dark to light).
use std::collections::BTreeMap;
use std::f64::consts::PI;

/// OKLab a, b (about +-0.3 for sRGB) -> about +-1.
pub const AB_SCALE: f64 = 3.0;
const NEUTRAL_C: f64 = 0.035;
const HUE_GAP: f64 = 28.0 * PI / 180.0;
const HUE_SPAN: f64 = 75.0 * PI / 180.0;
const MAX_RAMPS: i64 = 15;

/// sRGB 0..255 -> OKLab (f64 math, like the reference's numpy).
pub fn srgb_to_oklab(rgb: [u8; 3]) -> [f64; 3] {
    let lin = |v: u8| {
        let c = v as f64 / 255.0;
        if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
    };
    let (r, g, b) = (lin(rgb[0]), lin(rgb[1]), lin(rgb[2]));
    let l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b).max(0.0).cbrt();
    let m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b).max(0.0).cbrt();
    let s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b).max(0.0).cbrt();
    [
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    ]
}

/// OKLab -> sRGB 0..255 (clamped, rounded).
pub fn oklab_to_srgb(lab: [f64; 3]) -> [u8; 3] {
    let [l_, a, b] = lab;
    let l = (l_ + 0.3963377774 * a + 0.2158037573 * b).powi(3);
    let m = (l_ - 0.1055613458 * a - 0.0638541728 * b).powi(3);
    let s = (l_ - 0.0894841775 * a - 1.2914855480 * b).powi(3);
    let lin = [
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    ];
    lin.map(|c| {
        let c = c.clamp(0.0, 1.0);
        let c = if c <= 0.0031308 { c * 12.92 } else { 1.055 * c.max(1e-12).powf(1.0 / 2.4) - 0.055 };
        (c * 255.0).round().clamp(0.0, 255.0) as u8
    })
}

/// A model-space pixel [L', a', b', alpha] -> (OKLab, opaque).
pub fn from_model(x: [f32; 4]) -> ([f64; 3], bool) {
    ([(x[0] as f64 + 1.0) / 2.0, x[1] as f64 / AB_SCALE, x[2] as f64 / AB_SCALE], x[3] > 0.0)
}

/// An opaque palette colour in the model's pixel space.
pub fn to_model(rgb: [u8; 3]) -> [f32; 4] {
    let lab = srgb_to_oklab(rgb);
    [(2.0 * lab[0] - 1.0) as f32, (lab[1] * AB_SCALE) as f32, (lab[2] * AB_SCALE) as f32, 1.0]
}

/// The palette in the model's own guess [n*n] (flow.extract_palette): opaque pixels' OKLab binned at 0.015, the most
/// common bin means kept unless within `merge` of one already kept (at most `max_colours`), refined by a few k-means
/// passes, then as sRGB, deduplicated and sorted.
pub fn extract(x: &[[f32; 4]], max_colours: usize, merge: f64) -> Vec<[u8; 3]> {
    const BIN: f64 = 0.015;
    let pts: Vec<[f64; 3]> = x.iter().map(|p| from_model(*p)).filter(|(_, op)| *op).map(|(lab, _)| lab).collect();
    if pts.is_empty() {
        return vec![];
    }
    let mut bins: BTreeMap<[i64; 3], ([f64; 3], usize)> = BTreeMap::new();
    for p in &pts {
        let key = p.map(|v| (v / BIN).round() as i64);
        let e = bins.entry(key).or_insert(([0.0; 3], 0));
        for k in 0..3 {
            e.0[k] += p[k];
        }
        e.1 += 1;
    }
    let mut means: Vec<([f64; 3], usize)> = bins.into_values().map(|(s, n)| (s.map(|v| v / n as f64), n)).collect();
    means.sort_by_key(|m| std::cmp::Reverse(m.1)); // stable: ties keep the key order, like np.argsort(kind="stable")
    let dist = |a: &[f64; 3], b: &[f64; 3]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
    let mut centres: Vec<[f64; 3]> = vec![];
    for (c, _) in &means {
        if centres.iter().all(|k| dist(k, c) > merge) {
            centres.push(*c);
            if centres.len() >= max_colours {
                break;
            }
        }
    }
    for _ in 0..4 {
        let mut sums = vec![([0.0; 3], 0usize); centres.len()];
        for p in &pts {
            let d2 = |c: &[f64; 3]| (p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2);
            let (best, _) = centres.iter().enumerate().fold((0, f64::INFINITY), |acc, (i, c)| {
                let d = d2(c);
                if d < acc.1 { (i, d) } else { acc }
            });
            for k in 0..3 {
                sums[best].0[k] += p[k];
            }
            sums[best].1 += 1;
        }
        centres = sums.into_iter().filter(|s| s.1 > 0).map(|(s, n)| s.map(|v| v / n as f64)).collect();
    }
    let mut rgb: Vec<[u8; 3]> = centres.into_iter().map(oklab_to_srgb).collect();
    rgb.sort();
    rgb.dedup();
    rgb
}

/// numpy.unwrap: remove 2*pi jumps between neighbours.
fn unwrap(a: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(a.len());
    let mut corr = 0.0;
    for (i, &v) in a.iter().enumerate() {
        if i > 0 {
            let d = v - a[i - 1];
            let mut dd = (d + PI).rem_euclid(2.0 * PI) - PI;
            if dd == -PI && d > 0.0 {
                dd = PI;
            }
            if d.abs() >= PI {
                corr += dd - d;
            }
        }
        out.push(v + corr);
    }
    out
}

/// Shading ramps of a palette (ramps.ramps): (ramp id, level 0..1 dark to light). Greys are ramp 0; chromatic colours
/// group by hue around the circle, cut at the widest gap, broken at gaps over 28 degrees and spans over 75.
pub fn ramps(pal: &[[u8; 3]]) -> (Vec<i64>, Vec<f32>) {
    let n = pal.len();
    let mut ramp = vec![0i64; n];
    let mut level = vec![0.5f32; n];
    if n == 0 {
        return (ramp, level);
    }
    // float32 OKLab like the reference's srgb_to_oklab, then the f64 maths
    let lab: Vec<[f64; 3]> = pal.iter().map(|c| srgb_to_oklab(*c).map(|v| v as f32 as f64)).collect();
    let chroma: Vec<f64> = lab.iter().map(|l| l[1].hypot(l[2])).collect();
    let hue: Vec<f64> = lab.iter().map(|l| l[2].atan2(l[1]).rem_euclid(2.0 * PI)).collect();
    let mut chromatic: Vec<usize> = (0..n).filter(|&i| chroma[i] >= NEUTRAL_C).collect();
    let mut groups: Vec<Vec<usize>> = vec![];
    if !chromatic.is_empty() {
        chromatic.sort_by(|&a, &b| hue[a].partial_cmp(&hue[b]).unwrap());
        let h: Vec<f64> = chromatic.iter().map(|&i| hue[i]).collect();
        let gaps: Vec<f64> = (0..h.len()).map(|i| if i + 1 < h.len() { h[i + 1] - h[i] } else { h[0] + 2.0 * PI - h[i] }).collect();
        let widest = gaps.iter().enumerate().fold((0, f64::NEG_INFINITY), |acc, (i, &g)| if g > acc.1 { (i, g) } else { acc }).0;
        let start = (widest + 1) % chromatic.len();
        let order: Vec<usize> = chromatic[start..].iter().chain(&chromatic[..start]).copied().collect();
        let h = unwrap(&order.iter().map(|&i| hue[i]).collect::<Vec<_>>());
        let mut cur = vec![order[0]];
        for i in 1..order.len() {
            if h[i] - h[i - 1] > HUE_GAP {
                groups.push(std::mem::take(&mut cur));
            }
            cur.push(order[i]);
        }
        groups.push(cur);
        let mut split = vec![];
        for mut g in groups {
            while g.len() > 2 {
                let hg = unwrap(&g.iter().map(|&i| hue[i]).collect::<Vec<_>>());
                if hg[hg.len() - 1] - hg[0] <= HUE_SPAN {
                    break;
                }
                let k = (1..hg.len()).fold((1, f64::NEG_INFINITY), |acc, i| if hg[i] - hg[i - 1] > acc.1 { (i, hg[i] - hg[i - 1]) } else { acc }).0;
                split.push(g[..k].to_vec());
                g = g[k..].to_vec();
            }
            split.push(g);
        }
        groups = split;
    }
    let neutral: Vec<usize> = (0..n).filter(|&i| chroma[i] < NEUTRAL_C).collect();
    let all = std::iter::once((0i64, neutral)).chain(groups.into_iter().enumerate().map(|(i, g)| (((i + 1) as i64).min(MAX_RAMPS), g)));
    for (rid, g) in all {
        if g.is_empty() {
            continue;
        }
        for &i in &g {
            ramp[i] = rid;
        }
        if g.len() > 1 {
            let mut by_l: Vec<usize> = (0..g.len()).collect();
            by_l.sort_by(|&a, &b| lab[g[a]][0].partial_cmp(&lab[g[b]][0]).unwrap());
            for (rank, &j) in by_l.iter().enumerate() {
                level[g[j]] = rank as f32 / (g.len() - 1) as f32;
            }
        }
    }
    (ramp, level)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oklab_round_trips_and_matches_known_values() {
        let white = srgb_to_oklab([255, 255, 255]);
        assert!((white[0] - 1.0).abs() < 1e-4 && white[1].abs() < 1e-4 && white[2].abs() < 1e-4);
        let red = srgb_to_oklab([255, 0, 0]);
        assert!((red[0] - 0.62796).abs() < 1e-4 && (red[1] - 0.22486).abs() < 1e-4 && (red[2] - 0.12585).abs() < 1e-4);
        for c in [[0, 0, 0], [12, 200, 77], [255, 128, 3], [90, 90, 90]] {
            assert_eq!(oklab_to_srgb(srgb_to_oklab(c)), c);
        }
    }

    #[test]
    fn unwrap_follows_numpy() {
        let u = unwrap(&[6.0, 0.2, 0.5, 6.1]);
        assert!((u[1] - (0.2 + 2.0 * PI)).abs() < 1e-9 && (u[2] - (0.5 + 2.0 * PI)).abs() < 1e-9 && (u[3] - 6.1).abs() < 1e-9);
    }

    #[test]
    fn ramps_group_hues_and_order_by_lightness() {
        let pal = [[20, 20, 20], [200, 200, 200], [120, 10, 10], [230, 60, 60], [10, 30, 140], [70, 110, 240]];
        let (ramp, level) = ramps(&pal);
        assert_eq!((ramp[0], ramp[1]), (0, 0));
        assert_eq!(ramp[2], ramp[3]);
        assert_eq!(ramp[4], ramp[5]);
        assert_ne!(ramp[2], ramp[4]);
        assert!(ramp[2] > 0 && ramp[4] > 0);
        assert_eq!((level[0], level[1], level[2], level[3]), (0.0, 1.0, 0.0, 1.0));
    }

    #[test]
    fn extraction_finds_the_colours_of_a_flat_picture() {
        let mut x = vec![[0.0, 0.0, 0.0, -1.0f32]; 64 * 64];
        for (i, p) in x.iter_mut().enumerate() {
            if i % 3 == 0 {
                *p = to_model([200, 30, 30]);
            } else if i % 3 == 1 {
                *p = to_model([20, 160, 60]);
            }
        }
        assert_eq!(extract(&x, 64, 0.05), vec![[20, 160, 60], [200, 30, 30]]);
    }
}
