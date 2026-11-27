//! Ambient item animation over a static sprite, procedural (no model), as a stack of layers. Each layer is one effect
//! with its own colour, strength, repeats, area and origin, applied over the layers before it. Every frame is a pure
//! function of t mod n, so the loop is seamless; alpha stays binary.
use image::{Rgba, RgbaImage};
use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap};
use std::f32::consts::TAU;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Effect {
    Pulse,
    Glint,
    Shine,
    Motes,
    Plasma,
    Shock,
    Trail,
    Cycle,
    Rim,
    Heat,
    Flames,
    Frost,
    Icicles,
    Snow,
    Arcs,
    Drip,
    Bubbles,
    Smoke,
    Orbit,
    Wave,
    Beat,
    Blink,
    Hue,
    Twinkle,
}

impl Effect {
    pub const ALL: [Effect; 24] = [
        Effect::Pulse,
        Effect::Glint,
        Effect::Twinkle,
        Effect::Shine,
        Effect::Motes,
        Effect::Plasma,
        Effect::Shock,
        Effect::Trail,
        Effect::Cycle,
        Effect::Rim,
        Effect::Blink,
        Effect::Hue,
        Effect::Wave,
        Effect::Beat,
        Effect::Heat,
        Effect::Flames,
        Effect::Frost,
        Effect::Icicles,
        Effect::Snow,
        Effect::Arcs,
        Effect::Drip,
        Effect::Bubbles,
        Effect::Smoke,
        Effect::Orbit,
    ];

    pub fn key(self) -> &'static str {
        match self {
            Effect::Pulse => "pulse",
            Effect::Glint => "glint",
            Effect::Shine => "shine",
            Effect::Motes => "motes",
            Effect::Plasma => "plasma",
            Effect::Shock => "shock",
            Effect::Trail => "trail",
            Effect::Cycle => "cycle",
            Effect::Rim => "rim",
            Effect::Heat => "heat",
            Effect::Flames => "flames",
            Effect::Frost => "frost",
            Effect::Icicles => "icicles",
            Effect::Snow => "snow",
            Effect::Arcs => "arcs",
            Effect::Drip => "drip",
            Effect::Bubbles => "bubbles",
            Effect::Smoke => "smoke",
            Effect::Orbit => "orbit",
            Effect::Wave => "wave",
            Effect::Beat => "beat",
            Effect::Blink => "blink",
            Effect::Hue => "hue",
            Effect::Twinkle => "twinkle",
        }
    }

    pub fn from_key(k: &str) -> Option<Effect> {
        Effect::ALL.into_iter().find(|e| e.key() == k)
    }

    /// Starts from a point: the plasma front, the veins' flow, the wave's anchor.
    pub fn has_origin(self) -> bool {
        matches!(self, Effect::Plasma | Effect::Shock | Effect::Cycle | Effect::Wave | Effect::Beat)
    }

    /// What `Layer::amount` counts for this effect, if anything: (min, max, default).
    pub fn amount(self) -> Option<(usize, usize, usize)> {
        match self {
            Effect::Motes => Some((1, 16, 7)),
            Effect::Twinkle => Some((1, 8, 3)),
            Effect::Wave => Some((1, 4, 1)),
            Effect::Shock => Some((1, 3, 2)),
            Effect::Beat => Some((1, 3, 1)),
            Effect::Flames => Some((2, 14, 6)),
            Effect::Orbit => Some((1, 6, 3)),
            Effect::Frost => Some((1, 8, 3)),
            Effect::Icicles => Some((1, 10, 3)),
            Effect::Snow => Some((1, 12, 4)),
            _ => None,
        }
    }
}

/// Where a layer acts.
#[derive(Clone, Debug, PartialEq)]
pub enum Area {
    /// the effect's own choice: the glowing accents, the edge, or the whole item
    Auto,
    All,
    /// the glowing accents (gems, eyes, runes)
    Glow,
    /// pixels near this colour
    Color([u8; 3]),
    /// painted pixels, row-major over the sprite
    Mask(Vec<bool>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Layer {
    pub effect: Effect,
    /// None = the median colour of the layer's area
    pub color: Option<[u8; 3]>,
    /// 0..1, how much the layer changes what is under it
    pub strength: f32,
    /// times the effect runs per loop
    pub repeats: usize,
    pub area: Area,
    /// (x, y) where it starts; None = its own choice
    pub origin: Option<(u32, u32)>,
    /// Effect::amount
    pub amount: usize,
    /// off = kept in the list, left out of the loop
    pub on: bool,
}

impl Layer {
    pub fn new(effect: Effect) -> Layer {
        Layer { effect, color: None, strength: 1., repeats: 1, area: Area::Auto, origin: None, amount: effect.amount().map_or(0, |a| a.2), on: true }
    }

    pub fn colored(effect: Effect, color: [u8; 3]) -> Layer {
        Layer { color: Some(color), ..Layer::new(effect) }
    }
}

/// The game's elements as presets: effects + glow colour.
pub struct Element {
    pub key: &'static str,
    pub effects: &'static [Effect],
    pub color: [u8; 3],
}

impl Element {
    pub fn layers(&self) -> Vec<Layer> {
        self.effects.iter().map(|&e| Layer::colored(e, self.color)).collect()
    }
}

pub const ELEMENTS: [Element; 6] = [
    Element { key: "fire", effects: &[Effect::Heat, Effect::Flames, Effect::Motes], color: [240, 110, 20] },
    Element { key: "ice", effects: &[Effect::Frost, Effect::Icicles, Effect::Snow, Effect::Twinkle], color: [110, 200, 255] },
    Element { key: "energy", effects: &[Effect::Arcs, Effect::Trail], color: [215, 110, 235] },
    Element { key: "earth", effects: &[Effect::Drip, Effect::Motes], color: [100, 200, 50] },
    Element { key: "death", effects: &[Effect::Smoke, Effect::Plasma, Effect::Pulse], color: [150, 60, 200] },
    Element { key: "holy", effects: &[Effect::Glint, Effect::Shine, Effect::Orbit, Effect::Pulse], color: [255, 215, 100] },
];

#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    pub layers: Vec<Layer>,
    pub frames: usize,
    /// how readily pixels count as glowing (lower = more of them)
    pub threshold: f32,
    pub seed: u64,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { layers: [Effect::Pulse, Effect::Motes, Effect::Glint].map(Layer::new).to_vec(), frames: 12, threshold: 0.55, seed: 0 }
    }
}

/// splitmix64: the effects' placement, the same for the same seed.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    /// lo..hi, hi excluded
    fn int(&mut self, lo: i32, hi: i32) -> i32 {
        if hi <= lo { lo } else { lo + (self.next() % (hi - lo) as u64) as i32 }
    }
    fn index(&mut self, len: usize) -> usize {
        (self.next() % len.max(1) as u64) as usize
    }
}

type Col = [f32; 3];
type Pt = (i32, i32);

const N8: [Pt; 8] = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)];

struct Grid {
    w: i32,
    h: i32,
}

impl Grid {
    fn inside(&self, y: i32, x: i32) -> bool {
        (0..self.h).contains(&y) && (0..self.w).contains(&x)
    }
    fn i(&self, y: i32, x: i32) -> usize {
        (y * self.w + x) as usize
    }
}

pub fn hsv(c: Col) -> (f32, f32, f32) {
    let (r, g, b) = (c[0] / 255., c[1] / 255., c[2] / 255.);
    let (mx, mn) = (r.max(g).max(b), r.min(g).min(b));
    if mx == mn {
        return (0., 0., mx);
    }
    let d = mx - mn;
    let h = if mx == r {
        ((g - b) / d).rem_euclid(6.)
    } else if mx == g {
        (b - r) / d + 2.
    } else {
        (r - g) / d + 4.
    };
    (h / 6., d / mx, mx)
}

pub fn rgb_of(h: f32, s: f32, v: f32) -> Col {
    let i = (h * 6.).floor();
    let f = h * 6. - i;
    let (p, q, t) = (v * (1. - s), v * (1. - s * f), v * (1. - s * (1. - f)));
    let (r, g, b) = match (i as i32).rem_euclid(6) {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    [r * 255., g * 255., b * 255.]
}

fn tone(c: Col, s_mul: f32, v: f32) -> Col {
    let (h, s, _) = hsv(c);
    rgb_of(h, (s * s_mul).min(1.), v)
}

/// 3 x 3 mean, edges mirrored (scipy's uniform_filter).
fn mean3(g: &Grid, v: &[f32]) -> Vec<f32> {
    let cl = |a: i32, n: i32| a.clamp(0, n - 1);
    (0..g.h)
        .flat_map(|y| (0..g.w).map(move |x| (y, x)))
        .map(|(y, x)| {
            let mut s = 0.;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    s += v[g.i(cl(y + dy, g.h), cl(x + dx, g.w))];
                }
            }
            s / 9.
        })
        .collect()
}

/// 4-neighbour dilation.
fn dilate(g: &Grid, m: &[bool]) -> Vec<bool> {
    (0..g.h)
        .flat_map(|y| (0..g.w).map(move |x| (y, x)))
        .map(|(y, x)| m[g.i(y, x)] || [(0, 1), (1, 0), (0, -1), (-1, 0)].iter().any(|&(dy, dx)| g.inside(y + dy, x + dx) && m[g.i(y + dy, x + dx)]))
        .collect()
}

/// 4-neighbour erosion, outside the image counting as empty.
fn erode(g: &Grid, m: &[bool]) -> Vec<bool> {
    (0..g.h)
        .flat_map(|y| (0..g.w).map(move |x| (y, x)))
        .map(|(y, x)| m[g.i(y, x)] && [(0, 1), (1, 0), (0, -1), (-1, 0)].iter().all(|&(dy, dx)| g.inside(y + dy, x + dx) && m[g.i(y + dy, x + dx)]))
        .collect()
}

/// Distance from `seed` to every pixel of `mask`, travelling only through it (straight, diagonal and knight steps,
/// so the fronts come out round).
fn geodesic(g: &Grid, mask: &[bool], seed: Pt) -> Vec<f32> {
    let mut steps: Vec<Pt> = N8.to_vec();
    for dy in [-2, -1, 1, 2] {
        for dx in [-2, -1, 1, 2] {
            if i32::abs(dy) != i32::abs(dx) {
                steps.push((dy, dx));
            }
        }
    }
    let mut d = vec![f32::INFINITY; mask.len()];
    let mut q = BinaryHeap::new();
    d[g.i(seed.0, seed.1)] = 0.;
    q.push(Reverse((0u32, seed.0, seed.1)));
    while let Some(Reverse((c, y, x))) = q.pop() {
        let c = f32::from_bits(c);
        if c > d[g.i(y, x)] {
            continue;
        }
        for &(dy, dx) in &steps {
            let (ny, nx) = (y + dy, x + dx);
            if !g.inside(ny, nx) || !mask[g.i(ny, nx)] {
                continue;
            }
            let nc = c + ((dy * dy + dx * dx) as f32).sqrt();
            if nc < d[g.i(ny, nx)] {
                d[g.i(ny, nx)] = nc;
                q.push(Reverse((nc.to_bits(), ny, nx)));
            }
        }
    }
    d
}

/// The silhouette's inner edge as one ordered walk (nearest-neighbour chain; jumps only between separate parts).
fn contour_path(g: &Grid, a: &[bool]) -> Vec<Pt> {
    let inner = erode(g, a);
    let mut pts: BTreeSet<Pt> = (0..g.h).flat_map(|y| (0..g.w).map(move |x| (y, x))).filter(|&(y, x)| a[g.i(y, x)] && !inner[g.i(y, x)]).collect();
    let Some(&first) = pts.iter().next() else { return vec![] };
    pts.remove(&first);
    let mut path = vec![first];
    while !pts.is_empty() {
        let (y, x) = *path.last().unwrap();
        let next = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
            .iter()
            .map(|&(dy, dx)| (y + dy, x + dx))
            .find(|p| pts.contains(p))
            .unwrap_or_else(|| *pts.iter().min_by_key(|p| ((p.0 - y).pow(2) + (p.1 - x).pow(2), p.0, p.1)).unwrap());
        pts.remove(&next);
        path.push(next);
    }
    path
}

fn median(mut v: Vec<f32>) -> f32 {
    if v.is_empty() {
        return 0.;
    }
    v.sort_by(f32::total_cmp);
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2. }
}

fn quantile(mut v: Vec<f32>, q: f32) -> f32 {
    v.sort_by(f32::total_cmp);
    let pos = q * (v.len() - 1) as f32;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    v[lo] + (v[hi] - v[lo]) * (pos - lo as f32)
}

/// Pixels that should glow: saturated, bright, brighter than their neighbours (gems, eyes, runes, liquid), clamped to
/// 3-12 % of the sprite: accents only, like the game's animated items.
fn emissive(g: &Grid, a: &[bool], s: &[f32], v: &[f32], thr: f32) -> Vec<bool> {
    let m = mean3(g, v);
    let score: Vec<f32> = (0..a.len()).map(|i| s[i] * v[i] + (v[i] - m[i])).collect();
    let area = a.iter().filter(|&&b| b).count() as f32;
    let over = (0..a.len()).filter(|&i| a[i] && score[i] > thr).count() as f32;
    let k = over.max((area * 0.03).max(3.)).min(area * 0.12).max(1.) as usize;
    let mut on: Vec<f32> = (0..a.len()).filter(|&i| a[i]).map(|i| score[i]).collect();
    on.sort_by(|x, y| y.total_cmp(x));
    let cut = on[(k - 1).min(on.len() - 1)];
    (0..a.len()).map(|i| a[i] && score[i] >= cut).collect()
}

/// Largest 4-connected region of `m`.
fn largest_region(g: &Grid, m: &[bool]) -> Vec<bool> {
    let mut lab = vec![0usize; m.len()];
    let (mut best, mut best_n, mut next) = (0, 0, 0);
    for start in 0..m.len() {
        if !m[start] || lab[start] != 0 {
            continue;
        }
        next += 1;
        let (mut stack, mut n) = (vec![start], 0);
        lab[start] = next;
        while let Some(i) = stack.pop() {
            n += 1;
            let (y, x) = ((i as i32) / g.w, (i as i32) % g.w);
            for (dy, dx) in [(0, 1), (1, 0), (0, -1), (-1, 0)] {
                if g.inside(y + dy, x + dx) {
                    let j = g.i(y + dy, x + dx);
                    if m[j] && lab[j] == 0 {
                        lab[j] = next;
                        stack.push(j);
                    }
                }
            }
        }
        if n > best_n {
            (best, best_n) = (next, n);
        }
    }
    lab.iter().map(|&l| best != 0 && l == best).collect()
}

struct Mote {
    y0: f32,
    x0: f32,
    birth: f32,
    life: f32,
    vy: f32,
    sway: f32,
}

struct Wisp {
    y0: f32,
    x0: f32,
    birth: f32,
    sway: f32,
}

struct Flake {
    x0: f32,
    birth: f32,
    sway: f32,
    /// falls in front of the item (the others pass behind it)
    front: bool,
}

struct Icicle {
    y: i32,
    x: i32,
    len: i32,
    birth: f32,
    /// lets a drop fall from its tip
    drips: bool,
}

/// When each pixel of `mask` freezes as frost creeps in from `edge`: first-passage percolation. Crossing a pixel
/// costs an amount that varies smoothly over the item (about 1 on average), so the front advances in lobes, the way
/// rime grows, instead of as an even band.
fn freeze_order(g: &Grid, mask: &[bool], edge: &[Pt], seed: u64) -> Vec<f32> {
    let mut d = vec![f32::INFINITY; mask.len()];
    let mut q = BinaryHeap::new();
    for &(y, x) in edge {
        d[g.i(y, x)] = 0.;
        q.push(Reverse((0u32, y, x)));
    }
    while let Some(Reverse((c, y, x))) = q.pop() {
        let c = f32::from_bits(c);
        if c > d[g.i(y, x)] {
            continue;
        }
        for (dy, dx) in N8 {
            let (ny, nx) = (y + dy, x + dx);
            if !g.inside(ny, nx) || !mask[g.i(ny, nx)] {
                continue;
            }
            let r = noise(seed, nx as f32 * 0.45, ny as f32 * 0.45, 4096);
            let nc = c + (0.4 + 1.5 * r) * if dy != 0 && dx != 0 { 1.41 } else { 1. };
            if nc < d[g.i(ny, nx)] {
                d[g.i(ny, nx)] = nc;
                q.push(Reverse((nc.to_bits(), ny, nx)));
            }
        }
    }
    d
}

/// 4-connected regions of `m`, each as its pixel indices, in reading order of their first pixel.
fn regions(g: &Grid, m: &[bool]) -> Vec<Vec<usize>> {
    let mut seen = vec![false; m.len()];
    let mut out = vec![];
    for start in 0..m.len() {
        if !m[start] || seen[start] {
            continue;
        }
        let (mut stack, mut part) = (vec![start], vec![]);
        seen[start] = true;
        while let Some(i) = stack.pop() {
            part.push(i);
            let (y, x) = ((i as i32) / g.w, (i as i32) % g.w);
            for (dy, dx) in [(0, 1), (1, 0), (0, -1), (-1, 0)] {
                if g.inside(y + dy, x + dx) {
                    let j = g.i(y + dy, x + dx);
                    if m[j] && !seen[j] {
                        seen[j] = true;
                        stack.push(j);
                    }
                }
            }
        }
        out.push(part);
    }
    out
}

fn hash01(seed: u64, a: i64, b: i64) -> f32 {
    let mut r = Rng(seed ^ (a as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (b as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
    (r.next() >> 40) as f32 / (1u64 << 24) as f32
}

/// Smooth value noise in 0..1 that repeats every `period` in y: scrolled a whole number of periods per loop, it loops.
fn noise(seed: u64, x: f32, y: f32, period: i32) -> f32 {
    let y = y.rem_euclid(period as f32);
    let (x0, y0) = (x.floor(), y.floor());
    let ease = |f: f32| f * f * (3. - 2. * f);
    let (fx, fy) = (ease(x - x0), ease(y - y0));
    let at = |dx: i32, dy: i32| hash01(seed, x0 as i64 + dx as i64, ((y0 as i32 + dy).rem_euclid(period)) as i64);
    let (top, bottom) = (at(0, 0) + (at(1, 0) - at(0, 0)) * fx, at(0, 1) + (at(1, 1) - at(0, 1)) * fx);
    top + (bottom - top) * fy
}

/// What every layer reads from the sprite.
struct Sprite {
    g: Grid,
    img: Vec<[f32; 4]>,
    a: Vec<bool>,
    sat: Vec<f32>,
    v: Vec<f32>,
    glow: Vec<bool>,
}

impl Sprite {
    fn new(im: &RgbaImage, threshold: f32) -> Option<Sprite> {
        let g = Grid { w: im.width() as i32, h: im.height() as i32 };
        let img: Vec<[f32; 4]> = im.pixels().map(|p| p.0.map(|c| c as f32)).collect();
        let a: Vec<bool> = img.iter().map(|p| p[3] > 127.).collect();
        if !a.iter().any(|&b| b) {
            return None;
        }
        let (sat, v): (Vec<f32>, Vec<f32>) = img
            .iter()
            .map(|p| {
                let (mx, mn) = (p[0].max(p[1]).max(p[2]) / 255., p[0].min(p[1]).min(p[2]) / 255.);
                (if mx > 0. { (mx - mn) / mx } else { 0. }, mx)
            })
            .unzip();
        let glow = emissive(&g, &a, &sat, &v, threshold);
        Some(Sprite { g, img, a, sat, v, glow })
    }

    fn at(&self, i: usize) -> Pt {
        ((i as i32) / self.g.w, (i as i32) % self.g.w)
    }
}

struct Flame {
    y: i32,
    x: i32,
    /// 0..1, lower at the ends of a burning edge so the fire closes instead of ending in a wall
    taper: f32,
}

/// One layer made ready: where it acts, its colours and the placement of its parts.
struct Animator<'a> {
    sp: &'a Sprite,
    n: usize,
    layer: &'a Layer,
    seed: u64,
    /// the area was chosen, not left to the effect
    explicit: bool,
    /// what glows and where particles are born
    em: Vec<bool>,
    /// the shape the effect follows: its edge, its top, its inside
    body: Vec<bool>,
    hot: Col,
    light: Col,
    mid: Col,
    dark: Col,
    deep: Col,
    smoke: (Col, Col),
    /// dark, mid, light, hot: hotter is lighter and shifted towards yellow
    fire: [Col; 4],
    motes: Vec<Mote>,
    glints: Vec<(usize, Pt)>,
    dist: Vec<f32>,
    dmax: f32,
    veins: Vec<bool>,
    path: Vec<Pt>,
    heads: usize,
    flames: Vec<Flame>,
    embers: Vec<Mote>,
    bolts: Vec<(usize, Vec<Pt>)>,
    /// frost: when each pixel freezes (freeze_order)
    rime: Vec<f32>,
    icicles: Vec<Icicle>,
    flakes: Vec<Flake>,
    /// snow: the row it starts at and how far it falls
    fall: (f32, f32),
    drips: Vec<(i32, i32, f32)>,
    liquid: Vec<bool>,
    bubbles: Vec<(i32, i32, f32)>,
    orb: (f32, f32, f32, f32),
    wisps: Vec<Wisp>,
    /// wave: rows sway sideways (else columns sway up and down), the fixed line, the farthest line from it
    sway_rows: bool,
    anchor: f32,
    span: f32,
    parts: Vec<Vec<usize>>,
    sparkle: Vec<usize>,
    /// shockwave: the way the wave moves at each pixel (dy, dx), a unit vector, or zero where it has no direction
    flow: Vec<(f32, f32)>,
    /// heartbeat: the point it swells from (y, x) and half the area's size
    heart: (f32, f32, f32),
}

fn spread<T: Copy>(pts: &[T], k: usize) -> Vec<T> {
    (0..k.min(pts.len())).map(|i| pts[i * pts.len() / k]).collect()
}

impl<'a> Animator<'a> {
    /// None when the layer's area holds no pixel of the sprite.
    fn new(sp: &'a Sprite, layer: &'a Layer, n: usize, seed: u64) -> Option<Animator<'a>> {
        use Effect::*;
        let (g, a, v, e) = (&sp.g, &sp.a, &sp.v, layer.effect);
        let (w, h) = (g.w, g.h);
        let region: Option<Vec<bool>> = match &layer.area {
            Area::Auto => None,
            Area::All => Some(a.clone()),
            Area::Glow => Some(sp.glow.clone()),
            Area::Color(c) => Some((0..a.len()).map(|i| a[i] && (0..3).map(|k| (sp.img[i][k] - c[k] as f32).abs()).sum::<f32>() <= 72.).collect()),
            Area::Mask(m) => Some((0..a.len()).map(|i| a[i] && m.get(i).copied().unwrap_or(false)).collect()),
        };
        let explicit = region.is_some();
        let em = region.clone().unwrap_or_else(|| sp.glow.clone());
        let body = region.unwrap_or_else(|| a.clone());
        let em_pts: Vec<Pt> = (0..a.len()).filter(|&i| em[i]).map(|i| sp.at(i)).collect();
        let body_pts: Vec<Pt> = (0..a.len()).filter(|&i| body[i]).map(|i| sp.at(i)).collect();
        if em_pts.is_empty() {
            return None;
        }
        let base: Col = match layer.color {
            Some(c) => c.map(|v| v as f32),
            None if matches!(e, Heat | Flames) => [240., 110., 20.],
            None if matches!(e, Frost | Icicles | Snow) => [120., 205., 255.],
            None => [0, 1, 2].map(|k| median(em_pts.iter().map(|&(y, x)| sp.img[g.i(y, x)][k]).collect())),
        };
        let (hot, light, mid, dark) = (tone(base, 0.2, 1.), tone(base, 0.7, 1.), tone(base, 1., 0.85), tone(base, 1., 0.5));
        let deep = tone(mid, 1., 0.32);
        let smoke = (tone(mid, 0.3, 0.3), tone(mid, 0.3, 0.48));
        let (bh, bs, _) = hsv(base);
        let fire = [
            rgb_of((bh - 0.03).rem_euclid(1.), bs, 0.62),
            rgb_of(bh, bs, 0.92),
            rgb_of((bh + 0.035).rem_euclid(1.), bs * 0.85, 1.),
            rgb_of((bh + 0.075).rem_euclid(1.), bs * 0.4, 1.),
        ];
        let mut rng = Rng(seed ^ 0x5EED);
        let nf = n as f32;
        let solid = |y: i32, x: i32| g.inside(y, x) && body[g.i(y, x)];

        let motes = if e == Motes {
            let mc = layer.amount.max(1);
            (0..mc)
                .map(|i| {
                    let (y, x) = em_pts[rng.index(em_pts.len())];
                    Mote {
                        y0: y as f32,
                        x0: x as f32,
                        birth: i as f32 * nf / mc as f32 + rng.uniform(0., nf / mc as f32),
                        life: rng.int(n as i32 / 2, n as i32 - 1) as f32,
                        vy: -rng.uniform(0.5, 1.),
                        sway: rng.uniform(0., TAU),
                    }
                })
                .collect()
        } else {
            vec![]
        };
        let mut glints = if e == Glint {
            let vq = quantile(body_pts.iter().map(|&(y, x)| v[g.i(y, x)]).collect(), 0.93);
            let lit: Vec<Pt> = body_pts.iter().copied().filter(|&(y, x)| v[g.i(y, x)] >= vq).collect();
            vec![((nf * 0.15) as usize, lit[rng.index(lit.len())]), ((nf * 0.62) as usize, lit[rng.index(lit.len())])]
        } else {
            vec![]
        };
        let (mut dist, mut dmax, mut veins) = (vec![], 0., vec![]);
        if matches!(e, Plasma | Cycle | Shock) {
            // the wave starts at the chosen point, or at the glowing pixel nearest the glow's centre; either way on the item
            let (cy, cx, starts) = match layer.origin {
                Some((x, y)) => (y as f32, x as f32, &body_pts),
                None => {
                    let (sy, sx) = em_pts.iter().fold((0., 0.), |s, p| (s.0 + p.0 as f32, s.1 + p.1 as f32));
                    (sy / em_pts.len() as f32, sx / em_pts.len() as f32, &em_pts)
                }
            };
            let d2 = |p: &Pt| (p.0 as f32 - cy).powi(2) + (p.1 as f32 - cx).powi(2);
            let start = *starts.iter().min_by(|p, q| d2(p).total_cmp(&d2(q)))?;
            dist = geodesic(g, &body, start);
            dmax = dist.iter().copied().filter(|d| d.is_finite()).fold(0., f32::max);
            let m3 = mean3(g, v);
            let ridge: Vec<bool> = (0..a.len()).map(|i| body[i] && v[i] - m3[i] > 0.03 && v[i] > 0.15 && !(em[i] && !explicit)).collect();
            veins = if ridge.iter().filter(|&&b| b).count() as f32 > 0.05 * body_pts.len() as f32 {
                ridge
            } else {
                (0..a.len()).map(|i| body[i] && v[i] > 0.15).collect()
            };
        }
        let path = if matches!(e, Trail | Arcs | Frost | Icicles | Flames | Drip | Smoke) { contour_path(g, &body) } else { vec![] };
        let heads = (path.len() / 40).max(1);
        let edge_of = |dy: i32| {
            let mut pts: Vec<Pt> = path.iter().copied().filter(|&(y, x)| !solid(y + dy, x)).collect::<BTreeSet<_>>().into_iter().collect();
            pts.sort_by_key(|p| (p.1, p.0));
            pts
        };
        let (top, bottom) = (edge_of(-1), edge_of(1));

        let (mut flames, mut embers) = (vec![], vec![]);
        if e == Flames && !top.is_empty() {
            // beds: runs of upward edge that touch (a step of up to 2 px still belongs to the same fire)
            let open: Vec<Pt> = top.iter().copied().filter(|&(y, x)| explicit || !(g.inside(y - 2, x) && a[g.i(y - 2, x)])).collect();
            let mut bed = vec![usize::MAX; open.len()];
            let mut beds: Vec<Vec<usize>> = vec![];
            for start in 0..open.len() {
                if bed[start] != usize::MAX {
                    continue;
                }
                let (mut stack, mut members) = (vec![start], vec![]);
                bed[start] = beds.len();
                while let Some(i) = stack.pop() {
                    members.push(i);
                    for j in 0..open.len() {
                        if bed[j] == usize::MAX && (open[j].1 - open[i].1).abs() <= 1 && (open[j].0 - open[i].0).abs() <= 2 {
                            bed[j] = beds.len();
                            stack.push(j);
                        }
                    }
                }
                beds.push(members);
            }
            for members in &beds {
                let (lo, hi) = (members.iter().map(|&i| open[i].1).min().unwrap(), members.iter().map(|&i| open[i].1).max().unwrap());
                let width = (hi - lo + 1) as f32;
                // a wide bed carries a full fire, a ledge of a couple of pixels only a low flame
                let size = (0.3 + width / 9.).min(1.);
                for &i in members {
                    let (y, x) = open[i];
                    let end = (x - lo).min(hi - x) as f32;
                    flames.push(Flame { y, x, taper: size * ((end + 1.2) / 3.).min(1.) });
                }
            }
            let count = if flames.is_empty() { 0 } else { (flames.len() / 5).clamp(2, 8) };
            embers = (0..count)
                .map(|i| {
                    let f = &flames[rng.index(flames.len())];
                    Mote {
                        y0: f.y as f32 - layer.amount.max(2) as f32 * 0.5,
                        x0: f.x as f32,
                        birth: i as f32 * nf / count as f32 + rng.uniform(0., nf / count as f32),
                        life: rng.uniform(nf * 0.35, nf * 0.6),
                        vy: -rng.uniform(0.8, 1.3),
                        sway: rng.uniform(0., TAU),
                    }
                })
                .collect();
        }
        let mut bolts = vec![];
        if e == Arcs && !path.is_empty() {
            for i in 0..3 {
                let (mut p0, mut p1) = (path[0], path[0]);
                for _ in 0..40 {
                    (p0, p1) = (path[rng.index(path.len())], path[rng.index(path.len())]);
                    let span = (p1.0 - p0.0).abs().max((p1.1 - p0.1).abs());
                    if (6..=14).contains(&span) {
                        break;
                    }
                }
                let (dy, dx) = ((p1.0 - p0.0) as f32, (p1.1 - p0.1) as f32);
                let steps = (p1.0 - p0.0).abs().max((p1.1 - p0.1).abs());
                let len = dy.hypot(dx).max(1e-6);
                let nrm = (dx / len, -dy / len);
                let (mut off, mut pts) = (0, vec![]);
                for j in 0..=steps {
                    off = if j == 0 || j == steps { 0 } else { (off + rng.int(-1, 2)).clamp(-2, 2) };
                    let f = j as f32 / steps.max(1) as f32;
                    pts.push(((p0.0 as f32 + dy * f + nrm.0 * off as f32).round() as i32, (p0.1 as f32 + dx * f + nrm.1 * off as f32).round() as i32));
                }
                bolts.push(((nf * i as f32 / 3.) as usize + 1, pts));
            }
        }
        let mut rime = vec![];
        if e == Frost {
            rime = freeze_order(g, &body, &path, seed);
            // glitter on the frost that never melts
            let depth = layer.amount.max(1) as f32;
            let frozen: Vec<Pt> = body_pts.iter().copied().filter(|&(y, x)| rime[g.i(y, x)] <= depth * 0.55).collect();
            if !frozen.is_empty() {
                glints = [0.2, 0.55, 0.85].map(|at| ((nf * at) as usize, frozen[rng.index(frozen.len())])).to_vec();
            }
        }
        let mut icicles = vec![];
        if e == Icicles {
            // they hang from the edges that face down with a clear drop below, not from every ledge
            let room = |y: i32, x: i32| (1..).take_while(|&j| g.inside(y + j, x) && (explicit || !a[g.i(y + j, x)])).count() as i32;
            let longest = (h / 6).max(4);
            let clear: Vec<Pt> = bottom.iter().copied().filter(|&(y, x)| room(y, x) > longest).collect();
            let hooks: Vec<Pt> = if clear.is_empty() { bottom.iter().copied().filter(|&(y, x)| room(y, x) >= 3).collect() } else { clear };
            let count = layer.amount.max(1);
            icicles = spread(&hooks, count)
                .into_iter()
                .enumerate()
                .map(|(i, (y, x))| Icicle {
                    y,
                    x,
                    len: rng.int(3, longest + 1).min(room(y, x)),
                    birth: i as f32 * nf / count as f32 + rng.uniform(0., nf / count as f32),
                    drips: i % 2 == 0,
                })
                .collect();
        }
        let (mut flakes, mut fall) = (vec![], (0., 1.));
        if e == Snow {
            let edge = |k: fn(&Pt) -> i32| (body_pts.iter().map(k).min().unwrap() as f32, body_pts.iter().map(k).max().unwrap() as f32);
            let ((y0, y1), (x0, x1)) = (edge(|p| p.0), edge(|p| p.1));
            fall = (y0 - 3., y1 - y0 + 6.);
            let count = layer.amount.max(1);
            flakes = (0..count)
                .map(|i| Flake {
                    x0: rng.uniform(x0 - 2., x1 + 2.),
                    birth: i as f32 * nf / count as f32 + rng.uniform(0., nf / count as f32),
                    sway: rng.uniform(0., TAU),
                    front: i % 2 == 0,
                })
                .collect();
        }
        let mean_y = body_pts.iter().map(|p| p.0 as f32).sum::<f32>() / body_pts.len() as f32;
        let mean_x = body_pts.iter().map(|p| p.1 as f32).sum::<f32>() / body_pts.len() as f32;
        let drips = if e == Drip {
            let low: Vec<Pt> = bottom.iter().copied().filter(|p| p.0 as f32 >= mean_y).collect();
            let low = if low.is_empty() { bottom.clone() } else { low };
            spread(&low, 3).into_iter().enumerate().map(|(i, (y, x))| (y, x, i as f32 * nf / 3.)).collect()
        } else {
            vec![]
        };
        let (mut liquid, mut bubbles) = (vec![], vec![]);
        if e == Bubbles {
            liquid = if explicit {
                body.clone()
            } else {
                let wet: Vec<bool> = (0..a.len()).map(|i| a[i] && sp.sat[i] > 0.4 && v[i] > 0.3).collect();
                let l = largest_region(g, &wet);
                if l.iter().filter(|&&b| b).count() < 6 { em.clone() } else { l }
            };
            let cols: Vec<i32> = (0..w).filter(|&x| (0..h).any(|y| liquid[g.i(y, x)])).collect();
            bubbles = (0..5)
                .map(|i| {
                    let x = cols[rng.index(cols.len())];
                    let y0 = (0..h).rev().find(|&y| liquid[g.i(y, x)]).unwrap_or(0);
                    (x, y0, i as f32 * nf / 5. + rng.uniform(0., nf / 5.))
                })
                .collect();
        }
        let span_of = |k: fn(&Pt) -> i32| (body_pts.iter().map(k).max().unwrap() - body_pts.iter().map(k).min().unwrap()) as f32;
        let (span_y, span_x) = (span_of(|p| p.0), span_of(|p| p.1));
        let orb = (mean_y, mean_x, span_x / 2. + 2., (span_y / 6.).max(2.));
        let wisps = if e == Smoke {
            spread(&top, 5)
                .into_iter()
                .enumerate()
                .map(|(i, (y, x))| Wisp { y0: y as f32, x0: x as f32, birth: i as f32 * nf / 5. + rng.uniform(0., nf / 5.), sway: rng.uniform(0., TAU) })
                .collect()
        } else {
            vec![]
        };
        // wave: it hangs from the anchor (default: the top edge); a part that extends sideways from it ripples up and down
        let (ay, ax) = match layer.origin {
            Some((x, y)) => (y as f32, x as f32),
            None => (body_pts.iter().map(|p| p.0).min().unwrap() as f32, mean_x),
        };
        let reach = |k: fn(&Pt) -> i32, c: f32| body_pts.iter().map(|p| (k(p) as f32 - c).abs()).fold(0., f32::max);
        let (reach_y, reach_x) = (reach(|p| p.0, ay), reach(|p| p.1, ax));
        let sway_rows = reach_y >= reach_x;
        let (anchor, span) = if sway_rows { (ay, reach_y) } else { (ax, reach_x) };
        // which way the wave travels at each pixel: up the slope of the distance from its origin
        let flow: Vec<(f32, f32)> = if e == Shock {
            let at = |y: i32, x: i32| (g.inside(y, x) && dist[g.i(y, x)].is_finite()).then(|| dist[g.i(y, x)]);
            (0..a.len())
                .map(|i| {
                    let (y, x) = sp.at(i);
                    let Some(here) = at(y, x) else { return (0., 0.) };
                    let slope = |before: Option<f32>, after: Option<f32>| after.unwrap_or(here) - before.unwrap_or(here);
                    let (dy, dx) = (slope(at(y - 1, x), at(y + 1, x)), slope(at(y, x - 1), at(y, x + 1)));
                    let len = dy.hypot(dx);
                    if len < 1e-3 { (0., 0.) } else { (dy / len, dx / len) }
                })
                .collect()
        } else {
            vec![]
        };
        let heart = match layer.origin {
            Some((x, y)) => (y as f32, x as f32, span_y.max(span_x) / 2.),
            None => (mean_y, mean_x, span_y.max(span_x) / 2.),
        };
        let parts = if e == Blink { regions(g, &em) } else { vec![] };
        let sparkle = if e == Twinkle {
            let pool: Vec<usize> = (0..a.len()).filter(|&i| body[i]).collect();
            let cut = median(pool.iter().map(|&i| v[i]).collect());
            pool.into_iter().filter(|&i| v[i] >= cut).collect()
        } else {
            vec![]
        };
        Some(Animator {
            sp,
            n,
            layer,
            seed,
            explicit,
            em,
            body,
            hot,
            light,
            mid,
            dark,
            deep,
            smoke,
            fire,
            motes,
            glints,
            dist,
            dmax,
            veins,
            path,
            heads,
            flames,
            embers,
            bolts,
            rime,
            icicles,
            flakes,
            fall,
            drips,
            liquid,
            bubbles,
            orb,
            wisps,
            sway_rows,
            anchor,
            span,
            parts,
            sparkle,
            flow,
            heart,
        })
    }

    /// Draws the layer's state at frame `t` over `out`.
    fn apply(&self, out: &mut [[f32; 4]], t: usize) {
        let (g, a, em, n) = (&self.sp.g, &self.sp.a, &self.em, self.n as f32);
        let ti = t % self.n;
        let t = ti as f32;
        let blend = |out: &mut [[f32; 4]], i: usize, c: Col, k: f32| {
            for j in 0..3 {
                out[i][j] += (c[j] - out[i][j]) * k;
            }
        };
        let put = |out: &mut [[f32; 4]], y: i32, x: i32, c: Col| {
            if g.inside(y, x) {
                out[g.i(y, x)] = [c[0], c[1], c[2], 255.];
            }
        };
        let age_of = |birth: f32| (t - birth).rem_euclid(n);
        let p = (1. - (TAU * t / n).cos()) / 2.; // 0 -> 1 -> 0 over the loop
        let halo = || {
            let d = dilate(g, em);
            (0..a.len()).map(|i| d[i] && a[i] && !em[i]).collect::<Vec<bool>>()
        };
        match self.layer.effect {
            Effect::Cycle => {
                let ramp = [self.deep, self.dark, self.mid, self.light, self.hot];
                for i in 0..a.len() {
                    if self.veins[i] && self.dist[i].is_finite() {
                        let idx = ((self.dist[i] / 2.).floor() as i32 - (t * 8. / n) as i32).rem_euclid(4) as usize + 1;
                        blend(out, i, ramp[idx], 0.55);
                    }
                }
            }
            Effect::Pulse => {
                let halo = halo();
                for i in 0..a.len() {
                    if em[i] {
                        blend(out, i, self.light, 0.55 * p);
                    } else if halo[i] {
                        blend(out, i, self.light, 0.22 * p);
                    }
                }
            }
            Effect::Heat => {
                let (cool, white) = (self.fire[1], self.fire[3]);
                let target = [0, 1, 2].map(|k| cool[k] + (white[k] - cool[k]) * p);
                let halo = halo();
                for i in 0..a.len() {
                    if em[i] {
                        blend(out, i, target, 0.75 * p);
                    } else if halo[i] {
                        blend(out, i, cool, 0.3 * p);
                    }
                }
            }
            Effect::Plasma => {
                let tail = 4.;
                for i in 0..a.len() {
                    if !self.veins[i] {
                        continue;
                    }
                    let s = (self.dmax + tail + 1.) * t / (n * 0.75) - self.dist[i]; // the front travels 3/4 of the loop
                    if (0. ..1.).contains(&s) {
                        blend(out, i, self.hot, 0.9);
                    } else if (1. ..2.5).contains(&s) {
                        blend(out, i, self.light, 0.7);
                    } else if (2.5..tail).contains(&s) {
                        blend(out, i, self.mid, 0.45);
                    }
                }
            }
            Effect::Trail => {
                if self.path.is_empty() {
                    return;
                }
                let l = self.path.len() as f32;
                let tail = [(self.hot, 1.), (self.hot, 1.), (self.light, 0.9), (self.light, 0.75), (self.mid, 0.6), (self.mid, 0.45), (self.mid, 0.3)];
                for h in 0..self.heads {
                    let head = (h as f32 * l / self.heads as f32 + t * l / n).rem_euclid(l);
                    for (j, &(c, k)) in tail.iter().enumerate() {
                        let (y, x) = self.path[((head - j as f32).trunc() as i64).rem_euclid(self.path.len() as i64) as usize];
                        blend(out, g.i(y, x), c, k);
                        if j < 2
                            && let Some(&(dy, dx)) = N8.iter().find(|&&(dy, dx)| g.inside(y + dy, x + dx) && !a[g.i(y + dy, x + dx)])
                        {
                            put(out, y + dy, x + dx, self.light); // the head sparks 1 px outside the silhouette
                        }
                    }
                }
            }
            Effect::Rim => {
                // an outline around the area that breathes: dark, then mid, then light at the peak
                let tone = if p > 0.82 {
                    self.light
                } else if p > 0.55 {
                    self.mid
                } else if p > 0.28 {
                    self.dark
                } else {
                    return;
                };
                let around = dilate(g, &self.body);
                for i in 0..a.len() {
                    if around[i] && !a[i] {
                        let (y, x) = self.sp.at(i);
                        put(out, y, x, tone);
                    }
                }
            }
            Effect::Shine => {
                let (lo, hi) = (-(g.h as f32) - 3., g.w as f32 + 3.);
                let pos = lo + (hi - lo) * t / (n * 0.5); // sweeps in the first half of the loop
                for i in 0..a.len() {
                    let (y, x) = self.sp.at(i);
                    let d = ((x - y) as f32 - pos).abs();
                    if self.body[i] && d <= 1.5 && self.sp.v[i] > 0.25 {
                        blend(out, i, [255.; 3], 0.35);
                        if d <= 0.5 {
                            blend(out, i, [255.; 3], 0.35);
                        }
                    }
                }
            }
            Effect::Frost => {
                // the whole item turns a little colder; rime creeps in from the edge and draws back with a bright front.
                // The dark outline keeps its colour, so the item keeps its shape.
                let level = self.layer.amount.max(1) as f32 * (0.6 + 0.4 * p);
                for i in 0..a.len() {
                    if !self.body[i] || self.sp.v[i] < 0.2 {
                        continue;
                    }
                    let d = self.rime[i];
                    if d >= level {
                        blend(out, i, self.mid, 0.14);
                    } else if d > level - 1. {
                        blend(out, i, self.hot, 0.8);
                    } else {
                        blend(out, i, self.light, 0.5);
                    }
                }
                self.glint(out, ti);
            }
            Effect::Icicles => {
                for ic in &self.icicles {
                    let free = |y: i32, x: i32| g.inside(y, x) && (self.explicit || !a[g.i(y, x)]);
                    // a wedge: wider where it hangs, one pixel at the tip; lit in the middle, in shade on the right
                    for j in 1..=ic.len {
                        let left = ic.len - j + 1;
                        if left >= 3 && free(ic.y + j, ic.x + 1) {
                            put(out, ic.y + j, ic.x + 1, self.mid);
                        }
                        if left >= 5 && free(ic.y + j, ic.x - 1) {
                            put(out, ic.y + j, ic.x - 1, self.hot);
                        }
                        put(out, ic.y + j, ic.x, if left == 1 { self.hot } else { self.light });
                    }
                    // a glint slides down in the first half of the loop; then a drop falls from the tip
                    let (age, half) = (age_of(ic.birth), n * 0.5);
                    if age < half {
                        put(out, ic.y + 1 + (age / half * ic.len as f32) as i32, ic.x, self.hot);
                    } else if ic.drips {
                        let y = ic.y + ic.len + 1 + ((age - half).powi(2) * 0.3) as i32;
                        if free(y, ic.x) {
                            put(out, y, ic.x, self.light);
                        }
                    }
                }
            }
            Effect::Snow => {
                let (top, distance) = self.fall;
                for f in &self.flakes {
                    let age = age_of(f.birth);
                    let (y, x) = ((top + distance * age / n).round() as i32, (f.x0 + 1.3 * (f.sway + TAU * age / n).sin()).round() as i32);
                    if g.inside(y, x) && (f.front || !a[g.i(y, x)]) {
                        put(
                            out,
                            y,
                            x,
                            if age < n * 0.7 {
                                self.hot
                            } else if age < n * 0.88 {
                                self.light
                            } else {
                                self.mid
                            },
                        );
                    }
                }
            }
            Effect::Flames => self.flames(out, t),
            Effect::Drip => {
                for &(y, x, birth) in &self.drips {
                    let (age, form) = (age_of(birth), n * 0.45);
                    if age < form {
                        put(out, y + 1, x, self.mid);
                        if age >= form / 2. {
                            put(out, y + 1, x, self.dark);
                            put(out, y + 2, x, self.mid);
                        }
                    } else {
                        let fy = (y as f32 + 2. + (age - form).powi(2) * 0.35) as i32;
                        put(out, fy - 1, x, self.light);
                        put(out, fy, x, self.mid);
                    }
                }
            }
            Effect::Bubbles => {
                for &(x, y0, birth) in &self.bubbles {
                    let y = (y0 as f32 - 0.5 * age_of(birth)) as i32;
                    if g.inside(y, x) && self.liquid[g.i(y, x)] {
                        blend(out, g.i(y, x), self.hot, 0.8);
                    }
                }
            }
            Effect::Smoke => {
                for w in &self.wisps {
                    let (age, life) = (age_of(w.birth), n * 0.7);
                    if age > life {
                        continue;
                    }
                    let y = (w.y0 - 1. - 0.5 * age).round() as i32;
                    let x = (w.x0 + (w.sway + TAU * age / n).sin()).round() as i32;
                    put(out, y, x, self.smoke.1);
                    if age < life * 0.5 {
                        put(out, y, x + 1, self.smoke.0);
                        put(out, y + 1, x, self.smoke.0);
                    }
                }
            }
            Effect::Motes => {
                for m in &self.motes {
                    let age = age_of(m.birth);
                    if age > m.life {
                        continue;
                    }
                    let k = age / m.life.max(1.);
                    let y = (m.y0 + m.vy * age).round() as i32;
                    let x = (m.x0 + 0.6 * (m.sway + age * 0.7).sin()).round() as i32;
                    put(
                        out,
                        y,
                        x,
                        if k < 0.25 {
                            self.hot
                        } else if k < 0.6 {
                            self.light
                        } else if k < 0.85 {
                            self.mid
                        } else {
                            self.dark
                        },
                    );
                }
            }
            Effect::Glint => self.glint(out, ti),
            Effect::Orbit => {
                let (cy, cx, rx, ry) = self.orb;
                let count = self.layer.amount.max(1);
                for i in 0..count {
                    for (j, c) in [self.hot, self.light, self.mid].into_iter().enumerate() {
                        let ang = TAU * (t / n + i as f32 / count as f32) - j as f32 * 0.2;
                        let (y, x) = ((cy + ry * ang.sin()).round() as i32, (cx + rx * ang.cos()).round() as i32);
                        if ang.sin() < 0. && g.inside(y, x) && a[g.i(y, x)] {
                            continue; // the far side passes behind the item
                        }
                        put(out, y, x, c);
                    }
                }
            }
            Effect::Arcs => {
                for (t0, pts) in &self.bolts {
                    let k = (ti + self.n - t0 % self.n) % self.n;
                    if k < 2 {
                        for &(y, x) in pts {
                            put(out, y, x, if k == 0 { self.hot } else { self.light });
                        }
                    }
                }
            }
            Effect::Wave => self.wave(out, t),
            Effect::Shock => self.shock(out, t),
            Effect::Beat => self.beat(out, t),
            Effect::Blink => {
                // each part lights up in turn and rests dark, like eyes opening one after another
                let k = self.parts.len().max(1);
                let window = (n / k as f32).max(2.);
                for (i, part) in self.parts.iter().enumerate() {
                    let age = age_of(i as f32 * n / k as f32);
                    for &px in part {
                        if age < window {
                            blend(out, px, if age < 1. || age >= window - 1. { self.light } else { self.hot }, 0.9);
                        } else {
                            blend(out, px, self.deep, 0.6);
                        }
                    }
                }
            }
            Effect::Hue => {
                for i in 0..a.len() {
                    if em[i] {
                        let (h, s, v) = hsv([out[i][0], out[i][1], out[i][2]]);
                        blend(out, i, rgb_of((h + t / n).rem_euclid(1.), s, v), 1.);
                    }
                }
            }
            Effect::Twinkle => {
                if self.sparkle.is_empty() {
                    return;
                }
                // a few bright pixels flare for one frame and fade on the next
                let pick = |frame: usize, j: usize| {
                    self.sparkle[(hash01(self.seed, frame as i64, j as i64) * self.sparkle.len() as f32) as usize % self.sparkle.len()]
                };
                for j in 0..self.layer.amount.max(1) {
                    blend(out, pick((ti + self.n - 1) % self.n, j), self.light, 0.7);
                }
                for j in 0..self.layer.amount.max(1) {
                    blend(out, pick(ti, j), self.hot, 1.);
                }
            }
        }
    }

    /// A pixel flares, opens into a cross on the next frame and fades: twice per loop for a glint, on the frost too.
    fn glint(&self, out: &mut [[f32; 4]], ti: usize) {
        let g = &self.sp.g;
        let mut put = |y: i32, x: i32, c: Col| {
            if g.inside(y, x) {
                out[g.i(y, x)] = [c[0], c[1], c[2], 255.];
            }
        };
        for &(at, (y, x)) in &self.glints {
            let k = (ti + self.n - at % self.n) % self.n;
            if k == 1 {
                for (dy, dx) in [(0, 1), (0, -1), (1, 0), (-1, 0)] {
                    put(y + dy, x + dx, self.light);
                }
            }
            if k <= 2 {
                put(y, x, if k < 2 { self.hot } else { self.light });
            }
        }
    }

    /// Fire as tongues: along each burning edge a slow noise sets how high the fire reaches at every column, so licks
    /// rise and sink with gaps between them, and a finer noise scrolling upwards tears their tips. Colour follows how
    /// much flame is left above a pixel: only tall tongues have a bright core, low fire stays dark. Both noises go
    /// round a whole number of times per loop, so it repeats. Embers leave from the fire and the edge under it is lit.
    fn flames(&self, out: &mut [[f32; 4]], t: f32) {
        let (g, a, n) = (&self.sp.g, &self.sp.a, self.n as f32);
        let height = self.layer.amount.max(2) as f32;
        let period = 8;
        let (morph, rise) = (period as f32 * t / n, period as f32 * 2. * t / n);
        let mut left = vec![0f32; a.len()];
        for f in &self.flames {
            for j in 1..=(height * 1.3).ceil() as i32 {
                let (y, x) = (f.y - j, f.x);
                if !g.inside(y, x) || (!self.explicit && a[g.i(y, x)]) {
                    break;
                }
                let up = j as f32;
                let lean = (TAU * t / n + up * 0.5 + f.x as f32 * 0.3).sin() * up / height;
                let tongue = noise(self.seed, (f.x as f32 + lean) * 0.34, morph + f.y as f32 * 0.21, period);
                let reach = height * f.taper * (0.25 + 1.6 * tongue * tongue).min(1.25);
                let tear = noise(self.seed ^ 77, f.x as f32 * 0.8 + lean, up * 0.7 - rise, period);
                let i = g.i(y, x);
                left[i] = left[i].max(reach - up + (tear - 0.5) * 1.6);
            }
        }
        let band = |l: f32| {
            self.fire[if l > 3.4 {
                3
            } else if l > 2.1 {
                2
            } else if l > 0.9 {
                1
            } else {
                0
            }]
        };
        for (i, &l) in left.iter().enumerate() {
            if l > 0. {
                let c = band(l);
                out[i] = [c[0], c[1], c[2], 255.];
            }
        }
        for f in &self.flames {
            let glow = 0.3 + 0.2 * (TAU * 2. * t / n + f.x as f32 * 0.9).sin();
            for (dy, k) in [(0, glow), (1, glow * 0.45)] {
                if g.inside(f.y + dy, f.x) && a[g.i(f.y + dy, f.x)] && left[g.i(f.y + dy, f.x)] <= 0. {
                    let i = g.i(f.y + dy, f.x);
                    for c in 0..3 {
                        out[i][c] += (self.fire[2][c] - out[i][c]) * k * f.taper;
                    }
                }
            }
        }
        for m in &self.embers {
            let age = (t - m.birth).rem_euclid(n);
            if age > m.life {
                continue;
            }
            let (y, x) = ((m.y0 + m.vy * age).round() as i32, (m.x0 + 1.2 * (m.sway + age * 0.8).sin()).round() as i32);
            if g.inside(y, x) && (self.explicit || !a[g.i(y, x)]) && left[g.i(y, x)] <= 0. {
                let c = self.fire[if age < m.life * 0.4 {
                    2
                } else if age < m.life * 0.75 {
                    1
                } else {
                    0
                }];
                out[g.i(y, x)] = [c[0], c[1], c[2], 255.];
            }
        }
    }

    /// A shockwave runs through the item from its origin, following its shape (by distance inside it, like the plasma)
    /// and losing force on the way. It adds no colour: where it passes, the surface is pushed along the wave's path
    /// (each pixel shows what was just behind it) and stands out in relief, lit on the side of the crest that faces the
    /// light (top left) and shaded on the other, in the item's own colours. A layer colour tints the lit side.
    fn shock(&self, out: &mut [[f32; 4]], t: f32) {
        let (g, a, n) = (&self.sp.g, &self.sp.a, self.n as f32);
        let reach = self.dmax + 4.;
        let r = reach * t / (n * 0.8);
        let force = 1. - 0.7 * (r / reach).min(1.);
        let push = self.layer.amount.clamp(1, 3) as f32 * force;
        let lit_with = self.hot;
        let before = out.to_vec();
        for i in 0..a.len() {
            let behind = r - self.dist[i];
            if !self.body[i] || !(0. ..4.6).contains(&behind) {
                continue;
            }
            let (fy, fx) = self.flow[i];
            let (y, x) = self.sp.at(i);
            let shift = (push * (1. - (behind - 1.5).abs() / 1.5).max(0.)).round();
            if shift >= 1. {
                let (sy, sx) = ((y as f32 - fy * shift).round() as i32, (x as f32 - fx * shift).round() as i32);
                if g.inside(sy, sx) && a[g.i(sy, sx)] {
                    out[i] = before[g.i(sy, sx)];
                }
            }
            // the front slope faces the way the wave moves, the back slope the other way; light comes from the top left
            let facing = -(fy + fx) * 0.707;
            let relief = force * if behind < 2. { 0.3 + 0.7 * facing } else { -0.3 - 0.7 * facing };
            for j in 0..3 {
                // lit: the pixel's own colour, brighter (a layer colour tints it); shaded: the same colour, darker
                out[i][j] = if relief > 0. {
                    let own = out[i][j] * (1. + relief) + 46. * relief;
                    if self.layer.color.is_some() { own + (lit_with[j] - own) * 0.5 * relief } else { own }
                } else {
                    out[i][j] * (1. + 0.75 * relief)
                };
            }
        }
    }

    /// A heartbeat: a strong beat, a weaker one a quarter of a loop later, then rest. On each beat the area swells from
    /// its centre (every pixel shows what was a little closer to it, so it grows a pixel or two and falls back) and
    /// flushes: the glow lights up, the rest takes a little of the colour.
    fn beat(&self, out: &mut [[f32; 4]], t: f32) {
        let (g, a) = (&self.sp.g, &self.sp.a);
        let phase = t / self.n as f32;
        let bump = |at: f32, width: f32| (-((phase - at) / width).powi(2)).exp();
        let e = (bump(0.085, 0.06) + 0.65 * bump(0.33, 0.07)).min(1.);
        if e < 0.02 {
            return;
        }
        let (cy, cx, half) = self.heart;
        let scale = 1. + e * self.layer.amount.clamp(1, 3) as f32 / half.max(2.);
        let before = out.to_vec();
        for i in 0..a.len() {
            let (y, x) = self.sp.at(i);
            let (sy, sx) = ((cy + (y as f32 - cy) / scale).round() as i32, (cx + (x as f32 - cx) / scale).round() as i32);
            if !g.inside(sy, sx) || !self.body[g.i(sy, sx)] {
                continue;
            }
            let src = g.i(sy, sx);
            let (c, k) = if self.em[src] { (self.light, 0.6) } else { (self.mid, 0.12) };
            let mut px = before[src];
            for j in 0..3 {
                px[j] += (c[j] - px[j]) * k * e;
            }
            out[i] = px;
        }
    }

    /// The area sways: each line of it slides along itself by a travelling sine, more the farther from the anchor.
    /// What a line leaves behind turns transparent at the item's edge, or keeps the edge pixel inside the item.
    fn wave(&self, out: &mut [[f32; 4]], t: f32) {
        let (g, a, n) = (&self.sp.g, &self.sp.a, self.n as f32);
        let before = out.to_vec();
        let (lines, len) = if self.sway_rows { (g.h, g.w) } else { (g.w, g.h) };
        let at = |line: i32, pos: i32| if self.sway_rows { g.i(line, pos) } else { g.i(pos, line) };
        let amp = self.layer.amount.clamp(1, 4) as f32;
        let wavelength = (self.span * 1.5).max(6.);
        for line in 0..lines {
            let far = ((line as f32 - self.anchor).abs() / self.span.max(1.)).min(1.);
            let off = (amp * far * (TAU * t / n - line as f32 * TAU / wavelength).sin()).round() as i32;
            if off == 0 {
                continue;
            }
            let mut pos = 0;
            while pos < len {
                if !self.body[at(line, pos)] {
                    pos += 1;
                    continue;
                }
                let start = pos;
                while pos < len && self.body[at(line, pos)] {
                    pos += 1;
                }
                let end = pos - 1;
                let open = |beyond: i32| !(0..len).contains(&beyond) || !a[at(line, beyond)];
                for q in start..=end {
                    let src = q - off;
                    if src < start {
                        out[at(line, q)] = if open(start - 1) { [0.; 4] } else { before[at(line, start)] };
                    } else if src > end {
                        out[at(line, q)] = if open(end + 1) { [0.; 4] } else { before[at(line, end)] };
                    }
                }
                for q in start..=end {
                    if (0..len).contains(&(q + off)) {
                        out[at(line, q + off)] = before[at(line, q)];
                    }
                }
            }
        }
    }
}

/// The loop's frames, or None for a sprite with no visible pixels. Layers whose area is empty are skipped.
pub fn animate(img: &RgbaImage, s: &Settings) -> Option<Vec<RgbaImage>> {
    let sp = Sprite::new(img, s.threshold)?;
    let n = s.frames.max(3);
    let layers: Vec<Animator> =
        s.layers.iter().enumerate().filter(|(_, l)| l.on).filter_map(|(i, l)| Animator::new(&sp, l, n, s.seed.wrapping_add(i as u64 * 7919))).collect();
    let q = |c: f32| c.round().clamp(0., 255.) as u8;
    Some(
        (0..n)
            .map(|t| {
                let mut out = sp.img.clone();
                for an in &layers {
                    let k = an.layer.strength.clamp(0., 1.);
                    let before = (k < 1.).then(|| out.clone());
                    an.apply(&mut out, t * an.layer.repeats.max(1));
                    if let Some(before) = before {
                        for (o, b) in out.iter_mut().zip(&before) {
                            if o[3] > 127. && b[3] > 127. {
                                for c in 0..3 {
                                    o[c] = b[c] + (o[c] - b[c]) * k;
                                }
                            }
                        }
                    }
                }
                RgbaImage::from_fn(sp.g.w as u32, sp.g.h as u32, |x, y| {
                    let p = out[sp.g.i(y as i32, x as i32)];
                    Rgba([q(p[0]), q(p[1]), q(p[2]), if p[3] > 127. { 255 } else { 0 }])
                })
            })
            .collect(),
    )
}

/// The pixels `area` covers on `img`, row-major: what the editor tints.
pub fn area_mask(img: &RgbaImage, area: &Area, effect: Effect, threshold: f32) -> Vec<bool> {
    let Some(sp) = Sprite::new(img, threshold) else { return vec![false; (img.width() * img.height()) as usize] };
    let layer = Layer { area: area.clone(), ..Layer::new(effect) };
    match Animator::new(&sp, &layer, 3, 0) {
        Some(an) if matches!(area, Area::Auto) && matches!(effect, Effect::Pulse | Effect::Heat | Effect::Motes | Effect::Blink | Effect::Hue) => an.em,
        Some(an) => an.body,
        None => vec![false; sp.a.len()],
    }
}

/// Frames side by side, the sheet a game reads.
pub fn strip(frames: &[RgbaImage]) -> RgbaImage {
    let (w, h) = frames.first().map(|f| f.dimensions()).unwrap_or((1, 1));
    let mut out = RgbaImage::new(w * frames.len().max(1) as u32, h);
    for (i, f) in frames.iter().enumerate() {
        image::imageops::overlay(&mut out, f, (i as u32 * w) as i64, 0);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A round brown body with a green gem in the middle.
    fn gem_sprite() -> (RgbaImage, Vec<bool>) {
        let body = |x: i32, y: i32| (y - 18).pow(2) + (x - 16).pow(2) <= 64;
        let im = RgbaImage::from_fn(32, 32, |x, y| {
            let (x, y) = (x as i32, y as i32);
            if (y - 18).pow(2) + (x - 16).pow(2) <= 4 {
                Rgba([60, 220, 90, 255])
            } else if body(x, y) {
                Rgba([110, 74, 40, 255])
            } else {
                Rgba([0, 0, 0, 0])
            }
        });
        let mask = (0..32 * 32).map(|i| body(i % 32, i / 32)).collect();
        (im, mask)
    }

    fn one(layer: Layer) -> Settings {
        Settings { layers: vec![layer], ..Settings::default() }
    }

    fn ready<'a>(sp: &'a Sprite, layer: &'a Layer) -> Animator<'a> {
        Animator::new(sp, layer, 12, 0).unwrap()
    }

    #[test]
    fn the_gem_glows_and_plasma_starts_there() {
        let (im, _) = gem_sprite();
        let sp = Sprite::new(&im, 0.55).unwrap();
        let layer = Layer::new(Effect::Plasma);
        let an = ready(&sp, &layer);
        let g = &sp.g;
        assert!(an.em[g.i(18, 16)] && !an.em[g.i(18, 10)], "gem is emissive, wood is not");
        assert!(an.dist[g.i(18, 16)] < an.dist[g.i(18, 9)], "plasma starts at the gem");
    }

    #[test]
    fn the_contour_walk_visits_every_edge_pixel_once() {
        let (im, body) = gem_sprite();
        let sp = Sprite::new(&im, 0.55).unwrap();
        let layer = Layer::new(Effect::Trail);
        let an = ready(&sp, &layer);
        let inner = erode(&sp.g, &body);
        assert_eq!(an.path.len(), (0..body.len()).filter(|&i| body[i] && !inner[i]).count());
    }

    #[test]
    fn the_plasma_starts_where_it_is_told() {
        let (im, _) = gem_sprite();
        let sp = Sprite::new(&im, 0.55).unwrap();
        let layer = Layer { origin: Some((9, 18)), ..Layer::new(Effect::Plasma) };
        let an = ready(&sp, &layer);
        assert_eq!(an.dist[sp.g.i(18, 9)], 0.);
        assert!(an.dist[sp.g.i(18, 16)] > 0.);
        let off = Layer { origin: Some((0, 0)), ..Layer::new(Effect::Plasma) };
        let an = ready(&sp, &off);
        assert!(sp.a[an.dist.iter().position(|&d| d == 0.).unwrap()], "a point off the item snaps onto it");
    }

    #[test]
    fn plasma_fronts_are_round() {
        let g = Grid { w: 31, h: 31 };
        let d = geodesic(&g, &vec![true; 31 * 31], (15, 15));
        assert!((d[g.i(0, 0)] - 15. * 2f32.sqrt()).abs() < 0.5);
        assert!((d[g.i(0, 8)] - 15f32.hypot(7.)).abs() < 0.5);
    }

    fn changing(frames: &[RgbaImage]) -> usize {
        frames.iter().map(|f| f.as_raw().clone()).collect::<BTreeSet<_>>().len()
    }

    #[test]
    fn every_effect_loops_with_binary_alpha() {
        let (im, body) = gem_sprite();
        let mut runs: Vec<Settings> = ELEMENTS.iter().map(|e| Settings { layers: e.layers(), ..Settings::default() }).collect();
        runs.extend(Effect::ALL.map(|e| one(Layer::new(e))));
        runs.push(Settings { layers: Effect::ALL.map(Layer::new).to_vec(), ..Settings::default() });
        for s in runs {
            let names: Vec<&str> = s.layers.iter().map(|l| l.effect.key()).collect();
            let fr = animate(&im, &s).unwrap();
            assert_eq!(fr.len(), 12);
            assert!(fr.iter().all(|f| f.pixels().all(|p| p.0[3] == 0 || p.0[3] == 255)), "alpha stays binary: {names:?}");
            assert!(changing(&fr) > 1, "frames change: {names:?}");
            if !names.contains(&"wave") {
                assert!(fr.iter().all(|f| (0..1024).all(|i| !body[i] || f.get_pixel(i as u32 % 32, i as u32 / 32).0[3] == 255)), "{names:?}");
            }
            let twice = animate(&im, &Settings { frames: 12, ..s.clone() }).unwrap();
            assert_eq!(fr, twice, "the same options give the same loop: {names:?}");
        }
    }

    #[test]
    fn repeats_run_the_effect_more_often() {
        let (im, _) = gem_sprite();
        let once = animate(&im, &one(Layer::new(Effect::Pulse))).unwrap();
        let twice = animate(&im, &one(Layer { repeats: 2, ..Layer::new(Effect::Pulse) })).unwrap();
        assert_eq!(twice[3], once[6]);
        assert_eq!(twice[6], once[0]);
    }

    #[test]
    fn a_painted_area_keeps_the_layer_inside_it() {
        let (im, _) = gem_sprite();
        let left: Vec<bool> = (0..1024).map(|i| i % 32 < 14).collect();
        let s = one(Layer { area: Area::Mask(left.clone()), color: Some([255, 0, 0]), ..Layer::new(Effect::Pulse) });
        let fr = animate(&im, &s).unwrap();
        // the pulse lights the area and its 1 px halo, nothing farther
        let near: Vec<bool> = (0..1024).map(|i| i % 32 < 15).collect();
        assert!(fr.iter().all(|f| (0..1024usize).all(|i| near[i] || f.get_pixel(i as u32 % 32, i as u32 / 32) == im.get_pixel(i as u32 % 32, i as u32 / 32))));
        assert!(fr[6].get_pixel(10, 18) != im.get_pixel(10, 18));
    }

    #[test]
    fn strength_scales_the_change() {
        let (im, _) = gem_sprite();
        let full = animate(&im, &one(Layer::colored(Effect::Pulse, [255, 255, 255]))).unwrap();
        let half = animate(&im, &one(Layer { strength: 0.5, ..Layer::colored(Effect::Pulse, [255, 255, 255]) })).unwrap();
        let (b, f, h) = (im.get_pixel(16, 18).0[0] as i32, full[6].get_pixel(16, 18).0[0] as i32, half[6].get_pixel(16, 18).0[0] as i32);
        assert!(f > h && h > b, "{b} < {h} < {f}");
    }

    #[test]
    fn a_wave_hangs_from_its_anchor() {
        let (im, _) = gem_sprite();
        let s = one(Layer { amount: 2, ..Layer::new(Effect::Wave) });
        let fr = animate(&im, &s).unwrap();
        // anchored at the top edge by default: the top row never moves, lower rows do
        assert!(fr.iter().all(|f| (0..32).all(|x| f.get_pixel(x, 10) == im.get_pixel(x, 10))));
        assert!(fr.iter().any(|f| (0..32).any(|x| f.get_pixel(x, 24) != im.get_pixel(x, 24))));
    }

    #[test]
    fn flames_rise_above_the_item_and_light_its_edge() {
        let (im, body) = gem_sprite();
        let fr = animate(&im, &one(Layer::colored(Effect::Flames, [240, 110, 20]))).unwrap();
        let fire = |f: &RgbaImage| (0..1024usize).filter(|&i| !body[i] && f.get_pixel(i as u32 % 32, i as u32 / 32).0[3] == 255).count();
        assert!(fr.iter().all(|f| fire(f) >= 8), "fire in every frame: {:?}", fr.iter().map(fire).collect::<Vec<_>>());
        assert!(fr.iter().all(|f| (0..32).all(|x| f.get_pixel(x, 31).0[3] == 0)), "nothing burns below the item");
        assert!(changing(&fr) == 12, "no two frames alike");
    }

    #[test]
    fn frost_stays_on_the_item_and_icicles_hang_below_it() {
        let (im, body) = gem_sprite();
        let at = |f: &RgbaImage, i: usize| *f.get_pixel(i as u32 % 32, i as u32 / 32);
        let frost = animate(&im, &one(Layer::new(Effect::Frost))).unwrap();
        assert!(frost.iter().all(|f| (0..1024).all(|i| !body[i] || at(f, i).0[3] == 255)), "rime tints the item, it does not eat it");
        let rimed = |f: &RgbaImage| (0..1024).filter(|&i| body[i] && at(f, i).0[2] as i32 - at(&im, i).0[2] as i32 > 60).count();
        assert!(rimed(&frost[6]) > rimed(&frost[0]) && rimed(&frost[0]) > 0, "it creeps in and never melts away: {} {}", rimed(&frost[0]), rimed(&frost[6]));
        let ice = animate(&im, &one(Layer::new(Effect::Icicles))).unwrap();
        let hanging: Vec<usize> = (0..1024).filter(|&i| !body[i] && at(&ice[0], i).0[3] == 255).collect();
        assert!(hanging.len() >= 6 && hanging.iter().all(|&i| i / 32 > 18), "icicles are below the middle of the item: {hanging:?}");
    }

    #[test]
    fn a_shockwave_expands_from_its_origin() {
        let (im, body) = gem_sprite();
        let fr = animate(&im, &one(Layer { origin: Some((16, 18)), ..Layer::colored(Effect::Shock, [255, 255, 255]) })).unwrap();
        // the farthest pixel the wave has changed inside the item, frame by frame
        let front = |f: &RgbaImage| {
            (0..1024usize)
                .filter(|&i| body[i] && f.get_pixel(i as u32 % 32, i as u32 / 32) != im.get_pixel(i as u32 % 32, i as u32 / 32))
                .map(|i| ((i / 32) as f32 - 18.).hypot((i % 32) as f32 - 16.))
                .fold(0., f32::max)
        };
        assert!(front(&fr[1]) < front(&fr[3]) && front(&fr[3]) < front(&fr[5]), "{} {} {}", front(&fr[1]), front(&fr[3]), front(&fr[5]));
        assert_eq!(fr[11], im, "the loop rests once the wave has passed");
        assert!(fr.iter().all(|f| (0..1024usize).all(|i| body[i] == (f.get_pixel(i as u32 % 32, i as u32 / 32).0[3] == 255))), "it stays on the item");
    }

    #[test]
    fn a_heart_beats_twice_and_rests() {
        let (im, body) = gem_sprite();
        let fr = animate(&im, &one(Layer::new(Effect::Beat))).unwrap();
        let size = |f: &RgbaImage| f.pixels().filter(|p| p.0[3] == 255).count();
        let rest = body.iter().filter(|&&b| b).count();
        assert!(size(&fr[1]) > rest, "it swells on the beat: {} > {rest}", size(&fr[1]));
        assert!(size(&fr[1]) >= size(&fr[4]) && fr[4] != im, "the second beat is the weaker one");
        assert!(fr[7..].iter().all(|f| *f == im), "then it rests");
    }

    #[test]
    fn noise_repeats_with_its_period() {
        for (x, y) in [(0.3, 0.7), (4.2, 7.9), (-2.5, 3.1)] {
            assert!((noise(5, x, y, 8) - noise(5, x, y + 8., 8)).abs() < 1e-5);
            assert!((0. ..=1.).contains(&noise(5, x, y, 8)));
        }
    }

    #[test]
    fn an_empty_sprite_has_no_animation() {
        assert!(animate(&RgbaImage::new(8, 8), &Settings::default()).is_none());
    }
}

#[cfg(test)]
mod dump {
    use super::*;
    /// `ITEM_ANIM_IN="a.png;b.png" ITEM_ANIM_OUT=dir cargo test --release dump_item_anims -- --ignored`: one GIF per
    /// row (the elements, then every effect on its own), the given sprites side by side, plus the middle frame as PNG.
    #[test]
    #[ignore]
    fn dump_item_anims() {
        let out = std::path::PathBuf::from(std::env::var("ITEM_ANIM_OUT").unwrap());
        let items: Vec<RgbaImage> = std::env::var("ITEM_ANIM_IN").unwrap().split(';').map(|p| image::open(p).unwrap().to_rgba8()).collect();
        let mut rows: Vec<(String, Settings)> = ELEMENTS.iter().map(|e| (e.key.to_string(), Settings { layers: e.layers(), ..Settings::default() })).collect();
        rows.extend(Effect::ALL.map(|e| (format!("fx_{}", e.key()), Settings { layers: vec![Layer::new(e)], ..Settings::default() })));
        for (name, s) in rows {
            let anims: Vec<Vec<RgbaImage>> = items.iter().map(|im| animate(im, &s).unwrap()).collect();
            let frames: Vec<RgbaImage> = (0..s.frames)
                .map(|t| {
                    let w: u32 = items.iter().map(|i| i.width() + 4).sum();
                    let h = items.iter().map(|i| i.height()).max().unwrap();
                    let mut f = RgbaImage::from_pixel(w, h, Rgba([38, 44, 48, 255]));
                    let mut x = 0;
                    for a in &anims {
                        image::imageops::overlay(&mut f, &a[t], x, 0);
                        x += a[t].width() as i64 + 4;
                    }
                    f
                })
                .collect();
            std::fs::write(out.join(format!("{name}.gif")), crate::anim::gif(&frames, 10, 4).unwrap()).unwrap();
            frames[4].save(out.join(format!("{name}_f4.png"))).unwrap();
        }
    }
}
