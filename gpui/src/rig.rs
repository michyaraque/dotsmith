//! Pose / silhouette generator: a tiny 3D rig (humanoid or quadruped) of capsule bones, posed by a few procedural
//! animations and drawn with an orthographic high camera straight into sprite pixels, one flat colour per body part
//! (the colour-coded blockout used as an img2img origin). No anti-aliasing: every pixel is exactly one part or empty.
//!
//! World: X right, Y away from the viewer (north), Z up; units are sprite pixels at the 32 px scale.
//! Projections:
//! - Oblique (default): the ground is seen straight from above (north = up) and height is drawn diagonally up
//!   and to the left (`lean` px per unit on each axis). That is why a low rat lies vertical (N/S) or horizontal (E/W)
//!   while a tall player leans 45° with the head up-left in every direction (checked on exported player/rat sheets).
//! - Orthographic: looks north, tilted down by `pitch` (0 = eye level, 90 = straight down), for top-down 3/4 styles.
use image::{Rgba, RgbaImage};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Humanoid,
    Quadruped,
}
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dir {
    North,
    East,
    South,
    West,
}
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Anim {
    Idle,
    Walk,
    Attack,
}

/// (value, Spanish label, English label)
pub const KINDS: [(Kind, &str, &str); 2] = [(Kind::Humanoid, "Humanoide", "Humanoid"), (Kind::Quadruped, "Cuadrúpedo", "Quadruped")];
pub const DIRS: [(Dir, &str, &str); 4] =
    [(Dir::North, "Norte", "North"), (Dir::East, "Este", "East"), (Dir::South, "Sur", "South"), (Dir::West, "Oeste", "West")];
pub const ANIMS: [(Anim, &str, &str); 3] = [(Anim::Idle, "Quieto", "Idle"), (Anim::Walk, "Caminar", "Walk"), (Anim::Attack, "Atacar", "Attack")];
pub const FRAMES: u32 = 8;

impl Dir {
    pub fn key(self) -> &'static str {
        match self {
            Dir::North => "north",
            Dir::East => "east",
            Dir::South => "south",
            Dir::West => "west",
        }
    }
}
impl Kind {
    pub fn key(self) -> &'static str {
        match self {
            Kind::Humanoid => "humanoid",
            Kind::Quadruped => "quadruped",
        }
    }
}
impl Anim {
    pub fn key(self) -> &'static str {
        match self {
            Anim::Idle => "idle",
            Anim::Walk => "walk",
            Anim::Attack => "attack",
        }
    }
}

/// Body parts and their blockout colours (same scheme as the reference rig: cyan head, orange torso, red/purple arms,
/// green/pink legs; yellow tail).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Part {
    Head,
    Torso,
    ArmL,
    ArmR,
    LegL,
    LegR,
    Tail,
}

impl Part {
    pub fn color(self) -> [u8; 3] {
        match self {
            Part::Head => [0x5c, 0xc8, 0xe8],
            Part::Torso => [0xe0, 0x78, 0x20],
            Part::ArmL => [0x70, 0x40, 0xa0],
            Part::ArmR => [0xb0, 0x28, 0x38],
            Part::LegL => [0x60, 0xa8, 0x38],
            Part::LegR => [0xe0, 0x80, 0xc8],
            Part::Tail => [0xe8, 0xd0, 0x40],
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
pub struct Pose {
    pub kind: Kind,
    pub dir: Dir,
    pub anim: Anim,
    pub frame: u32,
    pub oblique: bool, // oblique projection; false = orthographic camera at `pitch`
    pub lean: f32,     // oblique: screen px left and up per unit of height
    pub pitch: f32,    // orthographic: degrees
    pub size: u32,     // sprite side in px (the rig is drawn at size / 32)
    pub silhouette: bool,
}

impl Default for Pose {
    fn default() -> Self {
        Pose { kind: Kind::Humanoid, dir: Dir::South, anim: Anim::Walk, frame: 0, oblique: true, lean: 0.55, pitch: 50.0, size: 32, silhouette: false }
    }
}

type V3 = [f32; 3];

struct Bone {
    a: V3,
    b: V3,
    ra: f32,
    rb: f32,
    part: Part,
}

fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: V3, s: f32) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Rotate a vector in the sagittal plane (Y forward, Z up) by `ang` radians: positive swings forward.
fn swing(len: f32, ang: f32) -> V3 {
    [0.0, len * ang.sin(), -len * ang.cos()]
}

/// Bones in the character's own frame: X = its right, Y = its forward, Z = up; feet at Z = 0.
fn bones(kind: Kind, anim: Anim, frame: u32) -> Vec<Bone> {
    let t = frame as f32 / FRAMES as f32 * std::f32::consts::TAU;
    let mut v = Vec::new();
    match kind {
        Kind::Humanoid => {
            let (step, bob) = match anim {
                Anim::Walk => (0.55, 0.5 * (2.0 * t).cos().abs()),
                Anim::Idle => (0.0, 0.25 * (t).sin().max(0.0)),
                Anim::Attack => (0.12, 0.0),
            };
            let hip_z = 7.5 + bob;
            let pelvis = [0.0, 0.0, hip_z];
            let chest = [0.0, if anim == Anim::Attack { 0.6 * t.sin().max(0.0) } else { 0.0 }, hip_z + 5.5];
            v.push(Bone { a: pelvis, b: chest, ra: 3.8, rb: 4.3, part: Part::Torso });
            v.push(Bone { a: add(chest, [0.0, 0.2, 4.6]), b: add(chest, [0.0, 0.2, 4.6]), ra: 4.2, rb: 4.2, part: Part::Head });
            for (side, leg, arm) in [(-1.0f32, Part::LegL, Part::ArmL), (1.0, Part::LegR, Part::ArmR)] {
                // legs: opposite phases; the knee bends while the leg swings forward
                let ph = if side < 0.0 { t } else { t + std::f32::consts::PI };
                let thigh = step * ph.sin();
                let knee_bend = if anim == Anim::Walk { 0.7 * (ph.cos().max(0.0)) } else { 0.0 };
                let hip = add(pelvis, [side * 1.8, 0.0, -0.5]);
                let knee = add(hip, swing(3.6, thigh));
                let foot = add(knee, swing(3.4 + bob, thigh - knee_bend));
                v.push(Bone { a: hip, b: knee, ra: 2.1, rb: 1.9, part: leg });
                v.push(Bone { a: knee, b: [foot[0], foot[1], foot[2].max(0.6)], ra: 1.9, rb: 1.8, part: leg });
                // arms swing against the legs; the right arm strikes forward when attacking
                let shoulder = add(chest, [side * 4.6, 0.0, -0.6]);
                let (upper, fore) = match anim {
                    Anim::Attack if side > 0.0 => {
                        let k = (0.5 - 0.5 * t.cos()).powf(0.7); // wind up -> strike
                        let up = -0.6 + 2.6 * k;
                        (up, up + 0.4)
                    }
                    _ => {
                        let a = -0.8 * step * ph.sin();
                        (a, a + 0.25)
                    }
                };
                let elbow = add(shoulder, swing(3.6, upper));
                let hand = add(elbow, swing(3.4, fore));
                v.push(Bone { a: shoulder, b: elbow, ra: 1.7, rb: 1.6, part: arm });
                v.push(Bone { a: elbow, b: hand, ra: 1.6, rb: 1.6, part: arm });
            }
        }
        Kind::Quadruped => {
            // a rat-like low body: Y forward
            let (step, bob) = match anim {
                Anim::Walk => (0.5, 0.35 * (2.0 * t).cos().abs()),
                Anim::Idle => (0.0, 0.2 * t.sin().max(0.0)),
                Anim::Attack => (0.1, 0.0),
            };
            let lunge = if anim == Anim::Attack { 1.6 * (0.5 - 0.5 * t.cos()) } else { 0.0 };
            let back = [0.0, -3.5 + lunge, 3.4 + bob];
            let front = [0.0, 3.0 + lunge, 3.6 + bob];
            v.push(Bone { a: back, b: front, ra: 2.9, rb: 2.4, part: Part::Torso });
            let neck = add(front, [0.0, 1.8, 0.4]);
            let head_dip = if anim == Anim::Attack { -0.6 * (0.5 - 0.5 * t.cos()) } else { 0.0 };
            v.push(Bone { a: neck, b: add(neck, [0.0, 3.4, -0.9 + head_dip]), ra: 2.1, rb: 0.9, part: Part::Head });
            // ears
            v.push(Bone { a: add(neck, [-1.2, 0.2, 1.8]), b: add(neck, [-1.2, 0.2, 1.8]), ra: 0.9, rb: 0.9, part: Part::Head });
            v.push(Bone { a: add(neck, [1.2, 0.2, 1.8]), b: add(neck, [1.2, 0.2, 1.8]), ra: 0.9, rb: 0.9, part: Part::Head });
            // legs: diagonal pairs move together (front-left with back-right)
            for (x, y, ph0, part) in [
                (-1.7f32, 2.6f32, 0.0f32, Part::ArmL),
                (1.7, 2.6, std::f32::consts::PI, Part::ArmR),
                (-1.7, -3.0, std::f32::consts::PI, Part::LegL),
                (1.7, -3.0, 0.0, Part::LegR),
            ] {
                let a = step * (t + ph0).sin();
                let top = [x, y + lunge, 2.6 + bob];
                let paw = add(top, swing(2.2 + bob, a));
                v.push(Bone { a: top, b: [paw[0], paw[1], paw[2].max(0.5)], ra: 0.9, rb: 0.7, part });
            }
            // tail: a gentle S curve behind, waving a little
            let mut p = add(back, [0.0, -2.2, -0.6]);
            for i in 0..6 {
                let wave = 0.5 * (t + i as f32 * 0.7).sin();
                let q = add(p, [wave, -1.6, -0.35]);
                v.push(Bone { a: p, b: [q[0], q[1], q[2].max(0.4)], ra: 0.55, rb: 0.5, part: Part::Tail });
                p = [q[0], q[1], q[2].max(0.4)];
            }
        }
    }
    v
}

/// Character frame -> world for a facing direction.
fn orient(p: V3, dir: Dir) -> V3 {
    let [x, y, z] = p;
    match dir {
        Dir::North => [x, y, z],   // forward = +Y (away)
        Dir::South => [-x, -y, z], // forward = toward the viewer
        Dir::East => [y, -x, z],   // forward = +X
        Dir::West => [-y, x, z],   // forward = -X
    }
}

/// One projected limb: both ends (x, y, depth), their radii and the body part.
type Segment = ((f32, f32, f32), (f32, f32, f32), f32, f32, Part);

/// One frame: draws the posed rig into a `size` x `size` sprite, feet on a fixed ground point so frames line up.
pub fn render(p: &Pose) -> RgbaImage {
    let n = p.size.max(8);
    let s = n as f32 / 32.0;
    let (sp, cp) = (p.pitch.to_radians().sin(), p.pitch.to_radians().cos());
    let (a, oblique) = (p.lean, p.oblique);
    // screen: u right, v up; depth toward the camera. Oblique: a point moved (a, -a, 1) lands on the same pixel, so the
    // camera sits above, south and east, and depth is the dot product with that direction.
    let project = move |w: V3| -> (f32, f32, f32) {
        if oblique {
            (w[0] - a * w[2], w[1] + a * w[2], (a * w[0] - a * w[1] + w[2]) / (1.0 + 2.0 * a * a).sqrt())
        } else {
            (w[0], w[2] * cp + w[1] * sp, -w[1] * cp + w[2] * sp)
        }
    };
    // where the world origin (between the feet) lands: oblique figures grow up-left, so the feet sit lower right
    let (gx, gy) = if oblique { (n as f32 * 0.5 + 6.0 * a * s, n as f32 - 4.0 * s) } else { (n as f32 / 2.0, n as f32 - 3.0 * s) };
    let segs: Vec<Segment> = bones(p.kind, p.anim, p.frame % FRAMES)
        .into_iter()
        .map(|b| {
            let (a, bb) = (project(orient(scale(b.a, s), p.dir)), project(orient(scale(b.b, s), p.dir)));
            ((gx + a.0, gy - a.1, a.2), (gx + bb.0, gy - bb.1, bb.2), b.ra * s, b.rb * s, b.part)
        })
        .collect();
    let mut img = RgbaImage::new(n, n);
    for py in 0..n {
        for px in 0..n {
            let (x, y) = (px as f32 + 0.5, py as f32 + 0.5);
            let mut best: Option<(f32, Part)> = None;
            for &(a, b, ra, rb, part) in &segs {
                let (dx, dy) = (b.0 - a.0, b.1 - a.1);
                let l2 = dx * dx + dy * dy;
                let k = if l2 < 1e-6 { 0.0 } else { (((x - a.0) * dx + (y - a.1) * dy) / l2).clamp(0.0, 1.0) };
                let (cx, cy) = (a.0 + k * dx, a.1 + k * dy);
                let r = (ra + (rb - ra) * k).max(0.55);
                let d2 = (x - cx).powi(2) + (y - cy).powi(2);
                if d2 <= r * r {
                    // depth of the capsule surface facing the camera
                    let depth = a.2 + k * (b.2 - a.2) + (r * r - d2).sqrt();
                    if best.is_none_or(|(bd, _)| depth > bd) {
                        best = Some((depth, part));
                    }
                }
            }
            if let Some((_, part)) = best {
                let c = if p.silhouette { [0x20, 0x20, 0x24] } else { part.color() };
                img.put_pixel(px, py, Rgba([c[0], c[1], c[2], 255]));
            }
        }
    }
    img
}

/// All 8 frames as a 4 x 2 sheet (the creatures_walk_v1 layout).
pub fn sheet(p: &Pose) -> RgbaImage {
    let n = p.size;
    let mut out = RgbaImage::new(n * 4, n * 2);
    for f in 0..FRAMES {
        let fr = render(&Pose { frame: f, ..*p });
        image::imageops::overlay(&mut out, &fr, ((f % 4) * n) as i64, ((f / 4) * n) as i64);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_pose_draws_something_inside_the_sprite() {
        for (kind, ..) in KINDS {
            for (dir, ..) in DIRS {
                for (anim, ..) in ANIMS {
                    for frame in 0..FRAMES {
                        let im = render(&Pose { kind, dir, anim, frame, ..Pose::default() });
                        let filled = im.pixels().filter(|p| p.0[3] > 0).count();
                        assert!(filled > 40, "{kind:?} {dir:?} {anim:?} {frame}: {filled} px");
                        // nothing touches the top row (the figure fits)
                        assert!((0..32).all(|x| im.get_pixel(x, 0).0[3] == 0), "{kind:?} {dir:?} {anim:?} clipped at the top");
                    }
                }
            }
        }
    }

    #[test]
    fn only_part_colours_no_blending() {
        let im = render(&Pose::default());
        let allowed: Vec<[u8; 3]> = [Part::Head, Part::Torso, Part::ArmL, Part::ArmR, Part::LegL, Part::LegR, Part::Tail].iter().map(|p| p.color()).collect();
        assert!(im.pixels().filter(|p| p.0[3] > 0).all(|p| allowed.contains(&[p.0[0], p.0[1], p.0[2]])));
    }
}

#[cfg(test)]
mod dump {
    use super::*;
    /// `POSE_OUT=dir cargo test --release dump_boards -- --ignored`: every kind x dir walk sheet at x6 on grey.
    #[test]
    #[ignore]
    fn dump_boards() {
        let out = std::path::PathBuf::from(std::env::var("POSE_OUT").unwrap());
        for (kind, ..) in KINDS {
            for (anim, ..) in ANIMS {
                let rows: Vec<RgbaImage> = DIRS.iter().map(|(dir, ..)| sheet(&Pose { kind, dir: *dir, anim, ..Pose::default() })).collect();
                let (w, h) = (rows[0].width(), rows[0].height());
                let mut b = RgbaImage::from_pixel(w, h * 4, Rgba([110, 128, 140, 255]));
                for (i, r) in rows.iter().enumerate() {
                    image::imageops::overlay(&mut b, r, 0, (i as u32 * h) as i64);
                }
                let b = image::imageops::resize(&b, b.width() * 5, b.height() * 5, image::imageops::FilterType::Nearest);
                b.save(out.join(format!("pose_{}_{}.png", kind.key(), anim.key()))).unwrap();
            }
        }
    }
}
