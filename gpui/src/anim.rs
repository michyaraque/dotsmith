//! Sprite-sheet animation: player state (play / pause / step / fps) and frame cutting. No UI and no mode knowledge:
//! the creature viewer and the GIF export both read frames through here.
use image::{RgbaImage, imageops};
use std::time::{Duration, Instant};

pub struct Anim {
    pub playing: bool,
    pub frame: usize,
    pub fps: u32,
    tick: Instant,
}

impl Default for Anim {
    fn default() -> Self {
        Self { playing: true, frame: 0, fps: 8, tick: Instant::now() }
    }
}

impl Anim {
    /// Advance by one frame when a frame interval has passed (returns true when the frame changed).
    pub fn advance(&mut self, frames: usize) -> bool {
        if !self.playing || frames == 0 || self.tick.elapsed() < Duration::from_millis(1000 / self.fps.max(1) as u64) {
            return false;
        }
        self.tick = Instant::now();
        self.frame = (self.frame + 1) % frames;
        true
    }

    /// Step by `d` frames and pause (frame-by-frame inspection).
    pub fn step(&mut self, d: i32, frames: usize) {
        self.playing = false;
        if frames > 0 {
            self.frame = (self.frame as i32 + d).rem_euclid(frames as i32) as usize;
        }
    }
}

/// Frame `col` of row `row` of a grid sheet with cells `cw` x `ch`.
pub fn frame(sheet: &RgbaImage, cw: u32, ch: u32, row: u32, col: u32) -> RgbaImage {
    imageops::crop_imm(sheet, col * cw, row * ch, cw, ch).to_image()
}

/// The `n` frames of a sheet laid out in reading order on a `cols` x `rows` grid.
pub fn frames_of(sheet: &RgbaImage, cols: u32, rows: u32) -> Vec<RgbaImage> {
    let (cw, ch) = (sheet.width() / cols, sheet.height() / rows);
    (0..rows).flat_map(|r| (0..cols).map(move |c| (r, c))).map(|(r, c)| frame(sheet, cw, ch, r, c)).collect()
}

/// Animated GIF of `frames` at `fps`, each pixel scaled `k` times (nearest) so it is viewable outside the game.
pub fn gif(frames: &[RgbaImage], fps: u32, k: u32) -> anyhow::Result<Vec<u8>> {
    use image::codecs::gif::{GifEncoder, Repeat};
    use image::{Delay, Frame};
    let mut out = vec![];
    {
        let mut enc = GifEncoder::new_with_speed(&mut out, 10);
        enc.set_repeat(Repeat::Infinite)?;
        let delay = Delay::from_numer_denom_ms(1000, fps.max(1));
        for f in frames {
            let big = imageops::resize(f, f.width() * k, f.height() * k, imageops::FilterType::Nearest);
            enc.encode_frame(Frame::from_parts(big, 0, 0, delay))?;
        }
    }
    Ok(out)
}
