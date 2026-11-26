//! Split a pack / sheet into one PNG per cell. No UI: the Tools menu calls `split` and shows the result.
//!
//! How the grid is found, most trusted first:
//! 1. the Dotsmith .json next to the PNG ("grid" + "cell" of Pack / Muros, "cell" + "cell_h" of creature sheets);
//! 2. a training caption .txt next to it ("format pack 3x3", "format wall set 2x2");
//! 3. transparent (or flat magenta) gutters between the items;
//! 4. a square sheet of 32 px cells (64 px when it is not a multiple of 32).
//!
//! Upscaled images (dataset x16 / x8 renders) are first brought back to their real pixels.
use image::{Rgba, RgbaImage};
use serde_json::Value;
use std::path::{Path, PathBuf};

pub struct Split {
    pub dir: PathBuf,
    pub saved: usize,
    pub cols: u32,
    pub rows: u32,
    pub cell: (u32, u32),
    /// how the grid was found, for the status line
    pub how: &'static str,
}

/// One cut: (column, row, image).
type Cut = (u32, u32, RgbaImage);

/// Cut `src` and save the non-empty cells as `<out_root>/<stem>_split/<stem>_r1c2.png`.
pub fn split(src: &Path, out_root: &Path) -> Result<Split, String> {
    let raw = image::open(src).map_err(|e| e.to_string())?.to_rgba8();
    let meta: Value = std::fs::read_to_string(src.with_extension("json")).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or(Value::Null);
    let caption = std::fs::read_to_string(src.with_extension("txt")).unwrap_or_default();

    // real pixels first: Dotsmith outputs are native; anything else may be a nearest-neighbour upscale
    let s = if meta["mode"].is_string() { 1 } else { block_scale(&raw) };
    let im = clear_magenta(&if s > 1 { downscale(&raw, s) } else { raw });
    let (w, h) = im.dimensions();

    let (cuts, how): (Vec<Cut>, &'static str) = if let Some((cw, ch)) = meta_cell(&meta, w, h) {
        (grid(&im, cw, ch), crate::tr!("rejilla del .json", "grid from the .json"))
    } else if let Some(k) = caption_grid(&caption).filter(|k| w % k == 0 && h % k == 0) {
        (grid(&im, w / k, h / k), crate::tr!("rejilla del caption", "grid from the caption"))
    } else {
        match gutters(&im) {
            Some(c) if c.len() > 1 => (c, crate::tr!("huecos entre objetos", "gaps between objects")),
            _ => {
                let cell = if w == h && w % 32 == 0 && w / 32 >= 2 && w / 32 <= 6 {
                    32
                } else if w % 64 == 0 && h % 64 == 0 {
                    64
                } else {
                    return Err(crate::tr!(
                        format!("{w} × {h} px: no se encuentra la rejilla (sin .json, sin huecos entre objetos)."),
                        format!("{w} × {h} px: no grid found (no .json, no gaps between objects).")
                    ));
                };
                (grid(&im, cell, cell), crate::tr!("rejilla de 32/64 px", "32/64 px grid"))
            }
        }
    };
    let (cols, rows) = (cuts.iter().map(|c| c.0).max().unwrap_or(0) + 1, cuts.iter().map(|c| c.1).max().unwrap_or(0) + 1);
    let cell = cuts.first().map(|c| c.2.dimensions()).unwrap_or((0, 0));
    let keep: Vec<&Cut> = cuts.iter().filter(|c| c.2.pixels().any(|p| p.0[3] > 0)).collect();
    if keep.is_empty() {
        return Err(crate::tr!("Todas las celdas están vacías.", "Every cell is empty.").into());
    }

    let stem = src.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "pack".into());
    let dir = out_root.join(format!("{stem}_split"));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    for (c, r, piece) in &keep {
        piece.save(dir.join(format!("{stem}_r{}c{}.png", r + 1, c + 1))).map_err(|e| e.to_string())?;
    }
    Ok(Split { dir, saved: keep.len(), cols, rows, cell, how })
}

/// Cell size from a Dotsmith .json, when it matches the image.
fn meta_cell(meta: &Value, w: u32, h: u32) -> Option<(u32, u32)> {
    let cell = meta["cell"].as_u64()? as u32;
    let cell_h = meta["cell_h"].as_u64().map(|v| v as u32).unwrap_or(cell);
    (cell > 0 && cell_h > 0 && w.is_multiple_of(cell) && h.is_multiple_of(cell_h) && (w > cell || h > cell_h)).then_some((cell, cell_h))
}

/// "format pack 3x3" / "format wall set 2x2" -> 3 / 2 (square grids only).
fn caption_grid(caption: &str) -> Option<u32> {
    let rest = caption.split_once("format pack ").or_else(|| caption.split_once("format wall set "))?.1;
    let mut it = rest.chars();
    let (a, x, b) = (it.next()?.to_digit(10)?, it.next()?, it.next()?.to_digit(10)?);
    (x == 'x' && a == b && a > 1).then_some(a)
}

fn grid(im: &RgbaImage, cw: u32, ch: u32) -> Vec<Cut> {
    let (cols, rows) = (im.width() / cw, im.height() / ch);
    (0..rows).flat_map(|r| (0..cols).map(move |c| (c, r))).map(|(c, r)| (c, r, image::imageops::crop_imm(im, c * cw, r * ch, cw, ch).to_image())).collect()
}

/// Items separated by fully transparent rows and columns: rows of items first, then the items inside each row.
/// Each cut keeps a common cell size (the largest item box) so pieces stay aligned like a grid.
fn gutters(im: &RgbaImage) -> Option<Vec<Cut>> {
    let (w, h) = im.dimensions();
    let bands = |n: u32, full: &dyn Fn(u32) -> bool| -> Vec<(u32, u32)> {
        let mut out = vec![];
        let mut start = None;
        for i in 0..n {
            match (full(i), start) {
                (true, None) => start = Some(i),
                (false, Some(s)) => {
                    out.push((s, i));
                    start = None;
                }
                _ => {}
            }
        }
        if let Some(s) = start {
            out.push((s, n));
        }
        out
    };
    let rows = bands(h, &|y| (0..w).any(|x| im.get_pixel(x, y).0[3] > 0));
    if rows.is_empty() {
        return None;
    }
    let mut boxes = vec![];
    for (ri, &(y0, y1)) in rows.iter().enumerate() {
        let cols = bands(w, &|x| (y0..y1).any(|y| im.get_pixel(x, y).0[3] > 0));
        for (ci, &(x0, x1)) in cols.iter().enumerate() {
            boxes.push((ci as u32, ri as u32, x0, y0, x1, y1));
        }
    }
    // stray single pixels would become their own "items": a real grid has similar sizes everywhere
    let big = boxes.iter().map(|b| (b.4 - b.2) * (b.5 - b.3)).max()?;
    if boxes.iter().any(|b| (b.4 - b.2) * (b.5 - b.3) * 25 < big) {
        return None;
    }
    let (cw, ch) = (boxes.iter().map(|b| b.4 - b.2).max()?, boxes.iter().map(|b| b.5 - b.3).max()?);
    Some(
        boxes
            .into_iter()
            .map(|(c, r, x0, y0, x1, y1)| {
                // centred horizontally, resting on the bottom like the trained items
                let mut cell = RgbaImage::new(cw, ch);
                let item = image::imageops::crop_imm(im, x0, y0, x1 - x0, y1 - y0).to_image();
                image::imageops::overlay(&mut cell, &item, ((cw - (x1 - x0)) / 2) as i64, (ch - (y1 - y0)) as i64);
                (c, r, cell)
            })
            .collect(),
    )
}

/// The largest nearest-neighbour upscale factor of `im` (1 when it is native): every s x s block is one colour.
fn block_scale(im: &RgbaImage) -> u32 {
    let (w, h) = im.dimensions();
    for s in [32u32, 16, 8, 4, 2] {
        if w % s != 0 || h % s != 0 || w / s < 8 || h / s < 8 {
            continue;
        }
        let mut bad = 0u64;
        for y in 0..h {
            for x in 0..w {
                if im.get_pixel(x, y) != im.get_pixel(x - x % s, y - y % s) {
                    bad += 1;
                }
            }
        }
        if bad * 1000 <= (w * h) as u64 {
            return s; // <= 0.1 % off: compression noise at most
        }
    }
    1
}

fn downscale(im: &RgbaImage, s: u32) -> RgbaImage {
    RgbaImage::from_fn(im.width() / s, im.height() / s, |x, y| *im.get_pixel(x * s + s / 2, y * s + s / 2))
}

/// Flat #FF00FF backgrounds (training sheets, magenta LoRAs) become transparent; nothing else changes.
fn clear_magenta(im: &RgbaImage) -> RgbaImage {
    let mut out = im.clone();
    for p in out.pixels_mut() {
        if p.0[0] > 230 && p.0[1] < 40 && p.0[2] > 230 {
            *p = Rgba([0, 0, 0, 0]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sheet(k: u32, cell: u32, gap: bool) -> RgbaImage {
        let mut im = RgbaImage::new(k * cell, k * cell);
        for r in 0..k {
            for c in 0..k {
                let m = if gap { 3 } else { 0 };
                for y in m..cell - m {
                    for x in m..cell - m {
                        im.put_pixel(c * cell + x, r * cell + y, Rgba([(c * 60) as u8, (r * 60) as u8, 90, 255]));
                    }
                }
            }
        }
        im
    }

    #[test]
    fn caption_parsing() {
        assert_eq!(caption_grid("tibiastyle, format pack 3x3. A grid"), Some(3));
        assert_eq!(caption_grid("format wall set 2x2"), Some(2));
        assert_eq!(caption_grid("format wall set 3x2"), None);
        assert_eq!(caption_grid("a sword"), None);
    }

    #[test]
    fn gutters_find_a_3x3() {
        let c = gutters(&sheet(3, 32, true)).unwrap();
        assert_eq!(c.len(), 9);
        assert_eq!(c[0].2.dimensions(), (26, 26));
        assert_eq!((c[8].0, c[8].1), (2, 2));
    }

    #[test]
    fn upscale_is_undone() {
        let im = sheet(2, 32, true);
        let big = image::imageops::resize(&im, 64 * 16, 64 * 16, image::imageops::FilterType::Nearest);
        assert_eq!(block_scale(&big), 16);
        assert_eq!(downscale(&big, 16), im);
        assert_eq!(block_scale(&im), 1);
    }

    #[test]
    fn json_grid_wins() {
        let meta = serde_json::json!({"mode": "pack", "grid": 3, "cell": 32});
        assert_eq!(meta_cell(&meta, 96, 96), Some((32, 32)));
        assert_eq!(grid(&sheet(3, 32, false), 32, 32).len(), 9);
    }
}
