//! Pixel-art images for gpui: nearest-neighbour upscaling (gpui samples textures smoothly) and history scanning.
use gpui::RenderImage;
use image::{Frame, RgbaImage, imageops::FilterType};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// RGBA -> a gpui image (BGRA), upscaled `k` times with nearest neighbour so pixels stay square.
pub fn render(im: &RgbaImage, k: u32) -> Arc<RenderImage> {
    let mut im = if k > 1 { image::imageops::resize(im, im.width() * k, im.height() * k, FilterType::Nearest) } else { im.clone() };
    for p in im.as_chunks_mut::<4>().0 {
        p.swap(0, 2);
    }
    Arc::new(RenderImage::new(vec![Frame::new(im)]))
}

/// Integer factor that brings the longest side to at least `target` px.
pub fn factor(w: u32, h: u32, target: u32) -> u32 {
    (target / w.max(h).max(1)).max(1)
}

pub struct Item {
    pub png: PathBuf,
    pub json: Value,
    pub thumb: Arc<RenderImage>,
    pub sprite: RgbaImage, // the real pixels; the big view paints them directly
}

impl Item {
    pub fn load(png: PathBuf, json: Value) -> Option<Item> {
        let sprite = image::open(&png).ok()?.to_rgba8();
        let (w, h) = sprite.dimensions();
        Some(Item { thumb: render(&sprite, factor(w, h, 160)), png, json, sprite })
    }
}

/// The newest results: every <name>.png in the output folder that has its <name>.json next to it.
pub fn scan(dir: &Path, limit: usize) -> Vec<Item> {
    let Ok(rd) = std::fs::read_dir(dir) else { return vec![] };
    let mut files: Vec<(std::time::SystemTime, PathBuf)> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "png") && p.with_extension("json").is_file())
        .filter_map(|p| Some((p.metadata().ok()?.modified().ok()?, p)))
        .collect();
    files.sort_by_key(|f| std::cmp::Reverse(f.0));
    files
        .into_iter()
        .take(limit)
        .filter_map(|(_, png)| {
            let json = std::fs::read_to_string(png.with_extension("json")).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or(Value::Null);
            Item::load(png, json)
        })
        .collect()
}

/// A source image for the image modes: shown at up to ~256 px.
pub fn load_source(path: &Path) -> Option<(Arc<RenderImage>, u32, u32)> {
    let im = image::open(path).ok()?.to_rgba8();
    let (w, h) = im.dimensions();
    let shown = if w.max(h) > 256 {
        let s = 256.0 / w.max(h) as f32;
        image::imageops::resize(&im, ((w as f32 * s) as u32).max(1), ((h as f32 * s) as u32).max(1), FilterType::Triangle)
    } else {
        im
    };
    let k = factor(shown.width(), shown.height(), 256);
    Some((render(&shown, k), w, h))
}

/// RGB -> (hue 0..1, saturation 0..1) like Python's colorsys.
pub fn hue_sat(r: u8, g: u8, b: u8) -> (f32, f32) {
    let (r, g, b) = (r as f32 / 255., g as f32 / 255., b as f32 / 255.);
    let mx = r.max(g).max(b);
    let mn = r.min(g).min(b);
    if mx == mn {
        return (0., 0.);
    }
    let d = mx - mn;
    let s = d / mx;
    let h = if mx == r {
        ((g - b) / d).rem_euclid(6.)
    } else if mx == g {
        (b - r) / d + 2.
    } else {
        (r - g) / d + 4.
    };
    (h / 6., s)
}

/// Moves files to the Recycle Bin; missing ones are skipped.
pub fn recycle(paths: &[PathBuf]) -> std::io::Result<()> {
    let existing: Vec<&PathBuf> = paths.iter().filter(|p| p.exists()).collect();
    if existing.is_empty() {
        return Ok(());
    }
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        #[repr(C)]
        struct ShFileOp {
            hwnd: isize,
            func: u32,
            from: *const u16,
            to: *const u16,
            flags: u16,
            aborted: i32,
            mappings: *mut std::ffi::c_void,
            title: *const u16,
        }
        unsafe extern "system" {
            fn SHFileOperationW(op: *mut ShFileOp) -> i32;
        }
        const FO_DELETE: u32 = 3;
        const FOF_SILENT: u16 = 0x4;
        const FOF_NOCONFIRMATION: u16 = 0x10;
        const FOF_ALLOWUNDO: u16 = 0x40;
        const FOF_NOERRORUI: u16 = 0x400;
        let mut from: Vec<u16> = vec![];
        for p in &existing {
            let full = std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
            let raw = full.to_string_lossy();
            let text = match raw.strip_prefix(r"\\?\UNC\") {
                Some(rest) => format!(r"\\{rest}"),
                None => raw.strip_prefix(r"\\?\").unwrap_or(&raw).to_string(),
            };
            from.extend(std::ffi::OsStr::new(&text).encode_wide());
            from.push(0);
        }
        from.push(0);
        let mut op = ShFileOp {
            hwnd: 0,
            func: FO_DELETE,
            from: from.as_ptr(),
            to: std::ptr::null(),
            flags: FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT | FOF_NOERRORUI,
            aborted: 0,
            mappings: std::ptr::null_mut(),
            title: std::ptr::null(),
        };
        match unsafe { SHFileOperationW(&mut op) } {
            0 => Ok(()),
            code => Err(std::io::Error::other(format!("SHFileOperation 0x{code:x}"))),
        }
    }
    #[cfg(not(windows))]
    {
        existing.into_iter().try_for_each(std::fs::remove_file)
    }
}
