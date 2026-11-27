//! The studio's generation modes on top of the Rust engine (port of studio/sprite_backend.py + sprite_ops.py):
//! prompt building, canvases, post-processing and the only files a generation leaves: <stamp>_<mode>_<seed>.png + .json.
use anyhow::{Context, Result, bail};
use dotsmith_engine::catalog;
use dotsmith_engine::engine::{Engine, Event, Request};
use dotsmith_engine::model::ModelKind;
use dotsmith_engine::sprite;
use image::{GrayImage, Luma, RgbaImage, imageops};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use crate::tr;

/// What Sprite, Pack and the image modes wrap the description in. `{prompt}` is the description; `{style}` (the
/// LoRA's trigger + "pixel art"), `{layout}`, `{background}`, `{width}` and `{height}` are filled per mode.
pub const TEMPLATE: &str = "{style}, modern detailed, {layout} on {background}. {prompt} --w {width} --h {height}";

/// A template without `{prompt}` gets it at the end.
fn templated(r: &Value, style: &str, layout: &str, background: &str, prompt: &str, w: usize, h: usize) -> String {
    let t = r["template"].as_str().map(str::trim).filter(|t| !t.is_empty()).unwrap_or(TEMPLATE);
    let t = if t.contains("{prompt}") { t.to_string() } else { format!("{t} {{prompt}}") };
    t.replace("{style}", style)
        .replace("{layout}", layout)
        .replace("{background}", background)
        .replace("{width}", &w.to_string())
        .replace("{height}", &h.to_string())
        .replace("{prompt}", prompt)
}

pub const NEG: &str = "smooth gradient, glossy, shiny, airbrushed, 3d render, soft shading, plastic, blurry";
/// Base model labels the UI offers, from the engine's family registry (never a hardcoded list).
pub fn bases() -> Vec<String> {
    ModelKind::all().into_iter().map(|k| k.label()).collect()
}

const CAMERA_45: &str = "Camera: top-down RPG view tilted about 45 degrees downward, axis-aligned and facing straight forward, not isometric. The top of the object is clearly visible and its front is foreshortened, shorter than when seen head-on. Brightest light on top surfaces, darker along the bottom edge.";

pub fn cameras() -> Vec<(&'static str, String)> {
    vec![
        ("Normal", String::new()),
        ("Cenital 45° (propia)", CAMERA_45.to_string()),
        (
            "Cenital 45° · objeto recto",
            format!(
                "{CAMERA_45} Same camera angle as a treasure chest in a top-down RPG, no perspective taper: the top surface is a lit band above the front face, both exactly the same width with straight vertical sides, no overhang."
            ),
        ),
    ]
}

pub fn with_camera(prompt: &str, camera: &str) -> String {
    let extra = cameras().into_iter().find(|c| c.0 == camera).map(|c| c.1).unwrap_or_default();
    format!("{} {}", prompt.trim(), extra).trim().to_string()
}

/// Camera sentence for a gizmo angle (pitch 0 = eye level .. 90 = top-down; yaw + shows the right side).
pub fn angle_text(pitch: f32, yaw: f32) -> String {
    let yaw = (yaw + 180.0).rem_euclid(360.0) - 180.0;
    if (40.0..=50.0).contains(&pitch) && yaw.abs() <= 7.0 {
        return CAMERA_45.to_string();
    }
    let p = pitch.round();
    let tilt = if pitch < 12.0 {
        "Camera at eye level, the object seen straight from the front, no top surface visible.".to_string()
    } else if pitch < 35.0 {
        format!("Camera slightly above, tilted about {p} degrees downward; a thin strip of the top surface is visible.")
    } else if pitch < 70.0 {
        format!("Camera: top-down RPG view tilted about {p} degrees downward. The top of the object is clearly visible and its front is foreshortened.")
    } else {
        format!("Camera almost straight top-down ({p} degrees): mostly the top surface is visible, the front is a thin band.")
    };
    let a = yaw.abs();
    let side = if yaw > 0.0 { "right" } else { "left" };
    let ar = a.round();
    let turn = if a <= 7.0 {
        "Axis-aligned, facing straight forward, not isometric.".to_string()
    } else if a < 30.0 {
        format!("Turned slightly, about {ar} degrees, so a narrow part of its {side} side shows.")
    } else if a < 60.0 {
        format!("Three-quarter view, rotated about {ar} degrees: its front and its {side} side are both visible.")
    } else if a < 120.0 {
        format!("Side view, rotated about {ar} degrees: mostly its {side} side is visible.")
    } else {
        "Seen from behind: its back faces the camera.".to_string()
    };
    format!("{tilt} {turn} Brightest light on top surfaces, darker along the bottom edge.")
}

// ------------------------------------------------------------------ trained runs on disk (engine catalog)
pub use dotsmith_engine::catalog::version_of;

// ------------------------------------------------------------------ one generation
pub struct Ctx<'a> {
    pub engine: &'a mut Engine,
    pub lora: PathBuf,
    pub lora_scale: f64,
    pub base: String,
    pub out_dir: PathBuf,
    pub meta: catalog::VersionMeta,
    pub cancel: &'a AtomicBool,
    pub emit: &'a mut dyn FnMut(Value),
    pub preview: &'a mut dyn FnMut(RgbaImage),
    pub result: &'a mut dyn FnMut(PathBuf, Value),
}

fn model_fields(c: &Ctx) -> Value {
    let version = (!c.lora.as_os_str().is_empty()).then(|| version_of(&c.lora).file_name().map(|n| n.to_string_lossy().to_string())).flatten();
    json!({"lora": c.lora.to_string_lossy(), "lora_scale": c.lora_scale, "base": c.base, "version": version, "scale": c.meta.scale, "engine": "rust",
           "app": format!("Dotsmith {}", env!("CARGO_PKG_VERSION"))})
}

fn save(c: &mut Ctx, mode: &str, seed: u64, img: &RgbaImage, mut settings: Value) -> Result<PathBuf> {
    let out = c.out_dir.clone();
    std::fs::create_dir_all(&out)?;
    let stamp = chrono_stamp();
    let mut png = out.join(format!("{stamp}_{mode}_{seed}.png"));
    let mut n = 2;
    while png.exists() {
        png = out.join(format!("{stamp}_{mode}_{seed}_{n}.png"));
        n += 1;
    }
    img.save(&png)?;
    if let (Some(s), Some(m)) = (settings.as_object_mut(), model_fields(c).as_object()) {
        for (k, v) in m {
            s.insert(k.clone(), v.clone());
        }
    }
    std::fs::write(png.with_extension("json"), serde_json::to_string_pretty(&settings)?)?;
    (c.result)(png.clone(), settings);
    Ok(png)
}

pub fn chrono_stamp() -> String {
    // local time without a date crate: Windows' GetLocalTime via the system clock offset is overkill; use UTC-free
    // formatting from SystemTime + the TZ offset the OS reports through `time` is not available -> std only.
    let now = std::time::SystemTime::now();
    let secs = now.duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) as i64 + local_offset_secs();
    let days = secs.div_euclid(86400);
    let rem = secs.rem_euclid(86400);
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}_{:02}{:02}{:02}", rem / 3600, rem % 3600 / 60, rem % 60)
}

fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(windows)]
fn local_offset_secs() -> i64 {
    #[repr(C)]
    struct Tzi {
        bias: i32,
        _rest: [u8; 168],
    }
    unsafe extern "system" {
        fn GetTimeZoneInformation(tzi: *mut Tzi) -> u32;
    }
    let mut t = Tzi { bias: 0, _rest: [0; 168] };
    let r = unsafe { GetTimeZoneInformation(&mut t) };
    // daylight bias lives in the rest of the struct; standard + daylight handled by the return code
    let daylight = if r == 2 { i32::from_le_bytes(t._rest[164..168].try_into().unwrap()) } else { 0 };
    -((t.bias + daylight) as i64) * 60
}

#[cfg(not(windows))]
fn local_offset_secs() -> i64 {
    0
}

/// The request as the loaded family and the LoRA's version need it: a version trained on a flat background asks for
/// that background instead of transparency; families marked `fixed_guidance` in the catalog keep their own guidance,
/// and those without `negative_prompt` get none.
fn adapt(c: &Ctx, req: &Request) -> Request {
    let mut r = req.clone();
    if c.meta.background == Some([255, 0, 255]) {
        r.prompt = r.prompt.replace("a transparent background", "a flat magenta background").replace("transparent background", "flat magenta background");
    }
    let kind = c.engine.kind();
    let spec = kind.family.spec();
    if !spec.negative_prompt {
        r.negative = String::new();
    }
    if spec.fixed_guidance {
        r.guidance = c.engine.defaults().guidance;
    }
    if r.steps == 0 {
        r.steps = c.engine.defaults().steps;
    }
    r
}

/// Families that read the description as written: no trigger, layout words or size flags, and no clean-up after (they
/// draw real pixel art at the final size).
fn plain(c: &Ctx) -> bool {
    c.engine.kind().family.spec().prompt == "plain"
}

/// The negative the engine really got (adapt() may drop it), for the settings json.
fn neg_used(c: &Ctx) -> &'static str {
    if c.engine.kind().family.spec().negative_prompt { NEG } else { "" }
}

/// depixel, then (magenta-keyed versions) the leftover background removed on the native sprite (sprite::drop_magenta):
/// block medians next to the background can land on a blended magenta.
/// When the #FF00FF key at full size missed the backdrop (third-party LoRAs such as frost paint "magenta" as a dull
/// pink, (182, 81, 141) and similar, varying per image), it is still there on the native sprite: measured on the border
/// and removed (sprite::remove_backdrop). Only a magenta-hued, mostly opaque border counts, so a backdrop the key already
/// removed, or an item touching the edge, is left alone.
fn depix(key: bool, im: &RgbaImage, colors: usize, scale: usize) -> RgbaImage {
    let d = sprite::depixel(im, colors, scale);
    if !key {
        return d;
    }
    let d = sprite::remove_magenta_backdrop(&d, 60).map(|r| r.0).unwrap_or(d);
    sprite::drop_magenta(&d)
}

fn run_engine(c: &mut Ctx, req: &Request, on_sample: impl FnMut(&mut Ctx, usize, RgbaImage) -> Result<()>) -> Result<bool> {
    run_engine_with(c, req, true, on_sample)
}

/// `key`: flood-key the version's flat background on the full-size sample. Off for modes whose silhouette comes from
/// the source image (img2sprite): there the key only ever removes parts of the object (pinks, purples, reddish shade).
fn run_engine_with(c: &mut Ctx, req: &Request, key: bool, mut on_sample: impl FnMut(&mut Ctx, usize, RgbaImage) -> Result<()>) -> Result<bool> {
    let req = &adapt(c, req);
    let key = if key { c.meta.background } else { None };
    let mut samples = vec![];
    {
        let emit = &mut *c.emit;
        let preview = &mut *c.preview;
        let mut last_phase = "";
        c.engine.generate(req, c.cancel, &mut |e| match e {
            Event::Phase(p) => {
                if p != last_phase {
                    emit(json!({"ev": "phase", "phase": "busy", "detail": p}));
                    last_phase = p;
                }
            }
            Event::Step { index, count, step, total } => emit(json!({"ev": "progress", "index": index, "count": count, "step": step, "total": total})),
            Event::Preview { image, .. } => preview(image),
            Event::Sample { index, image } => samples.push((index, image)),
        })?;
    }
    let stopped = c.cancel.load(std::sync::atomic::Ordering::Relaxed);
    for (i, img) in samples {
        let img = match key {
            Some(k) => sprite::key_color(&img, k, 90, 60),
            None => img,
        };
        on_sample(c, i, img)?;
    }
    Ok(stopped)
}

pub fn generate(c: &mut Ctx, r: &Value) -> Result<String> {
    let mode = r["mode"].as_str().unwrap_or("single").to_string();
    match mode.as_str() {
        "single" if r["pixel"].as_bool() == Some(false) => picture_mode(c, r),
        "single" | "pack" => text_mode(c, r, &mode),
        "creature" => creature_mode(c, r),
        "walls" => walls_mode(c, r),
        "photo" => image_to_sprite(c, r),
        _ => image_mode(c, r, &mode),
    }
}

fn pick_seed(v: &Value) -> u64 {
    match v.as_i64() {
        Some(s) if s >= 0 => s as u64,
        _ => {
            let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64;
            (t ^ (t >> 17)).wrapping_mul(0x9E3779B97F4A7C15) % (2u64.pow(31) - 16)
        }
    }
}

fn text_mode(c: &mut Ctx, r: &Value, mode: &str) -> Result<String> {
    let single = mode == "single";
    let desc = r["prompt"].as_str().unwrap_or("").trim().to_string();
    if desc.is_empty() {
        bail!(if single {
            tr!("Describe el objeto que quieres generar.", "Describe the object you want to generate.")
        } else {
            tr!("Describe los objetos del pack.", "Describe the objects of the pack.")
        });
    }
    let camera = r["camera"].as_str().unwrap_or("Normal").to_string();
    let opaque_ok = c.engine.kind().family.spec().opaque;
    let (opaque, center, mirror) =
        (opaque_ok && r["opaque"].as_bool().unwrap_or(false), r["center"].as_bool().unwrap_or(false), r["mirror"].as_bool().unwrap_or(false));
    let s = c.meta.scale as usize;
    let key = c.meta.background == Some([255, 0, 255]);
    let (size, k, height) = if single {
        let w = r["width"].as_u64().unwrap_or(32) as usize;
        (w, 1, r["height"].as_u64().map(|h| h as usize).unwrap_or(w))
    } else {
        (r["cell"].as_u64().unwrap_or(32) as usize, r["grid"].as_u64().unwrap_or(3) as usize, 0)
    };
    // x16 up to 2304 px (3x3·48, 2x2·64 fit with the tiled VAE decode); bigger falls back to the 1536 cap
    let mut sc = if s * size * k <= 2304 { s } else { (s * size * k).min(1536) / (size * k) };
    let side = sc * size * k;
    let (mut wpx, mut hpx) = (side, side);
    if single && height != size {
        wpx = s * size;
        hpx = s * height;
        sc = s;
    }
    let seed = pick_seed(&r["seed"]);
    let lay = if single {
        "format single. One fantasy inventory item".to_string()
    } else {
        format!("format pack {k}x{k}. A {k}x{k} grid of {} fantasy items", if k == 2 { "four" } else { "nine" })
    };
    let bg = if opaque { "a full painted opaque background" } else { "a transparent background" };
    let plain = plain(c);
    let text = if plain { desc.clone() } else { templated(r, &c.meta.style(), &lay, bg, &with_camera(&desc, &camera), wpx, hpx) };
    let steps = r["steps"].as_u64().unwrap_or(25) as usize;
    let req = Request {
        prompt: text.clone(),
        negative: NEG.into(),
        width: wpx,
        height: hpx,
        steps,
        guidance: 3.0,
        seed,
        count: 1,
        init: None,
        mask: None,
        strength: 1.0,
        preview_every: 1,
        opaque,
    };
    let req = adapt(c, &req); // what the loaded family really runs: that is what settings.json records
    let started = Instant::now();
    let mut saved = 0;
    let stopped = run_engine(c, &req, |c, _, raw| {
        let mut raw = raw;
        if opaque {
            for p in raw.pixels_mut() {
                p.0[3] = 255;
            }
        }
        let cleaned = if plain { raw.clone() } else { depix(key, &sprite::desaturate(&raw, 0.85), sprite::COLORS, sc) };
        let pieces = sprite::pieces_of(&cleaned, size as u32);
        if pieces.len() != k * k {
            bail!(tr!(
                format!("El modelo produjo {} celdas ocupadas en vez de {}. Prueba otra semilla.", pieces.len(), k * k),
                format!("The model produced {} filled cells instead of {}. Try another seed.", pieces.len(), k * k)
            ));
        }
        let (tw, th) = if hpx != wpx { (wpx / sc, hpx / sc) } else { (size, size) };
        let tiles: Vec<RgbaImage> = pieces
            .iter()
            .map(|p| {
                let t = sprite::fit_wh(&if plain { p.clone() } else { sprite::despeckle(p) }, tw as u32, th as u32, center);
                if mirror { sprite::mirror_right(&t) } else { t }
            })
            .collect();
        let img = if single {
            tiles[0].clone()
        } else {
            let mut sheet = RgbaImage::new((k * size) as u32, (k * size) as u32);
            for (i, t) in tiles.iter().enumerate() {
                imageops::overlay(&mut sheet, t, ((i % k) * size) as i64, ((i / k) * size) as i64);
            }
            sheet
        };
        let mut st = json!({"mode": mode, "prompt": desc, "camera": camera, "opaque": opaque, "center": center, "mirror": mirror,
            "seed": seed, "steps": req.steps, "guidance": req.guidance, "negative": neg_used(c), "full_prompt": req.prompt, "width_px": wpx, "height_px": hpx,
            "seconds": (started.elapsed().as_secs_f64() * 10.0).round() / 10.0});
        if single {
            st["width"] = tw.into();
            st["height"] = th.into();
        } else {
            st["grid"] = k.into();
            st["cell"] = size.into();
        }
        save(c, mode, seed, &img, st)?;
        saved += 1;
        Ok(())
    })?;
    let secs = started.elapsed().as_secs_f64();
    Ok(format!("{}{} {} · {:.1} s", if stopped { tr!("Detenido · ", "Stopped · ") } else { "" }, saved, if single { "sprite" } else { "pack" }, secs))
}

fn picture_mode(c: &mut Ctx, r: &Value) -> Result<String> {
    let desc = r["prompt"].as_str().unwrap_or("").trim().to_string();
    if desc.is_empty() {
        bail!(tr!("Describe la imagen que quieres generar.", "Describe the image you want to generate."));
    }
    let (w, h) = (r["width"].as_u64().unwrap_or(1024) as usize, r["height"].as_u64().unwrap_or(1024) as usize);
    let seed = pick_seed(&r["seed"]);
    let req = adapt(
        c,
        &Request {
            prompt: desc.clone(),
            negative: String::new(),
            width: w,
            height: h,
            steps: r["steps"].as_u64().unwrap_or(25) as usize,
            guidance: 3.0,
            seed,
            count: 1,
            init: None,
            mask: None,
            strength: 1.0,
            preview_every: 1,
            opaque: true,
        },
    );
    let started = Instant::now();
    let mut saved = 0;
    let stopped = run_engine_with(c, &req, false, |c, _, mut img| {
        for p in img.pixels_mut() {
            p.0[3] = 255;
        }
        let st = json!({"mode": "single", "pixel": false, "prompt": desc, "seed": seed, "steps": req.steps, "guidance": req.guidance,
            "full_prompt": req.prompt, "width": w, "height": h, "width_px": w, "height_px": h,
            "seconds": (started.elapsed().as_secs_f64() * 10.0).round() / 10.0});
        save(c, "single", seed, &img, st)?;
        saved += 1;
        Ok(())
    })?;
    Ok(format!(
        "{}{saved} {} · {w} × {h} · {:.1} s",
        if stopped { tr!("Detenido · ", "Stopped · ") } else { "" },
        tr!("imagen", "image"),
        started.elapsed().as_secs_f64()
    ))
}

/// "Imagen": any picture to an `out_size` x `out_size` sprite, the picture fitted inside without distortion (resting on
/// the bottom, or centred when opaque: the model paints the rest of the background). Background like the Sprite tab:
/// `opaque` keeps the picture whole; otherwise a flat background (the border mostly one colour) is keyed out and the
/// picture cropped to what is left, a transparent PNG keeps its alpha, and any other background stays. PixelOE
/// (outline expansion, contrast-based downscale, OKLab palette) makes the sprite; with `restyle` the loaded LoRA then
/// redraws it (img2sprite). The silhouette is always the source's, so nothing of the picture is cut away.
fn image_to_sprite(c: &mut Ctx, r: &Value) -> Result<String> {
    let source = PathBuf::from(r["source"].as_str().unwrap_or(""));
    if !source.is_file() {
        bail!(tr!("Carga una imagen de origen.", "Load a source image first."));
    }
    let started = Instant::now();
    let restyle = r["restyle"].as_bool().unwrap_or(false);
    if restyle && r["prompt"].as_str().unwrap_or("").trim().is_empty() {
        // measured: with a generic prompt the LoRA invents a different subject from the colour blobs
        bail!(tr!(
            "Describe lo que hay en la imagen (p. ej. «an orc warrior in iron armour»): sin descripción el modelo se inventa otra cosa.",
            "Describe what is in the image (e.g. «an orc warrior in iron armour»): without a description the model invents something else."
        ));
    }
    let mut src = image::open(&source)?.to_rgba8();
    if src.width().max(src.height()) > 2048 {
        let f = 2048.0 / src.width().max(src.height()) as f32;
        src = imageops::resize(&src, ((src.width() as f32 * f) as u32).max(1), ((src.height() as f32 * f) as u32).max(1), imageops::FilterType::Lanczos3);
    }
    let opaque = r["opaque"].as_bool().unwrap_or(false);
    let had_alpha = src.pixels().any(|p| p.0[3] < 128);
    // a flat backdrop, even a noisy one (a resampled screenshot's magenta), flood-keyed from the border; what is left
    // of a frame or the backdrop's rim along the edge goes too
    let matted = match sprite::flat_backdrop(&src, 120, 0.5).filter(|_| !opaque) {
        Some(key) => {
            src = sprite::drop_edge_islands(&sprite::defringe(&sprite::key_color(&src, key, 120, 0), key, 180, 2), 0.05);
            true
        }
        None => false,
    };
    let Some((x0, y0, x1, y1)) = sprite::bbox(&src) else {
        bail!(tr!("La imagen está vacía: todo es fondo.", "The image is empty: it is all background."));
    };
    src = imageops::crop_imm(&src, x0, y0, x1 - x0, y1 - y0).to_image();
    let (sw, sh) = src.dimensions();
    let long = r["out_size"].as_u64().unwrap_or(64).clamp(16, 128) as u32;
    // the LoRA canvas: its x scale (capped at 1536 px), sides a multiple of the model's alignment
    let (sc, step) = if restyle {
        let sc = c.meta.scale.min(1536 / long).max(1);
        let a = c.engine.align() as u32;
        (sc, a / gcd(a, sc))
    } else {
        (1, 1)
    };
    let fit = |v: u32| v.max(1).div_ceil(step) * step;
    let (ow, oh) = if sw >= sh {
        (fit(long), fit((long as f64 * sh as f64 / sw as f64).round() as u32))
    } else {
        (fit((long as f64 * sw as f64 / sh as f64).round() as u32), fit(long))
    };
    // PixelOE needs at least 2 source pixels per sprite pixel: a small source (an existing sprite) is enlarged first
    let need = 2 * ow.max(oh);
    if sw.max(sh) < need {
        let k = need.div_ceil(sw.max(sh));
        src = imageops::resize(&src, sw * k, sh * k, imageops::FilterType::Nearest);
    }
    let patch = (src.width() / ow).min(src.height() / oh).clamp(2, 16);
    let thickness = if patch >= 6 { 2 } else { 1 };
    let colors = r["colors"].as_u64().unwrap_or(32) as usize;
    // transparent areas take the nearest object colour, so the background never bleeds into the edges
    let rgb = if src.pixels().any(|p| p.0[3] < 128) { sprite::bleed_into_background(&src) } else { image::DynamicImage::ImageRgba8(src.clone()).to_rgb8() };
    let px = dotsmith_engine::pixeloe::pixelize(&rgb, ow, oh, patch, thickness, (colors > 0).then_some(colors));
    let alpha = imageops::resize(&src, ow, oh, imageops::FilterType::Triangle);
    // opaque from a quarter coverage up: at 50 % thin parts (a blade tip, a rope, antennae narrower than half an output
    // pixel) vanished; the colour there is already the object's (bleed_into_background), so the edge does not go dirty
    let img = RgbaImage::from_fn(ow, oh, |x, y| {
        let p = px.get_pixel(x, y).0;
        image::Rgba([p[0], p[1], p[2], if alpha.get_pixel(x, y).0[3] >= 64 { 255 } else { 0 }])
    });
    let side = fit(long);
    let mut cell = RgbaImage::new(side, side);
    imageops::overlay(&mut cell, &img, ((side - ow) / 2) as i64, if opaque { (side - oh) / 2 } else { side - oh } as i64);
    let img = cell;
    let info = Conversion { source: &source, patch, thickness, colors, matted, opaque, started };
    let note =
        if !opaque && !matted && !had_alpha { tr!(" · el fondo no es liso: se ha mantenido", " · the background is not flat: it was kept") } else { "" };
    if restyle {
        return restyle_sprite(c, r, &info, img, sc).map(|m| format!("{m}{note}"));
    }
    let st = json!({"mode": "photo", "direct": true, "prompt": "", "source": info.source_text(), "out_size": side, "width": side, "height": side,
        "patch": patch, "thickness": thickness, "colors": colors, "matte": matted, "opaque": opaque,
        "method": "pixeloe-contrast + OKLab k-means", "seed": 0, "seconds": (started.elapsed().as_secs_f64() * 10.0).round() / 10.0});
    save(c, "photo", 0, &img, st)?;
    Ok(tr!(
        format!("Convertida sin IA · {side} × {side} · {:.1} s{note}", started.elapsed().as_secs_f64()),
        format!("Converted without AI · {side} × {side} · {:.1} s{note}", started.elapsed().as_secs_f64())
    ))
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a.max(1) } else { gcd(b, a % b) }
}

/// What the PixelOE step did, recorded in every result's settings.
struct Conversion<'a> {
    source: &'a Path,
    patch: u32,
    thickness: u32,
    colors: usize,
    matted: bool,
    opaque: bool,
    started: Instant,
}

impl Conversion<'_> {
    fn source_text(&self) -> String {
        self.source.canonicalize().unwrap_or(self.source.to_path_buf()).to_string_lossy().replace("\\\\?\\", "")
    }
}

/// img2sprite strength from which the prompt may draw outside the source's silhouette.
pub const MIX_STRENGTH: f32 = 0.6;

/// img2sprite: the PixelOE sprite redrawn by the loaded LoRA (img2img at its x scale), so any picture comes out in
/// the trained style at the chosen size. Measured on a painted portrait: PixelOE alone keeps the drawing, Krea v1 at
/// strength ~0.45 adds the LoRA's block shading and clean outlines without changing the subject.
fn restyle_sprite(c: &mut Ctx, r: &Value, info: &Conversion, sprite_px: RgbaImage, s: u32) -> Result<String> {
    let (ow, oh) = sprite_px.dimensions();
    let started = info.started;
    let bg = c.meta.background;
    let has_alpha = sprite_px.pixels().any(|p| p.0[3] == 0);
    // flatten the transparent parts onto the version's background (opaque: a neutral grey the model paints a
    // background over; on the version's flat colour it kept that colour), then upscale to the LoRA's pixel scale
    let fill = if info.opaque { Some([110, 110, 110]) } else { bg };
    let flat = RgbaImage::from_fn(ow, oh, |x, y| {
        let p = sprite_px.get_pixel(x, y).0;
        match (p[3], fill) {
            (0, Some(b)) => image::Rgba([b[0], b[1], b[2], 255]),
            (0, None) => image::Rgba([0, 0, 0, 0]),
            _ => image::Rgba(p),
        }
    });
    let canvas = imageops::resize(&flat, ow * s, oh * s, imageops::FilterType::Nearest);
    let desc = r["prompt"].as_str().unwrap_or("").trim().to_string();
    let bgtext = if info.opaque {
        " on a full painted opaque background"
    } else if has_alpha {
        " on a transparent background"
    } else {
        ""
    };
    let text = format!("{}, modern detailed, format single{bgtext}. {desc} --w {} --h {}", c.meta.style(), ow * s, oh * s);
    let seed = pick_seed(&r["seed"]);
    let count = r["variants"].as_u64().unwrap_or(1).clamp(1, 9) as usize;
    let strength = r["strength"].as_f64().unwrap_or(0.45) as f32;
    let req = Request {
        prompt: text,
        negative: NEG.into(),
        width: (ow * s) as usize,
        height: (oh * s) as usize,
        steps: r["steps"].as_u64().unwrap_or(25) as usize,
        guidance: r["guidance"].as_f64().unwrap_or(3.0) as f32,
        seed,
        count,
        init: Some(canvas),
        mask: None,
        strength,
        preview_every: 1,
        opaque: false,
    };
    let req = adapt(c, &req);
    let src_s = info.source_text();
    let mut saved = 0;
    // the silhouette is the source's, never the model's: no background keying (it would eat pink, purple or reddish
    // parts of the object). Where the model painted its flat background inside the object, the PixelOE colour is used.
    let stopped = run_engine_with(c, &req, false, |c, i, raw| {
        let mut t = sprite::depixel(&sprite::desaturate(&raw, 0.85), sprite::COLORS, s as usize);
        if t.dimensions() != (ow, oh) {
            t = imageops::resize(&t, ow, oh, imageops::FilterType::Nearest);
        }
        let near_bg = |p: &image::Rgba<u8>| bg.is_some_and(|b| (0..3).map(|k| (p.0[k] as i32 - b[k] as i32).abs()).sum::<i32>() < 60);
        // from MIX_STRENGTH up the prompt may add to the picture: what the model clearly drew outside the source's
        // silhouette (its own background keyed out) is kept too; below it the silhouette is exactly the source's
        let drawn = (strength >= MIX_STRENGTH).then(|| {
            let d = match bg {
                Some(k) => depix(true, &sprite::key_color(&sprite::desaturate(&raw, 0.85), k, 90, 60), sprite::COLORS, s as usize),
                None => sprite::depixel(&sprite::desaturate(&raw, 0.85), sprite::COLORS, s as usize),
            };
            sprite::despeckle(&imageops::resize(&d, ow, oh, imageops::FilterType::Nearest))
        });
        for (x, y, p) in t.enumerate_pixels_mut() {
            let q = sprite_px.get_pixel(x, y);
            if q.0[3] == 0 && info.opaque {
                p.0[3] = 255; // the background the model painted around a transparent source
            } else if q.0[3] == 0 {
                p.0[3] = if drawn.as_ref().is_some_and(|d| d.get_pixel(x, y).0[3] >= 128) { 255 } else { 0 };
            } else {
                if near_bg(p) && !near_bg(q) {
                    *p = *q; // background leaked into the object: take the source sprite's colour
                }
                p.0[3] = 255;
            }
        }
        let st = json!({"mode": "photo", "direct": true, "restyle": true, "prompt": r["prompt"], "source": src_s, "out_size": ow.max(oh),
            "width": ow, "height": oh, "patch": info.patch, "thickness": info.thickness, "colors": info.colors, "matte": info.matted, "opaque": info.opaque, "canvas_scale": s,
            "strength": strength, "seed": seed + i as u64, "steps": req.steps, "guidance": req.guidance, "variant": i + 1, "variants": count,
            "full_prompt": req.prompt, "method": "pixeloe-contrast + OKLab k-means, then LoRA img2img",
            "seconds": (started.elapsed().as_secs_f64() * 10.0).round() / 10.0});
        save(c, "photo", seed + i as u64, &t, st)?;
        saved += 1;
        Ok(())
    })?;
    Ok(format!(
        "{}img2sprite · {saved}/{count} · {ow} × {oh} · {:.1} s",
        if stopped { tr!("Detenido · ", "Stopped · ") } else { "" },
        started.elapsed().as_secs_f64()
    ))
}

/// Wall-set pieces in sheet order (row-major 2x2: pile, north / east, corner), the v16 / krea2 wall captions' layout.
const WALL_PIECES: [(&str, &str); 4] =
    [("wall pile", "short post"), ("north wall", "long straight section"), ("east wall", "long straight section"), ("wall corner", "L-shaped join")];
pub const WALL_CELL: usize = 64;

/// The wall-set caption the wall LoRAs are trained on. `desc` is either a full
/// per-piece body (contains "wall pile:"), used as is, or "<material>, <details>…": the material names every piece and
/// the details repeat on each, like the training captions.
pub fn wall_prompt(style: &str, desc: &str) -> String {
    let head = format!(
        "{style}, format wall set 2x2, transparent background, matching wall set, tibia oblique view, top row: wall pile, north wall; bottom row: east wall, wall corner;"
    );
    let desc = desc.trim().trim_end_matches('.');
    if desc.contains("wall pile:") {
        let tail = if desc.contains("retro rpg style") { "" } else { ", pixel art, retro rpg style" };
        return format!("{head} {desc}{tail}");
    }
    let (name, details) = match desc.split_once(',') {
        Some((n, d)) => (n.trim(), d.trim()),
        None => (desc, ""),
    };
    let name = name.trim_end_matches(" walls").trim_end_matches(" wall");
    let d = if details.is_empty() { String::new() } else { format!(", {details}") };
    let pieces: Vec<String> = WALL_PIECES.iter().map(|(p, shape)| format!("{p}: {name} {p}, {shape}{d}")).collect();
    let unified = if details.is_empty() { "a matching look".to_string() } else { details.to_string() };
    format!("{head} {}; This fantasy wall set features {name} pieces, unified by {unified}, pixel art, retro rpg style", pieces.join("; "))
}

/// A wall piece moved so its opaque box touches the bottom-right of its cell: where 97 % of the training pieces sit
/// (oblique tile sprites are anchored there) and what the room assembly needs. Pack re-centres pieces; this undoes it.
pub fn anchor_bottom_right(p: &RgbaImage) -> RgbaImage {
    let (w, h) = p.dimensions();
    let (mut mx, mut my) = (None, None::<u32>);
    for (x, y, c) in p.enumerate_pixels() {
        if c.0[3] >= 128 {
            mx = Some(mx.map_or(x, |m: u32| m.max(x)));
            my = Some(my.map_or(y, |m| m.max(y)));
        }
    }
    let (Some(mx), Some(my)) = (mx, my) else { return p.clone() };
    let (dx, dy) = (w - 1 - mx, h - 1 - my);
    let mut out = RgbaImage::new(w, h);
    for (x, y, c) in p.enumerate_pixels() {
        if c.0[3] >= 128 && x + dx < w && y + dy < h {
            out.put_pixel(x + dx, y + dy, *c);
        }
    }
    out
}

/// A closed square room: 64 px pieces drawn with
/// their bottom-right 32 px on the tile, in row order; pile = NW post, north / east = straight runs, corner = SE join.
pub fn wall_room(pieces: &[RgbaImage], n: usize) -> RgbaImage {
    const T: usize = 32;
    let side = ((n + 1) * T + T) as u32;
    let mut room = RgbaImage::new(side, side);
    for y in 0..=n {
        for x in 0..=n {
            let k = if (x, y) == (0, 0) {
                0
            } else if (x, y) == (n, n) {
                3
            } else if y == 0 || (y == n && x > 0) {
                1
            } else if x == 0 || x == n {
                2
            } else {
                continue;
            };
            let p = &pieces[k];
            for (px, py, c) in p.enumerate_pixels() {
                if c.0[3] >= 128 {
                    room.put_pixel((x * T) as u32 + px, (y * T) as u32 + py, *c);
                }
            }
        }
    }
    room
}

/// Wall sets: one 2x2 sheet of 64 px pieces with the trained wall-set caption, cut by grid cell (pieces keep their
/// place in the cell: the room needs it). Optional: the 3x3 closed room assembled from those pieces, saved next to it.
fn walls_mode(c: &mut Ctx, r: &Value) -> Result<String> {
    let desc = r["prompt"].as_str().unwrap_or("").trim().to_string();
    if desc.is_empty() {
        bail!(tr!("Describe el material del muro.", "Describe the wall material."));
    }
    let with_room = r["room"].as_bool().unwrap_or(true);
    let s = c.meta.scale as usize;
    let key = c.meta.background == Some([255, 0, 255]);
    let sc = if s * 2 * WALL_CELL <= 2304 { s } else { 2304 / (2 * WALL_CELL) };
    let side = sc * 2 * WALL_CELL;
    let seed = pick_seed(&r["seed"]);
    let steps = r["steps"].as_u64().unwrap_or(30) as usize;
    let req = Request {
        prompt: wall_prompt(&c.meta.style(), &desc),
        negative: NEG.into(),
        width: side,
        height: side,
        steps,
        guidance: 3.0,
        seed,
        count: 1,
        init: None,
        mask: None,
        strength: 1.0,
        preview_every: 1,
        opaque: false,
    };
    let req = adapt(c, &req);
    let started = Instant::now();
    let mut saved = 0;
    let stopped = run_engine(c, &req, |c, _, raw| {
        let sheet = sprite::despeckle(&depix(key, &sprite::desaturate(&raw, 0.85), sprite::COLORS, sc));
        let w = WALL_CELL as u32;
        let pieces: Vec<RgbaImage> = (0..4u32).map(|i| anchor_bottom_right(&imageops::crop_imm(&sheet, (i % 2) * w, (i / 2) * w, w, w).to_image())).collect();
        let mut sheet = RgbaImage::new(2 * w, 2 * w);
        for (i, p) in pieces.iter().enumerate() {
            imageops::overlay(&mut sheet, p, ((i as u32 % 2) * w) as i64, ((i as u32 / 2) * w) as i64);
        }
        let empty: Vec<&str> = pieces.iter().zip(WALL_PIECES).filter(|(p, _)| p.pixels().filter(|q| q.0[3] >= 128).count() < 24).map(|(_, n)| n.0).collect();
        if !empty.is_empty() {
            bail!(tr!(
                format!("Faltan piezas ({}). Prueba otra semilla.", empty.join(", ")),
                format!("Missing pieces ({}). Try another seed.", empty.join(", "))
            ));
        }
        let mut st = json!({"mode": "walls", "prompt": desc, "room": with_room, "grid": 2, "cell": WALL_CELL, "seed": seed, "steps": req.steps,
            "guidance": req.guidance, "negative": neg_used(c), "full_prompt": req.prompt, "width_px": side, "height_px": side,
            "seconds": (started.elapsed().as_secs_f64() * 10.0).round() / 10.0});
        let png = save(c, "walls", seed, &sheet, st.clone())?;
        saved += 1;
        if with_room {
            st["room_of"] = png.file_name().map(|n| n.to_string_lossy().to_string()).into();
            st["room_tiles"] = 3.into();
            save(c, "walls_room", seed, &wall_room(&pieces, 3), st)?;
        }
        Ok(())
    })?;
    Ok(format!(
        "{}{} {}{} · {:.1} s",
        if stopped { tr!("Detenido · ", "Stopped · ") } else { "" },
        saved,
        tr!("set de muros", "wall set"),
        if with_room { tr!(" + habitación", " + room") } else { "" },
        started.elapsed().as_secs_f64()
    ))
}

/// Walk-sheet directions in the order the sheet rows are saved (the creature dataset's captions).
pub const DIRS: [&str; 4] = ["south", "west", "north", "east"];
/// Camera tokens of the creature dataset (models/krea2/datasets/krea2_creatures_x8_v1).
pub const CREATURE_CAMERAS: [&str; 2] = ["tibia oblique", "top-down 3/4"];

/// Animated characters / creatures: one 8-frame walk sheet (2 rows of 4) per direction, the caption format the Krea v1
/// LoRA was trained on, the same seed for every direction (same identity). Post: magenta keyed (run_engine), depixel at
/// the version's scale, cut into 8 frames; saved as one <stamp>_creature_<seed>.png sheet (8 columns x one row per
/// direction, S W N E) + .json, and a GIF per direction (x4) next to it.
fn creature_mode(c: &mut Ctx, r: &Value) -> Result<String> {
    let desc = r["prompt"].as_str().unwrap_or("").trim().trim_end_matches('.').to_string();
    if desc.is_empty() {
        bail!(tr!("Describe el personaje o la criatura.", "Describe the character or creature."));
    }
    // a LoRA that declares its own walk format (catalog::WalkFormat) draws every direction in one grid image
    if let Some(w) = c.meta.walk.clone() {
        return creature_grid(c, r, &desc, &w);
    }
    if !c.engine.kind().family.spec().can("walk_sheet") {
        bail!(tr!(
            "Este modelo no genera hojas de caminar. Elige uno compatible en el selector de modelo.",
            "This model can't make walk sheets. Pick a compatible one in the model picker."
        ));
    }
    let camera = r["camera"].as_str().filter(|c| CREATURE_CAMERAS.contains(c)).unwrap_or(CREATURE_CAMERAS[0]).to_string();
    // cells inside what the walk sheets were trained on: square 32 / 48 / 64 px (x8 -> up to 2048x1024)
    let cell = r["cell"].as_u64().unwrap_or(32).clamp(16, 64) as usize;
    let cell_h = r["cell_h"].as_u64().unwrap_or(cell as u64).clamp(16, 64) as usize;
    let s = c.meta.scale as usize;
    let key = c.meta.background == Some([255, 0, 255]);
    let (wpx, hpx) = (4 * cell * s, 2 * cell_h * s);
    let dirs: Vec<&str> = if r["all_dirs"].as_bool().unwrap_or(true) { DIRS.to_vec() } else { vec![DIRS[(r["dir"].as_u64().unwrap_or(0) as usize).min(3)]] };
    let seed = pick_seed(&r["seed"]);
    let steps = r["steps"].as_u64().unwrap_or(30) as usize;
    let started = Instant::now();
    let mut rows: Vec<Vec<RgbaImage>> = vec![];
    let mut prompts = vec![];
    let style = c.meta.style();
    let mut used = None;
    for (i, d) in dirs.iter().enumerate() {
        (c.emit)(
            json!({"ev": "toast", "level": "info", "message": tr!(format!("Dirección {} de {}: {d}", i + 1, dirs.len()), format!("Direction {} of {}: {d}", i + 1, dirs.len()))}),
        );
        let text = format!(
            "{style}, format walk sheet, {camera} camera, facing {d}. An 8 frame walk cycle, 2 rows of 4 frames, on a flat magenta background. {desc}, pixel art, retro rpg style"
        );
        let req = Request {
            prompt: text.clone(),
            negative: NEG.into(),
            width: wpx,
            height: hpx,
            steps,
            guidance: 4.5,
            seed,
            count: 1,
            init: None,
            mask: None,
            strength: 1.0,
            preview_every: 1,
            opaque: false,
        };
        let req = adapt(c, &req);
        let mut frames = None;
        let stopped = run_engine(c, &req, |_, _, raw| {
            let native = sprite::despeckle(&depix(key, &raw, sprite::COLORS, s)); // 4 x 2 cells of cell x cell_h
            frames = Some(crate::anim::frames_of(&native, 4, 2));
            Ok(())
        })?;
        if stopped {
            return Ok(tr!(format!("Detenido en la dirección {d}"), format!("Stopped at direction {d}")));
        }
        rows.push(frames.context(tr!("el modelo no devolvió imagen", "the model returned no image"))?);
        prompts.push(req.prompt.clone());
        used = Some((req.steps, req.guidance));
    }
    let (steps, guidance) = used.unwrap_or((steps, 4.5));
    let mut sheet = RgbaImage::new((8 * cell) as u32, (rows.len() * cell_h) as u32);
    for (ri, frames) in rows.iter().enumerate() {
        for (fi, f) in frames.iter().enumerate() {
            imageops::overlay(&mut sheet, f, (fi * cell) as i64, (ri * cell_h) as i64);
        }
    }
    let st = json!({"mode": "creature", "prompt": desc, "camera": camera, "dirs": dirs, "cell": cell, "cell_h": cell_h, "frames": 8, "fps": 8,
        "seed": seed, "steps": steps, "guidance": guidance, "negative": neg_used(c), "full_prompts": prompts, "width_px": wpx, "height_px": hpx,
        "seconds": (started.elapsed().as_secs_f64() * 10.0).round() / 10.0});
    let png = save(c, "creature", seed, &sheet, st)?;
    for (d, frames) in dirs.iter().zip(&rows) {
        let stem = png.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        std::fs::write(png.with_file_name(format!("{stem}_{d}.gif")), crate::anim::gif(frames, 8, 4)?)?;
    }
    Ok(tr!(
        format!("Personaje · {} direcci{} · {:.1} s", dirs.len(), if dirs.len() == 1 { "ón" } else { "ones" }, started.elapsed().as_secs_f64()),
        format!("Character · {} direction{} · {:.1} s", dirs.len(), if dirs.len() == 1 { "" } else { "s" }, started.elapsed().as_secs_f64())
    ))
}

/// Swap x and y (not a mirror): in the oblique projection east = transpose(south), west = transpose(north).
fn transpose(im: &RgbaImage) -> RgbaImage {
    RgbaImage::from_fn(im.height(), im.width(), |x, y| *im.get_pixel(y, x))
}

/// A direction made from a drawn one: "transpose:south" or "mirror:west" (horizontal flip).
fn derive_frames(rule: &str, drawn: &[(String, Vec<RgbaImage>)]) -> Option<Vec<RgbaImage>> {
    let (op, from) = rule.split_once(':')?;
    let src = &drawn.iter().find(|(d, _)| d == from.trim())?.1;
    match op.trim() {
        "transpose" if src.iter().all(|f| f.width() == f.height()) => Some(src.iter().map(transpose).collect()),
        "mirror" => Some(src.iter().map(imageops::flip_horizontal).collect()),
        _ => None,
    }
}

/// A native-size grid image -> (sheet with one row per direction in S W N E order, frames per direction): cut into
/// `cols` x `rows` cells of `cw` x `ch`, rows named by `row_dirs`, the `derive` directions added.
/// A sheet's frames per kind (row name -> frames).
pub type Frames = Vec<(String, Vec<RgbaImage>)>;

pub fn grid_sheet(native: &RgbaImage, w: &catalog::WalkFormat, cw: usize, ch: usize) -> Result<(RgbaImage, Frames)> {
    if native.width() < (w.cols * cw) as u32 || native.height() < (w.rows * ch) as u32 {
        bail!("the image is {}x{}, the format needs {}x{}", native.width(), native.height(), w.cols * cw, w.rows * ch);
    }
    let cells = crate::anim::frames_of(&imageops::crop_imm(native, 0, 0, (w.cols * cw) as u32, (w.rows * ch) as u32).to_image(), w.cols as u32, w.rows as u32);
    let mut drawn: Vec<(String, Vec<RgbaImage>)> =
        w.row_dirs.iter().enumerate().map(|(i, d)| (d.clone(), cells[i * w.cols..(i + 1) * w.cols].to_vec())).collect();
    let made: Vec<(String, Vec<RgbaImage>)> = w
        .derive
        .iter()
        .filter(|(d, _)| !drawn.iter().any(|(x, _)| x == *d))
        .filter_map(|(d, rule)| derive_frames(rule, &drawn).map(|f| (d.clone(), f)))
        .collect();
    drawn.extend(made);
    // rows in the built-in order (S W N E), then any other direction names the format uses
    drawn.sort_by_key(|(d, _)| DIRS.iter().position(|x| x == d).unwrap_or(DIRS.len()));
    let (fw, fh) = (drawn[0].1[0].width(), drawn[0].1[0].height());
    let mut sheet = RgbaImage::new(fw * w.cols as u32, fh * drawn.len() as u32);
    for (ri, (_, frames)) in drawn.iter().enumerate() {
        for (fi, f) in frames.iter().enumerate() {
            imageops::overlay(&mut sheet, f, (fi as u32 * fw) as i64, (ri as u32 * fh) as i64);
        }
    }
    Ok((sheet, drawn))
}

/// Walk sheets in a LoRA's own grid format (e.g. 3 x 2: south and north rows, idle + 2 steps): one image, cut into
/// frames, the other directions derived. Saved like the built-in format (one row per direction in S W N E order,
/// one column per drawn frame) plus the play order, so the viewer and GIFs animate it the same way.
fn creature_grid(c: &mut Ctx, r: &Value, desc: &str, w: &catalog::WalkFormat) -> Result<String> {
    let s = c.meta.scale.max(1) as usize;
    let (cw, ch) = w.cell(c.meta.scale);
    if w.cols == 0 || w.rows == 0 || w.row_dirs.len() != w.rows || cw == 0 || ch == 0 {
        bail!(tr!(
            "El formato de caminar de este LoRA está incompleto (revisa su .dotsmith.json).",
            "This LoRA's walk format is incomplete (check its .dotsmith.json)."
        ));
    }
    let kind = r["kind"].as_u64().unwrap_or(0) as usize;
    let seed = pick_seed(&r["seed"]);
    let steps = r["steps"].as_u64().unwrap_or(30) as usize;
    let started = Instant::now();
    let prompt = w.prompt_for(desc, kind);
    // the size the LoRA was trained at, snapped to the model's alignment and the pixel grid
    let req = Request {
        prompt,
        negative: String::new(),
        width: w.width,
        height: w.height,
        steps,
        guidance: 3.5,
        seed,
        count: 1,
        init: None,
        mask: None,
        strength: 1.0,
        preview_every: 1,
        opaque: false,
    };
    let req = adapt(c, &req);
    let key = c.meta.background == Some([255, 0, 255]);
    let mut native = None;
    let stopped = run_engine(c, &req, |_, _, raw| {
        native = Some(sprite::despeckle(&depix(key, &raw, sprite::COLORS, s)));
        Ok(())
    })?;
    if stopped {
        return Ok(tr!("Detenido", "Stopped").to_string());
    }
    let native = native.context(tr!("el modelo no devolvió imagen", "the model returned no image"))?;
    let (sheet, drawn) = grid_sheet(&native, w, cw, ch)?;
    let (fw, fh) = (drawn[0].1[0].width(), drawn[0].1[0].height());
    let dirs: Vec<&str> = drawn.iter().map(|(d, _)| d.as_str()).collect();
    let play = w.play_order();
    let fps = if w.fps > 0 { w.fps } else { 8 };
    let st = json!({"mode": "creature", "format": "grid", "prompt": desc, "kind": kind, "dirs": dirs, "cell": fw, "cell_h": fh,
        "frames": w.cols, "play": play, "fps": fps, "derived": w.derive, "seed": seed, "steps": req.steps, "guidance": req.guidance,
        "negative": req.negative, "full_prompt": req.prompt, "width_px": req.width, "height_px": req.height,
        "seconds": (started.elapsed().as_secs_f64() * 10.0).round() / 10.0});
    let png = save(c, "creature", seed, &sheet, st)?;
    let stem = png.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    for (d, frames) in &drawn {
        let seq: Vec<RgbaImage> = play.iter().map(|&i| frames[i].clone()).collect();
        std::fs::write(png.with_file_name(format!("{stem}_{d}.gif")), crate::anim::gif(&seq, fps, 4)?)?;
    }
    Ok(tr!(
        format!("Personaje · {} direcciones · {:.1} s", dirs.len(), started.elapsed().as_secs_f64()),
        format!("Character · {} directions · {:.1} s", dirs.len(), started.elapsed().as_secs_f64())
    ))
}

/// The 32x32 sprite the region mask paints on, exactly as the region mode builds it.
pub fn prepared_sprite(source: &Path) -> Result<RgbaImage> {
    let im = image::open(source).with_context(|| format!("open {}", source.display()))?.to_rgba8();
    if im.width() > 2048 || im.height() > 2048 {
        bail!(tr!("Usa una imagen de hasta 2048 × 2048 píxeles.", "Use an image of at most 2048 × 2048 pixels."));
    }
    if is_picture(im.width(), im.height()) {
        return Ok(smooth_canvas(&im, PICTURE_MASK));
    }
    Ok(sprite::fit_canvas(&im, im.width().max(im.height())))
}

/// Model canvas for a sprite of `out` px: the version's scale per sprite pixel, raised for tiny sprites (at least 256 px)
/// and lowered for big sheets (at most 1536 px), a multiple of 16 when one fits.
fn regions_side(scale: usize, out: usize) -> usize {
    let out = out.max(1);
    let k = scale.max(256usize.div_ceil(out)).min(1536 / out).max(1);
    (1..=k).rev().find(|k| (k * out).is_multiple_of(16)).unwrap_or(k) * out
}

pub const PICTURE_MASK: u32 = 64;
const PICTURE_SIDE: u32 = 1024;

pub fn is_picture(w: u32, h: u32) -> bool {
    w.max(h) > 256
}

fn smooth_canvas(im: &RgbaImage, side: u32) -> RgbaImage {
    let (nw, nh) = fitted(im.width(), im.height(), side);
    let s = imageops::resize(im, nw, nh, imageops::FilterType::Triangle);
    let mut c = RgbaImage::new(side, side);
    imageops::overlay(&mut c, &s, ((side - nw) / 2) as i64, ((side - nh) / 2) as i64);
    c
}

fn fitted(w: u32, h: u32, side: u32) -> (u32, u32) {
    let f = side as f32 / w.max(h) as f32;
    (((w as f32 * f).round() as u32).clamp(1, side), ((h as f32 * f).round() as u32).clamp(1, side))
}

/// Region edit blends the redrawn image in through a soft mask, so everything outside the area keeps its exact pixels.
fn picture_edit(c: &mut Ctx, r: &Value, mode: &str, text: &str, source: &Path, src: &RgbaImage) -> Result<String> {
    let prompt = r["prompt"].as_str().unwrap_or("").trim().to_string();
    let side = PICTURE_SIDE;
    let (soft, repaint) = if mode == "regions" {
        let n = PICTURE_MASK;
        let bits = r["mask"].as_str().unwrap_or("").as_bytes();
        if bits.len() != (n * n) as usize {
            bail!(tr!("Pinta la zona editable antes de generar.", "Paint the editable area before generating."));
        }
        let cells = smooth_canvas(src, n);
        let m = GrayImage::from_fn(n, n, |x, y| Luma([if bits[(y * n + x) as usize] == b'1' && cells.get_pixel(x, y).0[3] > 0 { 255 } else { 0 }]));
        if m.pixels().all(|p| p.0[0] == 0) {
            bail!(tr!("Selecciona al menos una celda de la imagen.", "Select at least one cell of the image."));
        }
        let soft = imageops::resize(&m, side, side, imageops::FilterType::Triangle);
        let repaint = GrayImage::from_fn(side, side, |x, y| Luma([if soft.get_pixel(x, y).0[0] > 0 { 255 } else { 0 }]));
        (soft, Some(repaint))
    } else {
        (GrayImage::from_pixel(side, side, Luma([255])), None)
    };
    let seed = pick_seed(&r["seed"]);
    let count = r["variants"].as_u64().unwrap_or(1).clamp(1, 9) as usize;
    let strength = r["strength"].as_f64().unwrap_or(0.8) as f32;
    let req = adapt(
        c,
        &Request {
            prompt: text.to_string(),
            negative: String::new(),
            width: side as usize,
            height: side as usize,
            steps: r["steps"].as_u64().unwrap_or(25) as usize,
            guidance: r["guidance"].as_f64().unwrap_or(3.0) as f32,
            seed,
            count,
            init: Some(smooth_canvas(src, side)),
            mask: repaint,
            strength,
            preview_every: 1,
            opaque: true,
        },
    );
    let (w, h) = src.dimensions();
    let (nw, nh) = fitted(w, h, side);
    let (ox, oy) = ((side - nw) / 2, (side - nh) / 2);
    let to_source = |im: &RgbaImage| imageops::resize(&imageops::crop_imm(im, ox, oy, nw, nh).to_image(), w, h, imageops::FilterType::Lanczos3);
    let alpha = imageops::resize(&imageops::crop_imm(&soft, ox, oy, nw, nh).to_image(), w, h, imageops::FilterType::Triangle);
    let started = Instant::now();
    let mut saved = 0;
    let src_s = source.canonicalize().unwrap_or(source.to_path_buf()).to_string_lossy().replace("\\\\?\\", "");
    let stopped = run_engine_with(c, &req, false, |c, i, raw| {
        let generated = to_source(&raw);
        let out = RgbaImage::from_fn(w, h, |x, y| {
            let (o, g, a) = (src.get_pixel(x, y).0, generated.get_pixel(x, y).0, alpha.get_pixel(x, y).0[0] as u32);
            let mix = |k: usize| ((o[k] as u32 * (255 - a) + g[k] as u32 * a) / 255) as u8;
            image::Rgba([mix(0), mix(1), mix(2), o[3]])
        });
        let mut st = json!({"mode": mode, "pixel": false, "original": true, "prompt": prompt, "source": src_s, "strength": strength,
            "guidance": req.guidance, "steps": req.steps, "seed": seed + i as u64, "variant": i + 1, "variants": count, "full_prompt": req.prompt,
            "width": w, "height": h, "seconds": (started.elapsed().as_secs_f64() * 10.0).round() / 10.0});
        if mode == "regions" {
            st["mask"] = r["mask"].clone();
        }
        if mode == "angle" {
            st["pitch"] = r["pitch"].clone();
            st["yaw"] = r["yaw"].clone();
        }
        save(c, mode, seed + i as u64, &out, st)?;
        saved += 1;
        Ok(())
    })?;
    Ok(format!(
        "{}{saved}/{count} {} · {w} × {h} · {:.1} s",
        if stopped { tr!("Detenido · ", "Stopped · ") } else { "" },
        tr!("variantes", "variants"),
        started.elapsed().as_secs_f64()
    ))
}

fn image_mode(c: &mut Ctx, r: &Value, mode: &str) -> Result<String> {
    let source = PathBuf::from(r["source"].as_str().unwrap_or(""));
    if !source.is_file() {
        bail!(tr!("Carga una imagen de origen.", "Load a source image first."));
    }
    let prompt = r["prompt"].as_str().unwrap_or("").trim().to_string();
    if prompt.is_empty() {
        bail!(tr!("Describe el resultado que quieres.", "Describe the result you want."));
    }
    let mut camera = r["camera"].as_str().unwrap_or("Normal").to_string();
    let mut full = with_camera(&prompt, &camera);
    let (pitch, yaw) = (r["pitch"].as_f64().unwrap_or(45.0) as f32, r["yaw"].as_f64().unwrap_or(0.0) as f32);
    if mode == "angle" {
        camera = format!("gizmo {}°/{}°", pitch.round(), yaw.round());
        full = format!("{prompt} {}", angle_text(pitch, yaw));
    }
    let src = image::open(&source)?.to_rgba8();
    if src.width() > 2048 || src.height() > 2048 {
        bail!(tr!("Usa una imagen de hasta 2048 × 2048 píxeles.", "Use an image of at most 2048 × 2048 pixels."));
    }
    // out_size 0 = "Original"
    let keep_size = mode == "regions" || r["out_size"].as_u64() == Some(0);
    if keep_size && is_picture(src.width(), src.height()) {
        let text = if mode == "angle" { full.clone() } else { prompt.clone() };
        return picture_edit(c, r, mode, &text, &source, &src);
    }
    let out = if keep_size { src.width().max(src.height()) as usize } else { r["out_size"].as_u64().unwrap_or(32) as usize };
    let s = c.meta.scale as usize;
    let key = c.meta.background == Some([255, 0, 255]);
    let side = regions_side(s, out);
    let k = side / out;
    let seed = pick_seed(&r["seed"]);
    let count = r["variants"].as_u64().unwrap_or(1).clamp(1, 9) as usize;
    let strength = r["strength"].as_f64().unwrap_or(0.45) as f32;
    let guidance = r["guidance"].as_f64().unwrap_or(3.0) as f32;
    let steps = r["steps"].as_u64().unwrap_or(25) as usize;
    let canvas = sprite::fit_canvas(&src, side as u32);
    let original = imageops::resize(&canvas, out as u32, out as u32, imageops::FilterType::Nearest);
    let canvas = if mode == "angle" && r["warp"].as_bool().unwrap_or(true) { sprite::warp_to_angle(&canvas, pitch, yaw) } else { canvas };
    let mut mask_img = None;
    let mut mask32 = None;
    if mode == "regions" {
        let bits = r["mask"].as_str().unwrap_or("");
        if bits.len() != out * out {
            bail!(tr!("Pinta la zona editable antes de generar.", "Paint the editable area before generating."));
        }
        let n = out as u32;
        let m = GrayImage::from_fn(n, n, |x, y| {
            let on = bits.as_bytes()[(y * n + x) as usize] == b'1' && original.get_pixel(x, y).0[3] > 0;
            Luma([if on { 255 } else { 0 }])
        });
        if m.pixels().all(|p| p.0[0] == 0) {
            bail!(tr!("Selecciona al menos un píxel del objeto.", "Select at least one pixel of the object."));
        }
        mask32 = Some(m.clone());
        mask_img = Some(m);
    }
    // versions trained on a flat background (an RGB-only VAE) start from the source flattened onto that colour,
    // the way their training images looked; alpha versions (Qwen 2.1) keep the transparent canvas
    let canvas = match c.meta.background {
        Some(bg) => RgbaImage::from_fn(canvas.width(), canvas.height(), |x, y| {
            let p = canvas.get_pixel(x, y).0;
            let a = p[3] as u32;
            let mix = |i: usize| ((p[i] as u32 * a + bg[i] as u32 * (255 - a)) / 255) as u8;
            image::Rgba([mix(0), mix(1), mix(2), 255])
        }),
        None => canvas,
    };
    let plain = plain(c);
    let text = if plain {
        full.clone()
    } else {
        templated(r, &c.meta.style(), "format single. One fantasy inventory item", "a transparent background", &full, side, side)
    };
    let req = Request {
        prompt: text.clone(),
        negative: String::new(),
        width: side,
        height: side,
        steps,
        guidance,
        seed,
        count,
        init: Some(canvas),
        mask: mask_img,
        strength,
        preview_every: 1,
        opaque: false,
    };
    let req = adapt(c, &req);
    let started = Instant::now();
    let mut saved = 0;
    let src_s = source.canonicalize().unwrap_or(source.clone()).to_string_lossy().replace("\\\\?\\", "");
    let stopped = run_engine(c, &req, |c, i, raw| {
        let tile = if let Some(m) = &mask32 {
            let orig = &original;
            let n = out as u32;
            let generated = imageops::resize(&raw, n, n, imageops::FilterType::Nearest);
            RgbaImage::from_fn(n, n, |x, y| {
                if m.get_pixel(x, y).0[0] > 127 {
                    // the version's flat background drawn inside the area is the object being cut there
                    let mut p = *generated.get_pixel(x, y);
                    p.0[3] = if key && sprite::is_magentaish(&p) { 0 } else { orig.get_pixel(x, y).0[3] };
                    p
                } else {
                    *orig.get_pixel(x, y)
                }
            })
        } else if plain {
            sprite::fit(&raw, out as u32)
        } else {
            sprite::fit(&sprite::despeckle(&depix(key, &sprite::desaturate(&raw, 0.85), sprite::COLORS, k)), out as u32)
        };
        let tile = if keep_size {
            let (w, h) = src.dimensions();
            imageops::crop_imm(&tile, (out as u32 - w) / 2, (out as u32 - h) / 2, w, h).to_image()
        } else {
            tile
        };
        let mut st = json!({"mode": mode, "prompt": prompt, "camera": camera, "source": src_s, "strength": strength, "guidance": req.guidance,
            "steps": req.steps, "seed": seed + i as u64, "variant": i + 1, "variants": count, "out_size": out, "original": keep_size,
            "full_prompt": req.prompt, "seconds": (started.elapsed().as_secs_f64() * 10.0).round() / 10.0});
        if mode == "regions" {
            st["mask"] = r["mask"].clone();
        }
        if mode == "angle" {
            st["pitch"] = pitch.into();
            st["yaw"] = yaw.into();
            st["warp"] = r["warp"].as_bool().unwrap_or(true).into();
        }
        save(c, mode, seed + i as u64, &tile, st)?;
        saved += 1;
        Ok(())
    })?;
    Ok(format!(
        "{}{saved}/{count} {} · {:.1} s",
        if stopped { tr!("Detenido · ", "Stopped · ") } else { "" },
        tr!("variantes", "variants"),
        started.elapsed().as_secs_f64()
    ))
}

#[cfg(test)]
mod grid_tests {
    use super::*;

    fn fmt() -> catalog::WalkFormat {
        serde_json::from_value(json!({"width": 48, "height": 32, "cols": 3, "rows": 2, "row_dirs": ["south", "north"],
            "derive": {"east": "transpose:south", "west": "transpose:north"}, "play": [1, 0, 2, 0]}))
        .unwrap()
    }

    #[test]
    fn a_3x2_grid_becomes_four_directions_with_transposed_sides() {
        // 2 x 2 cells of 16 px at scale 1: each cell gets its own colour, one marked pixel at (3, 1)
        let mut im = RgbaImage::new(48, 32);
        for (i, px) in im.pixels_mut().enumerate() {
            let (x, y) = (i as u32 % 48, i as u32 / 48);
            *px = image::Rgba([(x / 16 * 80) as u8, (y / 16 * 120) as u8, 0, 255]);
        }
        im.put_pixel(3, 1, image::Rgba([255, 255, 255, 255]));
        let (sheet, dirs) = grid_sheet(&im, &fmt(), 16, 16).unwrap();
        let names: Vec<&str> = dirs.iter().map(|d| d.0.as_str()).collect();
        assert_eq!(names, ["south", "west", "north", "east"]);
        assert_eq!(sheet.dimensions(), (48, 64));
        let east = &dirs[3].1[0];
        assert_eq!(east.get_pixel(1, 3).0, [255, 255, 255, 255], "east = transpose(south)");
        assert_eq!(dirs[2].1[0].get_pixel(0, 0).0[1], 120, "north = second drawn row");
    }

    /// DOTSMITH_GRID_SAMPLE=<raw 1536x1024 outfit png> [DOTSMITH_GRID_OUT=<dir>]: run the real post-processing on a frost
    /// outfit render (magenta keyed, depixel x8, cut, derive) and write the sheet and GIFs for a look.
    #[test]
    fn real_outfit_render_if_given() {
        let Ok(p) = std::env::var("DOTSMITH_GRID_SAMPLE") else { return };
        let raw = image::open(&p).unwrap().to_rgba8();
        let w: catalog::WalkFormat = serde_json::from_value(json!({"width": 1536, "height": 1024, "cols": 3, "rows": 2,
            "row_dirs": ["south", "north"], "derive": {"east": "transpose:south", "west": "transpose:north"}, "play": [1, 0, 2, 0]}))
        .unwrap();
        let keyed = sprite::key_color(&raw, [255, 0, 255], 90, 60);
        let native = sprite::despeckle(&depix(true, &keyed, sprite::COLORS, 8));
        let (cw, ch) = w.cell(8);
        let (sheet, dirs) = grid_sheet(&native, &w, cw, ch).unwrap();
        assert_eq!(dirs.len(), 4);
        let out = PathBuf::from(std::env::var("DOTSMITH_GRID_OUT").unwrap_or_else(|_| std::env::temp_dir().to_string_lossy().to_string()));
        let stem = Path::new(&p).file_stem().unwrap().to_string_lossy().to_string();
        sheet.save(out.join(format!("{stem}_sheet.png"))).unwrap();
        for (d, frames) in &dirs {
            let seq: Vec<RgbaImage> = w.play_order().iter().map(|&i| frames[i].clone()).collect();
            std::fs::write(out.join(format!("{stem}_{d}.gif")), crate::anim::gif(&seq, 6, 4).unwrap()).unwrap();
        }
    }
}

#[cfg(test)]
mod template_tests {
    use super::*;

    #[test]
    fn region_canvases_fit_the_model() {
        assert_eq!(regions_side(8, 32), 256);
        assert_eq!(regions_side(16, 32), 512);
        assert_eq!(regions_side(8, 96), 768);
        assert_eq!(regions_side(16, 96), 1536);
        assert_eq!(regions_side(8, 16), 256);
        assert_eq!(regions_side(16, 256), 1536);
        assert_eq!(regions_side(8, 50), 400);
    }

    #[test]
    fn the_default_template_is_the_built_in_prompt() {
        let t = templated(&json!({}), "pixel art", "format single. One fantasy inventory item", "a transparent background", "A wand.", 512, 512);
        assert_eq!(t, "pixel art, modern detailed, format single. One fantasy inventory item on a transparent background. A wand. --w 512 --h 512");
    }

    #[test]
    fn an_edited_template_keeps_the_description() {
        let r = json!({"template": "{style}, chunky outlines"});
        assert_eq!(templated(&r, "pixel art", "l", "b", "A wand.", 64, 64), "pixel art, chunky outlines A wand.");
        let r = json!({"template": "  "});
        assert!(templated(&r, "s", "l", "b", "p", 1, 1).starts_with("s, modern detailed"));
    }
}
