//! dotsmith-engine: the engine without the app, for scripted runs and engine checks.
//!
//!   generate <out.png> <prompt> [seed] [steps] [size] [lora|-] [guidance] [turbo 0|1]
//!   batch <jobs.tsv> [lora|-] [lora weight]
//!   mem-cycle <lora a> <lora b> [rounds]
//!   pixelize <in> <out.png> <width> <height> [patch] [thickness] [colors|0]
//!   post <raw.png> <out.png> [scale] [cell]
//!
//! MODEL=<family id> picks the family (default: the catalog's first).
use anyhow::{Context, Result, bail};
use dotsmith_engine::engine::{Engine, Event, Request};
use dotsmith_engine::model::{Family, ModelKind, ModelPaths, Variant};
use dotsmith_engine::sprite::{COLORS, depixel, desaturate, despeckle, fit_wh, pieces_of};
use std::sync::atomic::AtomicBool;
use std::time::Instant;

const NEGATIVE: &str = "smooth gradient, glossy, shiny, airbrushed, 3d render, soft shading, plastic, blurry";
const USAGE: &str = "usage: dotsmith-engine <generate|batch|mem-cycle|pixelize|post> ... (see the crate docs in src/main.rs)";

fn kind_from_env(turbo: bool) -> Result<ModelKind> {
    let family = match std::env::var("MODEL") {
        Ok(id) => Family::from_id(&id).with_context(|| format!("unknown MODEL {id}"))?,
        Err(_) => Family::ALL[0],
    };
    Ok(ModelKind::new(family, if turbo { Variant::Turbo } else { Variant::Base }))
}

fn env(key: &str) -> Option<String> {
    std::env::var(key).ok()
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let arg = |i: usize, default: &str| args.get(i).cloned().unwrap_or_else(|| default.to_string());
    match args.get(1).map(String::as_str) {
        // env: H (height, default = size), COUNT, INIT (img2img source), STRENGTH, SWAP=1 (load the other variant
        // first and switch, like the app's base / Turbo selector), ROUNDS (repeat to time warm runs)
        Some("generate") if args.len() >= 4 => {
            let t0 = Instant::now();
            let turbo = arg(9, "0") == "1";
            let swap = env("SWAP").is_some();
            let mut eng = Engine::load(kind_from_env(turbo ^ swap)?)?;
            if swap {
                eng.set_kind(kind_from_env(turbo)?)?;
            }
            let lora = arg(7, "-");
            eng.set_lora((lora != "-").then_some(lora.as_str()), 1.0)?;
            eng.dev.synchronize()?;
            println!("load {:.1} s", t0.elapsed().as_secs_f64());
            for line in dotsmith_engine::core::memory::report() {
                println!("  memory: {line}");
            }
            let size: usize = arg(6, "512").parse()?;
            let req = Request {
                prompt: args[3].clone(),
                negative: NEGATIVE.into(),
                width: size,
                height: env("H").map(|h| h.parse()).transpose()?.unwrap_or(size),
                steps: arg(5, "25").parse()?,
                guidance: arg(8, "3").parse()?,
                seed: arg(4, "7").parse()?,
                count: env("COUNT").map(|c| c.parse()).transpose()?.unwrap_or(1),
                init: env("INIT").map(|p| image::open(p).map(|i| i.to_rgba8())).transpose()?,
                mask: None,
                strength: env("STRENGTH").map(|s| s.parse()).transpose()?.unwrap_or(1.0),
                preview_every: 0,
                opaque: false,
            };
            let rounds: usize = env("ROUNDS").map(|r| r.parse()).transpose()?.unwrap_or(1);
            let cancel = AtomicBool::new(false);
            let out = args[2].clone();
            for round in 0..rounds {
                let t0 = Instant::now();
                let mut sampling = Instant::now();
                eng.generate(&req, &cancel, &mut |e| match e {
                    Event::Phase("sampling") => sampling = Instant::now(),
                    Event::Step { step, total, .. } if step == total => {
                        println!("  sampled {total} steps in {:.2} s", sampling.elapsed().as_secs_f64());
                    }
                    Event::Sample { image, index } => {
                        if let Err(e) = image.save(out.replace(".png", &format!("_{index}.png"))) {
                            eprintln!("could not save sample {index}: {e}");
                        }
                    }
                    _ => {}
                })?;
                eng.dev.synchronize()?;
                println!("round {round}: {:.2} s -> {out}", t0.elapsed().as_secs_f64());
            }
        }
        // One model load, many jobs. TSV columns (tab separated, '#' = comment):
        //   out.png  seed  width  height  strength  init.png|-  prompt
        // Writes <out>_raw.png (model output) and <out>.png (depixel + despeckle + fit, cell = side / 16).
        // env: TURBO=1, STEPS, GUIDANCE, NEG (negative prompt; empty for none), RAW_ONLY=1 (skip the post-processing)
        Some("batch") if args.len() >= 3 => {
            let lora = arg(3, "-");
            let weight: f64 = arg(4, "1").parse()?;
            if !weight.is_finite() {
                bail!("LoRA weight must be finite");
            }
            let t0 = Instant::now();
            let steps: usize = env("STEPS").map(|s| s.parse()).transpose()?.unwrap_or(25);
            let guidance: f32 = env("GUIDANCE").map(|s| s.parse()).transpose()?.unwrap_or(3.0);
            let raw_only = env("RAW_ONLY").as_deref() == Some("1");
            let mut eng = Engine::load(kind_from_env(env("TURBO").as_deref() == Some("1"))?)?;
            eng.set_lora((lora != "-" && weight != 0.0).then_some(lora.as_str()), weight)?;
            eng.dev.synchronize()?;
            println!("load {:.1} s", t0.elapsed().as_secs_f64());
            let cancel = AtomicBool::new(false);
            for line in std::fs::read_to_string(&args[2])?.lines() {
                let line = line.trim_end_matches('\r');
                if line.trim().is_empty() || line.starts_with('#') {
                    continue;
                }
                let c: Vec<&str> = line.splitn(7, '\t').collect();
                if c.len() < 7 {
                    bail!("bad job line: {line}");
                }
                let (out, w, h): (String, usize, usize) = (c[0].into(), c[2].parse()?, c[3].parse()?);
                let req = Request {
                    prompt: c[6].into(),
                    negative: env("NEG").unwrap_or_else(|| NEGATIVE.into()),
                    width: w,
                    height: h,
                    steps,
                    guidance,
                    seed: c[1].parse()?,
                    count: 1,
                    init: (c[5] != "-").then(|| image::open(c[5]).map(|i| i.to_rgba8())).transpose()?,
                    mask: None,
                    strength: c[4].parse()?,
                    preview_every: 0,
                    opaque: false,
                };
                let t = Instant::now();
                let mut raw = None;
                eng.generate(&req, &cancel, &mut |e| {
                    if let Event::Sample { image, .. } = e {
                        raw = Some(image)
                    }
                })?;
                let raw: image::RgbaImage = raw.with_context(|| format!("no sample for {out}"))?;
                raw.save(out.replace(".png", "_raw.png"))?;
                if raw_only {
                    println!("{out}: {:.1} s", t.elapsed().as_secs_f64());
                    continue;
                }
                let (cw, ch) = ((w / 16) as u32, (h / 16) as u32);
                let pieces = pieces_of(&depixel(&desaturate(&raw, 0.85), COLORS, 16), cw.max(ch));
                if let Some(p) = pieces.first() {
                    fit_wh(&despeckle(p), cw, ch, false).save(&out)?;
                }
                println!("{out}: {:.1} s, {} pieces", t.elapsed().as_secs_f64(), pieces.len());
            }
        }
        // Free VRAM (after a sync) through LoRA swaps, sizes, base <-> Turbo, a family round trip and the unload,
        // repeated: free VRAM falling from one round to the next = memory left behind.
        Some("mem-cycle") if args.len() >= 4 => {
            let fam = kind_from_env(true)?.family;
            let other =
                Family::ALL.into_iter().find(|f| *f != fam && f.has_turbo() && ModelPaths::default_for(*f).is_ok_and(|p| p.require(Variant::Turbo).is_ok()));
            let (la, lb) = (args[2].clone(), args[3].clone());
            let rounds: usize = arg(4, "3").parse()?;
            let mut eng = Engine::load(ModelKind::new(fam, Variant::Turbo))?;
            let report = |eng: &Engine, what: &str| {
                let _ = eng.dev.synchronize();
                println!("{what:<28} free {:>6} MB", dotsmith_engine::gpu::free_mb(&eng.dev).unwrap_or(0));
            };
            let cancel = AtomicBool::new(false);
            let run = |eng: &mut Engine, size: usize, prompt: &str| -> Result<()> {
                let req = Request {
                    prompt: prompt.into(),
                    negative: String::new(),
                    width: size,
                    height: size,
                    steps: 2,
                    guidance: 1.0,
                    seed: 1,
                    count: 1,
                    init: None,
                    mask: None,
                    strength: 1.0,
                    preview_every: 0,
                    opaque: false,
                };
                eng.generate(&req, &cancel, &mut |_| {})
            };
            report(&eng, "loaded");
            for round in 0..rounds {
                println!("-- round {round}");
                eng.set_lora(Some(&la), 1.0)?;
                report(&eng, "lora a");
                run(&mut eng, 512, "a red potion")?;
                report(&eng, "gen 512");
                eng.set_lora(Some(&lb), 1.0)?;
                report(&eng, "lora b");
                run(&mut eng, 1024, "a red potion")?;
                report(&eng, "gen 1024");
                run(&mut eng, 1536, "a blue sword")?;
                report(&eng, "gen 1536 (text enc. out)");
                run(&mut eng, 512, "a green shield")?;
                report(&eng, "gen 512 (text enc. back)");
                eng.set_kind(ModelKind::new(fam, Variant::Base))?;
                report(&eng, "base");
                run(&mut eng, 512, "a red potion")?;
                report(&eng, "gen 512 base");
                eng.set_kind(ModelKind::new(fam, Variant::Turbo))?;
                report(&eng, "turbo");
                eng.set_lora(None, 1.0)?;
                report(&eng, "no lora");
                if let Some(o) = other {
                    eng.set_kind(ModelKind::new(o, Variant::Turbo))?;
                    run(&mut eng, 512, "a red potion")?;
                    report(&eng, &format!("{} + gen", o.id()));
                    eng.set_kind(ModelKind::new(fam, Variant::Turbo))?;
                    report(&eng, &format!("back to {}", fam.id()));
                }
            }
            let dev = eng.dev.clone();
            drop(eng);
            dev.synchronize()?;
            println!("{:<28} free {:>6} MB", "unloaded", dotsmith_engine::gpu::free_mb(&dev).unwrap_or(0));
        }
        // PixelOE-style conversion (CPU)
        Some("pixelize") if args.len() >= 6 => {
            let im = image::open(&args[2])?.to_rgb8();
            let n = |i: usize, d: u32| -> Result<u32> { Ok(arg(i, &d.to_string()).parse()?) };
            let colors = n(8, 32)?;
            let t = Instant::now();
            let out = dotsmith_engine::pixeloe::pixelize(&im, n(4, 64)?, n(5, 64)?, n(6, 12)?, n(7, 2)?, (colors > 0).then_some(colors as usize));
            println!("pixelize {:.2} s -> {}x{}", t.elapsed().as_secs_f64(), out.width(), out.height());
            out.save(&args[3])?;
        }
        // depixel + despeckle + fit, like the app's single sprite
        Some("post") if args.len() >= 4 => {
            let raw = image::open(&args[2])?.to_rgba8();
            let scale: usize = arg(4, "16").parse()?;
            let cell: u32 = arg(5, "32").parse()?;
            let t = Instant::now();
            let pieces = pieces_of(&depixel(&desaturate(&raw, 0.85), COLORS, scale), cell);
            let first = pieces.first().context("nothing to cut out")?;
            println!("post {:.1} ms, {} pieces", t.elapsed().as_secs_f64() * 1e3, pieces.len());
            fit_wh(&despeckle(first), cell, cell, false).save(&args[3])?;
        }
        _ => bail!(USAGE),
    }
    Ok(())
}
