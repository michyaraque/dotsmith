//! The generation loop, generic over model families (model::DiffusionModel): prompts are encoded once and cached,
//! a LoRA swap only moves its small A/B matrices, a base <-> turbo swap reloads only the DiT, a family change reloads
//! the whole model. The loop knows latent geometry, CFG, Euler steps, SDEdit and masks; nothing about any one model.
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use image::{GrayImage, RgbaImage};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::core::rng;
use crate::model::{Defaults, DiffusionModel, ModelKind, ModelPaths, load_model};

#[derive(Clone)]
pub struct Request {
    pub prompt: String,
    pub negative: String,
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    /// Classic CFG scale (1 = off).
    pub guidance: f32,
    pub seed: u64,
    pub count: usize,
    pub init: Option<RgbaImage>, // img2img start (SDEdit), resized to width x height
    pub mask: Option<GrayImage>, // white = repaint (with init)
    pub strength: f32,
    pub preview_every: usize, // 0 = no previews
    /// Painted background instead of transparency (families that take it as a condition; the others read the prompt).
    pub opaque: bool,
}

pub enum Event {
    Phase(&'static str),
    Step { index: usize, count: usize, step: usize, total: usize },
    Preview { index: usize, image: RgbaImage },
    Sample { index: usize, image: RgbaImage },
}

/// Python's round(): half to even.
fn round_half_even(x: f64) -> i64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 { 2.0 * (x / 2.0).round() } else { r }.max(i64::MIN as f64) as i64
}

/// ai-toolkit strips `--w 512 --h 512` style flags from prompts before encoding.
pub fn strip_flags(p: &str) -> &str {
    p.split("--").next().unwrap_or("").trim()
}

/// [3|4, h, w] f32 in [-1, 1] -> RGBA8 (RGB gets an opaque alpha).
pub fn to_image(img: &Tensor) -> Result<RgbaImage> {
    let (c, h, w) = img.dims3()?;
    let v = ((img.clamp(-1f32, 1f32)? / 2.0)? + 0.5)?.affine(255.0, 0.0)?.round()?.permute((1, 2, 0))?.flatten_all()?.to_vec1::<f32>()?;
    let mut buf: Vec<u8> = v.iter().map(|x| x.clamp(0.0, 255.0) as u8).collect();
    if c == 3 {
        buf = buf.chunks(3).flat_map(|p| [p[0], p[1], p[2], 255]).collect();
    }
    Ok(RgbaImage::from_raw(w as u32, h as u32, buf).unwrap())
}

/// Below this much free dedicated VRAM the next allocation would land in system RAM.
const SPILL_MB: usize = 64;

/// The generation was stopped because the GPU ran out of dedicated memory: what the watch read at that moment (MB).
#[derive(Debug, Default, Clone, Copy)]
pub struct VramFull {
    pub used: u64,
    pub budget: u64,
    pub free: u64,
}

impl std::fmt::Display for VramFull {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "the GPU ran out of dedicated memory (VRAM): {} MB in use of a {} MB budget, {} MB free", self.used, self.budget, self.free)
    }
}

impl std::error::Error for VramFull {}

pub struct Engine {
    pub dev: Device,
    model: Box<dyn DiffusionModel>,
    lora: Option<(String, f64)>,
    prompts: HashMap<String, Tensor>,
}

/// The GPU works, and both the card and this build's kernels are new enough for the family (its catalog
/// `min_compute`): the legacy build for older cards has no bf16 kernels, whatever the card.
fn runs_here(kind: ModelKind) -> Result<()> {
    let gpu = crate::gpu::probe()?;
    let Some([major, minor]) = kind.family.spec().min_compute else { return Ok(()) };
    if crate::gpu::MIN_COMPUTE < (major, minor) {
        anyhow::bail!("{} needs Dotsmith's standard package; this one is for older cards.", kind.family.label());
    }
    if gpu.compute < (major, minor) {
        anyhow::bail!(
            "{} needs a {} series card or newer (compute {major}.{minor}+); {} is compute {}.{}.",
            kind.family.label(),
            crate::gpu::series((major, minor)),
            gpu.name,
            gpu.compute.0,
            gpu.compute.1
        );
    }
    Ok(())
}

impl Engine {
    /// Load `kind` from the default file layout of its family.
    pub fn load(kind: ModelKind) -> Result<Self> {
        Self::load_with(kind, ModelPaths::default_for(kind.family)?)
    }

    pub fn load_with(kind: ModelKind, paths: ModelPaths) -> Result<Self> {
        runs_here(kind)?;
        paths.require(kind.variant)?;
        let dev = Device::new_cuda(0)?;
        let model = load_model(kind, paths, &dev)?;
        Ok(Self { dev, model, lora: None, prompts: HashMap::new() })
    }

    pub fn kind(&self) -> ModelKind {
        self.model.kind()
    }

    /// The model's defaults, with the catalog's steps for the variant when it sets them.
    pub fn defaults(&self) -> Defaults {
        let mut d = self.model.defaults();
        let steps = self.kind().spec_steps();
        if steps > 0 {
            d.steps = steps;
        }
        d
    }

    /// Image pixels per latent pixel of the loaded family (16 Qwen 2.1, 8 Krea 2).
    pub fn latent_scale(&self) -> usize {
        self.model.latent_scale()
    }

    /// Canvas sides are rounded down to a multiple of this.
    pub fn align(&self) -> usize {
        self.model.align()
    }

    /// Switch model: same family = swap the DiT only (LoRA kept); another family = reload everything (LoRA dropped:
    /// a LoRA belongs to one family).
    pub fn set_kind(&mut self, kind: ModelKind) -> Result<()> {
        let cur = self.model.kind();
        if kind == cur {
            return Ok(());
        }
        runs_here(kind)?;
        let paths = ModelPaths::default_for(kind.family)?;
        paths.require(kind.variant)?;
        if kind.family == cur.family {
            return self.model.set_variant(kind.variant);
        }
        // free the old model before loading the new one (both resident would not fit 24 GB)
        self.model = Box::new(Placeholder(self.dev.clone()));
        self.prompts.clear();
        self.lora = None;
        self.dev.synchronize()?;
        self.model = load_model(kind, paths, &self.dev)?;
        Ok(())
    }

    /// Swap the LoRA in place (no base reload).
    pub fn set_lora(&mut self, path: Option<&str>, multiplier: f64) -> Result<usize> {
        let cur = self.lora.as_ref().map(|(p, m)| (p.as_str(), *m));
        if cur == path.map(|p| (p, multiplier)) {
            return Ok(0);
        }
        let n = self.model.set_lora(path, multiplier)?;
        self.lora = path.map(|p| (p.to_string(), multiplier));
        Ok(n)
    }

    fn embed(&mut self, prompt: &str) -> Result<Tensor> {
        let p = strip_flags(prompt).to_string();
        if let Some(t) = self.prompts.get(&p) {
            return Ok(t.clone());
        }
        let e = self.model.encode_prompt(&p)?;
        if self.prompts.len() > 64 {
            self.prompts.clear();
        }
        self.prompts.insert(p, e.clone());
        Ok(e)
    }

    /// Runs `req` while a watcher polls the VRAM (free memory and this process's Windows budget): the moment it runs out
    /// (Windows would silently continue in system RAM, which looks like a hang) the generation stops with `VramFull`
    /// within one DiT / VAE / text-encoder block. A cancel stops it the same way.
    pub fn generate(&mut self, req: &Request, cancel: &AtomicBool, on: &mut dyn FnMut(Event)) -> Result<()> {
        use crate::core::ABORT;
        ABORT.store(false, Ordering::SeqCst);
        let ctx = self.dev.as_cuda_device()?.cuda_stream().context().clone();
        let (done, full) = (AtomicBool::new(false), std::sync::Mutex::new(None::<VramFull>));
        let r = std::thread::scope(|s| {
            s.spawn(|| {
                let budget = crate::gpu::Budget::open();
                while !done.load(Ordering::Relaxed) {
                    if cancel.load(Ordering::Relaxed) {
                        ABORT.store(true, Ordering::SeqCst);
                    }
                    let (used, budget) = budget.as_ref().and_then(|b| b.read()).unwrap_or((0, u64::MAX));
                    let free = ctx.mem_get_info().map(|(f, _)| f >> 20).unwrap_or(usize::MAX);
                    if used > budget || free < SPILL_MB {
                        full.lock().unwrap().get_or_insert(VramFull { used, budget, free: free as u64 });
                        ABORT.store(true, Ordering::SeqCst);
                    }
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
            });
            let r = self.run(req, &ABORT, on);
            done.store(true, Ordering::Relaxed);
            r
        });
        ABORT.store(false, Ordering::SeqCst);
        if let Some(v) = full.into_inner().unwrap() {
            return Err(v.into()); // no synchronize: the queued work may be crawling through system RAM
        }
        if cancel.load(Ordering::SeqCst) {
            return Ok(());
        }
        r
    }

    fn run(&mut self, req: &Request, cancel: &AtomicBool, on: &mut dyn FnMut(Event)) -> Result<()> {
        if let Some(r) = self.model.sample_own(req, cancel, on) {
            return r;
        }
        let align = self.model.align();
        let (w, h) = (req.width / align * align, req.height / align * align);
        let s = self.model.latent_scale();
        let (lh, lw, c) = (h / s, w / s, self.model.latent_channels());
        let ldt = self.model.latent_dtype();
        let use_cfg = req.guidance > 1.0;
        on(Event::Phase("encoding"));
        let mut rows = vec![self.embed(&req.prompt)?];
        if use_cfg {
            rows.push(self.embed(&req.negative)?);
        }
        let b = rows.len();
        self.model.begin(&rows)?;
        // >= 1 MP: the text encoder leaves the GPU so sampling and decoding fit in 24 GB; encoded prompts stay cached
        if w * h >= 1024 * 1024 {
            self.model.release_text_encoder();
        }

        // img2img: init latents, schedule start, optional mask at latent resolution
        let init = match &req.init {
            Some(img) => {
                let img = image::imageops::resize(img, w as u32, h as u32, image::imageops::FilterType::Nearest);
                let x: Vec<f32> = img.pixels().flat_map(|p| p.0).map(|v| v as f32 / 127.5 - 1.0).collect();
                let x = Tensor::from_vec(x, (h, w, 4), &self.dev)?.permute((2, 0, 1))?.unsqueeze(0)?;
                on(Event::Phase("encoding image"));
                Some(self.model.encode_image(&x)?)
            }
            None => None,
        };
        let mask = match (&req.mask, &init) {
            (Some(m), Some(_)) => {
                let m = image::imageops::resize(m, lw as u32, lh as u32, image::imageops::FilterType::Nearest);
                let v: Vec<f32> = m.pixels().map(|p| p.0[0] as f32 / 255.0).collect();
                Some(Tensor::from_vec(v, (1, 1, lh, lw), &self.dev)?.to_dtype(ldt)?)
            }
            _ => None,
        };
        let sig = self.model.sigmas(req.steps, self.model.tokens(lh, lw), init.is_some());
        let steps = sig.len() - 1;
        let start = if init.is_some() { (round_half_even((1.0 - req.strength as f64) * steps as f64)).clamp(0, steps as i64 - 1) as usize } else { 0 };
        let total = steps - start;

        // lockstep (weight streaming): up to `group` images share every forward pass and so every weight upload
        let group = self.model.lockstep().min(req.count).max(1);
        if group > 1 {
            for g0 in (0..req.count).step_by(group) {
                if cancel.load(Ordering::Relaxed) {
                    break;
                }
                let n = group.min(req.count - g0);
                let mut noises = vec![];
                for i in 0..n {
                    let seed = req.seed + (g0 + i) as u64;
                    noises.push(Tensor::from_vec(rng::randn(seed, c * lh * lw), (1, c, lh, lw), &self.dev)?);
                }
                let noise = Tensor::cat(&noises, 0)?;
                let mut lat = match &init {
                    Some(x0) => {
                        let s0 = sig[start] as f64;
                        (x0.to_dtype(DType::F32)?.broadcast_mul(&Tensor::ones((n, 1, 1, 1), DType::F32, &self.dev)?)?.affine(1.0 - s0, 0.0)? + (&noise * s0)?)?
                            .to_dtype(ldt)?
                    }
                    None => noise.to_dtype(ldt)?,
                };
                on(Event::Phase("sampling"));
                for i in 0..n {
                    on(Event::Step { index: g0 + i, count: req.count, step: 0, total });
                }
                for k in start..steps {
                    if cancel.load(Ordering::Relaxed) {
                        break;
                    }
                    let batch = if b > 1 { Tensor::cat(&vec![&lat; b], 0)? } else { lat.clone() };
                    let pred = self.model.velocity(&batch, sig[k])?;
                    let v = if use_cfg {
                        let (p, ng) = (pred.narrow(0, 0, n)?, pred.narrow(0, n, n)?);
                        (&ng + ((p - &ng)? * req.guidance as f64)?)?
                    } else {
                        pred
                    };
                    let dt = sig[k + 1] - sig[k];
                    lat = (lat.to_dtype(DType::F32)? + (v.to_dtype(DType::F32)? * dt as f64)?)?.to_dtype(ldt)?;
                    if let (Some(m), Some(x0)) = (&mask, &init) {
                        let s1 = sig[k + 1] as f64;
                        let keep = (x0.to_dtype(DType::F32)?.broadcast_mul(&Tensor::ones((n, 1, 1, 1), DType::F32, &self.dev)?)?.affine(1.0 - s1, 0.0)?
                            + (&noise * s1)?)?
                            .to_dtype(ldt)?;
                        lat = (lat.broadcast_mul(m)? + keep.broadcast_mul(&(1.0 - m)?)?)?;
                    }
                    let done = k - start + 1;
                    for i in 0..n {
                        on(Event::Step { index: g0 + i, count: req.count, step: done, total });
                    }
                    if req.preview_every > 0 && done < total && done % req.preview_every == 0 {
                        let x0 = (lat.to_dtype(DType::F32)? - (v.to_dtype(DType::F32)? * sig[k + 1] as f64)?)?;
                        for i in 0..n {
                            on(Event::Preview { index: g0 + i, image: to_image(&self.model.preview(&x0.narrow(0, i, 1)?)?.squeeze(0)?)? });
                        }
                    }
                }
                if cancel.load(Ordering::Relaxed) {
                    break;
                }
                on(Event::Phase("decoding"));
                for i in 0..n {
                    let img = match self.model.decode(&lat.narrow(0, i, 1)?, Some(cancel)) {
                        Ok(img) => img,
                        Err(_) if cancel.load(Ordering::Relaxed) => break,
                        Err(e) => return Err(e),
                    };
                    on(Event::Sample { index: g0 + i, image: to_image(&img.squeeze(0)?).context("to image")? });
                }
            }
            return Ok(());
        }

        for index in 0..req.count {
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            let seed = req.seed + index as u64;
            let noise = Tensor::from_vec(rng::randn(seed, c * lh * lw), (1, c, lh, lw), &self.dev)?;
            let mut lat = match &init {
                Some(x0) => {
                    let s0 = sig[start] as f64;
                    ((x0.to_dtype(DType::F32)? * (1.0 - s0))? + (&noise * s0)?)?.to_dtype(ldt)?
                }
                None => noise.to_dtype(ldt)?,
            };
            on(Event::Phase("sampling"));
            on(Event::Step { index, count: req.count, step: 0, total });
            for k in start..steps {
                if cancel.load(Ordering::Relaxed) {
                    break;
                }
                let batch = if b > 1 { lat.repeat((b, 1, 1, 1))? } else { lat.clone() };
                let pred = self.model.velocity(&batch, sig[k])?;
                let v = if use_cfg {
                    let (p, n) = (pred.narrow(0, 0, 1)?, pred.narrow(0, 1, 1)?);
                    (&n + ((p - &n)? * req.guidance as f64)?)?
                } else {
                    pred
                };
                let dt = sig[k + 1] - sig[k];
                lat = (lat.to_dtype(DType::F32)? + (v.to_dtype(DType::F32)? * dt as f64)?)?.to_dtype(ldt)?;
                if let (Some(m), Some(x0)) = (&mask, &init) {
                    let s1 = sig[k + 1] as f64;
                    let keep = ((x0.to_dtype(DType::F32)? * (1.0 - s1))? + (&noise * s1)?)?.to_dtype(ldt)?;
                    lat = (lat.broadcast_mul(m)? + keep.broadcast_mul(&(1.0 - m)?)?)?;
                }
                let done = k - start + 1;
                on(Event::Step { index, count: req.count, step: done, total });
                if req.preview_every > 0 && done < total && done % req.preview_every == 0 {
                    // x0 estimate (the clean image the model is heading to), decoded small
                    let x0 = (lat.to_dtype(DType::F32)? - (v.to_dtype(DType::F32)? * sig[k + 1] as f64)?)?;
                    on(Event::Preview { index, image: to_image(&self.model.preview(&x0)?.squeeze(0)?)? });
                }
            }
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            if let Ok(p) = std::env::var("SAVE_LATENT") {
                lat.save_safetensors("lat", &p)?; // debug: the final latent (krea-vae-compare)
            }
            on(Event::Phase("decoding"));
            let img = match self.model.decode(&lat, Some(cancel)) {
                Ok(img) => img,
                Err(_) if cancel.load(Ordering::Relaxed) => break,
                Err(e) => return Err(e),
            };
            on(Event::Sample { index, image: to_image(&img.squeeze(0)?).context("to image")? });
        }
        Ok(())
    }
}

/// Holds the slot while a family change frees the old model before loading the new one.
struct Placeholder(Device);

impl DiffusionModel for Placeholder {
    fn kind(&self) -> ModelKind {
        ModelKind::new(crate::model::Family::Qwen21, crate::model::Variant::Base)
    }
    fn device(&self) -> &Device {
        &self.0
    }
    fn latent_channels(&self) -> usize {
        0
    }
    fn latent_scale(&self) -> usize {
        1
    }
    fn align(&self) -> usize {
        1
    }
    fn latent_dtype(&self) -> DType {
        DType::F32
    }
    fn defaults(&self) -> Defaults {
        Defaults { steps: 1, guidance: 1.0 }
    }
    fn set_variant(&mut self, _: crate::model::Variant) -> Result<()> {
        anyhow::bail!("no model loaded")
    }
    fn set_lora(&mut self, _: Option<&str>, _: f64) -> Result<usize> {
        anyhow::bail!("no model loaded")
    }
    fn encode_prompt(&mut self, _: &str) -> Result<Tensor> {
        anyhow::bail!("no model loaded")
    }
    fn begin(&mut self, _: &[Tensor]) -> Result<()> {
        anyhow::bail!("no model loaded")
    }
    fn velocity(&self, _: &Tensor, _: f32) -> Result<Tensor> {
        anyhow::bail!("no model loaded")
    }
    fn sigmas(&self, _: usize, _: usize, _: bool) -> Vec<f32> {
        vec![]
    }
    fn tokens(&self, _: usize, _: usize) -> usize {
        0
    }
    fn decode(&self, _: &Tensor, _: Option<&AtomicBool>) -> Result<Tensor> {
        anyhow::bail!("no model loaded")
    }
    fn preview(&self, _: &Tensor) -> Result<Tensor> {
        anyhow::bail!("no model loaded")
    }
}
