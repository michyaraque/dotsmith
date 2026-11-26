//! The contract every model family implements, and the identifiers used to pick one.
//!
//! The generation loop (engine.rs) only talks to `DiffusionModel`: it never knows which DiT, VAE, text encoder or
//! schedule sits behind it. A new family = a folder under models/ that implements this trait + an entry in
//! `Family` and `load_model`.
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use std::sync::atomic::AtomicBool;

/// A model family (architecture + its own text encoder, VAE and schedule).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Family {
    Qwen21,
    Krea2,
    LiliPixel,
    RdFlow,
}

impl Family {
    pub const ALL: [Family; 4] = [Family::Qwen21, Family::Krea2, Family::LiliPixel, Family::RdFlow];

    /// Folder name under models/ (training runs) and the `base` field of version.json.
    pub fn id(self) -> &'static str {
        match self {
            Family::Qwen21 => "qwen21",
            Family::Krea2 => "krea2",
            Family::LiliPixel => "lilipixel",
            Family::RdFlow => "rdflow",
        }
    }

    pub fn from_id(id: &str) -> Option<Family> {
        Family::ALL.into_iter().find(|f| f.id() == id)
    }

    /// Names, files, sources and defaults come from the catalog (models.json).
    pub fn spec(self) -> crate::manifest::FamilySpec {
        crate::manifest::family(self)
    }

    pub fn label(self) -> String {
        self.spec().label
    }

    /// Whether the catalog gives the family a few-step variant.
    pub fn has_turbo(self) -> bool {
        !self.spec().variants.turbo.name.is_empty()
    }
}

/// Which checkpoint of a family: the full model or its few-step distillation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Variant {
    Base,
    Turbo,
}

/// One loadable base model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ModelKind {
    pub family: Family,
    pub variant: Variant,
}

impl ModelKind {
    pub const fn new(family: Family, variant: Variant) -> Self {
        Self { family, variant }
    }

    /// Default steps from the catalog (0: not set there).
    pub fn spec_steps(self) -> usize {
        let v = self.family.spec().variants;
        if self.turbo() { v.turbo.steps } else { v.base.steps }
    }

    pub fn turbo(self) -> bool {
        self.variant == Variant::Turbo
    }

    pub fn label(self) -> String {
        let v = self.family.spec().variants;
        match self.variant {
            Variant::Base => v.base.label,
            Variant::Turbo => v.turbo.label,
        }
    }

    pub fn all() -> Vec<ModelKind> {
        Family::ALL
            .into_iter()
            .flat_map(|f| {
                let turbo = f.has_turbo().then(|| ModelKind::new(f, Variant::Turbo));
                std::iter::once(ModelKind::new(f, Variant::Base)).chain(turbo)
            })
            .collect()
    }

    pub fn from_label(label: &str) -> Option<ModelKind> {
        ModelKind::all().into_iter().find(|k| k.label() == label)
    }
}

/// Sampling defaults a family recommends for a variant (what the UI pre-fills).
#[derive(Clone, Copy, Debug)]
pub struct Defaults {
    pub steps: usize,
    /// Classic CFG scale (`uncond + g * (cond - uncond)`, 1 = off). Families with another formula convert inside.
    pub guidance: f32,
}

/// Everything the generation loop needs from a model family.
pub trait DiffusionModel: Send {
    fn kind(&self) -> ModelKind;
    fn device(&self) -> &Device;

    /// Latent geometry: channels and image pixels per latent pixel (also the alignment of width / height).
    fn latent_channels(&self) -> usize;
    fn latent_scale(&self) -> usize;
    /// Pixel alignment of the canvas (latent scale x patch size).
    fn align(&self) -> usize;
    /// dtype the latents are kept in between steps (the reference implementations differ).
    fn latent_dtype(&self) -> DType;
    fn defaults(&self) -> Defaults;

    /// Swap the base checkpoint inside the family (base <-> turbo), keeping text encoder, VAE and LoRA.
    fn set_variant(&mut self, variant: Variant) -> Result<()>;
    /// Attach (or with None remove) an ai-toolkit LoRA trained for this family; returns the modules it touched.
    fn set_lora(&mut self, path: Option<&str>, multiplier: f64) -> Result<usize>;

    /// One prompt -> the family's conditioning tensor (the engine caches it per prompt).
    fn encode_prompt(&mut self, prompt: &str) -> Result<Tensor>;
    /// Free the text encoder's VRAM before a big canvas; the next `encode_prompt` reloads it.
    fn release_text_encoder(&mut self) {}

    /// How many images of one request may be denoised together in one forward pass (lockstep). 1 = one after the
    /// other. Weight streaming pays one weight upload per forward, so batching shares it across the images.
    fn lockstep(&self) -> usize {
        1
    }

    /// Per-request conditioning prepared once from the encoded prompts (row 0 = cond, row 1 = uncond if CFG).
    fn begin(&mut self, prompts: &[Tensor]) -> Result<()>;
    /// Velocity for `latents` [b, c, h, w] at schedule time `sigma` (one row per prepared prompt).
    fn velocity(&self, latents: &Tensor, sigma: f32) -> Result<Tensor>;
    /// Schedule (with the terminal 0) for a canvas of `tokens` image tokens; `img2img` = SDEdit start.
    fn sigmas(&self, steps: usize, tokens: usize, img2img: bool) -> Vec<f32>;
    /// Image tokens of a latent [h, w] (what the schedule's resolution shift is measured in).
    fn tokens(&self, lh: usize, lw: usize) -> usize;

    /// RGB(A) image in [-1, 1] [1, 3|4, H, W] -> normalized latents (img2img). Not every family has an encoder yet.
    fn encode_image(&mut self, _image: &Tensor) -> Result<Tensor> {
        anyhow::bail!("{} does not support a starting image yet", self.kind().label())
    }
    /// Latents -> image [1, 3|4, H, W] f32 in [-1, 1].
    fn decode(&self, latents: &Tensor, cancel: Option<&AtomicBool>) -> Result<Tensor>;
    /// Quick small decode of an x0 estimate for live previews.
    fn preview(&self, x0: &Tensor) -> Result<Tensor>;

    /// A family with a sampler of its own (not the engine's velocity Euler loop) runs the whole request here;
    /// None = use the engine's loop.
    fn sample_own(&mut self, _req: &crate::engine::Request, _cancel: &AtomicBool, _on: &mut dyn FnMut(crate::engine::Event)) -> Option<Result<()>> {
        None
    }
}

/// Where a family's files live on this machine.
#[derive(Clone, Debug)]
pub struct ModelPaths {
    pub family: Family,
    pub base: String,
    pub turbo: String,
    pub text_encoder: String,
    pub tokenizer: String,
    pub vae: String,
}

impl ModelPaths {
    /// Missing files keep the path where they are expected; `require` reports them.
    pub fn default_for(family: Family) -> Result<Self> {
        let files = crate::locations::resolve(family);
        // the preferred precision's file when it is on disk, else the default file, else any other precision on disk
        // (someone may have downloaded only the bf16 one), else where the default file is expected
        let want = crate::core::memory::policy().precision;
        let get = |role: &str, variant: Option<&str>| {
            let same = |f: &&crate::locations::FileCheck| f.role == role && f.variant.as_deref() == variant;
            let of = |p: Option<&str>| files.iter().filter(same).find(|f| f.precision.as_deref() == p);
            want.as_deref()
                .and_then(|w| of(Some(w)).filter(|f| f.found))
                .or_else(|| of(None).filter(|f| f.found))
                .or_else(|| files.iter().filter(same).find(|f| f.found))
                .or_else(|| of(None))
                .map(|f| f.path.to_string_lossy().to_string())
                .unwrap_or_default()
        };
        Ok(Self {
            family,
            base: get("diffusion model", None),
            turbo: get("diffusion model", Some("turbo")),
            text_encoder: get("text encoder", None),
            tokenizer: get("tokenizer", None),
            vae: get("VAE", None),
        })
    }

    /// Bytes on disk a load reads: the variant's DiT, plus the text encoder and VAE unless only the DiT is swapped.
    pub fn load_bytes(&self, variant: Variant, dit_only: bool) -> u64 {
        let size = |p: &str| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
        size(self.checkpoint(variant)) + if dit_only { 0 } else { size(&self.text_encoder) + size(&self.vae) }
    }

    pub fn require(&self, variant: Variant) -> Result<()> {
        let roles: Vec<String> = crate::manifest::family(self.family).files.into_iter().map(|f| f.role).collect();
        let need: Vec<(&str, &str)> =
            [("diffusion model", self.checkpoint(variant)), ("text encoder", &self.text_encoder), ("VAE", &self.vae), ("tokenizer", &self.tokenizer)]
                .into_iter()
                .filter(|(role, _)| roles.iter().any(|r| r == role))
                .collect();
        let missing: Vec<String> = need
            .iter()
            .filter(|(_, p)| p.is_empty() || !std::path::Path::new(p).is_file())
            .map(|(role, p)| format!("{role}: {}", if p.is_empty() { "(no location configured)" } else { p }))
            .collect();
        if missing.is_empty() {
            Ok(())
        } else {
            anyhow::bail!(
                "model files missing (set the models folder in Settings):
{}",
                missing.join(
                    "
"
                )
            )
        }
    }

    pub fn checkpoint(&self, variant: Variant) -> &str {
        match variant {
            Variant::Base => &self.base,
            Variant::Turbo => &self.turbo,
        }
    }
}

/// The family registry: build the model for a kind.
pub fn load_model(kind: ModelKind, paths: ModelPaths, dev: &Device) -> Result<Box<dyn DiffusionModel>> {
    Ok(match kind.family {
        Family::Qwen21 => Box::new(crate::models::qwen21::Qwen21::load(paths, kind.variant, dev)?),
        Family::Krea2 => Box::new(crate::models::krea2::Krea2::load(paths, kind.variant, dev)?),
        Family::LiliPixel => Box::new(crate::models::lilipixel::LiliPixel::load(paths, kind.variant, dev)?),
        Family::RdFlow => Box::new(crate::models::rdflow::RdFlow::load(paths, kind.variant, dev)?),
    })
}
