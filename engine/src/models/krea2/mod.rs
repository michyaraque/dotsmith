//! Krea 2 (Raw / Turbo): single-stream MMDiT + Qwen-Image VAE (8x, 16 ch, RGB) + Qwen3-VL-4B stacked text layers.
//! Sampling follows krea2/src/pipeline.py: fp32 latents between steps, bf16 model input and timestep, Euler on the
//! resolution-shifted schedule (Turbo: fixed mu 1.15, 8 steps, no CFG). The reference CFG `cond + g (cond - uncond)`
//! equals the classic `uncond + (g + 1) (cond - uncond)` the engine applies, so the UI's guidance = Krea's g + 1.
pub mod dit;
pub mod sched;
pub mod text;
pub mod vae;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use std::sync::atomic::AtomicBool;

use crate::core::memory::{Needs, Stack, StackPlan};
use crate::model::{Defaults, DiffusionModel, Family, ModelKind, ModelPaths, Variant};
use crate::text::qwen3vl::TextEncoder;
use dit::{Krea, KreaText};
use vae::{VaeEnc, VaeQi};

pub struct Krea2 {
    dev: Device,
    paths: ModelPaths,
    variant: Variant,
    te: Option<TextEncoder>,
    dit: Option<Krea>,
    vae: VaeQi,
    enc: Option<VaeEnc>, // loaded on the first img2img request
    lora: Option<(String, f64)>,
    text: Option<KreaText>,
    /// memory plan of (DiT, text encoder), made at load from core::memory's policy and the free VRAM
    plan: (StackPlan, StackPlan),
}

/// Working memory at the largest canvas (attention chunks, SwiGLU, VAE tiles), MB: measured ~2.4 GB in low VRAM mode
/// and ~3.5 GB in normal mode at 1536 x 1024 (1 GB attention score chunks).
fn reserve_mb() -> u64 {
    if crate::core::low_vram() { 2500 } else { 4000 }
}

impl Krea2 {
    /// Place DiT and text encoder for the free VRAM (core::memory). Streams only when the model does not fit (or
    /// the policy says so); then both share the GPU at once, nothing is reloaded per prompt.
    fn plan(paths: &ModelPaths, variant: Variant, dev: &Device) -> Result<(StackPlan, StackPlan)> {
        let (dit_stack, dit_fixed, cold) = Krea::memory(paths.checkpoint(variant))?;
        let te_stack = TextEncoder::stack(&paths.text_encoder)?;
        let vae = std::fs::metadata(&paths.vae).map(|m| m.len()).unwrap_or(0);
        let stacks: Vec<Stack> = [dit_stack.clone(), te_stack.clone()].into_iter().flatten().collect();
        let needs = Needs { fixed: dit_fixed + vae, cold, reserve_mb: reserve_mb() };
        let free = crate::gpu::available_mb(dev).unwrap_or(0);
        let plans = crate::core::memory::plan(&stacks, &needs, free);
        let mut it = plans.into_iter();
        let dit = if dit_stack.is_some() { it.next().unwrap() } else { StackPlan::GPU };
        let te = if te_stack.is_some() { it.next().unwrap() } else { StackPlan::GPU };
        Ok((dit, te))
    }

    pub fn load(paths: ModelPaths, variant: Variant, dev: &Device) -> Result<Self> {
        crate::core::memory::clear_report();
        let plan = Self::plan(&paths, variant, dev)?;
        let te = TextEncoder::load_planned(&paths.text_encoder, &paths.tokenizer, dev, plan.1)?;
        // low VRAM: the DiT loads in begin(), after the prompts are encoded and the text encoder has left the GPU.
        // A streamed DiT (~1-2 GB of VRAM) loads now and stays: dropping it would re-read gigabytes per prompt
        let dit = if crate::core::low_vram() && !plan.0.stream { None } else { Some(Krea::load_planned(paths.checkpoint(variant), dev, plan.0)?) };
        let vae = VaeQi::load(&paths.vae, dev)?;
        Ok(Self { dev: dev.clone(), paths, variant, te: Some(te), dit, vae, enc: None, lora: None, text: None, plan })
    }

    /// Load the DiT if it is not on the GPU (low VRAM mode drops it while the text encoder runs) and re-apply the LoRA.
    fn ensure_dit(&mut self) -> Result<()> {
        if self.dit.is_none() {
            let mut dit = Krea::load_planned(self.paths.checkpoint(self.variant), &self.dev, self.plan.0)?;
            if let Some((p, m)) = self.lora.clone() {
                dit.set_lora(Some(&p), m)?;
            }
            self.dit = Some(dit);
        }
        Ok(())
    }
}

impl DiffusionModel for Krea2 {
    fn kind(&self) -> ModelKind {
        ModelKind::new(Family::Krea2, self.variant)
    }
    fn device(&self) -> &Device {
        &self.dev
    }
    fn latent_channels(&self) -> usize {
        16
    }
    fn latent_scale(&self) -> usize {
        8
    }
    fn align(&self) -> usize {
        16 // VAE 8 x patch 2
    }
    fn latent_dtype(&self) -> DType {
        DType::F32
    }
    fn defaults(&self) -> Defaults {
        match self.variant {
            // classic CFG 3.5 = what ai-toolkit's training samples use (it passes guidance_scale - 1 to Krea's formula)
            Variant::Base => Defaults { steps: 30, guidance: 3.5 },
            Variant::Turbo => Defaults { steps: 8, guidance: 1.0 },
        }
    }

    fn set_variant(&mut self, variant: Variant) -> Result<()> {
        if variant == self.variant {
            return Ok(());
        }
        if self.dit.is_none() {
            self.variant = variant; // low VRAM, DiT not loaded: begin() loads the right one
            return Ok(());
        }
        self.dit = None; // free the 13 GB DiT before loading the other one
        self.text = None;
        self.dev.synchronize()?;
        self.dit = Some(Krea::load_planned(self.paths.checkpoint(variant), &self.dev, self.plan.0)?);
        self.variant = variant;
        if let Some((p, m)) = self.lora.clone() {
            self.dit.as_mut().unwrap().set_lora(Some(&p), m)?;
        }
        Ok(())
    }

    fn set_lora(&mut self, path: Option<&str>, multiplier: f64) -> Result<usize> {
        self.lora = path.map(|p| (p.to_string(), multiplier));
        match self.dit.as_mut() {
            Some(dit) => dit.set_lora(path, multiplier),
            None => Ok(0), // low VRAM: applied when begin() loads the DiT
        }
    }

    fn encode_prompt(&mut self, prompt: &str) -> Result<Tensor> {
        if self.te.is_none() && crate::core::low_vram() && self.dit.as_ref().is_some_and(|d| !d.streaming()) {
            self.dit = None; // low VRAM: the text encoder and the DiT never share the GPU
            self.text = None;
            self.dev.synchronize()?;
        }
        if self.te.is_none() {
            self.te = Some(TextEncoder::load_planned(&self.paths.text_encoder, &self.paths.tokenizer, &self.dev, self.plan.1)?);
        }
        text::encode(self.te.as_ref().unwrap(), prompt)
    }

    fn release_text_encoder(&mut self) {
        if self.te.as_ref().is_some_and(|t| t.streaming()) {
            return; // streamed: ~0.3 GB of slots, cheaper to keep than to rebuild per prompt
        }
        if self.te.take().is_some() {
            let _ = self.dev.synchronize();
        }
    }

    fn begin(&mut self, prompts: &[Tensor]) -> Result<()> {
        if crate::core::low_vram() {
            self.release_text_encoder(); // (kept when it streams)
        }
        self.ensure_dit()?;
        self.text = Some(self.dit.as_ref().context("DiT not loaded")?.prepare_text(prompts)?);
        Ok(())
    }

    fn lockstep(&self) -> usize {
        match &self.dit {
            // the policy's images per forward while the DiT streams (each forward = one full weight upload)
            Some(d) if d.streaming() => crate::core::memory::policy().batch_max,
            _ => 1,
        }
    }

    fn velocity(&self, latents: &Tensor, sigma: f32) -> Result<Tensor> {
        let text = self.text.as_ref().context("begin() not called")?;
        let b = latents.dim(0)?;
        let t = half::bf16::from_f32(sigma).to_f32();
        self.dit.as_ref().context("DiT not loaded")?.step(text, latents, &vec![t; b])
    }

    fn sigmas(&self, steps: usize, tokens: usize, _img2img: bool) -> Vec<f32> {
        let mu = if self.variant == Variant::Turbo { Some(1.15) } else { None };
        sched::timesteps(tokens, steps, mu)
    }

    fn tokens(&self, lh: usize, lw: usize) -> usize {
        (lh / 2) * (lw / 2)
    }

    /// img2img: image [1, 3|4, H, W] in [-1, 1]. Krea's VAE is RGB only, so an alpha channel is flattened onto flat
    /// magenta (the v1 LoRA's background; Dotsmith already flattens onto the version's own background before this).
    fn encode_image(&mut self, image: &Tensor) -> Result<Tensor> {
        if self.enc.is_none() {
            self.enc = Some(VaeEnc::load(&self.paths.vae, &self.dev)?);
        }
        let (_, c, _, _) = image.dims4()?;
        let rgb = if c == 4 {
            let a = ((image.narrow(1, 3, 1)? + 1.0)? / 2.0)?; // [0, 1]
            let bg = Tensor::from_slice(&[1f32, -1.0, 1.0], (1, 3, 1, 1), &self.dev)?.to_dtype(image.dtype())?;
            (image.narrow(1, 0, 3)?.broadcast_mul(&a)? + bg.broadcast_mul(&(1.0 - &a)?)?)?
        } else {
            image.clone()
        };
        self.enc.as_ref().unwrap().encode(&rgb)
    }

    fn decode(&self, latents: &Tensor, cancel: Option<&AtomicBool>) -> Result<Tensor> {
        self.vae.decode_cancel(latents, cancel)
    }

    fn preview(&self, x0: &Tensor) -> Result<Tensor> {
        let (_, _, lh, lw) = x0.dims4()?;
        let k = (lh.max(lw) / 32).max(1);
        let small = if k > 1 && lh % k == 0 && lw % k == 0 { x0.to_dtype(DType::F32)?.avg_pool2d(k)? } else { x0.clone() };
        self.vae.decode(&small)
    }
}
