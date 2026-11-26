//! Qwen-Image 2.1 (base / turbo): single-stream DiT + RGBA VAE (16x, 64 ch) + Qwen3-VL-8B last hidden layer.
//! Numerics follow the PyTorch (ai-toolkit) sampling it was checked against: bf16 latents between steps,
//! bf16-rounded timesteps (t * 1000 rounded, / 1000 rounded), text K/V cached once per request (dit::TextCache).
pub mod dit;
pub mod sched;
pub mod vae;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use std::sync::atomic::AtomicBool;

use crate::core::memory::{Needs, Stack, StackPlan};
use crate::model::{Defaults, DiffusionModel, Family, ModelKind, ModelPaths, Variant};
use crate::text::qwen3vl::TextEncoder;
use dit::{TextCache, Transformer};
use vae::Vae;

fn bf16(v: f32) -> f32 {
    half::bf16::from_f32(v).to_f32()
}

pub struct Qwen21 {
    dev: Device,
    paths: ModelPaths,
    variant: Variant,
    te: Option<TextEncoder>, // dropped from the GPU during big canvases, reloaded (~2 s) for the next new prompt
    tr: Option<Transformer>, // None only while swapping base <-> turbo
    vae: Vae,
    lora: Option<(String, f64)>,
    cache: Option<TextCache>,
    /// memory plan of (DiT, text encoder), made at load from core::memory's policy and the free VRAM
    plan: (StackPlan, StackPlan),
}

impl Qwen21 {
    fn plan(paths: &ModelPaths, variant: Variant, dev: &Device) -> Result<(StackPlan, StackPlan)> {
        let (dit_stack, dit_fixed) = Transformer::memory(paths.checkpoint(variant))?;
        let te_stack = TextEncoder::stack(&paths.text_encoder)?;
        let vae = std::fs::metadata(&paths.vae).map(|m| m.len()).unwrap_or(0);
        let stacks: Vec<Stack> = [dit_stack.clone(), te_stack.clone()].into_iter().flatten().collect();
        // working memory: 1024 px canvases at x16 are small (4096 tokens); the text K/V cache and VAE decode dominate
        let needs = Needs { fixed: dit_fixed + vae, cold: 0, reserve_mb: 3000 };
        let plans = crate::core::memory::plan(&stacks, &needs, crate::gpu::available_mb(dev).unwrap_or(0));
        let mut it = plans.into_iter();
        let dit = if dit_stack.is_some() { it.next().unwrap() } else { StackPlan::GPU };
        let te = if te_stack.is_some() { it.next().unwrap() } else { StackPlan::GPU };
        Ok((dit, te))
    }

    pub fn load(paths: ModelPaths, variant: Variant, dev: &Device) -> Result<Self> {
        crate::core::memory::clear_report();
        let plan = Self::plan(&paths, variant, dev)?;
        let te = TextEncoder::load_planned(&paths.text_encoder, &paths.tokenizer, dev, plan.1)?;
        let tr = Transformer::load_planned(paths.checkpoint(variant), dev, plan.0)?;
        let vae = Vae::load(&paths.vae, dev)?;
        Ok(Self { dev: dev.clone(), paths, variant, te: Some(te), tr: Some(tr), vae, lora: None, cache: None, plan })
    }
}

impl DiffusionModel for Qwen21 {
    fn kind(&self) -> ModelKind {
        ModelKind::new(Family::Qwen21, self.variant)
    }
    fn device(&self) -> &Device {
        &self.dev
    }
    fn latent_channels(&self) -> usize {
        64
    }
    fn latent_scale(&self) -> usize {
        16
    }
    fn align(&self) -> usize {
        16
    }
    fn latent_dtype(&self) -> DType {
        DType::BF16
    }
    fn defaults(&self) -> Defaults {
        match self.variant {
            Variant::Base => Defaults { steps: 25, guidance: 3.0 },
            Variant::Turbo => Defaults { steps: 8, guidance: 1.0 },
        }
    }

    fn set_variant(&mut self, variant: Variant) -> Result<()> {
        if variant == self.variant {
            return Ok(());
        }
        // drop the old DiT (7.3 GB) BEFORE loading the new one: with both resident plus the text encoder the 24 GB card
        // overflows into WDDM shared memory and every step (and the VAE decode) becomes ~15x slower, i.e. a "hang"
        self.tr = None;
        self.cache = None;
        self.dev.synchronize()?;
        self.tr = Some(Transformer::load_planned(self.paths.checkpoint(variant), &self.dev, self.plan.0)?);
        self.variant = variant;
        if let Some((p, m)) = self.lora.clone() {
            self.tr.as_mut().unwrap().set_lora(Some(&p), m)?;
        }
        Ok(())
    }

    fn set_lora(&mut self, path: Option<&str>, multiplier: f64) -> Result<usize> {
        let n = self.tr.as_mut().context("DiT not loaded")?.set_lora(path, multiplier)?;
        self.lora = path.map(|p| (p.to_string(), multiplier));
        Ok(n)
    }

    fn encode_prompt(&mut self, prompt: &str) -> Result<Tensor> {
        if self.te.is_none() {
            self.te = Some(TextEncoder::load_planned(&self.paths.text_encoder, &self.paths.tokenizer, &self.dev, self.plan.1)?);
        }
        self.te.as_ref().unwrap().encode(prompt)
    }

    fn release_text_encoder(&mut self) {
        if self.te.as_ref().is_some_and(|t| t.streaming()) {
            return; // streamed: a few hundred MB of slots, cheaper to keep than to rebuild per prompt
        }
        if self.te.take().is_some() {
            let _ = self.dev.synchronize();
        }
    }

    fn begin(&mut self, prompts: &[Tensor]) -> Result<()> {
        // right-pad to one length (the worker's batched CFG), valid mask over real tokens
        let l = prompts.iter().map(|r| r.dim(0).unwrap()).max().unwrap_or(1);
        let (mut padded, mut valid) = (vec![], vec![]);
        for r in prompts {
            let n = r.dim(0)?;
            padded.push(r.pad_with_zeros(0, 0, l - n)?);
            valid.extend((0..l).map(|i| (i < n) as u8));
        }
        let embeds = Tensor::stack(&padded, 0)?;
        let valid = Tensor::from_vec(valid, (prompts.len(), l), &self.dev)?;
        self.cache = Some(self.tr.as_ref().context("DiT not loaded")?.prefill(&embeds, &valid)?);
        Ok(())
    }

    fn velocity(&self, latents: &Tensor, sigma: f32) -> Result<Tensor> {
        let cache = self.cache.as_ref().context("begin() not called")?;
        let b = latents.dim(0)?;
        let t = bf16(bf16(sigma * 1000.0) / 1000.0);
        self.tr.as_ref().context("DiT not loaded")?.step(cache, latents, &vec![t; b])
    }

    fn sigmas(&self, steps: usize, tokens: usize, img2img: bool) -> Vec<f32> {
        if img2img && self.variant == Variant::Turbo {
            // sdedit_run.py: linspace + resolution shift (the turbo config ignores mu and the terminal stretch)
            let n = steps.max(1);
            let mut s: Vec<f32> = (0..n).map(|i| (1.0 + (1.0 / n as f64 - 1.0) * i as f64 / (n.max(2) - 1) as f64) as f32).collect();
            s.push(0.0);
            s
        } else {
            sched::sigmas(steps, tokens, self.variant == Variant::Turbo && !img2img)
        }
    }

    fn tokens(&self, lh: usize, lw: usize) -> usize {
        lh * lw
    }

    fn encode_image(&mut self, image: &Tensor) -> Result<Tensor> {
        let (_, c, h, w) = image.dims4()?;
        let x = if c == 3 { Tensor::cat(&[image, &Tensor::ones((1, 1, h, w), image.dtype(), image.device())?], 1)? } else { image.clone() };
        self.vae.encode(&x)
    }

    fn decode(&self, latents: &Tensor, cancel: Option<&AtomicBool>) -> Result<Tensor> {
        self.vae.decode_cancel(latents, cancel)
    }

    fn preview(&self, x0: &Tensor) -> Result<Tensor> {
        let (_, _, lh, lw) = x0.dims4()?;
        let k = (lh.max(lw) / 16).max(1);
        let small = if k > 1 && lh % k == 0 && lw % k == 0 { x0.to_dtype(DType::F32)?.avg_pool2d(k)? } else { x0.clone() };
        self.vae.decode(&small)
    }
}
