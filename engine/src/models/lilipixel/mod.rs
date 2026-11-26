//! LiliPixel V1 (Pinkstack/LiliPixel-V1-190M): a 190M pixel-space flow DiT that draws 16, 32 or 64 px RGBA sprites
//! directly (no VAE), conditioned on MongoDB/voyage-4-nano. Euler flow matching, 30 steps, CFG 3 against its learned
//! "no caption" (a negative prompt is not used). One variant, no LoRAs.
pub mod dit;
pub mod text;

use anyhow::{Result, bail};
use candle_core::{DType, Device, Tensor};
use std::sync::atomic::AtomicBool;

use crate::model::{Defaults, DiffusionModel, Family, ModelKind, ModelPaths, Variant};
use dit::{Caption, Dit, TEXT_LEN};
use text::Voyage;

pub struct LiliPixel {
    dev: Device,
    te: Voyage,
    dit: Dit,
    caps: Vec<Caption>,
}

impl LiliPixel {
    pub fn load(paths: ModelPaths, variant: Variant, dev: &Device) -> Result<Self> {
        if variant != Variant::Base {
            bail!("LiliPixel has a single variant");
        }
        crate::core::memory::clear_report();
        Ok(Self { dev: dev.clone(), te: Voyage::load(&paths.text_encoder, &paths.tokenizer, dev)?, dit: Dit::load(&paths.base, dev)?, caps: vec![] })
    }
}

impl DiffusionModel for LiliPixel {
    fn kind(&self) -> ModelKind {
        ModelKind::new(Family::LiliPixel, Variant::Base)
    }
    fn device(&self) -> &Device {
        &self.dev
    }
    fn latent_channels(&self) -> usize {
        4
    }
    fn latent_scale(&self) -> usize {
        1
    }
    fn align(&self) -> usize {
        16
    }
    fn latent_dtype(&self) -> DType {
        DType::F32
    }
    fn defaults(&self) -> Defaults {
        Defaults { steps: 30, guidance: 3.0 }
    }
    fn set_variant(&mut self, variant: Variant) -> Result<()> {
        if variant != Variant::Base {
            bail!("LiliPixel has a single variant");
        }
        Ok(())
    }
    fn set_lora(&mut self, path: Option<&str>, _multiplier: f64) -> Result<usize> {
        match path {
            Some(p) if !p.is_empty() => bail!("LiliPixel does not take LoRAs ({p})"),
            _ => Ok(0),
        }
    }

    /// [96, 2048 + 1]: the encoder features with the real-token mask as the last column.
    fn encode_prompt(&mut self, prompt: &str) -> Result<Tensor> {
        let (feats, mask) = self.te.encode(prompt, TEXT_LEN)?;
        Ok(Tensor::cat(&[&feats, &mask.unsqueeze(1)?], 1)?)
    }

    fn begin(&mut self, prompts: &[Tensor]) -> Result<()> {
        self.caps.clear();
        for (i, p) in prompts.iter().enumerate() {
            // the second row is the CFG row: the learned "no caption", whatever the negative prompt said
            self.caps.push(if i == 0 {
                let w = p.dim(1)? - 1;
                self.dit.caption(Some((&p.narrow(1, 0, w)?, &p.narrow(1, w, 1)?.squeeze(1)?)))?
            } else {
                self.dit.caption(None)?
            });
        }
        Ok(())
    }

    fn velocity(&self, latents: &Tensor, sigma: f32) -> Result<Tensor> {
        let b = latents.dim(0)?;
        let rows = self.caps.len().max(1);
        let per = (b / rows).max(1);
        let caps: Vec<&Caption> = (0..b).map(|i| &self.caps[(i / per).min(self.caps.len() - 1)]).collect();
        self.dit.forward(&latents.to_dtype(DType::F32)?, sigma, &caps)
    }

    /// diffusers' FlowMatchEulerDiscreteScheduler with shift 1: linspace(1, 1/1000) then 0.
    fn sigmas(&self, steps: usize, _tokens: usize, _img2img: bool) -> Vec<f32> {
        let n = steps.max(1);
        let mut s: Vec<f32> = (0..n).map(|i| if n == 1 { 1.0 } else { 1.0 + (0.001 - 1.0) * i as f32 / (n - 1) as f32 }).collect();
        s.push(0.0);
        s
    }

    fn tokens(&self, lh: usize, lw: usize) -> usize {
        lh * lw
    }

    /// Pixel space: the picture is its own latent.
    fn encode_image(&mut self, image: &Tensor) -> Result<Tensor> {
        Ok(image.to_dtype(DType::F32)?)
    }

    fn decode(&self, latents: &Tensor, _cancel: Option<&AtomicBool>) -> Result<Tensor> {
        Ok(latents.to_dtype(DType::F32)?.clamp(-1f32, 1f32)?)
    }

    fn preview(&self, x0: &Tensor) -> Result<Tensor> {
        self.decode(x0, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::weights::Weights;

    fn max_diff(a: &Tensor, b: &Tensor) -> f32 {
        (a - b).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap()
    }

    /// DOTSMITH_LILI_REF=<parity/dump_ref_lilipixel.py output>; files from DOTSMITH_MODELS.
    #[test]
    fn matches_the_reference() {
        let Ok(r) = std::env::var("DOTSMITH_LILI_REF") else { return };
        let dev = Device::new_cuda(0).unwrap();
        let paths = ModelPaths::default_for(Family::LiliPixel).unwrap();
        let mut m = LiliPixel::load(paths, Variant::Base, &dev).unwrap();
        let rf = Weights::open(&r).unwrap();
        let get = |n: &str| rf.tensor(n, &dev).unwrap();
        let prompt = "a red health potion in a round glass bottle";

        let (ids, n) = m.te.tokens(prompt, TEXT_LEN).unwrap();
        let want: Vec<i64> = get("ids").flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(ids.iter().map(|&i| i as i64).collect::<Vec<_>>(), want, "token ids");
        let (feats, _) = m.te.encode(prompt, TEXT_LEN).unwrap();
        let fd = max_diff(&feats.narrow(0, 0, n).unwrap(), &get("feats").squeeze(0).unwrap().narrow(0, 0, n).unwrap());
        println!("text features: max |diff| {fd:.2e}");
        assert!(fd < 2e-2, "text features differ by {fd}");

        let cond = m.encode_prompt(prompt).unwrap();
        m.begin(&[cond.clone(), cond]).unwrap();
        let x = get("x");
        let v = m.velocity(&Tensor::cat(&[&x, &x], 0).unwrap(), 0.7).unwrap();
        let (dc, du) = (max_diff(&v.narrow(0, 0, 1).unwrap(), &get("v_cond")), max_diff(&v.narrow(0, 1, 1).unwrap(), &get("v_null")));
        println!("velocity: cond {dc:.2e}, no caption {du:.2e}");
        assert!(dc < 2e-2 && du < 2e-2);

        let sig = m.sigmas(30, 0, false);
        let mut lat = get("noise");
        for k in 0..30 {
            let v = m.velocity(&Tensor::cat(&[&lat, &lat], 0).unwrap(), sig[k]).unwrap();
            let (c, u) = (v.narrow(0, 0, 1).unwrap(), v.narrow(0, 1, 1).unwrap());
            let g = (&u + ((c - &u).unwrap() * 3.0).unwrap()).unwrap();
            lat = (lat + (g * (sig[k + 1] - sig[k]) as f64).unwrap()).unwrap();
        }
        let fd = max_diff(&lat, &get("final"));
        let mean = (&lat - get("final")).unwrap().abs().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap();
        println!("30 steps: max |diff| {fd:.2e}, mean {mean:.2e} (pixels are 2/255 = 7.8e-3 apart)");
        assert!(mean < 7.8e-3, "mean difference {mean}");
    }
}
