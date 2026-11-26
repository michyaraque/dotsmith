//! RD Flow 64 (Retro-Diffusion/rd-flow-64): a 271M model that draws 64 x 64 pixel art directly on the pixel grid, every
//! pixel an exact palette colour or transparent. Text: Qwen3-VL-4B's first 27 layers read through its final norm and
//! RMS-normalised per token (raw caption, no template). Its sampler is its own (rdflow/flow.py): x-prediction with
//! self-conditioning, CFG over most of the run, a palette taken from the model's guess at t = 0.6, then the pointer
//! head picks a palette slot for every pixel.
pub mod net;
pub mod palette;

use anyhow::{Result, bail};
use candle_core::{D, DType, Device, Tensor};
use image::RgbaImage;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::core::memory::{Needs, StackPlan};
use crate::core::rng;
use crate::engine::{Event, Request};
use crate::model::{Defaults, DiffusionModel, Family, ModelKind, ModelPaths, Variant};
use crate::text::qwen3vl::TextEncoder;
use net::{CHANNELS, Net, PIXELS, Palette, Row};

const TEXT_LEN: usize = 512;
const SHIFT: f32 = 3.0;
const GUIDE_UNTIL: f32 = 0.9;
const PALETTE_AT: f32 = 0.6;
const POINTER_FROM: f32 = 0.35;
const COMMIT_FROM: f32 = 0.55;
const COMMIT_THRESH: f32 = 0.9;
const MAX_COLOURS: usize = 64;
const MERGE: f64 = 0.05;
const SIZE: usize = 64;

pub struct RdFlow {
    dev: Device,
    te: TextEncoder,
    net: Net,
    cache: HashMap<String, Tensor>,
}

impl RdFlow {
    pub fn load(paths: ModelPaths, variant: Variant, dev: &Device) -> Result<Self> {
        if variant != Variant::Base {
            bail!("RD Flow 64 has a single variant");
        }
        crate::core::memory::clear_report();
        let te_stack = TextEncoder::stack(&paths.text_encoder)?;
        let net_bytes = std::fs::metadata(&paths.base).map(|m| m.len() * 2).unwrap_or(0);
        let plan = match &te_stack {
            Some(s) => crate::core::memory::plan(
                std::slice::from_ref(s),
                &Needs { fixed: net_bytes, cold: 0, reserve_mb: 2000 },
                crate::gpu::available_mb(dev).unwrap_or(0),
            )
            .into_iter()
            .next()
            .unwrap_or(StackPlan::GPU),
            None => StackPlan::GPU,
        };
        let te = TextEncoder::load_planned(&paths.text_encoder, &paths.tokenizer, dev, plan)?;
        Ok(Self { dev: dev.clone(), te, net: Net::load(&paths.base, dev)?, cache: HashMap::new() })
    }

    /// Caption -> features [l, 2560] f32: the final-norm hidden states RMS-normalised per token, rounded to bf16 as the
    /// model was trained on them.
    fn text(&mut self, caption: &str) -> Result<Tensor> {
        if let Some(t) = self.cache.get(caption) {
            return Ok(t.clone());
        }
        let h = self.te.encode_plain(caption, TEXT_LEN)?.to_dtype(DType::F32)?;
        let r = (h.sqr()?.mean_keepdim(D::Minus1)? + f32::EPSILON as f64)?.sqrt()?.recip()?;
        let t = h.broadcast_mul(&r)?.to_dtype(DType::BF16)?.to_dtype(DType::F32)?;
        if self.cache.len() > 32 {
            self.cache.clear();
        }
        self.cache.insert(caption.to_string(), t.clone());
        Ok(t)
    }

    /// [1 + k, 4]: transparent, then each palette colour, in model space.
    fn palette_space(&self, pal: &Palette) -> Result<Tensor> {
        let mut v = vec![0.0f32, 0.0, 0.0, -1.0];
        for c in &pal.rgb {
            v.extend_from_slice(&palette::to_model(*c));
        }
        Ok(Tensor::from_vec(v, (1 + pal.rgb.len(), CHANNELS), &self.dev)?)
    }

    fn pixels(x: &Tensor) -> Result<Vec<[f32; 4]>> {
        let v: Vec<f32> = x.flatten_all()?.to_vec1()?;
        Ok(v.as_chunks::<4>().0.iter().map(|c| [c[0], c[1], c[2], c[3]]).collect())
    }

    /// A continuous guess [1, 4096, 4] as an RGBA image (previews).
    fn guess_image(x: &Tensor) -> Result<RgbaImage> {
        let px = Self::pixels(x)?;
        Ok(RgbaImage::from_fn(SIZE as u32, SIZE as u32, |x, y| {
            let (lab, op) = palette::from_model(px[y as usize * SIZE + x as usize]);
            let [r, g, b] = palette::oklab_to_srgb(lab);
            image::Rgba([r, g, b, if op { 255 } else { 0 }])
        }))
    }

    fn sample_one(&mut self, req: &Request, index: usize, text: &Tensor, cancel: &AtomicBool, on: &mut dyn FnMut(Event)) -> Result<Option<RgbaImage>> {
        let steps = req.steps.max(1);
        let g = req.guidance as f64;
        let seed = req.seed + index as u64;
        let mut x = Tensor::from_vec(rng::randn(seed, PIXELS * CHANNELS), (1, PIXELS, CHANNELS), &self.dev)?;
        let ts: Vec<f32> = (0..=steps)
            .map(|i| {
                let s = i as f32 / steps as f32;
                s / (s + SHIFT * (1.0 - s))
            })
            .collect();
        let (mut pal, mut xpal): (Option<Palette>, Option<Tensor>) = (None, None);
        let (mut x_hat, mut sc, mut probs, mut feats): (Option<Tensor>, Option<Tensor>, Option<Tensor>, Option<Tensor>) = (None, None, None, None);
        for i in 0..steps {
            if cancel.load(Ordering::Relaxed) {
                return Ok(None);
            }
            let (t, t2) = (ts[i], ts[i + 1]);
            if pal.is_none()
                && t >= PALETTE_AT
                && let Some(xh) = &x_hat
            {
                let p = Palette::new(palette::extract(&Self::pixels(xh)?, MAX_COLOURS, MERGE));
                xpal = Some(self.palette_space(&p)?);
                pal = Some(p);
            }
            let guide = g != 1.0 && t <= GUIDE_UNTIL;
            let rows: Vec<Row> = if guide { vec![Row { text, dropped: false }, Row { text, dropped: true }] } else { vec![Row { text, dropped: false }] };
            let two = |t: &Tensor| -> Result<Tensor> { Ok(if guide { Tensor::cat(&[t, t], 0)? } else { t.clone() }) };
            let sc_in = sc.as_ref().map(two).transpose()?;
            let (xh, fe) = self.net.forward(&two(&x)?, t, &rows, sc_in.as_ref(), pal.as_ref(), req.opaque)?;
            let (mut guess, f, fu) = if guide {
                let (xc, xu) = (xh.narrow(0, 0, 1)?, xh.narrow(0, 1, 1)?);
                ((&xu + ((xc - &xu)? * g)?)?, fe.narrow(0, 0, 1)?, Some(fe.narrow(0, 1, 1)?))
            } else {
                (xh, fe, None)
            };
            if let (Some(p), Some(xp_space)) = (&pal, &xpal)
                && t >= POINTER_FROM
            {
                let expected = |f: &Tensor| -> Result<(Tensor, Tensor)> {
                    let pr = candle_nn::ops::softmax_last_dim(&self.net.pointer_logits(f, p)?)?;
                    let xp = pr.broadcast_matmul(xp_space)?;
                    Ok((pr, xp))
                };
                let (pr, mut xp) = expected(&f)?;
                if let Some(fu) = &fu {
                    let (_, xpu) = expected(fu)?;
                    xp = (&xpu + ((xp - &xpu)? * g)?)?;
                }
                if t >= COMMIT_FROM {
                    let conf = pr.max_keepdim(D::Minus1)?;
                    let idx = pr.argmax(D::Minus1)?.flatten_all()?;
                    let exact = xp_space.index_select(&idx, 0)?.unsqueeze(0)?;
                    let sure = conf.ge(COMMIT_THRESH)?.to_dtype(DType::F32)?;
                    xp = (exact.broadcast_mul(&sure)? + xp.broadcast_mul(&(1.0 - &sure)?)?)?;
                }
                probs = Some(pr);
                guess = xp;
            }
            sc = Some(guess.clone());
            x = (&x + ((&guess - &x)? * ((t2 - t) / (1.0 - t).max(1e-4)) as f64)?)?;
            on(Event::Step { index, count: req.count, step: i + 1, total: steps });
            if req.preview_every > 0 && i + 1 < steps {
                on(Event::Preview { index, image: Self::guess_image(&guess)? });
            }
            x_hat = Some(guess);
            feats = Some(f);
        }
        let x_hat = x_hat.expect("at least one step");
        let pal = match pal {
            Some(p) => p,
            None => Palette::new(palette::extract(&Self::pixels(&x_hat)?, MAX_COLOURS, MERGE)),
        };
        let probs = match probs {
            Some(p) => p,
            None => self.net.pointer_logits(feats.as_ref().expect("features"), &pal)?,
        };
        let tokens: Vec<u32> = probs.argmax(D::Minus1)?.flatten_all()?.to_vec1()?;
        Ok(Some(RgbaImage::from_fn(SIZE as u32, SIZE as u32, |x, y| match tokens[y as usize * SIZE + x as usize] as usize {
            0 => image::Rgba([0, 0, 0, 0]),
            s => {
                let [r, g, b] = pal.rgb[s - 1];
                image::Rgba([r, g, b, 255])
            }
        })))
    }
}

impl DiffusionModel for RdFlow {
    fn kind(&self) -> ModelKind {
        ModelKind::new(Family::RdFlow, Variant::Base)
    }
    fn device(&self) -> &Device {
        &self.dev
    }
    fn latent_channels(&self) -> usize {
        CHANNELS
    }
    fn latent_scale(&self) -> usize {
        1
    }
    fn align(&self) -> usize {
        SIZE
    }
    fn latent_dtype(&self) -> DType {
        DType::F32
    }
    fn defaults(&self) -> Defaults {
        Defaults { steps: 16, guidance: 4.0 }
    }
    fn set_variant(&mut self, variant: Variant) -> Result<()> {
        if variant != Variant::Base {
            bail!("RD Flow 64 has a single variant");
        }
        Ok(())
    }
    fn set_lora(&mut self, path: Option<&str>, _multiplier: f64) -> Result<usize> {
        match path {
            Some(p) if !p.is_empty() => bail!("RD Flow 64 does not take LoRAs ({p})"),
            _ => Ok(0),
        }
    }
    fn encode_prompt(&mut self, prompt: &str) -> Result<Tensor> {
        self.text(prompt)
    }
    fn begin(&mut self, _prompts: &[Tensor]) -> Result<()> {
        Ok(())
    }
    fn velocity(&self, _latents: &Tensor, _sigma: f32) -> Result<Tensor> {
        bail!("RD Flow 64 samples with its own loop")
    }
    fn sigmas(&self, steps: usize, _tokens: usize, _img2img: bool) -> Vec<f32> {
        (0..=steps).map(|i| 1.0 - i as f32 / steps.max(1) as f32).collect()
    }
    fn tokens(&self, lh: usize, lw: usize) -> usize {
        lh * lw
    }
    fn decode(&self, latents: &Tensor, _cancel: Option<&AtomicBool>) -> Result<Tensor> {
        Ok(latents.clone())
    }
    fn preview(&self, x0: &Tensor) -> Result<Tensor> {
        Ok(x0.clone())
    }

    fn sample_own(&mut self, req: &Request, cancel: &AtomicBool, on: &mut dyn FnMut(Event)) -> Option<Result<()>> {
        let mut run = || -> Result<()> {
            if req.init.is_some() {
                bail!("RD Flow 64 draws from text only");
            }
            on(Event::Phase("encoding"));
            let text = self.text(&req.prompt)?;
            on(Event::Phase("sampling"));
            for index in 0..req.count {
                on(Event::Step { index, count: req.count, step: 0, total: req.steps.max(1) });
                match self.sample_one(req, index, &text, cancel, on)? {
                    Some(image) => on(Event::Sample { index, image }),
                    None => break,
                }
            }
            Ok(())
        };
        Some(run())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::weights::Weights;

    fn max_diff(a: &Tensor, b: &Tensor) -> f32 {
        (a - b).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap()
    }

    /// DOTSMITH_RD_REF=<parity/dump_ref_rdflow.py output>: text features, palette ramps, one forward pass (cond + dropped
    /// caption, with a palette and self-conditioning) and the pointer logits against the PyTorch reference.
    #[test]
    fn matches_the_reference() {
        let Ok(r) = std::env::var("DOTSMITH_RD_REF") else { return };
        let dev = Device::new_cuda(0).unwrap();
        let mut m = RdFlow::load(ModelPaths::default_for(Family::RdFlow).unwrap(), Variant::Base, &dev).unwrap();
        let rf = Weights::open(&r).unwrap();
        let get = |n: &str| rf.tensor(n, &dev).unwrap();
        let caption = "A small red potion bottle with a cork stopper. The background is transparent.";
        let text = m.text(caption).unwrap();
        let td = max_diff(&text, &get("text"));
        let want = get("text");
        let mean = (&text - &want).unwrap().abs().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap();
        let peak = want.abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
        let dot = (&text * &want).unwrap().sum(1).unwrap();
        let norms = (text.sqr().unwrap().sum(1).unwrap().sqrt().unwrap() * want.sqr().unwrap().sum(1).unwrap().sqrt().unwrap()).unwrap();
        let cos: Vec<f32> = (dot / norms).unwrap().to_vec1().unwrap();
        let worst = cos.iter().cloned().fold(1f32, f32::min);
        println!("text features: max |diff| {td:.3e} of peak {peak:.1}, mean {mean:.3e}, worst token cosine {worst:.6}");

        let rgb: Vec<[u8; 3]> = get("pal_rgb").to_vec2::<i64>().unwrap().iter().map(|c| [c[0] as u8, c[1] as u8, c[2] as u8]).collect();
        let (ramp, level) = palette::ramps(&rgb);
        let want_ramp: Vec<i64> = get("ramp").to_vec1().unwrap();
        let want_level: Vec<f32> = get("level").to_vec1().unwrap();
        assert_eq!(ramp, want_ramp, "ramp ids");
        assert!(level.iter().zip(&want_level).all(|(a, b)| (a - b).abs() < 1e-6), "levels {level:?} vs {want_level:?}");

        let pal = Palette { rgb, ramp, level };
        let x_t = get("x_t").reshape((2, PIXELS, CHANNELS)).unwrap();
        let sc = get("sc").reshape((2, PIXELS, CHANNELS)).unwrap();
        let ref_text = get("text");
        let rows = [Row { text: &ref_text, dropped: false }, Row { text: &ref_text, dropped: true }];
        let (xh, feats) = m.net.forward(&x_t, 0.7, &rows, Some(&sc), Some(&pal), false).unwrap();
        let (xd, fd) = (max_diff(&xh, &get("x_hat")), max_diff(&feats, &get("feats")));
        let logits = m.net.pointer_logits(&feats, &pal).unwrap();
        let ld = max_diff(&logits, &get("logits"));
        println!("forward: x_hat {xd:.3e}, features {fd:.3e}, pointer logits {ld:.3e}");
        assert!(xd < 1e-2 && fd < 1e-2 && ld < 5e-2, "forward differs");
        // the encoder runs in bf16 (as RD Flow's own pipeline does on a GPU); the reference is f32
        assert!(worst > 0.999 && mean < 2e-2, "text features: worst cosine {worst}, mean {mean}");
    }

    /// DOTSMITH_RD_OUT=<dir> [DOTSMITH_RD_CAPTION=<file>]: draw two images with the real weights (files from DOTSMITH_MODELS).
    #[test]
    fn draws_with_the_real_weights() {
        let Ok(out) = std::env::var("DOTSMITH_RD_OUT") else { return };
        let caption = std::env::var("DOTSMITH_RD_CAPTION")
            .ok()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .unwrap_or_else(|| "A small red potion bottle with a cork. The background is transparent.".into());
        let dev = Device::new_cuda(0).unwrap();
        let mut m = RdFlow::load(ModelPaths::default_for(Family::RdFlow).unwrap(), Variant::Base, &dev).unwrap();
        let req = Request {
            prompt: caption,
            negative: String::new(),
            width: 64,
            height: 64,
            steps: 16,
            guidance: 4.0,
            seed: 7,
            count: 2,
            init: None,
            mask: None,
            strength: 1.0,
            preview_every: 0,
            opaque: false,
        };
        let t = std::time::Instant::now();
        let mut saved = 0;
        m.sample_own(&req, &AtomicBool::new(false), &mut |e| {
            if let Event::Sample { index, image } = e {
                image.save(format!("{out}/rd_{index}.png")).unwrap();
                saved += 1;
            }
        })
        .unwrap()
        .unwrap();
        println!("{saved} images in {:.1} s", t.elapsed().as_secs_f64());
        assert_eq!(saved, 2);
    }
}
