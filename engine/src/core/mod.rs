//! Shared building blocks: memory-mapped weights, int8 ConvRot linears + LoRA, attention, fused kernels, RNG.
pub mod attention;
pub mod fused;
pub mod int8;
pub mod kernels;
pub mod memory;
pub mod rng;
pub mod stream;
pub mod util;
pub mod weights;

use std::sync::atomic::{AtomicU8, Ordering};

/// Low-VRAM mode (ComfyUI --lowvram style): the text encoder and the DiT never share the GPU, the VAE decodes in
/// 512 px tiles and attention scores go in ~256 MB chunks. Automatic: DOTSMITH_LOW_VRAM=1, a GPU under 20 GB, or a
/// memory policy that limits this app to under 20 GB (core::memory).
pub fn low_vram() -> bool {
    match LOW_VRAM.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => {
            std::env::var("DOTSMITH_LOW_VRAM").is_ok_and(|v| v == "1")
                || (1..20_000).contains(&crate::gpu::total_mb())
                || memory::policy().vram_limit_mb.is_some_and(|l| l < 20_000)
        }
    }
}

static LOW_VRAM: AtomicU8 = AtomicU8::new(0);

/// Raised while a generation must stop (cancel, or the VRAM watch in engine.rs): the DiT block loops and the VAE tile
/// loops check it, so a step that started spilling to system RAM ends after one block instead of the whole step.
pub static ABORT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn check_abort() -> anyhow::Result<()> {
    if ABORT.load(Ordering::Relaxed) {
        anyhow::bail!("aborted");
    }
    Ok(())
}

/// None = automatic. Applies to the next load.
pub fn set_low_vram(on: Option<bool>) {
    LOW_VRAM.store(
        match on {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        Ordering::Relaxed,
    );
}
