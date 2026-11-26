//! Model memory policy: where a model's weights live, decided once per load from the user's policy and the free VRAM.
//! Generic over families: a family describes its repeated layer stacks (`Stack`) and the bytes that are not in a stack,
//! `plan` returns per stack how many layers stay on the GPU and whether the rest streams from RAM (core::stream).
//!
//! Automatic placement never changes a model that fits: everything stays on the GPU exactly as without this module.
//! Only when it does not fit (or streaming is forced) the stacks stream, the hot ones (run every step) first get the
//! spare VRAM as resident layers, so a 12 GB card streams a few blocks and a 6 GB card streams nearly all of them.
use serde::{Deserialize, Serialize};
use std::sync::Mutex;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Placement {
    /// GPU when the model fits the budget, streaming when it does not
    #[default]
    Auto,
    /// always fully on the GPU (the classic behaviour; may overflow small cards)
    Gpu,
    /// always stream the layer stacks from RAM (minimum VRAM)
    Stream,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HostMemory {
    /// pinned for the hot stacks when free RAM allows, staged otherwise
    #[default]
    Auto,
    /// page-locked copy of every streamed layer: fastest, locks that much RAM
    Pinned,
    /// layers stay in the OS file cache and pass through two small page-locked buffers: least RAM
    Staged,
}

/// The user's memory policy. Every field has an automatic default; the GUI stores it in its settings and the CLI
/// overrides it with environment variables (see `policy`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MemoryPolicy {
    pub placement: Placement,
    pub host: HostMemory,
    /// VRAM this app may use, MB (None = all that is free when the model loads)
    pub vram_limit_mb: Option<u64>,
    /// VRAM kept free for activations, attention and the VAE decode, MB (None = the model family's estimate)
    pub reserve_mb: Option<u64>,
    /// layers of each hot stack kept on the GPU while streaming (None = as many as fit)
    pub resident_layers: Option<usize>,
    /// images denoised together per forward pass while streaming (1 = one after the other)
    pub batch_max: usize,
    /// preferred weight precision ("bf16", ...): a family's alternative file in it is used when present
    /// (manifest::FileSpec::precision); None = the default (quantized) files
    pub precision: Option<String>,
}

impl Default for MemoryPolicy {
    fn default() -> Self {
        Self { placement: Placement::Auto, host: HostMemory::Auto, vram_limit_mb: None, reserve_mb: None, resident_layers: None, batch_max: 4, precision: None }
    }
}

static POLICY: Mutex<Option<MemoryPolicy>> = Mutex::new(None);
static REPORT: Mutex<Vec<Placed>> = Mutex::new(Vec::new());

/// Set by the app from its settings; applies on the next model load.
pub fn set_policy(p: MemoryPolicy) {
    *POLICY.lock().unwrap() = Some(p);
}

fn env<T: std::str::FromStr>(k: &str) -> Option<T> {
    std::env::var(k).ok().and_then(|v| v.trim().parse().ok())
}

/// The policy in force: the app's, else the default, with environment overrides on top (CLI and debugging):
/// DOTSMITH_STREAM=1|0 (placement stream|gpu), DOTSMITH_STREAM_HOST=pinned|staged, DOTSMITH_VRAM_LIMIT_MB, DOTSMITH_RESERVE_MB,
/// DOTSMITH_RESIDENT_BLOCKS, DOTSMITH_BATCH_MAX.
pub fn policy() -> MemoryPolicy {
    let mut p = POLICY.lock().unwrap().clone().unwrap_or_default();
    match std::env::var("DOTSMITH_STREAM").as_deref() {
        Ok("1") => p.placement = Placement::Stream,
        Ok("0") => p.placement = Placement::Gpu,
        _ => {}
    }
    match std::env::var("DOTSMITH_STREAM_HOST").as_deref() {
        Ok("pinned") => p.host = HostMemory::Pinned,
        Ok("staged") => p.host = HostMemory::Staged,
        _ => {}
    }
    p.vram_limit_mb = env("DOTSMITH_VRAM_LIMIT_MB").or(p.vram_limit_mb);
    p.reserve_mb = env("DOTSMITH_RESERVE_MB").or(p.reserve_mb);
    p.resident_layers = env("DOTSMITH_RESIDENT_BLOCKS").or(p.resident_layers);
    p.batch_max = env("DOTSMITH_BATCH_MAX").unwrap_or(p.batch_max).max(1);
    // DOTSMITH_PRECISION=bf16 prefers bf16 files, "default" (or empty) the default ones
    if let Ok(v) = std::env::var("DOTSMITH_PRECISION") {
        p.precision = Some(v.trim().to_string()).filter(|v| !v.is_empty() && v != "default");
    }
    p
}

/// One repeated layer stack of a model (DiT blocks, text encoder layers): only its streamable linears count.
#[derive(Clone, Debug)]
pub struct Stack {
    pub name: &'static str,
    pub layers: usize,
    pub layer_bytes: u64,
    /// runs every denoising step (gets resident layers first); false = once per prompt
    pub hot: bool,
}

/// What a model needs besides its stacks.
#[derive(Clone, Debug, Default)]
pub struct Needs {
    /// on the GPU in every placement (VAE, norms, small embedders), bytes
    pub fixed: u64,
    /// on the GPU when resident, in RAM while streaming (layers that run once per step or prompt), bytes
    pub cold: u64,
    /// the family's working-memory estimate at its largest canvas, MB (policy.reserve_mb overrides)
    pub reserve_mb: u64,
}

/// Per stack: the first `resident` layers stay on the GPU, the rest stream (if `stream`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StackPlan {
    pub stream: bool,
    pub resident: usize,
}

impl StackPlan {
    pub const GPU: StackPlan = StackPlan { stream: false, resident: usize::MAX };
}

const MB: u64 = 1 << 20;

/// Decide the placement of `stacks` for a GPU with `free_mb` free now. Returns one plan per stack (same order).
pub fn plan(stacks: &[Stack], needs: &Needs, free_mb: u64) -> Vec<StackPlan> {
    let out = plan_inner(stacks, needs, free_mb);
    if std::env::var("DOTSMITH_STREAM_STATS").is_ok() {
        eprintln!(
            "memory plan: free {free_mb} MB, fixed {} MB, cold {} MB, reserve {} MB, stacks {:?} -> {out:?}",
            needs.fixed / MB,
            needs.cold / MB,
            policy().reserve_mb.unwrap_or(needs.reserve_mb),
            stacks.iter().map(|s| (s.name, s.layers, s.layer_bytes / MB)).collect::<Vec<_>>()
        );
    }
    out
}

fn plan_inner(stacks: &[Stack], needs: &Needs, free_mb: u64) -> Vec<StackPlan> {
    let p = policy();
    // 5 % stays free: driver bookkeeping, cuBLAS workspaces and other apps' fluctuations
    let budget = p.vram_limit_mb.map_or(free_mb, |l| l.min(free_mb)) * MB / 100 * 95;
    let reserve = p.reserve_mb.unwrap_or(needs.reserve_mb) * MB;
    let stacks_total: u64 = stacks.iter().map(|s| s.layers as u64 * s.layer_bytes).sum();
    let resident_total = stacks_total + needs.fixed + needs.cold + reserve;
    let all_gpu = vec![StackPlan::GPU; stacks.len()];
    match p.placement {
        Placement::Gpu => return all_gpu,
        Placement::Auto if resident_total <= budget => return all_gpu,
        _ => {}
    }
    // streaming: every stack starts with its two slots; the spare VRAM becomes resident layers, hot stacks first
    let slots: u64 = stacks.iter().map(|s| 2 * s.layer_bytes).sum();
    let mut left = budget.saturating_sub(needs.fixed + reserve + slots) as i64;
    let mut out = vec![StackPlan { stream: true, resident: 0 }; stacks.len()];
    let mut order: Vec<usize> = (0..stacks.len()).collect();
    order.sort_by_key(|&i| !stacks[i].hot);
    for i in order {
        let s = &stacks[i];
        let lb = s.layer_bytes as i64;
        let fit = if lb > 0 { (left.max(0) / lb) as usize } else { s.layers };
        let mut r = match p.resident_layers {
            Some(n) if s.hot => n.min(s.layers),
            // forced streaming = minimum VRAM: no resident layers unless asked for; automatic fills the spare VRAM
            _ if p.placement == Placement::Stream => 0,
            _ => fit.min(s.layers),
        };
        // a stack streams an even number >= 4 of layers (double buffering), or none at all
        if s.layers - r < 4 {
            // all resident frees the two slots again: allowed when that fits
            if (s.layers - r) as i64 * lb <= left + 2 * lb && p.resident_layers.is_none() {
                left -= (s.layers - r) as i64 * lb - 2 * lb;
                out[i] = StackPlan { stream: false, resident: s.layers };
                left -= r as i64 * lb;
                continue;
            }
            r = s.layers.saturating_sub(4);
        }
        if (s.layers - r) % 2 == 1 {
            // one layer fewer on the GPU, or one more when none is (odd stacks such as a 27-layer text encoder)
            r = if r > 0 { r - 1 } else { r + 1 };
        }
        left -= r as i64 * lb;
        out[i] = StackPlan { stream: true, resident: r };
    }
    out
}

/// Where one loaded component's layer stack went (the app shows it, worded in its own language).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Placed {
    /// "dit" or "text_encoder"
    pub component: String,
    pub layers: usize,
    /// layers streamed from RAM (the rest are on the GPU)
    pub streamed: usize,
    /// streamed from page-locked RAM (else through the OS file cache)
    pub pinned: bool,
    /// page-locked RAM held, MB
    pub locked_mb: u64,
    /// weight format of the stack's linears: "int8", "bf16", "gguf" or "other"
    #[serde(default)]
    pub format: String,
}

impl Placed {
    pub fn gpu(component: &str, layers: usize, format: &str) -> Self {
        Self { component: component.into(), layers, streamed: 0, pinned: false, locked_mb: 0, format: format.into() }
    }
}

impl std::fmt::Display for Placed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.streamed == 0 {
            return write!(f, "{} ({}): all {} layers on the GPU", self.component, self.format, self.layers);
        }
        write!(
            f,
            "{} ({}): {} of {} layers streamed from {} RAM ({:.1} GB locked), {} on the GPU",
            self.component,
            self.format,
            self.streamed,
            self.layers,
            if self.pinned { "pinned" } else { "cached" },
            self.locked_mb as f64 / 1024.0,
            self.layers - self.streamed
        )
    }
}

/// Record where a component went (replaces its previous entry).
pub fn set_report(p: Placed) {
    let mut r = REPORT.lock().unwrap();
    r.retain(|l| l.component != p.component);
    r.push(p);
}

pub fn clear_report() {
    REPORT.lock().unwrap().clear();
}

/// The placement of the components loaded since the last model load.
pub fn report() -> Vec<Placed> {
    REPORT.lock().unwrap().clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn krea() -> (Vec<Stack>, Needs) {
        (
            vec![Stack { name: "dit", layers: 28, layer_bytes: 434 * MB, hot: true }, Stack { name: "te", layers: 36, layer_bytes: 100 * MB, hot: false }],
            Needs { fixed: 600 * MB, cold: 1000 * MB, reserve_mb: 2500 },
        )
    }

    #[test]
    fn a_model_that_fits_stays_on_the_gpu() {
        let (s, n) = krea();
        assert_eq!(plan(&s, &n, 23_000), vec![StackPlan::GPU; 2]);
    }

    #[test]
    fn small_cards_stream_and_fill_the_spare_vram_with_hot_layers() {
        let (s, n) = krea();
        let p = plan(&s, &n, 11_000);
        assert!(p[0].stream && p[0].resident > 0 && (28 - p[0].resident).is_multiple_of(2) && 28 - p[0].resident >= 4, "{p:?}");
        assert!(p[1].stream && p[1].resident < 36, "{p:?}"); // leftovers after rounding may keep a few text layers
        let tiny = plan(&s, &n, 4_000);
        assert_eq!(tiny[0], StackPlan { stream: true, resident: 0 });
    }
    #[test]
    fn an_odd_stack_streams_an_even_number_of_layers() {
        let stack = [Stack { name: "te", layers: 27, layer_bytes: 192 * MB, hot: false }];
        let p = plan(&stack, &Needs { fixed: 1000 * MB, cold: 0, reserve_mb: 2000 }, 3_000);
        assert_eq!(p, vec![StackPlan { stream: true, resident: 1 }]);
    }
}
