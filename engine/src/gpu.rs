//! CUDA device check, run before any model load so a missing driver or an old card is a message, not a crash.
use std::sync::OnceLock;

/// Ampere: the bf16 kernels need it.
pub const MIN_COMPUTE: (i32, i32) = (8, 0);

#[derive(Clone, Debug)]
pub struct GpuInfo {
    pub name: String,
    pub compute: (i32, i32),
    pub total_mb: u64,
}

#[derive(Clone, Debug)]
pub enum GpuError {
    NoDriver,
    MissingRuntime(Vec<String>),
    NoDevice(String),
    TooOld(GpuInfo),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::NoDriver => write!(f, "NVIDIA driver not found (nvcuda). Dotsmith needs an NVIDIA GPU with a current driver."),
            GpuError::MissingRuntime(l) => write!(
                f,
                "The CUDA files Dotsmith ships with are missing ({}). Install Dotsmith with its installer or unzip the whole portable folder: a copied dotsmith.exe alone cannot use the GPU.",
                l.join(", ")
            ),
            GpuError::NoDevice(e) => write!(f, "No usable NVIDIA GPU: {e}"),
            GpuError::TooOld(g) => write!(
                f,
                "{} (compute {}.{}) is too old: Dotsmith needs an RTX 30 series or newer (compute {}.{}+).",
                g.name, g.compute.0, g.compute.1, MIN_COMPUTE.0, MIN_COMPUTE.1
            ),
        }
    }
}

impl std::error::Error for GpuError {}

#[cfg(windows)]
fn loadable(name: &str) -> bool {
    unsafe extern "system" {
        fn LoadLibraryW(name: *const u16) -> *mut std::ffi::c_void;
    }
    let w: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
    !unsafe { LoadLibraryW(w.as_ptr()) }.is_null()
}

/// The CUDA 13 DLLs the build links against; the app delay-loads them.
#[cfg(windows)]
pub const RUNTIME_DLLS: [&str; 4] = ["cublas64_13.dll", "cublasLt64_13.dll", "curand64_10.dll", "nvrtc64_130_0.dll"];

fn panic_text(p: Box<dyn std::any::Any + Send>) -> String {
    p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()
}

fn probe_uncached() -> Result<GpuInfo, GpuError> {
    #[cfg(windows)]
    {
        if !loadable("nvcuda.dll") {
            return Err(GpuError::NoDriver);
        }
        let missing: Vec<String> = RUNTIME_DLLS.iter().filter(|d| !loadable(d)).map(|d| d.to_string()).collect();
        if !missing.is_empty() {
            return Err(GpuError::MissingRuntime(missing));
        }
    }
    let info = std::panic::catch_unwind(|| -> Result<GpuInfo, String> {
        let ctx = cudarc::driver::CudaContext::new(0).map_err(|e| e.to_string())?;
        Ok(GpuInfo {
            name: ctx.name().map_err(|e| e.to_string())?,
            compute: ctx.compute_capability().map_err(|e| e.to_string())?,
            total_mb: ctx.total_mem().map_err(|e| e.to_string())? as u64 / (1024 * 1024),
        })
    })
    .map_err(panic_text)
    .and_then(|r| r)
    .map_err(GpuError::NoDevice)?;
    if info.compute < MIN_COMPUTE {
        return Err(GpuError::TooOld(info));
    }
    Ok(info)
}

pub fn probe() -> Result<GpuInfo, GpuError> {
    static CACHE: OnceLock<Result<GpuInfo, GpuError>> = OnceLock::new();
    CACHE.get_or_init(probe_uncached).clone()
}

/// NVRTC target for the engine's own kernels: the device's architecture.
pub fn nvrtc_arch() -> &'static str {
    static ARCH: OnceLock<&'static str> = OnceLock::new();
    ARCH.get_or_init(|| {
        let (maj, min) = probe().map(|g| g.compute).unwrap_or(MIN_COMPUTE);
        Box::leak(format!("compute_{maj}{min}").into_boxed_str())
    })
}

/// Dedicated VRAM still free in MB, counting every process on the card. On Windows an allocation past it does not fail:
/// the driver backs it with system RAM and the GPU slows to a crawl (see also `Budget`).
pub fn free_mb(dev: &candle_core::Device) -> Option<u64> {
    let ctx = dev.as_cuda_device().ok()?.cuda_stream().context().clone();
    ctx.mem_get_info().ok().map(|(free, _)| free as u64 >> 20)
}

/// This process's dedicated VRAM use against the budget Windows grants it (DXGI, first NVIDIA adapter). Past the budget
/// Windows moves the process's GPU memory to system RAM and every kernel crawls; once that happened cuMemGetInfo
/// reports plenty free again (measured: 18 GB "free" on a full card), so this is the signal that stays true.
#[cfg(windows)]
pub struct Budget(windows::Win32::Graphics::Dxgi::IDXGIAdapter3);

#[cfg(windows)]
impl Budget {
    pub fn open() -> Option<Self> {
        use windows::Win32::Graphics::Dxgi::*;
        use windows::core::Interface;
        unsafe {
            let f: IDXGIFactory1 = CreateDXGIFactory1().ok()?;
            (0..).map_while(|i| f.EnumAdapters1(i).ok()).find(|a| a.GetDesc1().is_ok_and(|d| d.VendorId == 0x10DE))?.cast().ok().map(Budget)
        }
    }

    /// (in use, budget) in MB.
    pub fn read(&self) -> Option<(u64, u64)> {
        use windows::Win32::Graphics::Dxgi::*;
        let mut m = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        unsafe { self.0.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut m) }.ok()?;
        Some((m.CurrentUsage >> 20, m.Budget >> 20))
    }
}

#[cfg(not(windows))]
pub struct Budget;

#[cfg(not(windows))]
impl Budget {
    pub fn open() -> Option<Self> {
        None
    }
    pub fn read(&self) -> Option<(u64, u64)> {
        None
    }
}

/// VRAM this process can still allocate before Windows starts moving its memory to system RAM, MB: the smaller of
/// the CUDA free memory and this process's DXGI budget minus what it already uses (no budget off Windows).
pub fn available_mb(dev: &candle_core::Device) -> Option<u64> {
    let free = free_mb(dev)?;
    let budget = Budget::open().and_then(|b| b.read()).map(|(used, budget)| budget.saturating_sub(used));
    Some(budget.map_or(free, |b| b.min(free)))
}

pub fn total_mb() -> u64 {
    probe().map(|g| g.total_mb).unwrap_or(0)
}
