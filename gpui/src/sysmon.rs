//! RAM and VRAM in use, sampled once a second on its own thread for the status bar. VRAM comes from NVML (the driver's
//! management library, what nvidia-smi reads): it counts every process on the card and needs no CUDA context, so the
//! monitor costs no VRAM while no model is loaded.
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

#[derive(Clone, Copy, Default, PartialEq)]
pub struct Sample {
    pub ram_used: u64,
    pub ram_total: u64,
    /// Dotsmith's own committed memory.
    pub app_ram: u64,
    pub vram: Option<(u64, u64)>,
}

static LATEST: Mutex<Option<Sample>> = Mutex::new(None);

pub fn latest() -> Option<Sample> {
    static STARTED: OnceLock<()> = OnceLock::new();
    STARTED.get_or_init(|| {
        let _ = std::thread::Builder::new().name("sysmon".into()).spawn(|| {
            let nvml = nvml::open();
            loop {
                let s = sample(nvml.as_ref());
                *LATEST.lock().unwrap() = Some(s);
                std::thread::sleep(Duration::from_secs(1));
            }
        });
    });
    *LATEST.lock().unwrap()
}

#[cfg(windows)]
fn sample(nvml: Option<&nvml::Nvml>) -> Sample {
    use std::ffi::c_void;
    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        load: u32,
        total_phys: u64,
        avail_phys: u64,
        rest: [u64; 5],
    }
    #[repr(C)]
    #[derive(Default)]
    struct Counters {
        cb: u32,
        faults: u32,
        peak_ws: usize,
        ws: usize,
        q: [usize; 4],
        private: usize,
        peak_private: usize,
    }
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(b: *mut MemoryStatusEx) -> i32;
        fn GetCurrentProcess() -> *mut c_void;
        fn K32GetProcessMemoryInfo(p: *mut c_void, c: *mut Counters, cb: u32) -> i32;
    }
    let mut m = MemoryStatusEx { length: std::mem::size_of::<MemoryStatusEx>() as u32, load: 0, total_phys: 0, avail_phys: 0, rest: [0; 5] };
    let mut c = Counters { cb: std::mem::size_of::<Counters>() as u32, ..Default::default() };
    let ok = unsafe { GlobalMemoryStatusEx(&mut m) } != 0;
    let app = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) } != 0;
    Sample {
        ram_used: if ok { m.total_phys - m.avail_phys } else { 0 },
        ram_total: if ok { m.total_phys } else { 0 },
        app_ram: if app { c.private as u64 } else { 0 },
        vram: nvml.and_then(|n| n.memory()),
    }
}

#[cfg(not(windows))]
fn sample(nvml: Option<&nvml::Nvml>) -> Sample {
    Sample { vram: nvml.and_then(|n| n.memory()), ..Default::default() }
}

mod nvml {
    use std::ffi::c_void;

    type Init = unsafe extern "C" fn() -> i32;
    type Handle = unsafe extern "C" fn(u32, *mut *mut c_void) -> i32;
    type MemInfo = unsafe extern "C" fn(*mut c_void, *mut [u64; 3]) -> i32;

    pub struct Nvml {
        device: *mut c_void,
        mem: MemInfo,
    }

    unsafe impl Send for Nvml {}

    impl Nvml {
        /// (used, total) bytes of dedicated VRAM on the first GPU, all processes.
        pub fn memory(&self) -> Option<(u64, u64)> {
            let mut m = [0u64; 3]; // total, free, used
            (unsafe { (self.mem)(self.device, &mut m) } == 0).then_some((m[2], m[0]))
        }
    }

    #[cfg(windows)]
    pub fn open() -> Option<Nvml> {
        unsafe extern "system" {
            fn LoadLibraryW(name: *const u16) -> *mut c_void;
            fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
        }
        let name: Vec<u16> = "nvml.dll".encode_utf16().chain(std::iter::once(0)).collect();
        unsafe {
            let lib = LoadLibraryW(name.as_ptr());
            if lib.is_null() {
                return None;
            }
            let get = |s: &[u8]| Some(GetProcAddress(lib, s.as_ptr())).filter(|p| !p.is_null());
            let init: Init = std::mem::transmute(get(b"nvmlInit_v2\0")?);
            let handle: Handle = std::mem::transmute(get(b"nvmlDeviceGetHandleByIndex_v2\0")?);
            let mem: MemInfo = std::mem::transmute(get(b"nvmlDeviceGetMemoryInfo\0")?);
            if init() != 0 {
                return None;
            }
            let mut device = std::ptr::null_mut();
            (handle(0, &mut device) == 0).then_some(Nvml { device, mem })
        }
    }

    #[cfg(not(windows))]
    pub fn open() -> Option<Nvml> {
        None
    }
}
