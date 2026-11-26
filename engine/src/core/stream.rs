//! Weight streaming (DOTSMITH_STREAM=1): the int8 weights of a stack of identical layers (DiT blocks, text encoder
//! layers) stay in host RAM and are copied into two GPU slots on a second CUDA stream while the previous block computes (double buffering). VRAM holds two
//! blocks instead of all of them; the weights and the math are exactly the resident int8 path's, so the output is
//! bit-identical. Costs one full weight upload per denoising step, hidden behind the compute when a block's matmuls
//! take longer than its copy (PCIe 4.0 x8 ~13 GB/s vs ~430 MB per Krea block).
//!
//! Two host modes (core::memory::HostMemory; automatic = pinned for hot stacks when free RAM allows):
//! - pinned: every streamed block is packed once into its own page-locked buffer (~12 GB for Krea); the GPU's copy
//!   engine reads it directly, nothing on the CPU per step.
//! - staged: the weights stay in the memory-mapped checkpoint (OS page cache: shared, evictable, no extra RAM) and
//!   pass through two pinned staging buffers (~0.9 GB); the CPU copies each block into staging (8 threads) right
//!   before its upload. For machines whose RAM cannot spare 12 GB of locked pages.
//!
//! Order: streamed layer k computes from slot k % 2; when it is done the copy stream refills that slot with layer k + 2 (wrapping into the next step), after an event that says the
//! compute stream has finished reading it. Before a block runs, the compute stream waits for its slot's copy.
use anyhow::{Context, Result, bail};
use candle_core::cuda_backend::cudarc::driver::{CudaEvent, CudaStream, DevicePtr, PinnedHostSlice};
use candle_core::{DType, Device, Storage, Tensor};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::core::weights::Weights;

const ALIGN: usize = 256;
const STAGE_THREADS: usize = 8;

#[cfg(windows)]
fn free_ram_bytes() -> usize {
    #[repr(C)]
    struct MemStatus {
        len: u32,
        load: u32,
        total_phys: u64,
        avail_phys: u64,
        rest: [u64; 5],
    }
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(m: *mut MemStatus) -> i32;
    }
    let mut m = MemStatus { len: std::mem::size_of::<MemStatus>() as u32, load: 0, total_phys: 0, avail_phys: 0, rest: [0; 5] };
    if unsafe { GlobalMemoryStatusEx(&mut m) } != 0 { m.avail_phys as usize } else { 0 }
}

#[cfg(not(windows))]
fn free_ram_bytes() -> usize {
    0
}

enum Host {
    /// one packed page-locked buffer per streamed block
    Pinned(Vec<PinnedHostSlice<u8>>),
    /// the mapped checkpoint + two page-locked staging buffers; `src[k]` = (file offset, packed offset, len) per linear
    Staged { map: memmap2::Mmap, stage: [PinnedHostSlice<u8>; 2], src: Vec<Vec<(usize, usize, usize)>> },
}

struct State {
    holds: [Option<usize>; 2],      // streamed index whose copy was issued into each slot
    copied: [Option<CudaEvent>; 2], // copy stream: slot filled (staged: also = staging buffer free again)
    freed: [Option<CudaEvent>; 2],  // compute stream: slot no longer read
}

pub struct BlockStream {
    copy: Arc<CudaStream>,
    compute: Arc<CudaStream>,
    host: Host,
    host_ptr: Vec<usize>, // pinned: per block; staged: per staging buffer
    slots: [Tensor; 2],
    slot_ptr: [u64; 2],
    n: usize,
    /// element type of the streamed weights: U8 (int8 codes, read as i8 by the kernels) or BF16 (plain checkpoints)
    pub dtype: DType,
    bytes: usize,                                   // one block's packed size
    layout: HashMap<String, (usize, usize, usize)>, // linear suffix -> (offset, out, in)
    state: Mutex<State>,
    pub uploaded: std::sync::atomic::AtomicUsize,
    pub stage_ns: std::sync::atomic::AtomicU64, // staged: CPU time spent filling staging buffers
}

unsafe impl Send for BlockStream {}
unsafe impl Sync for BlockStream {}

impl BlockStream {
    /// `blocks`: the tensor prefix of each streamed layer in run order (e.g. "blocks.3"; an even count >= 4);
    /// `linears`: the linears of one layer as suffixes (e.g. "attn.wq"), all int8 or all bf16; `path`: the checkpoint
    /// `wt` came from.
    /// `hot`: runs many times per image (DiT: every step), so it gets pinned RAM when there is room; a cold stack
    /// (text encoder: once per prompt) stays staged unless the policy says pinned.
    pub fn new(wt: &Weights, path: &Path, dev: &Device, blocks: &[String], linears: &[&str], hot: bool) -> Result<Self> {
        if blocks.len() < 4 || blocks.len() % 2 == 1 {
            bail!("stream: need an even number >= 4 of streamed blocks, got {}", blocks.len());
        }
        let Device::Cuda(cd) = dev else { bail!("stream needs CUDA") };
        let compute = cd.cuda_stream();
        let ctx = compute.context().clone();
        let copy = ctx.new_stream()?;
        // layout from the first block; every block has the same shapes
        let mut layout = HashMap::new();
        let mut order = vec![];
        let mut off = 0;
        let mut dtype = None;
        for l in linears {
            let e = wt.entries.get(&format!("{}.{l}.weight", blocks[0])).with_context(|| format!("stream: no {l}"))?;
            let dt = match e.dtype.as_str() {
                "I8" | "U8" => DType::U8,
                "BF16" => DType::BF16,
                other => bail!("stream: {l} is {other} (int8 and bf16 checkpoints stream)"),
            };
            if dtype.is_some_and(|d| d != dt) {
                bail!("stream: {l} mixes int8 and bf16 in one layer");
            }
            dtype = Some(dt);
            let (o, i) = (e.shape[0], e.shape[1]);
            let len = o * i * dt.size_in_bytes();
            layout.insert(l.to_string(), (off, o, i));
            order.push((l.to_string(), off, len));
            off = (off + len).div_ceil(ALIGN) * ALIGN;
        }
        let dtype = dtype.context("stream: no linears")?;
        let bytes = off;
        let n = blocks.len();
        use crate::core::memory::HostMemory;
        let want = crate::core::memory::policy().host;
        // pinned locks the whole packed stack in RAM, and filling it reads the same bytes through the file cache: Auto
        // only pins with room for both plus 8 GB (a 32 GB laptop pinning a 12 GB stack thrashes instead)
        let mut pinned = match want {
            HostMemory::Pinned => true,
            HostMemory::Staged => false,
            HostMemory::Auto => hot && free_ram_bytes() > 2 * bytes * n + (8 << 30),
        };
        let mut host_bufs = Vec::with_capacity(n);
        if pinned {
            for _ in blocks {
                match unsafe { ctx.alloc_pinned::<u8>(bytes) } {
                    Ok(h) => host_bufs.push(h),
                    Err(e) if want == HostMemory::Auto => {
                        eprintln!("stream: pinned RAM refused ({e}), staging instead");
                        host_bufs.clear();
                        pinned = false;
                        break;
                    }
                    Err(e) => return Err(anyhow::Error::from(e).context("stream: pinned RAM allocation failed (choose the low-RAM mode)")),
                }
            }
        }
        let host = if pinned {
            let host = host_bufs;
            let ptrs: Vec<usize> = host.iter().map(|h| h.as_ptr().map(|p| p as usize)).collect::<std::result::Result<_, _>>()?;
            // fill in parallel straight from the memory-mapped file (first touch reads the disk)
            std::thread::scope(|s| -> Result<()> {
                let mut hs = vec![];
                for (k, b) in blocks.iter().enumerate() {
                    let (order, hp) = (&order, ptrs[k]);
                    hs.push(s.spawn(move || -> Result<()> {
                        let dst = unsafe { std::slice::from_raw_parts_mut(hp as *mut u8, bytes) };
                        for (l, o, len) in order {
                            let (src, _) = wt.bytes(&format!("{b}.{l}.weight"))?;
                            dst[*o..*o + len].copy_from_slice(&src[..*len]);
                        }
                        Ok(())
                    }));
                }
                for h in hs {
                    h.join().map_err(|_| anyhow::anyhow!("stream: fill thread panicked"))??;
                }
                Ok(())
            })?;
            Host::Pinned(host)
        } else {
            let map = unsafe { memmap2::Mmap::map(&std::fs::File::open(path)?)? };
            let mut src = vec![];
            for b in blocks {
                let mut v = vec![];
                for (l, o, len) in &order {
                    let (s0, s1) = wt.file_range(&format!("{b}.{l}.weight"))?;
                    if s1 - s0 != *len {
                        bail!("stream: {b}.{l} has another shape than {}", blocks[0]);
                    }
                    v.push((s0, *o, *len));
                }
                src.push(v);
            }
            // warm the OS file cache in the background, in run order, so the first run does not wait on the disk
            // (a cold SATA SSD reads ~0.5 GB/s); best effort, the reads are shared with the staging copies
            let ranges: Vec<(usize, usize)> = src.iter().flatten().map(|&(f, _, n)| (f, n)).collect();
            let wpath = path.to_path_buf();
            std::thread::spawn(move || {
                let Ok(m) = std::fs::File::open(&wpath).and_then(|f| unsafe { memmap2::Mmap::map(&f) }) else { return };
                let mut sum = 0u8;
                for (f, n) in ranges {
                    for i in (f..f + n).step_by(4096) {
                        sum = sum.wrapping_add(m[i]);
                    }
                }
                std::hint::black_box(sum);
            });
            // page-locked but not write-combined: the CPU writes it every step
            let alloc = || unsafe { ctx.alloc_pinned_with_flags::<u8>(bytes, 0) };
            Host::Staged { map, stage: [alloc()?, alloc()?], src }
        };
        let host_ptr: Vec<usize> = match &host {
            Host::Pinned(h) => h.iter().map(|h| h.as_ptr().map(|p| p as usize)).collect::<std::result::Result<_, _>>()?,
            Host::Staged { stage, .. } => stage.iter().map(|h| h.as_ptr().map(|p| p as usize)).collect::<std::result::Result<_, _>>()?,
        };
        let es = dtype.size_in_bytes();
        let slots = [Tensor::zeros(bytes / es, dtype, dev)?, Tensor::zeros(bytes / es, dtype, dev)?];
        let ptr = |t: &Tensor| -> Result<u64> {
            let (st, l) = t.storage_and_layout();
            let Storage::Cuda(cs) = &*st else { bail!("slot not on CUDA") };
            let p = if dtype == DType::BF16 {
                cs.as_cuda_slice::<half::bf16>()?.slice(l.start_offset()..).device_ptr(&compute).0
            } else {
                cs.as_cuda_slice::<u8>()?.slice(l.start_offset()..).device_ptr(&compute).0
            };
            Ok(p)
        };
        let slot_ptr = [ptr(&slots[0])?, ptr(&slots[1])?];
        dev.synchronize()?;
        let s = Self {
            copy,
            compute,
            host,
            host_ptr,
            slots,
            slot_ptr,
            n,
            dtype,
            bytes,
            layout,
            state: Mutex::new(State { holds: [None, None], copied: [None, None], freed: [None, None] }),
            uploaded: Default::default(),
            stage_ns: Default::default(),
        };
        {
            let mut st = s.state.lock().unwrap();
            s.issue(&mut st, 0)?;
            s.issue(&mut st, 1)?;
        }
        Ok(s)
    }

    pub fn count(&self) -> usize {
        self.n
    }

    pub fn mode(&self) -> &'static str {
        match self.host {
            Host::Pinned(_) => "pinned",
            Host::Staged { .. } => "staged",
        }
    }

    /// Page-locked host bytes this streamer holds.
    pub fn pinned_bytes(&self) -> usize {
        match self.host {
            Host::Pinned(_) => self.bytes * self.n,
            Host::Staged { .. } => self.bytes * 2,
        }
    }

    pub fn slot_bytes(&self) -> usize {
        self.bytes * 2
    }

    /// The weight of `linear` in streamed layer `k` as an [out, in] view into its slot (u8 int8 codes or bf16;
    /// the contents change per run).
    pub fn view(&self, k: usize, linear: &str) -> Result<Tensor> {
        let &(o, out, inp) = self.layout.get(linear).with_context(|| format!("stream: {linear} not packed"))?;
        Ok(self.slots[k % 2].narrow(0, o / self.dtype.size_in_bytes(), out * inp)?.reshape((out, inp))?)
    }

    fn issue(&self, st: &mut State, k: usize) -> Result<()> {
        let s = k % 2;
        let src_ptr = match &self.host {
            Host::Pinned(_) => self.host_ptr[k],
            Host::Staged { map, src, .. } => {
                // staging buffer s is free once its previous upload (into slot s) has finished
                if let Some(e) = &st.copied[s] {
                    e.synchronize()?;
                }
                let t = std::time::Instant::now();
                let dst = self.host_ptr[s];
                let parts = &src[k];
                // split every linear into ~even pieces over the threads
                let total: usize = parts.iter().map(|p| p.2).sum();
                let piece = total.div_ceil(STAGE_THREADS).div_ceil(ALIGN) * ALIGN;
                let mut jobs = vec![]; // (file off, packed off, len)
                for &(f, o, len) in parts {
                    let mut d = 0;
                    while d < len {
                        let n = piece.min(len - d);
                        jobs.push((f + d, o + d, n));
                        d += n;
                    }
                }
                let map: &[u8] = map;
                std::thread::scope(|sc| {
                    for chunk in jobs.chunks(jobs.len().div_ceil(STAGE_THREADS)) {
                        sc.spawn(move || {
                            for &(f, o, n) in chunk {
                                let d = unsafe { std::slice::from_raw_parts_mut((dst + o) as *mut u8, n) };
                                d.copy_from_slice(&map[f..f + n]);
                            }
                        });
                    }
                });
                self.stage_ns.fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                dst
            }
        };
        if let Some(e) = &st.freed[s] {
            self.copy.wait(e)?;
        }
        let src = unsafe { std::slice::from_raw_parts(src_ptr as *const u8, self.bytes) };
        unsafe { candle_core::cuda_backend::cudarc::driver::result::memcpy_htod_async(self.slot_ptr[s], src, self.copy.cu_stream()) }?;
        st.copied[s] = Some(self.copy.record_event(None)?);
        st.holds[s] = Some(k);
        self.uploaded.fetch_add(self.bytes, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Before streamed layer `k` computes: make sure its slot holds it and the compute stream waits for the copy.
    pub fn prepare(&self, k: usize) -> Result<()> {
        let mut st = self.state.lock().unwrap();
        if st.holds[k % 2] != Some(k) {
            self.issue(&mut st, k)?; // out of order (first call after an abort): fetch it now
        }
        if let Some(e) = &st.copied[k % 2] {
            self.compute.wait(e)?;
        }
        Ok(())
    }

    /// After streamed layer `k`'s kernels are queued: free its slot and prefetch the layer two ahead.
    pub fn done(&self, k: usize) -> Result<()> {
        let mut st = self.state.lock().unwrap();
        st.freed[k % 2] = Some(self.compute.record_event(None)?);
        self.issue(&mut st, (k + 2) % self.n)
    }
}

/// A layer stack placed by a memory plan: layers `first..` stream through a `BlockStream`, the earlier ones are
/// ordinary GPU layers. Families call `linear` while loading and `prepare` / `done` around each layer's forward.
pub struct StreamedStack {
    pub stream: BlockStream,
    pub first: usize,
}

impl StreamedStack {
    /// `prefixes`: every layer's tensor prefix in run order; `plan` from core::memory::plan. None = not streaming.
    pub fn build(
        wt: &Weights,
        path: &Path,
        dev: &Device,
        prefixes: &[String],
        linears: &[&str],
        plan: crate::core::memory::StackPlan,
        hot: bool,
    ) -> Result<Option<Self>> {
        if !plan.stream {
            return Ok(None);
        }
        let first = plan.resident.min(prefixes.len());
        let stream = BlockStream::new(wt, path, dev, &prefixes[first..], linears, hot)?;
        Ok(Some(Self { stream, first }))
    }

    pub fn covers(&self, layer: usize) -> bool {
        layer >= self.first
    }

    /// The int8 linear `{prefix}.{suffix}` of `layer`, reading its codes from the layer's slot.
    pub fn linear(&self, wt: &Weights, layer: usize, prefix: &str, suffix: &str, dev: &Device) -> Result<crate::core::int8::AnyLinear> {
        let w = self.stream.view(layer - self.first, suffix)?;
        if self.stream.dtype == DType::BF16 {
            // a plain bf16 checkpoint: the block linears have no bias (the same as Linear::load(.., false))
            return Ok(crate::core::int8::AnyLinear::F(crate::core::int8::Linear { w, b: None, loras: vec![], q: None }));
        }
        Ok(crate::core::int8::AnyLinear::Q(crate::core::int8::QLinear::load_with(wt, &format!("{prefix}.{suffix}"), w, dev)?))
    }

    pub fn prepare(&self, layer: usize) -> Result<()> {
        if layer >= self.first { self.stream.prepare(layer - self.first) } else { Ok(()) }
    }

    pub fn done(&self, layer: usize) -> Result<()> {
        if layer >= self.first { self.stream.done(layer - self.first) } else { Ok(()) }
    }

    /// Where this stack went, for core::memory's report.
    pub fn placed(&self, component: &str, total: usize) -> crate::core::memory::Placed {
        crate::core::memory::Placed {
            component: component.into(),
            layers: total,
            streamed: self.stream.count(),
            pinned: self.stream.mode() == "pinned",
            locked_mb: (self.stream.pinned_bytes() >> 20) as u64,
            format: if self.stream.dtype == DType::BF16 { "bf16" } else { "int8" }.into(),
        }
    }
}
