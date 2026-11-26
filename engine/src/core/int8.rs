//! ConvRot int8 linear layers (ComfyUI `int8_tensorwise` + `convrot`), run exactly like ai-toolkit's
//! ConvRotInt8Quantizer inference path (W8A8):
//!   x_rot = x @ blockdiag(H)            (regular Hadamard, block = min(256, largest power-of-4 divisor of in))
//!   aq, a_s = per-row int8 quant of x_rot (amax / 127, round half to even)
//!   y = (int32(aq @ Wq^T) * w_s) * a_s   on the int8 tensor cores (cuBLAS GemmEx, 32-bit integer accumulate)
//! plus unmerged LoRA deltas: y += (x @ A^T) @ B^T * scale, in bf16.
use anyhow::{Context, Result, bail};
use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::CudaStorage;
use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg};
use candle_core::{CustomOp3, DType, Device, Layout, Shape, Tensor};
use cudarc::cublas::sys as cbs;
use std::collections::HashMap;
use std::sync::Mutex;

use crate::core::kernels;
use crate::core::weights::Weights;

pub fn largest_pow4_divisor(d: usize) -> usize {
    let mut p = 1;
    while d.is_multiple_of(p * 4) {
        p *= 4;
    }
    p
}

/// The ConvRot rotation: Kronecker powers of the 4x4 regular Hadamard, orthonormal and symmetric (self-inverse).
fn regular_hadamard(n: usize) -> Vec<f32> {
    let r4 = [[1f32, 1., 1., -1.], [1., 1., -1., 1.], [1., -1., 1., 1.], [-1., 1., 1., 1.]];
    let mut h = vec![1f32];
    let mut size = 1;
    while size < n {
        let mut next = vec![0f32; size * 4 * size * 4];
        for i in 0..size {
            for j in 0..size {
                for a in 0..4 {
                    for b in 0..4 {
                        next[(i * 4 + a) * size * 4 + (j * 4 + b)] = h[i * size + j] * r4[a][b];
                    }
                }
            }
        }
        h = next;
        size *= 4;
    }
    assert_eq!(size, n, "rot size {n} is not a power of 4");
    let norm = (n as f32).sqrt();
    h.iter().map(|v| v / norm).collect()
}

static HADAMARD: Mutex<Option<HashMap<(usize, DType), Tensor>>> = Mutex::new(None);

fn hadamard(n: usize, dtype: DType, dev: &Device) -> Result<Tensor> {
    let mut cache = HADAMARD.lock().unwrap();
    let map = cache.get_or_insert_with(HashMap::new);
    if let Some(t) = map.get(&(n, dtype)) {
        return Ok(t.clone());
    }
    let t = Tensor::from_vec(regular_hadamard(n), (n, n), dev)?.to_dtype(dtype)?;
    map.insert((n, dtype), t.clone());
    Ok(t)
}

/// Block rotation along the last dim (torch: matmul(x.reshape(-1, d/rot, rot), H)).
pub fn rotate(x: &Tensor, rot: usize) -> Result<Tensor> {
    if rot == 1 {
        return Ok(x.clone());
    }
    let dims = x.dims().to_vec();
    let h = hadamard(rot, x.dtype(), x.device())?;
    let flat = x.contiguous()?.reshape((x.elem_count() / rot, rot))?;
    Ok(flat.matmul(&h)?.reshape(dims.as_slice())?)
}

/// int8 GEMM + act quant + epilogue as one candle op: (x_rot bf16 [m,k], w u8-as-i8 [n,k], w_scale f32 [n]) -> bf16 [m,n].
struct W8A8;

impl CustomOp3 for W8A8 {
    fn name(&self) -> &'static str {
        "convrot_w8a8"
    }
    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("convrot_w8a8 needs CUDA")
    }
    fn cuda_fwd(
        &self,
        xs: &CudaStorage,
        xl: &Layout,
        ws: &CudaStorage,
        wl: &Layout,
        ss: &CudaStorage,
        sl: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let dev = xs.device().clone();
        let stream = dev.cuda_stream();
        let (m, k) = xl.shape().dims2()?;
        let (n, k2) = wl.shape().dims2()?;
        if k != k2 || !xl.is_contiguous() || !wl.is_contiguous() || !sl.is_contiguous() {
            candle_core::bail!("w8a8: bad layout x{:?} w{:?}", xl.shape(), wl.shape());
        }
        let x = xs.as_cuda_slice::<half::bf16>()?.slice(xl.start_offset()..);
        let w = ws.as_cuda_slice::<u8>()?.slice(wl.start_offset()..);
        let s = ss.as_cuda_slice::<f32>()?.slice(sl.start_offset()..);
        let mp = m.div_ceil(4) * 4; // GemmEx int8: keep every leading dim a multiple of 4
        let mut aq = unsafe { stream.alloc::<i8>(mp * k) }.map_err(candle_core::Error::wrap)?;
        if mp != m {
            stream.memset_zeros(&mut aq.slice_mut(m * k..)).map_err(candle_core::Error::wrap)?;
        }
        let mut a_s = unsafe { stream.alloc::<f32>(mp) }.map_err(candle_core::Error::wrap)?;
        let f = kernels::func(&dev, "act_quant_bf16")?;
        let mut b = stream.launch_builder(&f);
        let (ki, qmax) = (k as i32, 127f32);
        b.arg(&x).arg(&mut aq).arg(&mut a_s).arg(&ki).arg(&qmax);
        unsafe { b.launch(LaunchConfig { grid_dim: (m as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }.map_err(candle_core::Error::wrap)?;
        let mut acc = unsafe { stream.alloc::<i32>(mp * n) }.map_err(candle_core::Error::wrap)?;
        {
            let blas = dev.cublas_handle();
            let (wp, _g1) = w.device_ptr(&stream);
            let (ap, _g2) = aq.device_ptr(&stream);
            let (cp, _g3) = acc.device_ptr_mut(&stream);
            let (alpha, beta) = (1i32, 0i32);
            // column-major view: C^T[n, mp] = W[n, k] (op T on the k x n col-major view) * A^T[k, mp]
            unsafe {
                cudarc::cublas::result::gemm_ex(
                    *blas.handle(),
                    cbs::cublasOperation_t::CUBLAS_OP_T,
                    cbs::cublasOperation_t::CUBLAS_OP_N,
                    n as i32,
                    mp as i32,
                    k as i32,
                    &alpha as *const i32 as *const _,
                    wp as *const _,
                    cbs::cudaDataType::CUDA_R_8I,
                    k as i32,
                    ap as *const _,
                    cbs::cudaDataType::CUDA_R_8I,
                    k as i32,
                    &beta as *const i32 as *const _,
                    cp as *mut _,
                    cbs::cudaDataType::CUDA_R_32I,
                    n as i32,
                    cbs::cublasComputeType_t::CUBLAS_COMPUTE_32I,
                    cbs::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                )
            }
            .map_err(candle_core::Error::wrap)?;
        }
        let mut out = unsafe { stream.alloc::<half::bf16>(m * n) }.map_err(candle_core::Error::wrap)?;
        let f = kernels::func(&dev, "int8_epilogue")?;
        let mut b = stream.launch_builder(&f);
        let (mi, ni) = (m as i32, n as i32);
        b.arg(&acc).arg(&a_s).arg(&s).arg(&mut out).arg(&mi).arg(&ni);
        let total = (m * n) as u32;
        unsafe { b.launch(LaunchConfig { grid_dim: (total.div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }
            .map_err(candle_core::Error::wrap)?;
        Ok((CudaStorage::wrap_cuda_slice(out, dev), Shape::from((m, n))))
    }
}

pub struct Lora {
    pub down: Tensor, // [r, in]
    pub up: Tensor,   // [out, r]
    pub scale: f64,
}

pub struct QLinear {
    pub w: Tensor,  // u8 [out, in] holding int8 codes in the rotated basis
    pub ws: Tensor, // f32 [out]
    pub rot: usize,
    pub out_f: usize,
    pub in_f: usize,
    pub loras: Vec<Lora>,
}

impl QLinear {
    pub fn load(wt: &Weights, prefix: &str, dev: &Device) -> Result<Self> {
        let conf = wt.json(&format!("{prefix}.comfy_quant")).with_context(|| format!("{prefix}: no comfy_quant"))?;
        if conf["format"] != "int8_tensorwise" {
            bail!("{prefix}: unsupported quant format {}", conf["format"]);
        }
        let w = wt.raw(&format!("{prefix}.weight"), DType::U8, dev)?;
        Self::load_with(wt, prefix, w, dev)
    }

    /// Like `load`, with the int8 codes supplied by the caller (a streamed block's slot view, see core::stream).
    pub fn load_with(wt: &Weights, prefix: &str, w: Tensor, dev: &Device) -> Result<Self> {
        let conf = wt.json(&format!("{prefix}.comfy_quant")).with_context(|| format!("{prefix}: no comfy_quant"))?;
        let (out_f, in_f) = w.dims2()?;
        let ws = wt.tensor(&format!("{prefix}.weight_scale"), dev)?.to_dtype(DType::F32)?.reshape(out_f)?;
        let rot = if conf["convrot"].as_bool().unwrap_or(false) {
            (conf["convrot_groupsize"].as_u64().unwrap_or(256) as usize).min(largest_pow4_divisor(in_f))
        } else {
            1
        };
        Ok(Self { w, ws, rot, out_f, in_f, loras: vec![] })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let m = x.elem_count() / self.in_f;
        let x2 = x.to_dtype(DType::BF16)?.reshape((m, self.in_f))?;
        let xr = rotate(&x2, self.rot)?.contiguous()?;
        // weights kept in RAM (weight streaming's cold layers): uploaded for this call only, same numerics
        let mut y = if self.w.device().is_cpu() && !x.device().is_cpu() {
            xr.apply_op3_no_bwd(&self.w.to_device(x.device())?, &self.ws.to_device(x.device())?, &W8A8)?
        } else {
            xr.apply_op3_no_bwd(&self.w, &self.ws, &W8A8)?
        };
        for l in &self.loras {
            let d = x2.matmul(&l.down.t()?)?.matmul(&l.up.t()?)?;
            y = (y + (d * l.scale)?)?;
        }
        let mut out = dims;
        *out.last_mut().unwrap() = self.out_f;
        Ok(y.reshape(out)?)
    }
}

/// A plain bf16 linear (the modules the checkpoint keeps unquantized), with the same LoRA hook.
pub struct Linear {
    pub w: Tensor, // [out, in] ([out, 0] placeholder when `q` holds the weight)
    pub b: Option<Tensor>,
    pub loras: Vec<Lora>,
    /// GGUF: the weight kept quantized, dequantized per call like GLinear (safetensors: None)
    pub q: Option<candle_core::quantized::QTensor>,
}

impl Linear {
    pub fn load(wt: &Weights, prefix: &str, dev: &Device, bias: bool) -> Result<Self> {
        let name = format!("{prefix}.weight");
        let (w, q) = if wt.is_quantized(&name) {
            let q = wt.qtensor(&name, dev)?;
            (Tensor::zeros((q.shape().dims()[0], 0), DType::BF16, dev)?, Some(q))
        } else {
            (wt.tensor(&name, dev)?.to_dtype(DType::BF16)?, None)
        };
        let b = if bias { Some(wt.tensor(&format!("{prefix}.bias"), dev)?.to_dtype(DType::BF16)?) } else { None };
        Ok(Self { w, b, loras: vec![], q })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let k = *dims.last().unwrap();
        let x2 = x.to_dtype(DType::BF16)?.reshape((x.elem_count() / k, k))?;
        // weights kept in RAM (weight streaming's cold layers) are uploaded for this call only, same numerics
        let host = self.w.device().is_cpu() && !x.device().is_cpu();
        let mut y = match &self.q {
            Some(q) => x2.matmul(&q.dequantize_f16(x.device())?.to_dtype(DType::BF16)?.t()?)?,
            None if host => x2.matmul(&self.w.to_device(x.device())?.t()?)?,
            None => x2.matmul(&self.w.t()?)?,
        };
        if let Some(b) = &self.b {
            y = if host { y.broadcast_add(&b.to_device(x.device())?)? } else { y.broadcast_add(b)? };
        }
        for l in &self.loras {
            y = (y + (x2.matmul(&l.down.t()?)?.matmul(&l.up.t()?)? * l.scale)?)?;
        }
        let mut out = dims;
        *out.last_mut().unwrap() = self.w.dim(0)?;
        Ok(y.reshape(out)?)
    }
}

/// A GGUF linear (Q3_K, Q4_K, Q8_0 ...): the weight stays quantized on the device (about 3.4 bits per weight for
/// Q3_K) and is dequantized to a transient bf16 copy for each matmul, so peak VRAM grows by one layer, not the model.
/// The activations keep the bf16 numerics of the int8 path; only the weights carry the quantization error.
pub struct GLinear {
    pub w: candle_core::quantized::QTensor, // [out, in]
    pub out_f: usize,
    pub in_f: usize,
    pub loras: Vec<Lora>,
}

impl GLinear {
    pub fn load(wt: &Weights, prefix: &str, dev: &Device) -> Result<Self> {
        let w = wt.qtensor(&format!("{prefix}.weight"), dev)?;
        let (out_f, in_f) = w.shape().dims2()?;
        Ok(Self { w, out_f, in_f, loras: vec![] })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let m = x.elem_count() / self.in_f;
        let x2 = x.to_dtype(DType::BF16)?.reshape((m, self.in_f))?;
        let w = if self.w.device().is_cpu() {
            // block swap: the quantized weight lives in RAM and is uploaded for this call only
            let g = candle_core::quantized::ggml_file::qtensor_from_ggml(self.w.dtype(), &self.w.data()?, self.w.shape().dims().to_vec(), x.device())?;
            g.dequantize_f16(x.device())?.to_dtype(DType::BF16)?
        } else {
            self.w.dequantize_f16(x.device())?.to_dtype(DType::BF16)?
        };
        let mut y = x2.matmul(&w.t()?)?;
        drop(w);
        for l in &self.loras {
            y = (y + (x2.matmul(&l.down.t()?)?.matmul(&l.up.t()?)? * l.scale)?)?;
        }
        let mut out = dims;
        *out.last_mut().unwrap() = self.out_f;
        Ok(y.reshape(out)?)
    }
}

/// Either kind, as the checkpoint stores it.
pub enum AnyLinear {
    Q(QLinear),
    F(Linear),
    G(GLinear),
}

impl AnyLinear {
    pub fn load(wt: &Weights, prefix: &str, dev: &Device) -> Result<Self> {
        if wt.has(&format!("{prefix}.comfy_quant")) {
            Ok(AnyLinear::Q(QLinear::load(wt, prefix, dev)?))
        } else if wt.is_quantized(&format!("{prefix}.weight")) {
            Ok(AnyLinear::G(GLinear::load(wt, prefix, dev)?))
        } else {
            Ok(AnyLinear::F(Linear::load(wt, prefix, dev, false)?))
        }
    }
    /// Like `load`, but a GGUF-quantized weight stays in RAM (block swap) and is uploaded per call; anything else
    /// loads on `dev` as usual.
    pub fn load_swapped(wt: &Weights, prefix: &str, dev: &Device) -> Result<Self> {
        if wt.is_quantized(&format!("{prefix}.weight")) { Ok(AnyLinear::G(GLinear::load(wt, prefix, &Device::Cpu)?)) } else { Self::load(wt, prefix, dev) }
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            AnyLinear::Q(q) => q.forward(x),
            AnyLinear::F(f) => f.forward(x),
            AnyLinear::G(g) => g.forward(x),
        }
    }
    pub fn loras_mut(&mut self) -> &mut Vec<Lora> {
        match self {
            AnyLinear::Q(q) => &mut q.loras,
            AnyLinear::F(f) => &mut f.loras,
            AnyLinear::G(g) => &mut g.loras,
        }
    }
    pub fn out_features(&self) -> usize {
        match self {
            AnyLinear::Q(q) => q.out_f,
            AnyLinear::F(f) => f.w.dims()[0],
            AnyLinear::G(g) => g.out_f,
        }
    }
}
