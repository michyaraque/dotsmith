//! Scaled dot-product attention for bf16 q/k/v with fp32 scores and softmax (what flash attention does), in
//! query chunks so a 1536 px canvas never materialises a full multi-GB score matrix.
//! QK^T runs on the tensor cores with fp32 output (cuBLAS strided-batched GemmEx 16BF -> 32F).
use anyhow::Result;
use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::CudaStorage;
use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
use candle_core::{CustomOp2, D, DType, Layout, Shape, Tensor};
use cudarc::cublas::sys as cbs;

/// (q [n, sq, d] bf16, k [n, sk, d] bf16) -> scale * q @ k^T as f32 [n, sq, sk].
struct ScoresF32 {
    scale: f32,
}

impl CustomOp2 for ScoresF32 {
    fn name(&self) -> &'static str {
        "scores_f32"
    }
    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("scores_f32 needs CUDA")
    }
    fn cuda_fwd(&self, qs: &CudaStorage, ql: &Layout, ks: &CudaStorage, kl: &Layout) -> candle_core::Result<(CudaStorage, Shape)> {
        let dev = qs.device().clone();
        let stream = dev.cuda_stream();
        let (n, sq, d) = ql.shape().dims3()?;
        let (n2, sk, d2) = kl.shape().dims3()?;
        if n != n2 || d != d2 || !ql.is_contiguous() || !kl.is_contiguous() {
            candle_core::bail!("scores_f32: bad shapes {:?} {:?}", ql.shape(), kl.shape());
        }
        let q = qs.as_cuda_slice::<half::bf16>()?.slice(ql.start_offset()..);
        let k = ks.as_cuda_slice::<half::bf16>()?.slice(kl.start_offset()..);
        let mut out = unsafe { stream.alloc::<f32>(n * sq * sk) }.map_err(candle_core::Error::wrap)?;
        {
            let blas = dev.cublas_handle();
            let (kp, _a) = k.device_ptr(&stream);
            let (qp, _b) = q.device_ptr(&stream);
            let (op, _c) = out.device_ptr_mut(&stream);
            let (alpha, beta) = (self.scale, 0f32);
            // column-major: S^T [sk, sq] = K [sk, d] (op T on the d x sk view) * Q^T [d, sq]
            unsafe {
                cudarc::cublas::result::gemm_strided_batched_ex(
                    *blas.handle(),
                    cbs::cublasOperation_t::CUBLAS_OP_T,
                    cbs::cublasOperation_t::CUBLAS_OP_N,
                    sk as i32,
                    sq as i32,
                    d as i32,
                    &alpha as *const f32 as *const _,
                    kp as *const _,
                    cbs::cudaDataType::CUDA_R_16BF,
                    d as i32,
                    (sk * d) as i64,
                    qp as *const _,
                    cbs::cudaDataType::CUDA_R_16BF,
                    d as i32,
                    (sq * d) as i64,
                    &beta as *const f32 as *const _,
                    op as *mut _,
                    cbs::cudaDataType::CUDA_R_32F,
                    sk as i32,
                    (sq * sk) as i64,
                    n as i32,
                    cbs::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    cbs::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                )
            }
            .map_err(candle_core::Error::wrap)?;
        }
        Ok((CudaStorage::wrap_cuda_slice(out, dev), Shape::from((n, sq, sk))))
    }
}

/// q [b, h, sq, d], k/v [b, h, sk, d] (bf16); `valid` [b, sk] u8 masks keys; `causal` limits query i to keys <= i.
/// fp32 scores on the tensor cores, fused masked softmax straight to bf16, then P @ V.
pub fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor, valid: Option<&Tensor>, causal: bool) -> Result<Tensor> {
    let (b, h, sq, d) = q.dims4()?;
    let sk = k.dim(2)?;
    let scale = 1.0 / (d as f32).sqrt();
    let q3 = q.contiguous()?.reshape((b * h, sq, d))?;
    let k3 = k.contiguous()?.reshape((b * h, sk, d))?;
    let v3 = v.contiguous()?.reshape((b * h, sk, d))?;
    let dummy;
    let valid = match valid {
        Some(v) => v.contiguous()?,
        None => {
            dummy = Tensor::ones(1, DType::U8, q.device())?;
            dummy.clone()
        }
    };
    // ~1 GB of fp32 scores per chunk
    let budget = if crate::core::low_vram() { 1usize << 26 } else { 1usize << 28 }; // fp32 scores per chunk
    let chunk = (budget / (b * h * sk).max(1)).clamp(16, sq.max(16));
    let mut outs = Vec::new();
    let mut s0 = 0;
    while s0 < sq {
        let n = chunk.min(sq - s0);
        let qc = q3.narrow(1, s0, n)?.contiguous()?;
        let s = qc.apply_op2_no_bwd(&k3, &ScoresF32 { scale })?;
        let p = crate::core::fused::softmax_mask(&s, &valid, h, causal, s0)?;
        outs.push(p.matmul(&v3)?);
        s0 += n;
    }
    let o = if outs.len() == 1 { outs.pop().unwrap() } else { Tensor::cat(&outs, 1)? };
    Ok(o.reshape((b, h, sq, d))?)
}

/// -inf where `valid` is false: [b, sk] bool-as-u8 -> f32 bias [b, 1, 1, sk].
pub fn key_bias(valid: &Tensor) -> Result<Tensor> {
    let (b, sk) = valid.dims2()?;
    let ninf = Tensor::full(f32::NEG_INFINITY, (b, sk), valid.device())?;
    let zero = Tensor::zeros((b, sk), DType::F32, valid.device())?;
    Ok(valid.ne(0u8)?.where_cond(&zero, &ninf)?.reshape((b, 1, 1, sk))?)
}

/// Causal (q >= k) bias [1, 1, s, s] combined with key validity [b, 1, 1, s].
pub fn causal_bias(valid: &Tensor) -> Result<Tensor> {
    let (_, s) = valid.dims2()?;
    let dev = valid.device();
    let mask: Vec<f32> = (0..s * s).map(|i| if i % s <= i / s { 0. } else { f32::NEG_INFINITY }).collect();
    let causal = Tensor::from_vec(mask, (1, 1, s, s), dev)?;
    Ok(causal.broadcast_add(&key_bias(valid)?)?)
}

#[allow(dead_code)]
fn _last(t: &Tensor) -> Result<usize> {
    Ok(t.dim(D::Minus1)?)
}
