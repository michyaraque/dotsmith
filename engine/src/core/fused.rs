//! Candle wrappers for the fused kernels in kernels.rs (bf16 in / bf16 out, contiguous inputs).
use anyhow::Result;
use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::CudaStorage;
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::{CpuStorage, CustomOp1, CustomOp2, CustomOp3, Layout, Shape, Tensor};
use half::bf16;

use crate::core::kernels;

type R<T> = candle_core::Result<T>;

fn cpu() -> R<(CpuStorage, Shape)> {
    candle_core::bail!("fused op needs CUDA")
}
fn cfg(grid: u32, block: u32) -> LaunchConfig {
    LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 }
}
fn w<E: std::fmt::Display>(e: E) -> candle_core::Error {
    candle_core::Error::Msg(e.to_string())
}

/// bf16(bf16(LayerNorm(x)) * bf16(1 + scale)); x [b, s, d], scale [b|1, 1, d].
struct LnMod;
impl CustomOp2 for LnMod {
    fn name(&self) -> &'static str {
        "ln_mod"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> R<(CpuStorage, Shape)> {
        cpu()
    }
    fn cuda_fwd(&self, xs: &CudaStorage, xl: &Layout, ss: &CudaStorage, sl: &Layout) -> R<(CudaStorage, Shape)> {
        let dev = xs.device().clone();
        let st = dev.cuda_stream();
        let (b, s, d) = xl.shape().dims3()?;
        let sb = sl.shape().dims()[0];
        let x = xs.as_cuda_slice::<bf16>()?.slice(xl.start_offset()..);
        let sc = ss.as_cuda_slice::<bf16>()?.slice(sl.start_offset()..);
        let mut out = unsafe { st.alloc::<bf16>(b * s * d) }.map_err(w)?;
        let f = kernels::func(&dev, "ln_mod")?;
        let (di, si, sbi) = (d as i32, s as i32, (sb > 1) as i32);
        let mut l = st.launch_builder(&f);
        l.arg(&x).arg(&sc).arg(&mut out).arg(&di).arg(&si).arg(&sbi);
        unsafe { l.launch(cfg((b * s) as u32, 256)) }.map_err(w)?;
        Ok((CudaStorage::wrap_cuda_slice(out, dev), xl.shape().clone()))
    }
}
pub fn ln_mod(x: &Tensor, scale: &Tensor) -> Result<Tensor> {
    Ok(x.contiguous()?.apply_op2_no_bwd(&scale.contiguous()?, &LnMod)?)
}

/// Per-head RMSNorm * w then RoPE, [b, s, h, 128] -> [b, h, s, 128]; cs [s, 128] = cos | sin (f32).
struct QkNormRope;
impl CustomOp3 for QkNormRope {
    fn name(&self) -> &'static str {
        "qk_norm_rope"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> R<(CpuStorage, Shape)> {
        cpu()
    }
    fn cuda_fwd(&self, xs: &CudaStorage, xl: &Layout, ws: &CudaStorage, wl: &Layout, cs: &CudaStorage, cl: &Layout) -> R<(CudaStorage, Shape)> {
        let dev = xs.device().clone();
        let st = dev.cuda_stream();
        let (b, s, h, d) = xl.shape().dims4()?;
        if d != 128 {
            candle_core::bail!("qk_norm_rope: head dim {d}");
        }
        let x = xs.as_cuda_slice::<bf16>()?.slice(xl.start_offset()..);
        let wt = ws.as_cuda_slice::<bf16>()?.slice(wl.start_offset()..);
        let c = cs.as_cuda_slice::<f32>()?.slice(cl.start_offset()..);
        let mut out = unsafe { st.alloc::<bf16>(b * s * h * d) }.map_err(w)?;
        let f = kernels::func(&dev, "qk_norm_rope")?;
        let (si, hi) = (s as i32, h as i32);
        let mut l = st.launch_builder(&f);
        l.arg(&x).arg(&wt).arg(&c).arg(&mut out).arg(&si).arg(&hi);
        unsafe { l.launch(cfg((b * s * h) as u32, 64)) }.map_err(w)?;
        Ok((CudaStorage::wrap_cuda_slice(out, dev), Shape::from((b, h, s, d))))
    }
}
pub fn qk_norm_rope(x: &Tensor, w: &Tensor, cs: &Tensor) -> Result<Tensor> {
    Ok(x.contiguous()?.apply_op3_no_bwd(w, cs, &QkNormRope)?)
}

/// Masked softmax of f32 scores [b*h, n, sk] -> bf16 probabilities.
struct SoftmaxMask {
    n: usize,
    heads: usize,
    causal: bool,
    q0: usize,
}
impl CustomOp2 for SoftmaxMask {
    fn name(&self) -> &'static str {
        "softmax_mask"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> R<(CpuStorage, Shape)> {
        cpu()
    }
    fn cuda_fwd(&self, ss: &CudaStorage, sl: &Layout, vs: &CudaStorage, vl: &Layout) -> R<(CudaStorage, Shape)> {
        let dev = ss.device().clone();
        let st = dev.cuda_stream();
        let sk = *sl.shape().dims().last().unwrap();
        let rows = sl.shape().elem_count() / sk;
        let sc = ss.as_cuda_slice::<f32>()?.slice(sl.start_offset()..);
        let v = vs.as_cuda_slice::<u8>()?.slice(vl.start_offset()..);
        let has_valid = (vl.shape().elem_count() > 1) as i32;
        let mut out = unsafe { st.alloc::<bf16>(rows * sk) }.map_err(w)?;
        let f = kernels::func(&dev, "softmax_mask")?;
        let (ski, ni, hi, ci, qi) = (sk as i32, self.n as i32, self.heads as i32, self.causal as i32, self.q0 as i32);
        let mut l = st.launch_builder(&f);
        l.arg(&sc).arg(&v).arg(&mut out).arg(&ski).arg(&ni).arg(&hi).arg(&ci).arg(&qi).arg(&has_valid);
        unsafe { l.launch(cfg(rows as u32, 256)) }.map_err(w)?;
        Ok((CudaStorage::wrap_cuda_slice(out, dev), sl.shape().clone()))
    }
}
/// scores [b*h, n, sk] f32, valid [b, sk] u8 (a 1-element tensor = every key valid).
pub fn softmax_mask(scores: &Tensor, valid: &Tensor, heads: usize, causal: bool, q0: usize) -> Result<Tensor> {
    let n = scores.dim(1)?;
    Ok(scores.contiguous()?.apply_op2_no_bwd(&valid.contiguous()?, &SoftmaxMask { n, heads, causal, q0 })?)
}

/// gu [.., 2F] -> bf16(bf16(silu(gate)) * up) [.., F]
struct SwiGlu;
impl CustomOp1 for SwiGlu {
    fn name(&self) -> &'static str {
        "swiglu"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> R<(CpuStorage, Shape)> {
        cpu()
    }
    fn cuda_fwd(&self, gs: &CudaStorage, gl: &Layout) -> R<(CudaStorage, Shape)> {
        let dev = gs.device().clone();
        let st = dev.cuda_stream();
        let mut dims = gl.shape().dims().to_vec();
        let f2 = *dims.last().unwrap();
        let m = gl.shape().elem_count() / f2;
        let fh = f2 / 2;
        let g = gs.as_cuda_slice::<bf16>()?.slice(gl.start_offset()..);
        let mut out = unsafe { st.alloc::<bf16>(m * fh) }.map_err(w)?;
        let f = kernels::func(&dev, "swiglu")?;
        let (mi, fi) = (m as i32, fh as i32);
        let mut l = st.launch_builder(&f);
        l.arg(&g).arg(&mut out).arg(&mi).arg(&fi);
        unsafe { l.launch(cfg(((m * fh) as u32).div_ceil(256), 256)) }.map_err(w)?;
        *dims.last_mut().unwrap() = fh;
        Ok((CudaStorage::wrap_cuda_slice(out, dev), Shape::from(dims)))
    }
}
pub fn swiglu(gu: &Tensor) -> Result<Tensor> {
    Ok(gu.contiguous()?.apply_op1_no_bwd(&SwiGlu)?)
}

/// x + a * tanh(g); x, a [b, s, d], g [b, 1, d]
struct GatedResidual;
impl CustomOp3 for GatedResidual {
    fn name(&self) -> &'static str {
        "gated_residual"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> R<(CpuStorage, Shape)> {
        cpu()
    }
    fn cuda_fwd(&self, xs: &CudaStorage, xl: &Layout, aas: &CudaStorage, al: &Layout, gs: &CudaStorage, gl: &Layout) -> R<(CudaStorage, Shape)> {
        let dev = xs.device().clone();
        let st = dev.cuda_stream();
        let (_, s, d) = xl.shape().dims3()?;
        let total = xl.shape().elem_count();
        let x = xs.as_cuda_slice::<bf16>()?.slice(xl.start_offset()..);
        let a = aas.as_cuda_slice::<bf16>()?.slice(al.start_offset()..);
        let g = gs.as_cuda_slice::<bf16>()?.slice(gl.start_offset()..);
        let mut out = unsafe { st.alloc::<bf16>(total) }.map_err(w)?;
        let f = kernels::func(&dev, "gated_residual")?;
        let (di, si, ti) = (d as i32, s as i32, total);
        let mut l = st.launch_builder(&f);
        l.arg(&x).arg(&a).arg(&g).arg(&mut out).arg(&di).arg(&si).arg(&ti);
        unsafe { l.launch(cfg((total as u32).div_ceil(256), 256)) }.map_err(w)?;
        Ok((CudaStorage::wrap_cuda_slice(out, dev), xl.shape().clone()))
    }
}
pub fn gated_residual(x: &Tensor, a: &Tensor, g: &Tensor) -> Result<Tensor> {
    Ok(x.contiguous()?.apply_op3_no_bwd(&a.contiguous()?, &g.contiguous()?, &GatedResidual)?)
}
