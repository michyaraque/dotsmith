//! Hand-written CUDA kernels, compiled once with NVRTC at runtime (no nvcc needed on the user's machine).
//! bf16 is handled as raw u16 so the source needs no CUDA headers and builds for any card. Kernels the f32 activation
//! path also uses (cards without bf16, see core::act_dtype) are templates over the storage type: `name` is the bf16
//! one, `name_f32` the f32 one.
use candle_core::Result;
use candle_core::cuda_backend::CudaDevice;
use candle_core::cuda_backend::cudarc::driver::{CudaFunction, CudaModule};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

const SRC: &str = r#"
__device__ __forceinline__ float bf2f(unsigned short h) { return __uint_as_float(((unsigned int)h) << 16); }
__device__ __forceinline__ unsigned short f2bf(float f) {
    unsigned int u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (unsigned short)((u >> 16) | 0x40);  // NaN stays NaN
    u += 0x7fffu + ((u >> 16) & 1u);                                                 // round to nearest even
    return (unsigned short)(u >> 16);
}

// storage types: load as f32, store from f32
struct Bf16 { typedef unsigned short T; static __device__ float ld(T h) { return bf2f(h); } static __device__ T st(float f) { return f2bf(f); } };
struct F32 { typedef float T; static __device__ float ld(T v) { return v; } static __device__ T st(float f) { return f; } };

// Per-row symmetric int8 activation quantization (toolkit convrot int8_act_quant_kernel):
// scale = amax / qmax (1 when amax == 0), q = rint(x / scale) clamped. One block per row.
template <class D> __device__ void act_quant_t(const typename D::T* x, signed char* q, float* scales, int K, float qmax) {
    int row = blockIdx.x;
    const typename D::T* xr = x + (size_t)row * K;
    signed char* qr = q + (size_t)row * K;
    __shared__ float red[256];
    float m = 0.f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) m = fmaxf(m, fabsf(D::ld(xr[i])));
    red[threadIdx.x] = m;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    float amax = red[0];
    float scale = amax > 0.f ? amax / qmax : 1.f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float v = rintf(D::ld(xr[i]) / scale);
        v = fminf(fmaxf(v, -qmax), qmax);
        qr[i] = (signed char)v;
    }
    if (threadIdx.x == 0) scales[row] = scale;
}
extern "C" __global__ void act_quant(const unsigned short* x, signed char* q, float* s, int K, float m) { act_quant_t<Bf16>(x, q, s, K, m); }
extern "C" __global__ void act_quant_f32(const float* x, signed char* q, float* s, int K, float m) { act_quant_t<F32>(x, q, s, K, m); }

// out[r, c] = (float(acc) * w_scale[c]) * a_scale[r] in the activation type (toolkit _int8_epilogue, same op order)
template <class D> __device__ void int8_epilogue_t(const int* acc, const float* a_s, const float* w_s, typename D::T* out, int M, int N) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)M * N) return;
    int r = i / N, c = i % N;
    out[i] = D::st(((float)acc[i] * w_s[c]) * a_s[r]);
}
extern "C" __global__ void int8_epilogue(const int* acc, const float* a, const float* w, unsigned short* o, int M, int N) { int8_epilogue_t<Bf16>(acc, a, w, o, M, N); }
extern "C" __global__ void int8_epilogue_f32(const int* acc, const float* a, const float* w, float* o, int M, int N) { int8_epilogue_t<F32>(acc, a, w, o, M, N); }

// Signed int8 rows times their per-row scale -> f32 (int8 embedding rows before the inverse rotation).
extern "C" __global__ void i8_rows_to_f32(const signed char* q, const float* s, float* out, int rows, int K) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * K) return;
    out[i] = (float)q[i] * s[i / K];
}

// ---- fused transformer ops (one memory pass each, same bf16 rounding points as the PyTorch graph) ----
__device__ __forceinline__ float bfr(float f) { return bf2f(f2bf(f)); }   // round through bf16

// out = bf16(bf16(LN(x)) * bf16(1 + s)); x [rows, D], s [B, D] with row -> batch = row / S (S = rows per batch)
extern "C" __global__ void ln_mod(const unsigned short* x, const unsigned short* s, unsigned short* out, int D, int S, int sb) {
    int row = blockIdx.x;
    const unsigned short* xr = x + (size_t)row * D;
    const unsigned short* sr = s + (size_t)(sb ? row / S : 0) * D;
    __shared__ float red[256];
    float a = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) a += bf2f(xr[i]);
    red[threadIdx.x] = a; __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) { if (threadIdx.x < k) red[threadIdx.x] += red[threadIdx.x + k]; __syncthreads(); }
    float mean = red[0] / D; __syncthreads();
    float v = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) { float d = bf2f(xr[i]) - mean; v += d * d; }
    red[threadIdx.x] = v; __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) { if (threadIdx.x < k) red[threadIdx.x] += red[threadIdx.x + k]; __syncthreads(); }
    float rstd = rsqrtf(red[0] / D + 1e-6f);
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float y = bfr((bf2f(xr[i]) - mean) * rstd);
        float m = bfr(1.f + bf2f(sr[i]));
        out[(size_t)row * D + i] = f2bf(y * m);
    }
}

// per-head RMSNorm (fp32 -> bf16, * w -> bf16) then complex RoPE on (even, odd) pairs (fp32 -> bf16),
// x [B, S, H, 128] -> out [B, H, S, 128]; cs [S, 128] = cos(64) | sin(64). One warp-sized block of 64 threads per head.
extern "C" __global__ void qk_norm_rope(const unsigned short* x, const unsigned short* w, const float* cs,
                                        unsigned short* out, int S, int H) {
    int vec = blockIdx.x;                 // (b * S + s) * H + h
    int h = vec % H, bs = vec / H, s = bs % S, b = bs / S;
    const unsigned short* xv = x + (size_t)vec * 128;
    int j = threadIdx.x;                  // pair index 0..63
    float x0 = bf2f(xv[2 * j]), x1 = bf2f(xv[2 * j + 1]);
    __shared__ float red[64];
    red[j] = x0 * x0 + x1 * x1; __syncthreads();
    for (int k = 32; k > 0; k >>= 1) { if (j < k) red[j] += red[j + k]; __syncthreads(); }
    float r = rsqrtf(red[0] / 128.f + 1e-6f);
    float y0 = bfr(bfr(x0 * r) * bf2f(w[2 * j])), y1 = bfr(bfr(x1 * r) * bf2f(w[2 * j + 1]));
    float c = cs[(size_t)s * 128 + j], sn = cs[(size_t)s * 128 + 64 + j];
    unsigned short* o = out + (((size_t)b * H + h) * S + s) * 128;
    o[2 * j] = f2bf(y0 * c - y1 * sn);
    o[2 * j + 1] = f2bf(y0 * sn + y1 * c);
}

// masked softmax of fp32 scores [B*H*n, sk] -> P in the activation type. Keys with valid[b, k] == 0 get 0;
// causal: k <= q0 + qi.
template <class D> __device__ void softmax_mask_t(const float* sc, const unsigned char* valid, typename D::T* p,
                                                int sk, int n, int H, int causal, int q0, int has_valid) {
    int row = blockIdx.x;
    int qi = row % n, b = (row / n) / H;
    const float* r = sc + (size_t)row * sk;
    typename D::T* pr = p + (size_t)row * sk;
    const unsigned char* vb = valid + (size_t)b * sk;
    int lim = causal ? min(sk, q0 + qi + 1) : sk;
    __shared__ float red[256];
    float m = __int_as_float(0xff800000);
    for (int k = threadIdx.x; k < lim; k += blockDim.x) if (!has_valid || vb[k]) m = fmaxf(m, r[k]);
    red[threadIdx.x] = m; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]); __syncthreads(); }
    m = red[0]; __syncthreads();
    float sum = 0.f;
    for (int k = threadIdx.x; k < lim; k += blockDim.x) if (!has_valid || vb[k]) sum += __expf(r[k] - m);
    red[threadIdx.x] = sum; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s]; __syncthreads(); }
    float inv = 1.f / red[0];
    for (int k = threadIdx.x; k < sk; k += blockDim.x) {
        bool ok = k < lim && (!has_valid || vb[k]);
        pr[k] = D::st(ok ? __expf(r[k] - m) * inv : 0.f);
    }
}
extern "C" __global__ void softmax_mask(const float* sc, const unsigned char* v, unsigned short* p, int sk, int n, int H, int c, int q0, int hv) {
    softmax_mask_t<Bf16>(sc, v, p, sk, n, H, c, q0, hv);
}
extern "C" __global__ void softmax_mask_f32(const float* sc, const unsigned char* v, float* p, int sk, int n, int H, int c, int q0, int hv) {
    softmax_mask_t<F32>(sc, v, p, sk, n, H, c, q0, hv);
}

// gu [M, 2F] -> bf16(bf16(silu(gate)) * up)
extern "C" __global__ void swiglu(const unsigned short* gu, unsigned short* out, int M, int F) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)M * F) return;
    size_t r = i / F, c = i % F;
    float g = bf2f(gu[r * 2 * F + c]), u = bf2f(gu[r * 2 * F + F + c]);
    out[i] = f2bf(bfr(g / (1.f + __expf(-g))) * u);
}

// out = x + a * tanh(g) with the bf16 rounding of each torch op; g [B, D] by row / S
extern "C" __global__ void gated_residual(const unsigned short* x, const unsigned short* a, const unsigned short* g,
                                          unsigned short* out, int D, int S, size_t total) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    size_t row = i / D, c = i % D;
    float t = bfr(tanhf(bf2f(g[(row / S) * D + c])));
    out[i] = f2bf(bf2f(x[i]) + bfr(bf2f(a[i]) * t));
}
"#;

static PTX: OnceLock<String> = OnceLock::new();
static MODULE_CACHE: Mutex<Option<Arc<CudaModule>>> = Mutex::new(None);
static FUNCS: Mutex<Option<HashMap<String, CudaFunction>>> = Mutex::new(None);

/// A kernel of SRC, compiled (NVRTC) and loaded once per process. Single-GPU engine: one module.
pub fn func(dev: &CudaDevice, name: &str) -> Result<CudaFunction> {
    let mut funcs = FUNCS.lock().unwrap();
    let map = funcs.get_or_insert_with(HashMap::new);
    if let Some(f) = map.get(name) {
        return Ok(f.clone());
    }
    let ptx = PTX.get_or_init(|| {
        let opts = cudarc::nvrtc::CompileOptions { arch: Some(crate::gpu::nvrtc_arch()), ..Default::default() };
        cudarc::nvrtc::compile_ptx_with_opts(SRC, opts).expect("NVRTC failed to compile the engine kernels").to_src()
    });
    let mut m = MODULE_CACHE.lock().unwrap();
    if m.is_none() {
        let module = dev.cuda_stream().context().load_module(cudarc::nvrtc::Ptx::from_src(ptx.clone())).map_err(candle_core::Error::wrap)?;
        *m = Some(module);
    }
    let f = m.as_ref().unwrap().load_function(name).map_err(candle_core::Error::wrap)?;
    map.insert(name.to_string(), f.clone());
    Ok(f)
}

#[cfg(test)]
mod tests {
    /// The engine's kernels compile for every architecture this NVRTC targets: RTX 20 (sm_75), 30 (sm_86), 40 (sm_89),
    /// 50 (sm_120), A100 / H100 / B200 (sm_80 / 90 / 100). Needs NVRTC, not the GPU.
    #[test]
    fn kernels_compile_for_every_supported_gpu() {
        for arch in ["compute_75", "compute_80", "compute_86", "compute_89", "compute_90", "compute_100", "compute_120"] {
            let opts = cudarc::nvrtc::CompileOptions { arch: Some(arch), ..Default::default() };
            if let Err(e) = cudarc::nvrtc::compile_ptx_with_opts(super::SRC, opts) {
                panic!("{arch}: {e:?}");
            }
        }
    }
}
