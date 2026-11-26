// Pre-included into every kernel of the legacy build (NVCC_APPEND_FLAGS=--pre-include) for GTX 10 cards: candle's f16
// reductions call atomicAdd on __half, which only exists natively from compute 7.0.
#pragma once
#include <cuda_fp16.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
__device__ __forceinline__ __half atomicAdd(__half* address, __half val) {
    unsigned int* word = (unsigned int*)((size_t)address & ~(size_t)2);
    const bool high = (size_t)address & 2;
    unsigned int old = *word, assumed;
    do {
        assumed = old;
        unsigned short bits = high ? (unsigned short)(assumed >> 16) : (unsigned short)(assumed & 0xffff);
        unsigned short sum = __half_as_ushort(__hadd(__ushort_as_half(bits), val));
        unsigned int next = high ? ((assumed & 0xffffu) | ((unsigned int)sum << 16)) : ((assumed & 0xffff0000u) | sum);
        old = atomicCAS(word, assumed, next);
    } while (assumed != old);
    return __ushort_as_half(high ? (unsigned short)(old >> 16) : (unsigned short)(old & 0xffff));
}
#endif

// candle also builds mixture-of-experts kernels on tensor cores (compute 7.0+). Dotsmith never runs them; before 7.0
// they compile against these empty stand-ins.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
namespace nvcuda {
namespace wmma {
struct matrix_a {};
struct matrix_b {};
struct accumulator {};
struct row_major {};
struct col_major {};
enum layout_t { mem_row_major, mem_col_major };
template <typename Use, int M, int N, int K, typename T, typename Layout = void>
struct fragment {
    T x[1];
};
template <typename F, typename V>
__device__ inline void fill_fragment(F&, const V&) {}
template <typename F, typename P>
__device__ inline void load_matrix_sync(F&, const P*, unsigned) {}
template <typename F, typename P>
__device__ inline void load_matrix_sync(F&, const P*, unsigned, layout_t) {}
template <typename D, typename A, typename B, typename C>
__device__ inline void mma_sync(D&, const A&, const B&, const C&) {}
template <typename P, typename F>
__device__ inline void store_matrix_sync(P*, const F&, unsigned, layout_t) {}
}  // namespace wmma
}  // namespace nvcuda
#endif
