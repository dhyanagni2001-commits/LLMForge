// Fused residual-add + RMSNorm. One block per row; z is written once and re-read from
// global memory (L2-resident for typical hidden sizes) for the normalization pass.
#include "fused_add_rmsnorm.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace llmforge {
namespace {

template <typename T> __device__ __forceinline__ float to_f(T v);
template <> __device__ __forceinline__ float to_f(float v) { return v; }
template <> __device__ __forceinline__ float to_f(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float to_f(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T> __device__ __forceinline__ T from_f(float v);
template <> __device__ __forceinline__ float from_f(float v) { return v; }
template <> __device__ __forceinline__ __half from_f(float v) { return __float2half_rn(v); }
template <> __device__ __forceinline__ __nv_bfloat16 from_f(float v) { return __float2bfloat16_rn(v); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  return v;
}

__device__ float block_sum(float v) {
  __shared__ float partial[32];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  v = warp_sum(v);
  if (lane == 0) partial[warp] = v;
  __syncthreads();
  const int n_warps = (blockDim.x + 31) >> 5;
  v = (threadIdx.x < n_warps) ? partial[threadIdx.x] : 0.f;
  if (warp == 0) v = warp_sum(v);
  if (threadIdx.x == 0) partial[0] = v;
  __syncthreads();
  return partial[0];
}

// Vector width: 16-byte accesses (4 floats or 8 halfs) when hidden is divisible and
// pointers are aligned; otherwise scalar. VEC=1 handles arbitrary widths.
template <typename T, int VEC>
__global__ void fused_add_rmsnorm_kernel(const T* __restrict__ x, const T* __restrict__ r,
                                         const T* __restrict__ w, T* __restrict__ y,
                                         T* __restrict__ z, int64_t hidden, float eps) {
  struct alignas(sizeof(T) * VEC) Vec { T v[VEC]; };
  const int64_t row = blockIdx.x;
  const int64_t n_vec = hidden / VEC;
  const Vec* xv = reinterpret_cast<const Vec*>(x + row * hidden);
  const Vec* rv = reinterpret_cast<const Vec*>(r + row * hidden);
  const Vec* wv = reinterpret_cast<const Vec*>(w);
  Vec* yv = reinterpret_cast<Vec*>(y + row * hidden);
  Vec* zv = reinterpret_cast<Vec*>(z + row * hidden);

  float sq = 0.f;
  for (int64_t i = threadIdx.x; i < n_vec; i += blockDim.x) {
    Vec a = xv[i], b = rv[i], o;
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
      // Round z to T before squaring so y is computed from the z we return.
      o.v[j] = from_f<T>(to_f(a.v[j]) + to_f(b.v[j]));
      const float f = to_f(o.v[j]);
      sq += f * f;
    }
    zv[i] = o;
  }
  const float inv_rms = rsqrtf(block_sum(sq) / static_cast<float>(hidden) + eps);

  for (int64_t i = threadIdx.x; i < n_vec; i += blockDim.x) {
    Vec a = zv[i], g = wv[i], o;
#pragma unroll
    for (int j = 0; j < VEC; ++j) o.v[j] = from_f<T>(to_f(a.v[j]) * inv_rms * to_f(g.v[j]));
    yv[i] = o;
  }
}

int pick_threads(int64_t work) {
  int t = 32;
  while (t < 1024 && t < work) t <<= 1;
  return t;
}

bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; }

}  // namespace

template <typename T>
void launch_fused_add_rmsnorm(const T* x, const T* r, const T* w, T* y, T* z, int64_t rows,
                              int64_t hidden, float eps, cudaStream_t stream) {
  if (rows == 0) return;
  constexpr int VEC = 16 / sizeof(T);
  const bool vec_ok = hidden % VEC == 0 && aligned16(x) && aligned16(r) && aligned16(w) &&
                      aligned16(y) && aligned16(z);
  if (vec_ok) {
    fused_add_rmsnorm_kernel<T, VEC><<<rows, pick_threads(hidden / VEC), 0, stream>>>(
        x, r, w, y, z, hidden, eps);
  } else {
    fused_add_rmsnorm_kernel<T, 1><<<rows, pick_threads(hidden), 0, stream>>>(
        x, r, w, y, z, hidden, eps);
  }
}

template void launch_fused_add_rmsnorm<float>(const float*, const float*, const float*, float*,
                                              float*, int64_t, int64_t, float, cudaStream_t);
template void launch_fused_add_rmsnorm<__half>(const __half*, const __half*, const __half*,
                                               __half*, __half*, int64_t, int64_t, float,
                                               cudaStream_t);
template void launch_fused_add_rmsnorm<__nv_bfloat16>(const __nv_bfloat16*, const __nv_bfloat16*,
                                                      const __nv_bfloat16*, __nv_bfloat16*,
                                                      __nv_bfloat16*, int64_t, int64_t, float,
                                                      cudaStream_t);

}  // namespace llmforge
