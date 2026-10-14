#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace llmforge {

// Computes z = x + r and y = z * rsqrt(mean(z^2) + eps) * w over the last dimension.
// All pointers are contiguous row-major [rows, hidden]; w is [hidden]. Accumulation is FP32.
// T is float, __half, or __nv_bfloat16.
template <typename T>
void launch_fused_add_rmsnorm(const T* x, const T* r, const T* w, T* y, T* z,
                              int64_t rows, int64_t hidden, float eps, cudaStream_t stream);

}  // namespace llmforge
