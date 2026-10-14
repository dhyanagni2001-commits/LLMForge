#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include "fused_add_rmsnorm.h"

namespace {

std::vector<torch::Tensor> fused_add_rmsnorm(torch::Tensor x, torch::Tensor r, torch::Tensor w,
                                             double eps) {
  TORCH_CHECK(x.is_cuda() && r.is_cuda() && w.is_cuda(), "all inputs must be CUDA tensors");
  TORCH_CHECK(x.device() == r.device() && x.device() == w.device(), "inputs on different devices");
  TORCH_CHECK(x.scalar_type() == r.scalar_type() && x.scalar_type() == w.scalar_type(),
              "x, r, w must share a dtype");
  TORCH_CHECK(x.sizes() == r.sizes(), "x and r must have the same shape");
  TORCH_CHECK(x.dim() >= 1 && w.dim() == 1 && w.size(0) == x.size(-1),
              "w must be 1-D with size equal to x.size(-1)");
  TORCH_CHECK(x.is_contiguous() && r.is_contiguous() && w.is_contiguous(),
              "inputs must be contiguous");

  const c10::cuda::CUDAGuard guard(x.device());
  auto y = torch::empty_like(x);
  auto z = torch::empty_like(x);
  const int64_t hidden = x.size(-1);
  const int64_t rows = hidden == 0 ? 0 : x.numel() / hidden;
  auto stream = at::cuda::getCurrentCUDAStream();
  const float e = static_cast<float>(eps);

  switch (x.scalar_type()) {
    case at::kFloat:
      llmforge::launch_fused_add_rmsnorm<float>(x.data_ptr<float>(), r.data_ptr<float>(),
                                                w.data_ptr<float>(), y.data_ptr<float>(),
                                                z.data_ptr<float>(), rows, hidden, e, stream);
      break;
    case at::kHalf:
      llmforge::launch_fused_add_rmsnorm<__half>(
          reinterpret_cast<const __half*>(x.data_ptr()), reinterpret_cast<const __half*>(r.data_ptr()),
          reinterpret_cast<const __half*>(w.data_ptr()), reinterpret_cast<__half*>(y.data_ptr()),
          reinterpret_cast<__half*>(z.data_ptr()), rows, hidden, e, stream);
      break;
    case at::kBFloat16:
      llmforge::launch_fused_add_rmsnorm<__nv_bfloat16>(
          reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
          reinterpret_cast<const __nv_bfloat16*>(r.data_ptr()),
          reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),
          reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
          reinterpret_cast<__nv_bfloat16*>(z.data_ptr()), rows, hidden, e, stream);
      break;
    default:
      TORCH_CHECK(false, "unsupported dtype: ", x.scalar_type());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y, z};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_add_rmsnorm", &fused_add_rmsnorm, "Fused residual-add + RMSNorm -> (y, z)",
        py::arg("x"), py::arg("r"), py::arg("w"), py::arg("eps") = 1e-6);
}
