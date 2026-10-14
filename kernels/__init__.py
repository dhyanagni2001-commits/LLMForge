"""Kernel entry points. The CUDA extension is JIT-built on first use via torch.utils.cpp_extension."""
from functools import lru_cache
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=None)
def load_cuda_extension(verbose: bool = False):
    from torch.utils.cpp_extension import load

    return load(
        name="llmforge_cuda",
        sources=[str(_ROOT / "bindings/torch_extension.cpp"),
                 str(_ROOT / "kernels/cuda/fused_add_rmsnorm.cu")],
        extra_include_paths=[str(_ROOT / "kernels/cuda")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=verbose,
    )


def cuda_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        load_cuda_extension()
        return True
    except Exception:
        return False


def fused_add_rmsnorm(x, r, w, eps: float = 1e-6):
    """CUDA fused residual-add + RMSNorm. Returns (y, z)."""
    y, z = load_cuda_extension().fused_add_rmsnorm(x, r, w, eps)
    return y, z
