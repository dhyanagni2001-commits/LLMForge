"""vLLM integration of the LLMForge fused residual-add + RMSNorm kernel.

Pinned target: vLLM 0.6.x-0.8.x, where `vllm.model_executor.layers.layernorm.RMSNorm`
is a CustomOp whose `forward_cuda(x, residual=None)` handles both plain and fused-add
norm. When `residual` is given, vLLM's contract is to return (normed, x + residual).

Usage (in the server process, before the model is built):

    import runtimes.vllm_custom_op as op
    op.install()            # patches RMSNorm.forward_cuda
    ...                     # start vLLM (e.g. LLM(...) or the OpenAI server entrypoint)
    op.stats()              # {'custom': N, 'fallback': M}  -> proves the kernel executed

Or via env: `LLMFORGE_VLLM_PATCH=1 python -m runtimes.vllm_custom_op serve <model> ...`.

Fallback: inputs the kernel does not support (non-contiguous, mismatched dtypes, CPU,
or variance_size_override) go through vLLM's original implementation. Counters are plain
Python ints incremented on the host, so under CUDA graph capture they count captures,
not replays; compare runs with `--enforce-eager` when using counts as execution evidence,
and use Nsight Systems to confirm the `fused_add_rmsnorm_kernel` symbol in graph mode.
"""
import os
import sys

_STATS = {"custom": 0, "fallback": 0}
_ORIGINAL = None


def _supported(layer, x, residual):
    w = layer.weight
    return (residual is not None and x.is_cuda and x.is_contiguous() and residual.is_contiguous()
            and x.dtype == residual.dtype == w.dtype and x.shape == residual.shape
            and getattr(layer, "variance_size_override", None) is None)


def _forward_cuda(self, x, residual=None):
    if not _supported(self, x, residual):
        _STATS["fallback"] += 1
        return _ORIGINAL(self, x, residual)
    import kernels
    _STATS["custom"] += 1
    y, z = kernels.fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
    return y, z


def install():
    """Patch vLLM's RMSNorm. Idempotent. Builds the extension eagerly so failures are loud."""
    global _ORIGINAL
    import kernels
    from vllm.model_executor.layers.layernorm import RMSNorm

    if _ORIGINAL is None:
        kernels.load_cuda_extension()
        _ORIGINAL = RMSNorm.forward_cuda
        RMSNorm.forward_cuda = _forward_cuda


def uninstall():
    global _ORIGINAL
    if _ORIGINAL is not None:
        from vllm.model_executor.layers.layernorm import RMSNorm
        RMSNorm.forward_cuda = _ORIGINAL
        _ORIGINAL = None


def stats():
    return dict(_STATS)


if os.environ.get("LLMFORGE_VLLM_PATCH") == "1":
    install()


if __name__ == "__main__":
    # Launch the vLLM CLI with the patch applied in this process. Note: with multiprocessing
    # engines (V1 / tensor parallel), set VLLM_ENABLE_V1_MULTIPROCESSING=0 or export
    # LLMFORGE_VLLM_PATCH=1 and import this module from a sitecustomize so workers patch too.
    install()
    from vllm.entrypoints.cli.main import main as vllm_main
    sys.argv = ["vllm"] + sys.argv[1:]
    vllm_main()
