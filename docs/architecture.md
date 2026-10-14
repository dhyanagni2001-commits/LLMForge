# Architecture and Implementation Decisions

## Semantics
`models/reference_ops.py` is the source of truth. `fused_add_rmsnorm(x, r, w, eps)` returns
`(y, z)` with `z = x + r` (in the storage dtype) and `y = rmsnorm(z) * w` computed in FP32 and
cast back. Attention uses layout `(batch, heads, seq, head_dim)`; causal masks align queries to
the end of the key sequence.

## CUDA fused residual-add + RMSNorm (`kernels/cuda/`)
- One thread block per row; threads = next power of two >= work items, capped at 1024.
- Pass 1 computes `z`, writes it, and accumulates `sum(z^2)` in FP32 from the *rounded* `z`, so
  `y` is consistent with the `z` that is returned. Pass 2 re-reads `z` (L2-resident for typical
  hidden sizes) and writes `y`.
- Reduction: warp shuffle (`__shfl_xor_sync`) then one shared-memory step across warps.
- 16-byte vectorized loads/stores (4 x fp32, 8 x fp16/bf16) when `hidden` is divisible and all
  pointers are 16-byte aligned; otherwise a scalar path that handles any width.
- Runs on the current PyTorch CUDA stream; validates device, dtype, shape, contiguity.
- Built JIT via `torch.utils.cpp_extension.load` (`kernels.load_cuda_extension()`); `CMakeLists.txt`
  is an optional alternative.

## Triton attention (`kernels/triton/attention.py`)
- Grid `(ceil(seq / BLOCK_M), batch * heads)`. Each program holds a Q tile and streams K/V tiles.
- Online softmax with `exp2` and the scale folded with `log2(e)`; running max/sum in FP32.
- Causal programs stop at the diagonal tile, skipping fully masked K/V tiles.
- Supported: forward only, equal Q/K/V heads, head_dim in {16, 32, 64, 128}, no dropout.
- Not FlashAttention-2: no work partitioning across warps for the backward, no split-KV, no
  warp specialization.

## Model (`models/transformer.py`)
Pre-norm decoder whose residual stream is carried explicitly, so every norm is a fused
add+norm. An `Ops` table selects reference or custom kernels; `runtimes/pytorch_runner.py --custom`
compares logits and greedy tokens between the two paths.

## vLLM (`runtimes/vllm_custom_op.py`)
Patches `RMSNorm.forward_cuda` for the residual case only (vLLM's contract: return
`(normed, x + residual)`). Unsupported inputs fall back to vLLM's original method. Host-side
counters record custom vs fallback calls; with CUDA graphs they count captures, so use
`--enforce-eager` or Nsight Systems as execution evidence.

## TVM (`compiler/tvm/`)
- `FuseAddRMSNorm` is a Relax module pass. A visitor finds `add -> nn.rms_norm` pairs in one
  dataflow block and checks: static equal shapes (no broadcasting), last-axis normalization,
  1-D weight of matching size/dtype, weight is a parameter or constant. A mutator emits a single
  `call_tir` to a TE-defined fused kernel at the add's position and remaps both the norm result
  and every use of `z` to its outputs. Rejection reasons are recorded in `pass.stats`.
- `schedule_rmsnorm.py` defines the TE compute and a hand-written TIR schedule: rows bound to
  `blockIdx.x`, a coalesced strided split bound to `threadIdx.x`, a cross-thread reduction for
  `sum(z^2)` stored in shared memory, and `z`/`y` stages placed under the same row loop.
- Developed against TVM 0.27 (`tirx`/`s_tir`, `TensorType`); a small shim keeps older
  `tir`/`StructInfo` names working.
