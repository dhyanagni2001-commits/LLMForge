"""Tiled causal attention forward pass in Triton with online softmax.

Layout: q, k, v are contiguous (batch, heads, seq, head_dim) with equal head counts.
Supported head_dim: 16, 32, 64, 128. Dtypes: float16, bfloat16, float32. No dropout.
The full (seq x seq) score matrix is never materialized. This is a tiled online-softmax
kernel, not a complete FlashAttention-2 implementation (no warp specialization, no
sequence-parallel backward, etc.).
"""
import math

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:  # CPU-only dev machines
    HAS_TRITON = False

SUPPORTED_HEAD_DIMS = (16, 32, 64, 128)

if HAS_TRITON:

    @triton.jit
    def _attn_fwd(Q, K, V, O, sm_scale, N_CTX,
                  stride_bh, stride_s,
                  CAUSAL: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  HEAD_DIM: tl.constexpr):
        start_m = tl.program_id(0)
        bh = tl.program_id(1)
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, HEAD_DIM)
        base = bh * stride_bh

        q = tl.load(Q + base + offs_m[:, None] * stride_s + offs_d[None, :],
                    mask=offs_m[:, None] < N_CTX, other=0.0)
        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
        qk_scale = sm_scale * 1.44269504  # fold log2(e) so we can use exp2

        hi = tl.minimum((start_m + 1) * BLOCK_M, N_CTX) if CAUSAL else N_CTX
        for start_n in range(0, hi, BLOCK_N):
            cols = start_n + offs_n
            k = tl.load(K + base + cols[None, :] * stride_s + offs_d[:, None],
                        mask=cols[None, :] < N_CTX, other=0.0)
            s = tl.dot(q, k) * qk_scale
            valid = cols[None, :] < N_CTX
            if CAUSAL:
                valid = valid & (offs_m[:, None] >= cols[None, :])
            s = tl.where(valid, s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            # Rows past N_CTX keep m=-inf; guard to avoid (-inf) - (-inf) = nan.
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            p = tl.math.exp2(s - m_safe[:, None])
            alpha = tl.math.exp2(m_i - m_safe)
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            v = tl.load(V + base + cols[:, None] * stride_s + offs_d[None, :],
                        mask=cols[:, None] < N_CTX, other=0.0)
            acc += tl.dot(p.to(v.dtype), v)
            m_i = m_new

        acc = acc / tl.where(l_i == 0, 1.0, l_i)[:, None]
        tl.store(O + base + offs_m[:, None] * stride_s + offs_d[None, :],
                 acc.to(O.dtype.element_ty), mask=offs_m[:, None] < N_CTX)


def triton_attention(q, k, v, causal: bool = True, sm_scale: float = None,
                     block_m: int = 64, block_n: int = 64):
    if not HAS_TRITON:
        raise RuntimeError("triton is not installed")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("inputs must be CUDA tensors")
    if q.dim() != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, v must share shape (batch, heads, seq, head_dim)")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32) or \
            not (q.dtype == k.dtype == v.dtype):
        raise ValueError(f"unsupported dtype {q.dtype}")
    b, h, n, d = q.shape
    if d not in SUPPORTED_HEAD_DIMS:
        raise ValueError(f"head_dim {d} not in {SUPPORTED_HEAD_DIMS}")
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    o = torch.empty_like(q)
    sm_scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(d)
    grid = (triton.cdiv(n, block_m), b * h)
    _attn_fwd[grid](q, k, v, o, sm_scale, n, q.stride(1), q.stride(2),
                    CAUSAL=causal, BLOCK_M=block_m, BLOCK_N=block_n, HEAD_DIM=d,
                    num_warps=4 if d <= 64 else 8, num_stages=2)
    return o
