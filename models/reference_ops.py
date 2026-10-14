"""Readable PyTorch reference operations. These define the semantics every kernel must match."""
import math

import torch


def residual_add(x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    return x + r


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm over the last dim, computed in FP32 and cast back to x.dtype."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (y * w.float()).to(x.dtype)


def fused_add_rmsnorm(x: torch.Tensor, r: torch.Tensor, w: torch.Tensor, eps: float = 1e-6):
    """Returns (y, z) where z = x + r and y = rmsnorm(z) * w. Inputs are not modified."""
    z = residual_add(x, r)
    return rmsnorm(z, w, eps), z


def causal_mask(q_len: int, k_len: int, device=None) -> torch.Tensor:
    """Boolean mask, True where attention is allowed. Queries are aligned to the end of keys."""
    q = torch.arange(q_len, device=device)[:, None] + (k_len - q_len)
    k = torch.arange(k_len, device=device)[None, :]
    return k <= q


def attention(q, k, v, causal: bool = True, scale: float = None) -> torch.Tensor:
    """Explicit attention. Layout: (batch, heads, seq, head_dim). Computed in FP32."""
    scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    s = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    if causal:
        s = s.masked_fill(~causal_mask(q.shape[-2], k.shape[-2], q.device), float("-inf"))
    p = torch.softmax(s, dim=-1)
    return torch.matmul(p, v.float()).to(q.dtype)
