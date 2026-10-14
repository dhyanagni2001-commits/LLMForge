"""Small decoder-only transformer whose norm and attention ops can be swapped for custom kernels."""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import reference_ops


@dataclass
class Config:
    vocab_size: int = 256
    dim: int = 128
    n_layers: int = 2
    n_heads: int = 4
    max_seq_len: int = 256
    eps: float = 1e-6


class Ops:
    """Pluggable op table. Replace entries to route the model through custom kernels."""

    def __init__(self, fused_add_rmsnorm=None, attention=None):
        self.fused_add_rmsnorm = fused_add_rmsnorm or reference_ops.fused_add_rmsnorm
        self.attention = attention or reference_ops.attention


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.attn_norm = nn.Parameter(torch.ones(cfg.dim))
        self.mlp_norm = nn.Parameter(torch.ones(cfg.dim))
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.up = nn.Linear(cfg.dim, 4 * cfg.dim, bias=False)
        self.down = nn.Linear(4 * cfg.dim, cfg.dim, bias=False)

    def forward(self, x, residual, ops: Ops):
        # Pre-norm block expressed with fused add+norm: residual stream carried separately.
        h, residual = ops.fused_add_rmsnorm(x, residual, self.attn_norm, self.cfg.eps)
        b, t, _ = h.shape
        q, k, v = self.qkv(h).view(b, t, 3, self.cfg.n_heads, -1).permute(2, 0, 3, 1, 4)
        a = ops.attention(q.contiguous(), k.contiguous(), v.contiguous(), causal=True)
        x = self.proj(a.transpose(1, 2).reshape(b, t, -1))
        h, residual = ops.fused_add_rmsnorm(x, residual, self.mlp_norm, self.cfg.eps)
        return self.down(F.silu(self.up(h))), residual


class TinyTransformer(nn.Module):
    def __init__(self, cfg: Config = Config()):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.final_norm = nn.Parameter(torch.ones(cfg.dim))
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.ops = Ops()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        residual = torch.zeros_like(x)
        for blk in self.blocks:
            x, residual = blk(x, residual, self.ops)
        h, _ = self.ops.fused_add_rmsnorm(x, residual, self.final_norm, self.cfg.eps)
        return self.head(h)

    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Greedy decoding without KV cache (deterministic, for output comparison)."""
        for _ in range(max_new_tokens):
            logits = self(tokens[:, -self.cfg.max_seq_len:])
            tokens = torch.cat([tokens, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
        return tokens
