"""Causal attention forward benchmark: Triton kernel vs PyTorch explicit / SDPA /
FlashAttention / FlashInfer, on identical (batch, heads, seq, head_dim) inputs.
Library baselines are skipped (and recorded as such) when not installed or incompatible.
"""
import argparse

import torch
import torch.nn.functional as F

from benchmarks.common import cuda_time_ms, environment, summarize, write
from models import reference_ops as ref


def baselines(causal):
    impls = {"torch_explicit": lambda q, k, v: ref.attention(q, k, v, causal=causal),
             "torch_sdpa": lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=causal)}
    try:
        from kernels.triton.attention import triton_attention
        impls["triton"] = lambda q, k, v: triton_attention(q, k, v, causal=causal)
    except Exception as e:
        print("triton unavailable:", e)
    try:
        from flash_attn import flash_attn_func  # expects (b, s, h, d)
        impls["flash_attn"] = lambda q, k, v: flash_attn_func(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=causal).transpose(1, 2)
    except Exception:
        pass
    try:
        import flashinfer  # single_prefill expects (s, h, d) per sequence

        def fi(q, k, v):
            return torch.stack([flashinfer.single_prefill_with_kv_cache(
                q[i].transpose(0, 1), k[i].transpose(0, 1), v[i].transpose(0, 1),
                causal=causal).transpose(0, 1) for i in range(q.shape[0])])
        impls["flashinfer"] = fi
    except Exception:
        pass
    return impls


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dtype", default="float16")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4])
    p.add_argument("--seq-lengths", type=int, nargs="+", default=[128, 512, 2048])
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--causal", action="store_true")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--output", default="results/attention.json")
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")

    dtype = getattr(torch, args.dtype)
    impls = baselines(args.causal)
    results = {"config": vars(args), "env": environment(), "impls": list(impls), "runs": []}
    for b in args.batch_sizes:
        for n in args.seq_lengths:
            q, k, v = (torch.randn(b, args.heads, n, args.head_dim, device="cuda", dtype=dtype)
                       for _ in range(3))
            expected = ref.attention(q.float(), k.float(), v.float(), causal=args.causal)
            # Causal FLOPs are ~half of dense.
            flops = 4 * b * args.heads * n * n * args.head_dim * (0.5 if args.causal else 1.0)
            for name, fn in impls.items():
                entry = {"impl": name, "batch": b, "seq": n}
                try:
                    out = fn(q, k, v)
                    entry["max_abs_err"] = (out.float() - expected).abs().max().item()
                    entry["correct"] = entry["max_abs_err"] < 2e-2
                    if entry["correct"]:
                        entry.update(summarize(cuda_time_ms(lambda: fn(q, k, v),
                                                            args.warmup, args.iterations)))
                        entry["tflops"] = flops / (entry["median_ms"] / 1e3) / 1e12
                except Exception as e:  # e.g. OOM for explicit attention at long seq
                    entry["error"] = f"{type(e).__name__}: {e}"[:200]
                results["runs"].append(entry)
                print(f"{name:15s} b={b} n={n:5d} median={entry.get('median_ms', float('nan')):.4f}ms "
                      f"err={entry.get('max_abs_err', float('nan')):.2e}")
    write(args.output, results)


if __name__ == "__main__":
    main()
