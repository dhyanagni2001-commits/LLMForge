"""Fused residual-add + RMSNorm benchmark: CUDA kernel vs PyTorch eager vs torch.compile.

All implementations allocate their outputs inside the timed region (same contract).
Correctness is checked before timing; failing configurations are recorded and not timed.
"""
import argparse
import time

import torch

from benchmarks.common import cuda_time_ms, environment, summarize, write
from models import reference_ops as ref


def eager(x, r, w, eps):
    z = x + r
    zf = z.float()
    return (zf * torch.rsqrt(zf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype), z


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dtype", default="float16")
    p.add_argument("--rows", type=int, nargs="+", default=[1, 32, 256])
    p.add_argument("--hidden-sizes", type=int, nargs="+", default=[768, 2048, 4096])
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--output", default="results/rmsnorm.json")
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")

    import kernels
    dtype = getattr(torch, args.dtype)
    t0 = time.perf_counter()
    kernels.load_cuda_extension()
    impls = {"cuda_fused": kernels.fused_add_rmsnorm, "torch_eager": eager}
    if not args.no_compile:
        impls["torch_compile"] = torch.compile(eager, dynamic=False)
    results = {"config": vars(args), "env": environment(),
               "extension_build_s": time.perf_counter() - t0,
               "timing": "CUDA events; output allocation included; compile excluded via warmup",
               "runs": []}

    for rows in args.rows:
        for hidden in args.hidden_sizes:
            x, r = (torch.randn(rows, hidden, device="cuda", dtype=dtype) for _ in range(2))
            w = torch.randn(hidden, device="cuda", dtype=dtype)
            y_ref, z_ref = ref.fused_add_rmsnorm(x, r, w, args.eps)
            for name, fn in impls.items():
                y, z = fn(x, r, w, args.eps)
                err = (y.float() - y_ref.float()).abs().max().item()
                ok = torch.allclose(y.float(), y_ref.float(), rtol=2e-2, atol=2e-2) and \
                    torch.equal(z, z_ref)
                entry = {"impl": name, "rows": rows, "hidden": hidden, "max_abs_err": err,
                         "correct": ok}
                if ok:
                    times = cuda_time_ms(lambda: fn(x, r, w, args.eps), args.warmup, args.iterations)
                    entry.update(summarize(times))
                    # bytes: read x, r, w; write y, z
                    gb = (4 * rows * hidden + hidden) * x.element_size() / 1e9
                    entry["effective_gbps"] = gb / (entry["median_ms"] / 1e3)
                results["runs"].append(entry)
                print(f"{name:14s} rows={rows:5d} hidden={hidden:5d} "
                      f"median={entry.get('median_ms', float('nan')):.4f}ms err={err:.2e} ok={ok}")
            base = {e["impl"]: e for e in results["runs"] if e["rows"] == rows and e["hidden"] == hidden}
            for b in ("torch_eager", "torch_compile"):
                if b in base and "median_ms" in base[b] and "median_ms" in base["cuda_fused"]:
                    base["cuda_fused"][f"speedup_vs_{b}"] = base[b]["median_ms"] / base["cuda_fused"]["median_ms"]
    write(args.output, results)


if __name__ == "__main__":
    main()
