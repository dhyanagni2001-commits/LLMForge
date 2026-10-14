"""Shared timing and environment-recording helpers."""
import json
import platform
import statistics
import subprocess
from pathlib import Path

import torch


def cuda_time_ms(fn, warmup: int, iterations: int):
    """Per-iteration latencies (ms) using CUDA events. Inputs must be pre-allocated by caller."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iterations):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return times


def summarize(times):
    s = sorted(times)
    return {"median_ms": statistics.median(s), "p10_ms": s[len(s) // 10],
            "p90_ms": s[(9 * len(s)) // 10], "stdev_ms": statistics.pstdev(s), "n": len(s)}


def environment():
    def run(cmd):
        try:
            return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    env = {"commit": run(["git", "rev-parse", "HEAD"]), "python": platform.python_version(),
           "torch": torch.__version__, "cuda_build": torch.version.cuda}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        env.update(gpu=p.name, gpu_memory_gb=round(p.total_memory / 2**30, 1),
                   driver=run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]))
    for mod in ("triton", "vllm", "sglang", "flash_attn", "flashinfer", "tvm"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            pass
    return env


def write(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2))
    print(f"wrote {path}")
