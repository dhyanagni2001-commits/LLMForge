"""Launch an SGLang server with settings matched to the vLLM baseline.

SGLang is benchmarked as an unmodified baseline runtime; LLMForge kernels are not integrated
into it. Both servers expose an OpenAI-compatible API, so benchmarks/bench_serving.py is
used against each.

    python -m runtimes.sglang_runner --model MODEL --revision REV --port 30000
    python -m benchmarks.bench_serving --base-url http://localhost:30000/v1 --model MODEL ...
"""
import argparse
import subprocess
import sys


def build_command(args):
    return [sys.executable, "-m", "sglang.launch_server",
            "--model-path", args.model, "--revision", args.revision,
            "--dtype", args.dtype, "--port", str(args.port),
            "--mem-fraction-static", str(args.mem_fraction),
            "--context-length", str(args.context_length)] + \
           (["--disable-cuda-graph"] if args.eager else [])


def vllm_equivalent(args):
    """The matching vLLM baseline command, printed for the run log."""
    return ["vllm", "serve", args.model, "--revision", args.revision, "--dtype", args.dtype,
            "--gpu-memory-utilization", str(args.mem_fraction),
            "--max-model-len", str(args.context_length)] + (["--enforce-eager"] if args.eager else [])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default="main")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--mem-fraction", type=float, default=0.85)
    p.add_argument("--context-length", type=int, default=4096)
    p.add_argument("--eager", action="store_true", help="disable CUDA graphs")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    cmd = build_command(args)
    print("sglang:", " ".join(cmd))
    print("matching vllm:", " ".join(vllm_equivalent(args)))
    if not args.dry_run:
        sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
