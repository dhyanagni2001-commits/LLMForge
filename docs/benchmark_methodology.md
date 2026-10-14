# Benchmark Methodology

## Kernel benchmarks (`bench_rmsnorm.py`, `bench_attention.py`)
- Inputs allocated outside the timed region; output allocation is *inside* it for every
  implementation (identical contract).
- Correctness is checked against the reference before timing; failing configs are recorded
  with `correct: false` and not timed.
- Extension build time is recorded separately (`extension_build_s`). `torch.compile` and Triton
  autotuning/JIT are excluded by warmup.
- Timing: one CUDA event pair per iteration; report median, p10, p90, stdev, n.
- Derived: effective GB/s for RMSNorm (reads x, r, w; writes y, z), TFLOP/s for attention
  (causal counted as half of dense).
- Baselines: PyTorch eager and `torch.compile` for RMSNorm; explicit PyTorch, SDPA,
  FlashAttention and FlashInfer (when installed) for attention.
- Every JSON includes `env`: commit, GPU, driver, CUDA, and library versions.

## Serving benchmarks (`bench_serving.py`)
- Same prompts (seeded), `temperature=0`, `ignore_eos=true`, fixed max_tokens per run.
- Streaming `/v1/completions`. TTFT = request start to first non-empty chunk; ITL = gaps
  between non-empty chunks; output token counts from the server's final `usage` block.
- Warmup requests are sent and discarded before each run.
- Reported per concurrency level: p50/p95 TTFT, ITL, end-to-end; output and request
  throughput; completed/failed counts.
- Matched configs: use `runtimes/sglang_runner.py`, which prints the equivalent vLLM command
  (same model revision, dtype, memory fraction, context length, CUDA-graph setting).

## Reporting
Speedup = baseline / candidate latency. Always name the metric, workload and baseline.
Kernel speedups are never reported as serving speedups.
