# Profiling

No GPU profiling has been run yet. Commands to use:

```bash
# Timeline: launch overhead, idle gaps, and proof the custom kernel runs in serving
nsys profile -t cuda,nvtx -o results/rmsnorm python -m benchmarks.bench_rmsnorm --iterations 50
nsys stats results/rmsnorm.nsys-rep | grep fused_add_rmsnorm_kernel

# Kernel metrics: memory throughput, occupancy, registers, stalls
ncu --kernel-name regex:fused_add_rmsnorm_kernel --set full -o results/rmsnorm_ncu \
    python -m benchmarks.bench_rmsnorm --rows 256 --hidden-sizes 4096 --warmup 5 --iterations 5
ncu --kernel-name regex:_attn_fwd --set full -o results/attn_ncu \
    python -m benchmarks.bench_attention --batch-sizes 1 --seq-lengths 2048 --causal --iterations 5
```

Profile runs are separate from timing runs. For each optimization record:
observed bottleneck, change, profiling evidence, correctness, benchmark outcome, tradeoffs.

## Expectations to verify
- RMSNorm is memory-bound: compare achieved DRAM throughput to peak (`dram__throughput`).
- Fusion should remove one kernel launch and one round-trip of `z` vs eager.
- Attention at head_dim 128 may be register-limited; check `launch__registers_per_thread`
  and achieved occupancy.
