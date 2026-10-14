"""TE definition and hand-written GPU schedule for fused residual-add + RMSNorm.

Schedule (one CUDA block per row, `threads` threads per block):
  - z[i, j] = x[i, j] + r[i, j]                 j strided by threads -> coalesced
  - sq[i] = sum_j z[i, j]^2  (fp32)             cross-thread reduction (warp shuffle + smem)
  - y[i, j] = z[i, j] * rsqrt(sq[i]/d + eps) * w[j]
All three stages are placed under the same blockIdx.x loop; each thread re-reads only the z
elements it reads. sq and y recompute x + r (bit-identical to z: the add is done in the
storage dtype) instead of reading z, because TIR cannot compute_at an output block; this keeps
z as an independent stage that reverse_compute_at can place in the same kernel. sq lives in shared memory so
every thread sees the reduced value.
"""
import tvm
from tvm import te

try:  # TVM >= 0.26 split TIR into tirx (IR) and s_tir (scheduling)
    from tvm import s_tir as _sched
    from tvm import tirx as tir
except ImportError:
    from tvm import tir
    _sched = tir


def fused_add_rmsnorm_te(x: te.Tensor, r: te.Tensor, w: te.Tensor, eps: float):
    """x, r: (..., d); w: (d,). Returns [y, z]. Leading dims are flattened logically."""
    *lead, d = x.shape

    def last(idx):
        return idx[-1]

    z = te.compute(x.shape, lambda *i: x(*i) + r(*i), name="z")
    k = te.reduce_axis((0, d), name="k")
    sq = te.compute(lead, lambda *i: te.sum(
        (x(*i, k) + r(*i, k)).astype("float32") * (x(*i, k) + r(*i, k)).astype("float32"),
        axis=k), name="sq")
    y = te.compute(x.shape, lambda *i: (
        (x(*i) + r(*i)).astype("float32") * tir.rsqrt(sq(*i[:-1]) / tir.const(int(d), "float32")
                                            + tir.const(eps, "float32"))
        * w(last(i)).astype("float32")).astype(x.dtype), name="y")
    return [y, z]


def make_prim_func(shape, dtype="float32", eps=1e-6):
    x = te.placeholder(shape, dtype, "x")
    r = te.placeholder(shape, dtype, "r")
    w = te.placeholder((shape[-1],), dtype, "w")
    y, z = fused_add_rmsnorm_te(x, r, w, eps)
    return te.create_prim_func([x, r, w, y, z])


def schedule_gpu(func, threads: int = 256):
    sch = _sched.Schedule(func)
    get = getattr(sch, "get_sblock", None) or sch.get_block
    b_z, b_sq, b_y = get("z"), get("sq"), get("y")
    # Collapse leading dims into one row loop on every stage.
    for blk in (b_z, b_sq, b_y):
        loops = sch.get_loops(blk)
        if len(loops) > 2:
            sch.fuse(*loops[:-1])

    row, k = sch.get_loops(b_sq)
    k_o, k_i = sch.split(k, [None, threads])
    sch.reorder(k_i, k_o)
    sch.bind(row, "blockIdx.x")
    sch.bind(k_i, "threadIdx.x")
    sch.set_scope(b_sq, 0, "shared")

    sch.reverse_compute_at(b_z, row)
    sch.reverse_compute_at(b_y, row)
    for blk in (b_z, b_y):
        j = sch.get_loops(blk)[-1]
        j_o, j_i = sch.split(j, [None, threads])
        sch.reorder(j_i, j_o)
        sch.bind(j_i, "threadIdx.x")
    return sch


def lower_cross_thread_reduction(mod):
    return _sched.transform.LowerCrossThreadReduction()(mod)


def build(shape, dtype="float32", eps=1e-6, target="cuda", threads=256):
    sch = schedule_gpu(make_prim_func(shape, dtype, eps), threads)
    return tvm.compile(sch.mod, target=target), sch
