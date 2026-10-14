import numpy as np
import pytest

tvm = pytest.importorskip("tvm")
from tvm import relax  # noqa: E402

from compiler.tvm.fuse_add_rmsnorm import FuseAddRMSNorm  # noqa: E402
from compiler.tvm.schedule_rmsnorm import (  # noqa: E402
    lower_cross_thread_reduction, make_prim_func, schedule_gpu)


T = getattr(relax, "TensorStructInfo", None) or relax.TensorType


def build_mod(x_shape, r_shape, w_shape, axes=(-1,), use_z=True, w_param=True):
    bb = relax.BlockBuilder()
    x = relax.Var("x", T(x_shape, "float32"))
    r = relax.Var("r", T(r_shape, "float32"))
    w = relax.Var("w", T(w_shape, "float32"))
    with bb.function("main", [x, r, w]):
        with bb.dataflow():
            z = bb.emit(relax.op.add(x, r))
            wt = w if w_param else bb.emit(relax.op.multiply(w, w))
            y = bb.emit(relax.op.nn.rms_norm(z, wt, axes=list(axes), epsilon=1e-6))
            out = bb.emit_output(relax.Tuple([y, z]) if use_z else y)
        bb.emit_func_output(out)
    return bb.get()


def ops_in(mod):
    names = []
    relax.analysis.post_order_visit(mod["main"], lambda e: isinstance(e, relax.Call) and
                                    isinstance(e.op, tvm.ir.Op) and names.append(e.op.name))
    return names


def numpy_ref(x, r, w, eps=1e-6):
    z = x + r
    return z / np.sqrt((z * z).mean(-1, keepdims=True) + eps) * w, z


def run_cpu(mod, *args):
    mod = relax.transform.LegalizeOps()(mod)
    ex = tvm.compile(mod, target="llvm")
    vm = relax.VirtualMachine(ex, tvm.cpu())
    out = vm["main"](*[tvm.runtime.tensor(a) for a in args])
    return [o.numpy() for o in out] if hasattr(out, "__len__") else [out.numpy()]


@pytest.mark.parametrize("shape", [(4, 64), (2, 3, 33)])
def test_fuses_valid_pattern_and_matches(shape):
    mod = build_mod(shape, shape, (shape[-1],))
    p = FuseAddRMSNorm()
    fused = p(mod)
    assert p.stats["main"][0] == 1
    ops = ops_in(fused)
    assert "relax.nn.rms_norm" not in ops and "relax.add" not in ops and "relax.call_tir" in ops
    assert any(gv.name_hint.startswith("fused_add_rmsnorm") for gv in fused.get_global_vars())

    rng = np.random.default_rng(0)
    x, r = (rng.standard_normal(shape).astype("float32") for _ in range(2))
    w = rng.standard_normal(shape[-1]).astype("float32")
    y_ref, z_ref = numpy_ref(x, r, w)
    for out in (run_cpu(mod, x, r, w), run_cpu(fused, x, r, w)):
        np.testing.assert_allclose(out[0], y_ref, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(out[1], z_ref, rtol=1e-6, atol=1e-6)


def test_fuses_when_z_unused():
    fused = FuseAddRMSNorm()(build_mod((4, 64), (4, 64), (64,), use_z=False))
    assert "relax.add" not in ops_in(fused)


@pytest.mark.parametrize("case,kwargs", [
    ("broadcast", dict(x_shape=(4, 64), r_shape=(1, 64), w_shape=(64,))),
    ("axes", dict(x_shape=(4, 64), r_shape=(4, 64), w_shape=(4, 64), axes=(0, 1))),
    ("weight_computed", dict(x_shape=(4, 64), r_shape=(4, 64), w_shape=(64,), w_param=False)),
])
def test_rejects_invalid(case, kwargs):
    p = FuseAddRMSNorm()
    out = p(build_mod(**kwargs))
    assert p.stats["main"][0] == 0 and p.stats["main"][1], case
    assert "relax.nn.rms_norm" in ops_in(out)


def test_gpu_schedule_lowers_with_thread_bindings():
    sch = schedule_gpu(make_prim_func((8, 1000)), threads=128)
    script = sch.mod.script()
    assert "blockIdx.x" in script and "threadIdx.x" in script and '"shared"' in script
    lowered = lower_cross_thread_reduction(sch.mod)
    assert "tvm_thread_allreduce" in lowered.script()


@pytest.mark.skipif(not tvm.cuda().exist, reason="CUDA device required")
@pytest.mark.parametrize("shape", [(1, 31), (32, 4096), (7, 1000)])
def test_gpu_kernel_matches(shape):
    from compiler.tvm.schedule_rmsnorm import build
    f, _ = build(shape)
    rng = np.random.default_rng(1)
    x, r = (rng.standard_normal(shape).astype("float32") for _ in range(2))
    w = rng.standard_normal(shape[-1]).astype("float32")
    dev = tvm.cuda()
    args = [tvm.runtime.tensor(a, dev) for a in (x, r, w)] + \
           [tvm.runtime.empty(shape, "float32", dev) for _ in range(2)]
    f(*args)
    y_ref, z_ref = numpy_ref(x, r, w)
    np.testing.assert_allclose(args[3].numpy(), y_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(args[4].numpy(), z_ref, rtol=1e-6, atol=1e-6)
