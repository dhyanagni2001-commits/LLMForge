"""Relax pass: rewrite `z = add(x, r); y = nn.rms_norm(z, w)` into one fused TIR kernel
that produces both y and z.

Fusion is applied only when all of these hold (otherwise the graph is left unchanged):
  - x and r have identical static shapes and dtypes (no broadcasting),
  - rms_norm normalizes over the last axis only,
  - w is 1-D with length equal to the last dimension and the same dtype,
  - w is a function parameter or constant (so it is available where the add is emitted),
  - add and rms_norm are bound in the same dataflow block.
Other users of z are redirected to the fused kernel's second output.
"""
import tvm
from tvm import relax
from tvm.relax.expr_functor import PyExprMutator, PyExprVisitor, mutator, visitor

from compiler.tvm.schedule_rmsnorm import fused_add_rmsnorm_te

_ADD = tvm.ir.Op.get("relax.add")
_RMS = tvm.ir.Op.get("relax.nn.rms_norm")

# TVM >= 0.26 renamed StructInfo -> Type (expr.struct_info -> expr.ty, var.vid -> var.id_).
_TensorInfo = getattr(relax, "TensorStructInfo", None) or relax.TensorType


def _info(expr):
    return expr.struct_info if hasattr(expr, "struct_info") else expr.ty


def _remap_key(var):
    return var.vid if hasattr(var, "vid") else var  # newer TVM keys the remap by Var


def _is_constant(expr):
    return type(expr).__name__ in ("Constant", "GenericConst")


def _static_shape(expr):
    sinfo = _info(expr)
    if not isinstance(sinfo, _TensorInfo) or sinfo.shape is None:
        return None
    try:
        return tuple(int(s) for s in sinfo.shape.values), sinfo.dtype
    except (TypeError, AttributeError):
        return None


@visitor
class _Collector(PyExprVisitor):
    """Finds (add_var -> rms_norm binding) pairs that satisfy the fusion conditions."""

    def __init__(self, params):
        super().__init__()
        self.params = set(params)
        self.adds = {}      # add var -> add call
        self.matches = {}   # add var -> (norm var, norm call)
        self.rejected = []  # reasons, for tests and debugging

    def visit_binding_block_(self, block):
        self.adds = {}  # matches must stay within one block
        super().visit_binding_block_(block)

    def visit_dataflow_block_(self, block):
        self.adds = {}
        super().visit_dataflow_block_(block)

    def visit_var_binding_(self, binding):
        val = binding.value
        if isinstance(val, relax.Call) and val.op == _ADD:
            self.adds[binding.var] = val
        elif isinstance(val, relax.Call) and val.op == _RMS:
            src = val.args[0]
            if src in self.adds and src not in self.matches:
                reason = self._check(self.adds[src], val)
                if reason is None:
                    self.matches[src] = (binding.var, val)
                else:
                    self.rejected.append(reason)
        super().visit_var_binding_(binding)

    def _check(self, add, norm):
        x, r = add.args
        w = norm.args[1]
        xs, rs, ws = _static_shape(x), _static_shape(r), _static_shape(w)
        if xs is None or rs is None or ws is None:
            return "dynamic shape"
        if xs != rs:
            return "broadcasting add"
        rank = len(xs[0])
        axes = [int(a) % rank for a in norm.attrs.axes]
        if axes != [rank - 1]:
            return "normalized axes are not the last axis"
        if ws != ((xs[0][-1],), xs[1]):
            return "weight shape/dtype mismatch"
        if not (w in self.params or _is_constant(w)):
            return "weight not a parameter or constant"
        return None


@mutator
class _Fuser(PyExprMutator):
    def __init__(self, mod, matches):
        super().__init__(mod)
        self.matches = matches
        self.norm_vars = {nv for nv, _ in matches.values()}

    def _emit(self, var, value):
        if isinstance(var, relax.DataflowVar):
            return self.builder_.emit(value)
        return self.builder_.emit_output(value)

    def visit_var_binding_(self, binding):
        if binding.var in self.norm_vars:
            return  # already emitted together with its add
        if binding.var not in self.matches:
            return super().visit_var_binding_(binding)
        norm_var, norm = self.matches[binding.var]
        x, r = (self.visit_expr(a) for a in binding.value.args)
        w = self.visit_expr(norm.args[1])
        fused = self.builder_.emit(self.builder_.call_te(
            fused_add_rmsnorm_te, x, r, w, float(norm.attrs.epsilon),
            primfunc_name_hint="fused_add_rmsnorm"))
        y = self._emit(norm_var, relax.TupleGetItem(fused, 0))
        z = self._emit(binding.var, relax.TupleGetItem(fused, 1))
        self.set_var_remap(_remap_key(norm_var), y)
        self.set_var_remap(_remap_key(binding.var), z)


@tvm.transform.module_pass(opt_level=0, name="FuseAddRMSNorm")
class FuseAddRMSNorm:
    """Module pass. After running, `self.stats` maps function name -> (fused, rejected reasons)."""

    def __init__(self):
        self.stats = {}

    def transform_module(self, mod, ctx):
        for gv, func in list(mod.functions.items()):
            if not isinstance(func, relax.Function):
                continue
            col = _Collector(func.params)
            col.visit_expr(func)
            self.stats[gv.name_hint] = (len(col.matches), list(col.rejected))
            if not col.matches:
                continue
            fuser = _Fuser(mod, col.matches)
            new_func = fuser.visit_expr(func)
            mod = fuser.builder_.get()
            mod[gv] = relax.analysis.remove_all_unused(new_func)
        return mod
