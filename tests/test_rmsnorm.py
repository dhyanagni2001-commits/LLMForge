import pytest
import torch

from models import reference_ops as ref

# Tolerances relative to an FP64 reference, per dtype.
TOL = {torch.float32: (1e-5, 1e-5), torch.float16: (2e-3, 2e-3), torch.bfloat16: (1.6e-2, 1.6e-2)}


def fp64_reference(x, r, w, eps):
    z = x.double() + r.double()
    y = z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + eps) * w.double()
    return y, z


def make(rows, hidden, dtype, device, scale=1.0):
    g = torch.Generator(device="cpu").manual_seed(rows * 7919 + hidden)
    mk = lambda *s: (torch.randn(*s, generator=g) * scale).to(device=device, dtype=dtype)
    return mk(rows, hidden), mk(rows, hidden), mk(hidden)


# --- Reference semantics (CPU, always runs) -------------------------------------------------

@pytest.mark.parametrize("rows,hidden", [(1, 1), (3, 7), (32, 768), (5, 4097)])
def test_reference_matches_fp64(rows, hidden):
    x, r, w = make(rows, hidden, torch.float32, "cpu")
    y, z = ref.fused_add_rmsnorm(x, r, w, 1e-6)
    y64, z64 = fp64_reference(x, r, w, 1e-6)
    torch.testing.assert_close(z.double(), z64, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(y.double(), y64, rtol=1e-5, atol=1e-5)


def test_reference_zero_input_is_finite():
    x = torch.zeros(4, 64)
    y, z = ref.fused_add_rmsnorm(x, x, torch.ones(64))
    assert torch.all(y == 0) and torch.all(z == 0)


# --- CUDA kernel ---------------------------------------------------------------------------

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.fixture(scope="module")
def kernel():
    from kernels import fused_add_rmsnorm, load_cuda_extension
    load_cuda_extension()
    return fused_add_rmsnorm


@cuda
@pytest.mark.parametrize("dtype", list(TOL))
@pytest.mark.parametrize("rows,hidden", [(1, 1), (1, 31), (3, 7), (32, 768), (17, 1000),
                                         (256, 2048), (8, 4096), (2, 8193)])
@pytest.mark.parametrize("scale", [1e-3, 1.0, 100.0])
def test_cuda_matches_reference(kernel, dtype, rows, hidden, scale):
    x, r, w = make(rows, hidden, dtype, "cuda", scale)
    x0, r0 = x.clone(), r.clone()
    y, z = kernel(x, r, w, 1e-6)
    torch.testing.assert_close(x, x0, rtol=0, atol=0)  # inputs preserved
    torch.testing.assert_close(r, r0, rtol=0, atol=0)
    rtol, atol = TOL[dtype]
    y64, z64 = fp64_reference(x, r, w, 1e-6)
    torch.testing.assert_close(z.double(), z64, rtol=rtol, atol=atol * scale)
    # y is computed from the dtype-rounded z, so compare against reference on that z.
    y_ref = fp64_reference(z, torch.zeros_like(z), w, 1e-6)[0]
    torch.testing.assert_close(y.double(), y_ref, rtol=rtol, atol=atol)


@cuda
def test_cuda_3d_input_and_zero(kernel):
    x = torch.zeros(2, 3, 128, device="cuda")
    y, z = kernel(x, x, torch.ones(128, device="cuda"))
    assert y.shape == x.shape and torch.all(y == 0)


@cuda
def test_cuda_unaligned_view_uses_scalar_path(kernel):
    base = torch.randn(4 * 64 + 1, device="cuda", dtype=torch.float16)
    x = base[1:].view(4, 64)  # contiguous but not 16-byte aligned
    r, w = torch.randn_like(x), torch.randn(64, device="cuda", dtype=torch.float16)
    y, _ = kernel(x, r, w)
    torch.testing.assert_close(y, ref.fused_add_rmsnorm(x, r, w)[0], rtol=2e-3, atol=2e-3)


@cuda
@pytest.mark.parametrize("bad", ["cpu", "dtype", "shape", "weight", "noncontig"])
def test_cuda_rejects_invalid(kernel, bad):
    x, r, w = make(4, 64, torch.float32, "cuda")
    if bad == "cpu":
        x = x.cpu()
    elif bad == "dtype":
        r = r.half()
    elif bad == "shape":
        r = r[:2]
    elif bad == "weight":
        w = w[:32]
    elif bad == "noncontig":
        x = torch.randn(64, 4, device="cuda").t()
    with pytest.raises(RuntimeError):
        kernel(x, r, w)
