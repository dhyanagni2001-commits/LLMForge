import math

import pytest
import torch
import torch.nn.functional as F

from models import reference_ops as ref
from models.transformer import Config, Ops, TinyTransformer


# --- Reference semantics (CPU) -------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 7, 64, 130])
def test_reference_matches_sdpa(n):
    torch.manual_seed(n)
    q, k, v = (torch.randn(2, 3, n, 32) for _ in range(3))
    torch.testing.assert_close(ref.attention(q, k, v, causal=True),
                               F.scaled_dot_product_attention(q, k, v, is_causal=True),
                               rtol=1e-5, atol=1e-5)


def test_causal_mask_blocks_future():
    q, k = torch.randn(1, 1, 5, 16), torch.randn(1, 1, 5, 16)
    v = torch.zeros(1, 1, 5, 16)
    v[..., 4, :] = 1.0  # only the last position carries signal
    out = ref.attention(q, k, v, causal=True)
    assert torch.all(out[..., :4, :] == 0) and torch.all(out[..., 4, :] > 0)


def test_reference_stable_with_large_logits():
    q = torch.full((1, 1, 8, 16), 1e3)
    out = ref.attention(q, q, torch.randn(1, 1, 8, 16))
    assert torch.isfinite(out).all()


def test_model_logits_identical_through_op_table():
    torch.manual_seed(0)
    model = TinyTransformer(Config()).eval()
    tokens = torch.randint(0, 256, (2, 16))
    base = model(tokens)
    model.ops = Ops(attention=lambda q, k, v, causal: F.scaled_dot_product_attention(
        q, k, v, is_causal=causal))
    torch.testing.assert_close(model(tokens), base, rtol=1e-4, atol=1e-4)


# --- Triton kernel -------------------------------------------------------------------------

try:
    from kernels.triton.attention import HAS_TRITON, triton_attention
except Exception:
    HAS_TRITON = False

gpu = pytest.mark.skipif(not (torch.cuda.is_available() and HAS_TRITON),
                         reason="CUDA + Triton required")
TOL = {torch.float16: 2e-3, torch.bfloat16: 2e-2, torch.float32: 2e-3}  # fp32 dot may use TF32


@gpu
@pytest.mark.parametrize("dtype", list(TOL))
@pytest.mark.parametrize("n", [1, 17, 64, 65, 128, 300, 1024])
@pytest.mark.parametrize("d", [16, 32, 64, 128])
@pytest.mark.parametrize("causal", [True, False])
def test_triton_matches_reference(dtype, n, d, causal):
    torch.manual_seed(n + d)
    q, k, v = (torch.randn(2, 3, n, d, device="cuda", dtype=dtype) for _ in range(3))
    out = triton_attention(q, k, v, causal=causal)
    expected = ref.attention(q.float(), k.float(), v.float(), causal=causal)
    torch.testing.assert_close(out.float(), expected, rtol=TOL[dtype], atol=TOL[dtype])


@gpu
def test_triton_large_logits_stable():
    q = torch.full((1, 1, 128, 64), 30.0, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 1, 128, 64, device="cuda", dtype=torch.float16)
    assert torch.isfinite(triton_attention(q, q, v)).all()


@gpu
def test_triton_custom_scale():
    q, k, v = (torch.randn(1, 2, 96, 64, device="cuda", dtype=torch.float16) for _ in range(3))
    out = triton_attention(q, k, v, sm_scale=0.5 / math.sqrt(64))
    expected = ref.attention(q.float(), k.float(), v.float(), scale=0.5 / math.sqrt(64))
    torch.testing.assert_close(out.float(), expected, rtol=2e-3, atol=2e-3)


@gpu
@pytest.mark.parametrize("bad", ["head_dim", "shape", "cpu", "dtype"])
def test_triton_rejects_unsupported(bad):
    q = torch.randn(1, 1, 16, 64, device="cuda", dtype=torch.float16)
    k = v = q
    if bad == "head_dim":
        q = k = v = torch.randn(1, 1, 16, 48, device="cuda", dtype=torch.float16)
    elif bad == "shape":
        k = q[:, :, :8]
    elif bad == "cpu":
        q = k = v = q.cpu()
    elif bad == "dtype":
        q = k = v = q.double()
    with pytest.raises(ValueError):
        triton_attention(q, k, v)
