"""MSDA backward bias against float64 ground truth, on the backends picked for training.

Reduced-precision error must be rounding noise, not an offset along the gradient:
a non-zero ``cos(bias, gt)`` accumulates linearly over training. This exposes the
SM 8.0 bfloat16 atomic discontinuity; see ``docs/precision.md``.
"""

from __future__ import annotations

import pytest
import torch
from deformops.core.runtime import Method
from deformops.registry import Backend
from deformops.registry import msda as msda_forward

pytestmark = pytest.mark.gpu


if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)


# The smallest shape that exposes the bfloat16 bias on g_v; it grows with (L, P).
_Q, _L, _K = 1024, 3, 4
_B, _H, _D = 2, 8, 32

# Atomic-add backends round differently per launch; averaging separates noise from bias.
_N_RUNS = 4

# Training-safe directional bias, with headroom for run-to-run jitter.
_COS_TOL_GV = 0.05


def _make_inputs(dtype: torch.dtype):
    device = torch.device("cuda")
    torch.manual_seed(1958)
    levels = [(48, 48), (24, 24), (12, 12)]
    sizes = [h * w for h, w in levels]
    n = sum(sizes)
    spatial_shapes = torch.tensor(levels, dtype=torch.long, device=device)
    level_start_index = torch.tensor(
        [0] + sizes[:-1], dtype=torch.long, device=device
    ).cumsum(0)
    # Generate in float64 then cast: ``randn`` draws different bits per dtype.
    value64 = torch.randn(_B, n, _H, _D, device=device, dtype=torch.float64)
    loc64 = (
        torch.rand(_B, _Q, _H, _L, _K, 2, device=device, dtype=torch.float64) * 0.8
        + 0.1
    )
    attn64 = torch.randn(_B, _Q, _H, _L * _K, device=device, dtype=torch.float64)
    return (
        value64.to(dtype).detach().requires_grad_(True),
        loc64.to(dtype).detach().requires_grad_(True),
        attn64.to(dtype).detach().requires_grad_(True),
        spatial_shapes,
        level_start_index,
    )


def _run(backend: Backend, dtype: torch.dtype):
    value, loc, attn, shapes, lsi = _make_inputs(dtype)
    out = msda_forward(
        value,
        shapes,
        lsi,
        loc,
        attn,
        method=Method.SOFTMAX,
        backend=backend,
    )
    out.sum().backward()
    assert value.grad is not None
    assert loc.grad is not None
    assert attn.grad is not None
    return value.grad.detach().double()


def _cos_bias_vs_fp64(backend: Backend, dtype: torch.dtype) -> float:
    gt = _run(Backend.EAGER, torch.float64)
    errs = [_run(backend, dtype) - gt for _ in range(_N_RUNS)]
    mean_err = torch.stack(errs).mean(dim=0)
    return (mean_err * gt).sum().item() / (
        mean_err.norm().item() * gt.norm().item() + 1e-30
    )


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(Backend.CUDA, id="cuda"),
        pytest.param(Backend.TRITON, id="triton"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_msda_backward_bias_within_tolerance(
    backend: Backend,
    dtype: torch.dtype,
) -> None:
    """Backends picked for training keep the ``g_v`` bias close to zero.

    ``cuda`` covers SM 8.0 and 9.0; query-tiled with ``fp32acc`` is the
    half-precision training pick below SM 9.0.
    """
    cos = _cos_bias_vs_fp64(backend, dtype)
    assert abs(cos) < _COS_TOL_GV, (
        f"{backend.name} at {dtype} shows directional g_v bias "
        f"cos(bias, gt)={cos:+.3f} (tolerance {_COS_TOL_GV}). "
        f"This is the signature of an atomic-path regression."
    )
