"""Gradcheck coverage for ``DeformAttn2d`` and ``DeformSample2d``.

Both aggregates are checked, since a lost softmax flag yields plausible gradients.
Offsets sit a quarter-cell off the grid, where bilinear interpolation kinks, and the
forward is non-zero so gradcheck cannot pass trivially.
"""

from collections.abc import Callable
from typing import Literal

import deformops  # noqa: F401  # pyright: ignore[reportUnusedImport]  # registers extensions on import
import pytest
import torch
from deformops.nn import DeformAttn2d, DeformSample2d
from deformops.registry import Backend
from torch import Tensor


def _resolve_attn_backend(backend: str) -> Backend | None:
    if backend == "auto":
        return None
    if backend == "triton_query_tiled":
        return Backend.TRITON
    return Backend(backend)


# Backends computing true float64; Triton upcasts to float32 and needs relaxed
# tolerance.
_NATIVE_FP64 = frozenset({"auto", "pytorch", "cuda"})


def _run_gradcheck(
    func: Callable[..., Tensor],
    inputs: tuple[Tensor, ...],
    *,
    backend: str,
) -> None:
    """Run gradcheck with the tolerance the backend's precision allows.

    Triton computes in float32 and accumulates with order-dependent
    ``tl.atomic_add``, so it gets the relaxed tolerance and ``nondet_tol``.
    """
    if backend in _NATIVE_FP64:
        torch.autograd.gradcheck(func, inputs, eps=1e-6, atol=1e-5, fast_mode=False)
    else:
        # Triton + atomic_add backward: non-deterministic order of accumulation.
        torch.autograd.gradcheck(
            func, inputs, eps=1e-3, atol=1e-2, nondet_tol=1e-3, fast_mode=False
        )


_GPU = (
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
)


# ---------------------------------------------------------------------------
# DeformAttn2d: the registry picks the backend; parametrize the layer arguments.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "center_gate", "drop_center", "backend"),
    [
        pytest.param(
            method,
            cg,
            ex,
            backend,
            id=f"{method}-{'gate' if cg else 'plain'}{'-ex' if ex else ''}-{backend}",
            marks=list(_GPU) if backend != "auto" else [],
        )
        for method in ("linear", "softmax")
        for (cg, ex) in [(False, False), (True, False), (False, True)]
        for backend in ("auto", "triton_query_tiled")
    ],
)
def test_deform_attn2d_gradcheck(
    method: Literal["linear", "softmax"],
    center_gate: bool,
    drop_center: bool,
    backend: str,
) -> None:
    """Gradcheck DeformAttn2d for every (method, gate, exclude, backend) combo.

    ``auto`` rows run float64, which Triton lacks, so they also cover the fallback to
    eager. ``triton_query_tiled`` rows run Triton at float32 with a relaxed tolerance.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64 if backend == "auto" else torch.float32

    # A larger extent keeps samples away from the image edges.
    C, H, W = 16, 12, 12
    B = 1
    torch.manual_seed(0)
    m = DeformAttn2d(
        dim=C,
        kernel_size=3,
        groups=1,
        method=method,
        center_gate=center_gate,
        drop_center=drop_center,
        project=None,  # avoid the extra Linear layer for speed
        backend=_resolve_attn_backend(backend),
    ).to(device, dtype=dtype)
    # The projection is zero-initialized; perturb it so a linear forward is non-zero.
    torch.nn.init.normal_(m.proj_offset_mask.weight, std=0.01)
    # Bias each (dy, dx) a quarter-cell off the grid; the mask channel is untouched.
    num_points = m.kernel_size * m.kernel_size - int(m.drop_center)
    with torch.no_grad():
        offset_bias = m.proj_offset_mask.bias[: m.groups * num_points * 3].view(-1, 3)
        offset_bias[:, :2] = 0.25

    q = torch.randn(B, C, H, W, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, C, H, W, device=device, dtype=dtype, requires_grad=True)

    def func(q_: torch.Tensor, v_: torch.Tensor) -> torch.Tensor:
        return m(q_, v_)

    assert func(q, v).abs().max() > 1e-3, (
        "the forward is ~zero, so gradcheck would pass without testing anything"
    )
    _run_gradcheck(func, (q, v), backend=backend)


# ---------------------------------------------------------------------------
# DeformSample2d: only ``method`` varies at this level.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["linear", "softmax"])
def test_deform_sample2d_gradcheck(method: Literal["linear", "softmax"]) -> None:
    on_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if on_cuda else "cpu")
    # float64 has no Triton path, so ``auto`` also covers the fallback to eager.
    backend = "auto"

    C, H, W = 16, 12, 12
    B = 1
    m = DeformSample2d(
        dim=C,
        kernel_size=3,
        groups=1,
        method=method,
        project=None,
        backend=_resolve_attn_backend(backend),
    ).to(device, dtype=torch.float64)

    x = torch.randn(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)

    def func(x_: torch.Tensor) -> torch.Tensor:
        return m(x_)

    _run_gradcheck(func, (x,), backend=backend)
