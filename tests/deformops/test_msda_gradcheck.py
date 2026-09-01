"""Gradcheck of ``MultiScaleDeformAttn2d`` for every backend and method.

Eager and CUDA run in float64 with the tight tolerance. Triton has no float64 kernel,
so it runs in float32 with a relaxed tolerance that allows nondeterminism. Reference
points are in ``[0.25, 0.75]``, away from the edge discontinuity; see
``docs/testing.md``.
"""

import importlib
from collections.abc import Callable
from typing import Literal

import deformops  # noqa: F401  # pyright: ignore[reportUnusedImport]  # registers extensions on import
import pytest
import torch
from deformops.nn import MultiScaleDeformAttn2d
from deformops.registry import Backend, MSDABackend
from torch import Tensor


def _resolve_backend(backend: str) -> Backend | MSDABackend:
    """The module's ``backend`` argument for one row of the table below.

    A Triton configuration uses its own backend object, imported lazily because a
    CPU host may lack ``triton``.
    """
    if backend == "pytorch":
        return Backend.EAGER
    if backend == "cuda":
        return Backend.CUDA
    module = importlib.import_module(f"deformops.backends.{_TRITON_MODULES[backend]}")
    return module.Backend()


#: The Triton configuration module of each ``triton_*`` row.
_TRITON_MODULES = {
    "triton_point_parallel": "triton.experimental.msda_point_parallel",
    "triton_query_tiled": "triton.msda_query_tiled",
    "triton_query_tiled_fp32acc": "triton.msda_query_tiled_fp32acc",
}


# Backends that compute in float64; Triton casts to float32.
_NATIVE_FP64 = frozenset({"pytorch", "cuda"})


def _run_gradcheck(
    func: Callable[..., Tensor],
    inputs: tuple[Tensor, ...],
    *,
    backend: str,
) -> None:
    """Run gradcheck, relaxed for Triton's float32, order-dependent backward."""
    if backend in _NATIVE_FP64:
        torch.autograd.gradcheck(func, inputs, eps=1e-6, atol=1e-5, fast_mode=False)
    else:
        # Triton + atomic_add backward: non-deterministic order of accumulation.
        torch.autograd.gradcheck(
            func, inputs, eps=1e-3, atol=1e-2, nondet_tol=1e-3, fast_mode=False
        )


# Every backend and method; only pytorch runs without CUDA.

_GPU = (
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
)

# Strict xfails for broken (method, backend) backwards, so a fix must remove the entry.
_KNOWN_BROKEN_MSDA: dict[tuple[str, str], str] = {}


# Backends that hand float64 operands to a compiled extension, which a build without
# double dispatch breaks.
_COMPILED_EXTENSION_BACKENDS = frozenset({"cuda"})


def _msda_marks(backend: str, method: str) -> list[pytest.MarkDecorator]:
    """Marks for a (backend, method) MSDA gradcheck configuration."""
    marks: list[pytest.MarkDecorator] = []
    if backend != "pytorch":
        marks.extend(_GPU)
    if backend in _COMPILED_EXTENSION_BACKENDS:
        marks.append(pytest.mark.fp64)
    if (method, backend) in _KNOWN_BROKEN_MSDA:
        marks.append(
            pytest.mark.xfail(reason=_KNOWN_BROKEN_MSDA[(method, backend)], strict=True)
        )
    return marks


# One row per distinct kernel; ``Backend.CUDA`` names only the production CUDA one.
_MSDA_BACKENDS: tuple[str, ...] = (
    "pytorch",
    "cuda",
    "triton_point_parallel",
    "triton_query_tiled",
    "triton_query_tiled_fp32acc",
)


@pytest.mark.parametrize(
    ("method", "backend"),
    [
        pytest.param(
            method,
            backend,
            id=f"{method}-{backend}",
            marks=_msda_marks(backend, method),
        )
        for method in ("linear", "softmax")
        for backend in _MSDA_BACKENDS
    ],
)
def test_msda_module_gradcheck(
    backend: str,
    method: Literal["linear", "softmax"],
) -> None:
    """Full module gradcheck: gradients reach the query, locations and value."""
    # A fixed seed keeps float32 backends off the sampler's boundary discontinuity.
    torch.manual_seed(0)

    B, Q, dim = 1, 2, 16
    num_heads, num_levels, num_points = 1, 1, 4
    H = W = 16

    device = torch.device("cuda" if backend != "pytorch" else "cpu")
    dtype = torch.float64 if backend in _NATIVE_FP64 else torch.float32

    m = MultiScaleDeformAttn2d(
        dim=dim,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
        method=method,
        backend=_resolve_backend(backend),
    ).to(device, dtype=dtype)
    # Zero offsets keep samples off the boundary; nonzero weights keep gradcheck from
    # passing on all-zero gradients.
    torch.nn.init.zeros_(m.proj_offset.bias)
    torch.nn.init.normal_(m.proj_weights.weight, std=0.01)
    torch.nn.init.normal_(m.proj_weights.bias, std=0.01)

    q = torch.randn(B, Q, dim, device=device, dtype=dtype, requires_grad=True)
    p = (
        torch.rand(B, Q, num_levels, 2, device=device, dtype=dtype) * 0.5 + 0.25
    ).requires_grad_(True)
    v = torch.randn(B, H * W, dim, device=device, dtype=dtype, requires_grad=True)
    shapes = torch.tensor([[H, W]], dtype=torch.long, device=device)
    level_index = torch.tensor([0], dtype=torch.long, device=device)

    def func(q_: torch.Tensor, p_: torch.Tensor, v_: torch.Tensor) -> torch.Tensor:
        return m(q_, p_, v_, shapes, level_index)

    _run_gradcheck(func, (q, p, v), backend=backend)
