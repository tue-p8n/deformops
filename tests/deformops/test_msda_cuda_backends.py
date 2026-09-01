"""The factory-generated dispatch adapter of every CUDA MSDA configuration.

The adapter is how ``deformops.nn`` and an explicit ``backend=`` reach these kernels;
the functional entry point is covered elsewhere.
"""

from __future__ import annotations

import importlib

import pytest
import torch
from deformops.core.runtime import Method
from deformops.registry.msda import (
    MSDAContext,
    resolve_msda_attn,
)

#: Every CUDA MSDA configuration module, mapped to its adapter's backend name.
MSDA_MODULES = {
    "msda_fp32acc": "cuda",
    "experimental.msda_serialreduce": "cuda_serialreduce",
    "experimental.msda_half2atomic": "cuda_half2atomic",
    "experimental.msda_half2atomic_serialreduce": "cuda_half2atomic_serialreduce",
    "experimental.msda_scalaratomic": "cuda_scalaratomic",
    "experimental.msda_scalaratomic_serialreduce": "cuda_scalaratomic_serialreduce",
    "experimental.msda_morton": "cuda_morton",
    "experimental.msda_hilbert": "cuda_hilbert",
}

_LEVELS = [(16, 24), (8, 12), (4, 6)]
_SPATIAL = sum(h * w for h, w in _LEVELS)
_L = len(_LEVELS)
_K = 4
_B, _Q, _G, _D = 2, 32, 8, 32

_NEEDS_GPU = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available"
)


def _module(name: str):
    return importlib.import_module(f"deformops.backends.cuda.{name}")


def _tensors(device: torch.device):
    torch.manual_seed(0)
    shapes = torch.tensor(_LEVELS, dtype=torch.long, device=device)
    start_index = torch.cat(
        [
            torch.zeros(1, dtype=torch.long, device=device),
            shapes.prod(1).cumsum(0)[:-1],
        ]
    )
    values = torch.randn(_B, _SPATIAL, _G, _D, device=device)
    locs = torch.rand(_B, _Q, _G, _L, _K, 2, device=device)
    attn = torch.randn(_B, _Q, _G, _L * _K, device=device)
    return shapes, start_index, values, locs, attn


def _context(device: torch.device, method: Method) -> MSDAContext:
    shapes, _, values, locs, attn = _tensors(device)
    return MSDAContext.from_call(values, shapes, locs, attn, method)


@pytest.mark.parametrize("name", MSDA_MODULES)
def test_backend_constructs_and_reports_its_backend_name(name: str) -> None:
    """Construction and name, without a device.

    A missing abstract member raises ``TypeError`` here, which no kernel test reaches.
    """
    backend = _module(name).Backend()
    assert backend.name == MSDA_MODULES[name]


@pytest.mark.parametrize("name", MSDA_MODULES)
def test_backend_does_not_claim_cpu_contexts(name: str) -> None:
    """``supports`` is a real predicate, not a constant ``True``."""
    backend = _module(name).Backend()
    context = _context(torch.device("cpu"), Method.SOFTMAX)
    assert not backend.supports(context)


@_NEEDS_GPU
@pytest.mark.gpu
@pytest.mark.parametrize("name", MSDA_MODULES)
def test_backend_supports_a_cuda_context_at_a_compiled_point_count(name: str) -> None:
    backend = _module(name).Backend()
    context = _context(torch.device("cuda"), Method.SOFTMAX)
    assert backend.supports(context)


@_NEEDS_GPU
@pytest.mark.gpu
@pytest.mark.parametrize("name", MSDA_MODULES)
@pytest.mark.parametrize("method", [Method.LINEAR, Method.SOFTMAX])
def test_backend_dispatch_matches_the_functional_entry_point(
    name: str, method: Method
) -> None:
    """Both method branches of ``forward`` match the functional entry point."""
    module = _module(name)
    device = torch.device("cuda")
    shapes, start_index, values, locs, attn = _tensors(device)

    dispatched = module.Backend().forward(
        values, shapes, start_index, locs, attn, 64, method
    )

    if method == Method.SOFTMAX:
        expected = module.forward(
            values, shapes, start_index, locs, attn, 64, num_points=_K, softmax=True
        )
    else:
        normalized, _ = resolve_msda_attn(attn, method)
        expected = module.forward(
            values,
            shapes,
            start_index,
            locs,
            normalized,
            64,
            num_points=_K,
            softmax=False,
        )

    torch.testing.assert_close(dispatched, expected)
