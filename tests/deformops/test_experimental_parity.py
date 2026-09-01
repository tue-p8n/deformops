"""Every experimental MSDA configuration computes the same values as eager.

Outputs and value, location and weight gradients through each ``Backend`` adapter,
for both methods, in float32 and half precision.
"""

from __future__ import annotations

import importlib

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA is not available", allow_module_level=True)

pytestmark = pytest.mark.gpu

from deformops.core.runtime import Method  # noqa: E402
from deformops.registry import Backend, msda  # noqa: E402

_TRITON = (
    "msda_query_tiled_cached",
    "msda_query_tiled_cached_fp32acc",
    "msda_query_tiled_autotuned",
    "msda_query_tiled_autotuned_fp32acc",
    "msda_point_parallel",
    "msda_atomic_wide",
    "msda_block_reduce",
    "msda_block_reduce_wide",
    "msda_warp_reduce",
    "msda_warp_reduce_wide",
    "msda_sort_segscan",
    "msda_sort_segscan_wide",
)
_CUDA = (
    "msda_serialreduce",
    "msda_half2atomic",
    "msda_half2atomic_serialreduce",
    "msda_scalaratomic",
    "msda_scalaratomic_serialreduce",
    "msda_morton",
    "msda_hilbert",
)
_CONFIGURATIONS = [
    *(
        pytest.param(f"deformops.backends.triton.experimental.{n}", id=n)
        for n in _TRITON
    ),
    *(pytest.param(f"deformops.backends.cuda.experimental.{n}", id=n) for n in _CUDA),
]

_LEVELS = ((19, 23), (9, 11), (5, 6))
_HEADS, _CHANNELS, _POINTS, _QUERIES, _BATCH = 4, 16, 4, 37, 2

#: Relative tolerance per dtype; half precision compares against eager on the same
#: rounded operands.
_RTOL = {torch.float32: 1e-4, torch.float16: 2e-2, torch.bfloat16: 2e-2}


def _operands(dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cuda").manual_seed(0)
    shapes = torch.tensor(_LEVELS, device="cuda")
    areas = shapes.prod(1)
    starts = torch.cat([areas.new_zeros(1), areas.cumsum(0)[:-1]])
    value = torch.randn(
        _BATCH, int(areas.sum()), _HEADS, _CHANNELS, device="cuda", generator=generator
    ).to(dtype)
    loc = torch.rand(
        _BATCH,
        _QUERIES,
        _HEADS,
        len(_LEVELS),
        _POINTS,
        2,
        device="cuda",
        generator=generator,
    )
    attn = torch.randn(
        _BATCH,
        _QUERIES,
        _HEADS,
        len(_LEVELS) * _POINTS,
        device="cuda",
        generator=generator,
    ).to(dtype)
    return value, shapes, starts, loc, attn


def _run(backend, method: Method, operands) -> list[torch.Tensor]:
    value, shapes, starts, loc, attn = operands
    leaves = [t.clone().requires_grad_() for t in (value, loc, attn)]
    if method is Method.LINEAR:
        weights = leaves[2].float().softmax(-1).to(attn.dtype)
    else:
        weights = leaves[2]
    if backend is Backend.EAGER:
        out = msda(
            leaves[0].float(),
            shapes,
            starts,
            leaves[1],
            weights.float(),
            backend=Backend.EAGER,
            method=method,
        )
    else:
        out = backend.forward(leaves[0], shapes, starts, leaves[1], weights, 64, method)
    upstream = torch.linspace(-1.0, 1.0, out.numel(), device="cuda").view_as(out)
    (out.float() * upstream).sum().backward()
    return [out.detach().float(), *(leaf.grad.float() for leaf in leaves)]


@pytest.mark.parametrize("method", [Method.LINEAR, Method.SOFTMAX], ids=str)
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16], ids=str
)
@pytest.mark.parametrize("module_name", _CONFIGURATIONS)
def test_experimental_configuration_matches_eager(
    module_name: str, dtype: torch.dtype, method: Method
) -> None:
    if dtype is torch.bfloat16 and torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("bfloat16 needs SM 8.0")
    module = importlib.import_module(module_name)
    if hasattr(module, "load"):
        module.load(download=False, compile=True)
    backend = module.Backend()
    operands = _operands(dtype)
    expected = _run(Backend.EAGER, method, operands)
    got = _run(backend, method, operands)
    for name, a, b in zip(
        ("output", "value", "location", "weights"), got, expected, strict=True
    ):
        assert torch.isfinite(a).all(), f"{module_name}: non-finite {name}"
        scale = b.abs().max().clamp_min(1e-6)
        rel = ((a - b).abs().max() / scale).item()
        assert rel < _RTOL[dtype], (
            f"{module_name} {dtype} {method}: {name} rel={rel:.2e}"
        )
