"""MSDA backward on valid but awkward operands, and rejection of malformed ones.

An infinite location contributes nothing, so the value gradient must be finite
everywhere. That query's location and attention gradients are undefined (NaN in the
reference too), so only the other queries must be finite. Malformed operands raise
``ValueError`` before any kernel launches.
"""

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

if not torch.cuda.is_available():
    pytest.skip("CUDA is not available", allow_module_level=True)

pytestmark = pytest.mark.gpu

from deformops.backends.cuda.experimental import msda_half2atomic  # noqa: E402
from deformops.backends.triton.experimental.msda_point_parallel import (  # noqa: E402
    forward as msda_point_parallel,
)
from deformops.registry import Backend, msda  # noqa: E402

device = torch.device("cuda")


@st.composite
def robustness_inputs(draw):
    # Stress-test shapes and corner cases
    N = draw(st.integers(min_value=1, max_value=2))
    M = draw(st.sampled_from([1, 2, 4]))
    D = draw(st.sampled_from([8, 16, 32]))
    L = draw(st.integers(min_value=1, max_value=4))
    P = draw(st.sampled_from([1, 2, 4, 8]))
    Q = draw(st.integers(min_value=1, max_value=128))

    dtype = draw(st.sampled_from([torch.float32, torch.float16]))
    # Seeded by hypothesis, so a failing example replays and shrinks.
    generator = torch.Generator(device=device).manual_seed(
        draw(st.integers(min_value=0, max_value=2**31 - 1))
    )

    value_list = []
    shapes = []
    start_index = []
    offset = 0
    for _ in range(L):
        h = draw(st.integers(min_value=1, max_value=32))
        w = draw(st.integers(min_value=1, max_value=32))
        shapes.append([h, w])
        start_index.append(offset)
        value_list.append(
            torch.randn(N, h * w, M, D, device=device, dtype=dtype, generator=generator)
        )
        offset += h * w

    value = torch.cat(value_list, dim=1)
    shapes = torch.tensor(shapes, device=device, dtype=torch.long)
    start_index = torch.tensor(start_index, device=device, dtype=torch.long)

    locs = torch.rand(N, Q, M, L, P, 2, device=device, dtype=dtype, generator=generator)
    attn = torch.randn(N, Q, M, L, P, device=device, dtype=dtype, generator=generator)

    # Push one query's locations to infinity.
    infinite_query = None
    if draw(st.booleans()):
        infinite_query = draw(st.integers(min_value=0, max_value=Q - 1))
        locs[:, infinite_query, ...] = float("inf")

    return (
        value,
        shapes,
        start_index,
        locs,
        attn,
        dtype,
        P,
        infinite_query,
    )


def _assert_finite(label: str, gradients, infinite_query: int | None) -> None:
    grad_value, grad_loc, grad_attn = gradients
    for name, gradient in (
        ("value", grad_value),
        ("location", grad_loc),
        ("attention", grad_attn),
    ):
        assert gradient is not None, f"{label}: no {name} gradient"
    assert torch.isfinite(grad_value).all(), f"{label}: non-finite value gradient"
    keep = torch.ones(grad_loc.shape[1], dtype=torch.bool, device=grad_loc.device)
    if infinite_query is not None:
        keep[infinite_query] = False
    for name, gradient in (("location", grad_loc), ("attention", grad_attn)):
        assert torch.isfinite(gradient[:, keep]).all(), (
            f"{label}: non-finite {name} gradient for a query with finite locations"
        )


@settings(
    deadline=None,
    suppress_health_check=[HealthCheck.data_too_large],
    max_examples=20,
)
@given(inputs=robustness_inputs())
def test_backward_robustness(inputs):
    value, shapes, start_index, locs, attn, _dtype, P, infinite_query = inputs

    # Scale inputs for float16
    if _dtype == torch.float16:
        val_scale = 1e-2
        loc_range = 10.0
        value *= val_scale
        locs = (locs * 2 - 1) * loc_range

    value.requires_grad_(True)
    locs.requires_grad_(True)
    attn.requires_grad_(True)

    attn_sm = attn.softmax(-1)

    # The half-precision-atomic CUDA configuration instantiates 4 and 8 points.
    if P in (4, 8):
        out_v5 = msda_half2atomic.forward(
            value, shapes, start_index, locs, attn_sm, 64, num_points=P
        )
        gradients = torch.autograd.grad(
            out_v5.sum(), [value, locs, attn], retain_graph=True
        )
        _assert_finite("msda_half2atomic", gradients, infinite_query)

    out_v6 = msda_point_parallel(value, shapes, start_index, locs, attn_sm)
    gradients = torch.autograd.grad(out_v6.sum(), [value, locs, attn])
    _assert_finite("triton_point_parallel", gradients, infinite_query)


def _valid_operands() -> dict[str, torch.Tensor]:
    batch, heads, channels, queries, levels, points = 1, 2, 16, 8, 2, 4
    shapes = torch.tensor([[4, 4], [2, 2]], device=device, dtype=torch.long)
    return {
        "value": torch.randn(batch, 20, heads, channels, device=device),
        "shapes": shapes,
        "start_index": torch.tensor([0, 16], device=device, dtype=torch.long),
        "locs": torch.rand(batch, queries, heads, levels, points, 2, device=device),
        "attn": torch.rand(batch, queries, heads, levels * points, device=device),
    }


_MALFORMED = {
    # (x, y, z) triples where the operator takes (x, y) pairs.
    "three-coordinate-locations": lambda ops: {
        **ops,
        "locs": torch.rand(*ops["locs"].shape[:-1], 3, device=device),
    },
    # One weight too few for levels x points.
    "attention-width": lambda ops: {**ops, "attn": ops["attn"][..., :-1]},
    # A half-precision weight beside a single-precision value.
    "mixed-dtype": lambda ops: {**ops, "attn": ops["attn"].half()},
}


@pytest.mark.parametrize(
    "backend",
    [pytest.param(Backend.CUDA, id="cuda"), pytest.param(Backend.TRITON, id="triton")],
)
@pytest.mark.parametrize("malformation", sorted(_MALFORMED))
def test_malformed_operands_raise_value_error(backend: Backend, malformation: str):
    """The structural checks run on the device backends as well as on eager."""
    ops = _MALFORMED[malformation](_valid_operands())
    with pytest.raises(ValueError):  # noqa: PT011
        msda(
            ops["value"],
            ops["shapes"],
            ops["start_index"],
            ops["locs"],
            ops["attn"],
            backend=backend,
        )
