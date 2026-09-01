"""The eager MSDA backend: reference parity, dispatch, and tracing without graph breaks.

The reference breaks the graph (``.item()`` per level), so tests use it only for
parity.
"""

from __future__ import annotations

import pytest
import torch
from deformops.backends.eager.msda_pytorch import Backend as EagerBackend
from deformops.backends.eager.msda_pytorch import forward as eager_forward
from deformops.backends.reference.msda import reference as base_reference
from deformops.nn import MultiScaleDeformAttn2d
from deformops.registry import Backend

DEVICE = torch.device("cpu")


def _make_inputs(N, M, D, levels, P, Q, *, dtype=torch.float32, seed=0):
    """Build (values, shapes, start_index, locs, attn) with some locs out of bounds."""
    torch.manual_seed(seed)
    L = len(levels)
    sizes = [h * w for h, w in levels]
    S = sum(sizes)
    shapes = torch.tensor(levels, dtype=torch.long, device=DEVICE)
    start_index = torch.tensor(
        [0, *sizes[:-1]], dtype=torch.long, device=DEVICE
    ).cumsum(0)
    values = torch.randn(N, S, M, D, device=DEVICE, dtype=dtype)
    locs = torch.rand(N, Q, M, L, P, 2, device=DEVICE, dtype=dtype) * 1.2 - 0.1
    attn = torch.randn(N, Q, M, L, P, device=DEVICE, dtype=dtype).softmax(-1)
    return values, shapes, start_index, locs, attn


@pytest.mark.parametrize(
    ("N", "M", "D", "levels", "P", "Q"),
    [
        (1, 1, 16, [(8, 8)], 4, 5),
        (2, 4, 32, [(8, 8), (4, 4), (2, 2)], 4, 12),
        (1, 8, 16, [(7, 5), (3, 3)], 8, 9),
        (3, 2, 16, [(1, 1)], 1, 4),  # degenerate 1x1 level: all-corner clamp
    ],
)
def test_eager_matches_reference(N, M, D, levels, P, Q):
    """The eager implementation matches the reference implementation."""
    values, shapes, start_index, locs, attn = _make_inputs(N, M, D, levels, P, Q)
    expected = base_reference(values, shapes, locs, attn)
    actual = eager_forward(values, shapes, start_index, locs, attn)
    assert actual.shape == expected.shape == (N, Q, M * D)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_eager_matches_reference_at_boundaries():
    """Locations at exact pixel transitions match the reference in float32.

    Tests the two-sided zero-pad mask, unlike the CUDA kernel's one-sided mask.
    """
    W = H = 4
    N, M, D, L = 1, 1, 1, 1
    shapes = torch.tensor([[H, W]], dtype=torch.long, device=DEVICE)
    start_index = torch.tensor([0], dtype=torch.long, device=DEVICE)
    # Known feature map: values[0, s, 0, 0] = s, so each pixel is identifiable.
    values = torch.arange(H * W, dtype=torch.float32, device=DEVICE).view(
        1, H * W, 1, 1
    )
    # loc -> pix = loc*4 - 0.5: 0->-0.5, 0.125->0, 0.375->1(int), 1.0->3.5(=W-0.5),
    # -0.125->-1(fully out), 1.25->4.5(fully out), 0.5->1.5.
    coords = [0.0, 0.125, 0.375, 0.5, 1.0, -0.125, 1.25]
    pts = [(cx, cy) for cx in coords for cy in coords]
    P = len(pts)
    locs = torch.tensor(
        [[[[[[cx, cy] for cx, cy in pts]]]]],
        dtype=torch.float32,
        device=DEVICE,
    ).view(N, 1, M, L, P, 2)
    attn = torch.ones(N, 1, M, L, P, dtype=torch.float32, device=DEVICE)
    expected = base_reference(values, shapes, locs, attn)
    actual = eager_forward(values, shapes, start_index, locs, attn)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    ("dtype", "tol"), [(torch.float16, 5e-3), (torch.bfloat16, 2e-2)]
)
def test_eager_matches_reference_reduced_precision(dtype, tol):
    """Half-precision parity on GPU, which ``supports()`` accepts.

    The tolerance covers two independent rounding errors.
    """
    dev = torch.device("cuda")
    N, M, D, L, P, Q = 2, 4, 32, 3, 4, 12
    levels = [(8, 8), (4, 4), (2, 2)]
    sizes = [h * w for h, w in levels]
    S = sum(sizes)
    torch.manual_seed(0)
    shapes = torch.tensor(levels, dtype=torch.long, device=dev)
    start_index = torch.tensor([0, *sizes[:-1]], dtype=torch.long, device=dev).cumsum(0)
    values = torch.randn(N, S, M, D, device=dev, dtype=dtype)
    locs = torch.rand(N, Q, M, L, P, 2, device=dev, dtype=dtype) * 1.2 - 0.1
    attn = torch.randn(N, Q, M, L, P, device=dev, dtype=dtype).softmax(-1)
    expected = base_reference(values, shapes, locs, attn)
    actual = eager_forward(values, shapes, start_index, locs, attn)
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)


def test_eager_backward_matches_reference():
    """Input-gradient parity (implicit autograd through the clamp + mask gather)
    vs the reference implementation, CPU fp32."""
    values, shapes, start_index, locs, attn = _make_inputs(
        2, 4, 32, [(6, 6), (4, 4)], 4, 6, seed=7
    )

    def _reference(values, locs, attn):
        return base_reference(values, shapes, locs, attn)

    def _eager(values, locs, attn):
        return eager_forward(values, shapes, start_index, locs, attn)

    grads = {}
    for tag, forward in (("reference", _reference), ("eager", _eager)):
        vi, li, ai = (t.clone().requires_grad_(True) for t in (values, locs, attn))
        forward(vi, li, ai).sum().backward()
        grads[tag] = (vi.grad, li.grad, ai.grad)

    for g_reference, g_eager in zip(grads["reference"], grads["eager"], strict=True):
        torch.testing.assert_close(g_eager, g_reference, atol=1e-4, rtol=1e-4)


def test_eager_all_out_of_bounds_is_zero():
    """Samples entirely outside every level contribute nothing (zero-pad)."""
    values, shapes, start_index, locs, attn = _make_inputs(1, 2, 16, [(4, 4)], 4, 6)
    # Push all sample locations far outside [0, 1] in both axes.
    locs = torch.full_like(locs, 5.0)
    out = eager_forward(values, shapes, start_index, locs, attn)
    torch.testing.assert_close(out, torch.zeros_like(out))


def test_backend_eager_dispatch_matches_concrete_backend():
    """``Backend.EAGER`` is the eager implementation, registered and identical
    to the concrete backend object."""
    dim, num_heads, num_levels, num_points = 32, 4, 2, 4
    B, Q = 2, 8
    levels = [(6, 6), (4, 4)]
    S = sum(h * w for h, w in levels)
    shapes = torch.tensor(levels, dtype=torch.long, device=DEVICE)
    level_index = torch.tensor(
        [0, levels[0][0] * levels[0][1]], dtype=torch.long, device=DEVICE
    )
    torch.manual_seed(1958)
    q = torch.randn(B, Q, dim, device=DEVICE)
    p = torch.rand(B, Q, num_levels, 2, device=DEVICE)
    v = torch.randn(B, S, dim, device=DEVICE)

    kw = {
        "dim": dim,
        "num_heads": num_heads,
        "num_levels": num_levels,
        "num_points": num_points,
        "method": "softmax",
    }
    m_registered = MultiScaleDeformAttn2d(**kw, backend=Backend.EAGER).to(DEVICE)
    m_concrete = MultiScaleDeformAttn2d(**kw, backend=EagerBackend()).to(DEVICE)
    m_concrete.load_state_dict(m_registered.state_dict())

    out_registered = m_registered(q, p, v, shapes, level_index)
    out_concrete = m_concrete(q, p, v, shapes, level_index)
    assert out_registered.shape == out_concrete.shape == (B, Q, dim)
    torch.testing.assert_close(out_registered, out_concrete, atol=1e-6, rtol=1e-6)


def test_auto_resolver_selects_eager_on_cpu():
    """``backend=None`` on CPU selects ``Backend.EAGER``."""
    from deformops.registry import msda as msda_forward

    dim, num_heads, num_levels, num_points = 16, 2, 2, 4
    B, Q = 1, 6
    levels = [(4, 4), (2, 2)]
    S = sum(h * w for h, w in levels)
    shapes = torch.tensor(levels, dtype=torch.long, device=DEVICE)
    level_index = torch.tensor(
        [0, levels[0][0] * levels[0][1]], dtype=torch.long, device=DEVICE
    )
    torch.manual_seed(42)
    v = torch.randn(B, S, num_heads, dim // num_heads, device=DEVICE)
    loc = torch.rand(B, Q, num_heads, num_levels, num_points, 2, device=DEVICE)
    attn = torch.randn(B, Q, num_heads, num_levels * num_points, device=DEVICE)

    out_auto = msda_forward(v, shapes, level_index, loc, attn, backend=None)
    out_eager = msda_forward(v, shapes, level_index, loc, attn, backend=Backend.EAGER)
    torch.testing.assert_close(out_auto, out_eager)


def test_backend_self_reports_registry_name():
    """The eager Backend's ``name`` matches the registry key it is filed under."""
    assert EagerBackend().name == "pytorch"


def test_eager_is_fullgraph_traceable():
    """``fullgraph=True`` traces without a graph break.

    ``backend="eager"`` checks tracing alone, without the Inductor toolchain.
    """
    values, shapes, start_index, locs, attn = _make_inputs(
        1, 2, 16, [(6, 6), (3, 3)], 4, 8
    )
    compiled = torch.compile(eager_forward, fullgraph=True, backend="eager")
    out = compiled(values, shapes, start_index, locs, attn)
    expected = eager_forward(values, shapes, start_index, locs, attn)
    torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)
