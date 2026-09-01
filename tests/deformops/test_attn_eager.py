"""The sampling operator's eager backend, and its traceability.

The eager backend must compile without a graph break, which only an explicit
assertion detects. ``aot_eager`` tests tracing without needing a C++ toolchain.
"""

from __future__ import annotations

import torch
import torch._dynamo
from deformops.registry import Backend
from deformops.registry import attn as apply

DEVICE = torch.device("cpu")


def _operands(seed: int = 0):
    """Channels-last operands for deformable sampling."""
    torch.manual_seed(seed)
    return (
        torch.randn(1, 8, 8, 32, device=DEVICE),
        torch.randn(1, 8, 8, 2 * 9 * 3, device=DEVICE),
    )


_KWARGS = {
    "kernel_size": (3, 3),
    "padding": (1, 1),
    "groups": 2,
    "group_dims": 16,
}


def test_operator_has_a_cpu_reachable_reference() -> None:
    """No operator ships kernels without something that runs anywhere."""
    assert apply.is_registered(Backend.EAGER)


def test_reference_traces_without_breaking_the_graph() -> None:
    """The sampling reference traces into a single graph without branching on a tensor value."""
    value, offset_mask = _operands(seed=11)
    eager = apply(value, offset_mask, backend=Backend.EAGER, **_KWARGS)
    compiled = torch.compile(apply, fullgraph=True, backend="aot_eager")(
        value, offset_mask, backend=Backend.EAGER, **_KWARGS
    )
    torch.testing.assert_close(compiled, eager)


def test_reference_reports_no_graph_breaks() -> None:
    """Warm the backend's lazy provider with an eager call before tracing."""
    value, offset_mask = _operands(seed=11)
    apply(value, offset_mask, backend=Backend.EAGER, **_KWARGS)
    explanation = torch._dynamo.explain(apply)(
        value, offset_mask, backend=Backend.EAGER, **_KWARGS
    )
    assert explanation.graph_break_count == 0


def test_an_out_of_bounds_point_contributes_exactly_what_a_zero_mask_does() -> None:
    """A point sampled far out of bounds contributes the same as a zero weight."""
    value, offset_mask = _operands(seed=13)
    points = 9
    outside = (0, 3, 7)  # a subset, so the rest still contribute

    far = offset_mask.clone()
    zeroed = offset_mask.clone()
    for point in outside:
        for group in range(2):
            base = (group * points + point) * 3
            far[..., base] = 1e4  # x offset, far outside
            far[..., base + 1] = 1e4  # y offset
            zeroed[..., base + 2] = 0.0  # modulation weight instead

    out_far = apply(value, far, backend=Backend.EAGER, **_KWARGS)
    out_zeroed = apply(value, zeroed, backend=Backend.EAGER, **_KWARGS)
    torch.testing.assert_close(out_far, out_zeroed)
    # The remaining points contribute, so this is not 0 == 0.
    assert out_far.abs().max() > 0
