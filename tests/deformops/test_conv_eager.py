"""Classic deformable convolution's eager backend, and its traceability.

The eager backend must compile without a graph break, which only an explicit
assertion detects. ``aot_eager`` tests tracing without needing a C++ toolchain.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch._dynamo
from deformops.backends.reference.conv import reference as conv_reference
from deformops.core.errors import NoBackendError
from deformops.integrations.torchvision import deform_conv2d
from deformops.registry import Backend
from deformops.registry import conv as apply

DEVICE = torch.device("cpu")

#: Where a cross-backend comparison runs: the device kernel exists only on CUDA.
COMPARE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _conv_operands(  # noqa: PLR0913
    batch: int = 2,
    in_channels: int = 8,
    out_channels: int = 8,
    height: int = 7,
    width: int = 6,
    kernel: int = 3,
    *,
    groups: int = 1,
    offset_groups: int = 1,
    mask: bool = True,
    seed: int = 0,
    device: torch.device = DEVICE,
):
    """Torchvision-shaped operands for classic deformable convolution."""
    torch.manual_seed(seed)
    out_h, out_w = height - kernel + 1, width - kernel + 1
    points = kernel * kernel
    return {
        "input": torch.randn(batch, in_channels, height, width, device=device),
        "offset": torch.randn(
            batch, 2 * offset_groups * points, out_h, out_w, device=device
        )
        * 0.5,
        "mask": (
            torch.rand(batch, offset_groups * points, out_h, out_w, device=device)
            if mask
            else None
        ),
        "weight": torch.randn(
            out_channels, in_channels // groups, kernel, kernel, device=device
        ),
        "bias": torch.randn(out_channels, device=device),
    }


def _call(operands):
    return deform_conv2d(
        operands["input"],
        operands["offset"],
        operands["weight"],
        operands["bias"],
        (1, 1),
        (0, 0),
        (1, 1),
        operands["mask"],
    )


# ---------------------------------------------------------------------------
# Every operator with a kernel has a reference
# ---------------------------------------------------------------------------


def test_operator_has_a_cpu_reachable_reference() -> None:
    """No operator ships kernels without something that runs anywhere."""
    assert apply.is_registered(Backend.EAGER)


def test_reference_is_selected_off_cuda() -> None:
    """``backend=None`` resolves to the eager backend when there is no device kernel."""
    operands = _conv_operands()
    assert _call(operands).shape == (2, 8, 5, 4)


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("groups", "offset_groups", "mask"),
    [
        (1, 1, True),
        (1, 1, False),
        (2, 1, True),
        (1, 2, True),
        (2, 2, False),
    ],
)
def test_float32_evaluation_carries_only_rounding_error(
    groups: int, offset_groups: int, mask: bool
) -> None:
    """The float32 reference matches itself evaluated at float64.

    A rounding check across groups, offset groups and the mask; parity against
    torchvision is in ``test_conv_torchvision_parity.py``.
    """
    operands = _conv_operands(
        groups=groups, offset_groups=offset_groups, mask=mask, seed=3
    )
    got = _call(operands).double()

    exact = deform_conv2d(
        operands["input"].double(),
        operands["offset"].double(),
        operands["weight"].double(),
        operands["bias"].double(),
        (1, 1),
        (0, 0),
        (1, 1),
        None if operands["mask"] is None else operands["mask"].double(),
    )
    scale = exact.abs().max().clamp_min(1e-12)
    assert ((got - exact).abs().max() / scale).item() < 1e-6


def test_native_and_torchvision_spellings_agree() -> None:
    """The channels-last entry point and the compat wrapper are one operator."""
    operands = _conv_operands(seed=5)
    batch, _, out_h, out_w = operands["offset"].shape
    points = 9

    native = apply(
        operands["input"].permute(0, 2, 3, 1).contiguous(),
        operands["offset"]
        .view(batch, 1, points, 2, out_h, out_w)
        .permute(0, 4, 5, 1, 2, 3)
        .contiguous(),
        operands["weight"],
        operands["bias"],
        operands["mask"]
        .view(batch, 1, points, out_h, out_w)
        .permute(0, 3, 4, 1, 2)
        .contiguous(),
    )
    torch.testing.assert_close(native.permute(0, 3, 1, 2), _call(operands))


# ---------------------------------------------------------------------------
# torch.compile
# ---------------------------------------------------------------------------


def test_reference_traces_without_breaking_the_graph() -> None:
    operands = _conv_operands(seed=7)
    eager = _call(operands)
    compiled = torch.compile(deform_conv2d, fullgraph=True, backend="aot_eager")(
        operands["input"],
        operands["offset"],
        operands["weight"],
        operands["bias"],
        (1, 1),
        (0, 0),
        (1, 1),
        operands["mask"],
    )
    torch.testing.assert_close(compiled, eager)


def test_reference_reports_no_graph_breaks() -> None:
    """``fullgraph`` raises on a break; this reports how many there were.

    An eager call first imports the backend, whose import Dynamo would break on.
    """
    operands = _conv_operands(seed=7)
    _call(operands)
    explanation = torch._dynamo.explain(deform_conv2d)(
        operands["input"],
        operands["offset"],
        operands["weight"],
        operands["bias"],
        (1, 1),
        (0, 0),
        (1, 1),
        operands["mask"],
    )
    assert explanation.graph_break_count == 0


def test_reference_is_importable_without_a_device() -> None:
    """The reference is callable directly, outside dispatch, so tests can use it."""
    operands = _conv_operands(seed=17)
    batch, _, out_h, out_w = operands["offset"].shape
    direct = conv_reference(
        operands["input"].permute(0, 2, 3, 1).contiguous(),
        operands["offset"]
        .view(batch, 1, 9, 2, out_h, out_w)
        .permute(0, 4, 5, 1, 2, 3)
        .contiguous(),
        operands["weight"],
        operands["bias"],
        operands["mask"]
        .view(batch, 1, 9, out_h, out_w)
        .permute(0, 3, 4, 1, 2)
        .contiguous(),
        1,
        1,
        0,
        0,
        1,
        1,
        1,
        1,
    )
    torch.testing.assert_close(direct.permute(0, 3, 1, 2), _call(operands))


def test_float64_falls_back_to_the_reference() -> None:
    """A float64 call routes to eager through the device backend's ``supports``."""
    operands = _conv_operands(seed=19)
    out = deform_conv2d(
        operands["input"].double(),
        operands["offset"].double(),
        operands["weight"].double(),
        operands["bias"].double(),
        (1, 1),
        (0, 0),
        (1, 1),
        operands["mask"].double(),
    )
    assert out.dtype == torch.float64


def test_backends_agree_across_the_operator() -> None:
    """Every kernel of this operator reads the caller's tensors the same way.

    Each backend can pass reference parity with its own split yet disagree on the
    packed layout, so they share one tensor. A second backend exists only on CUDA.
    """
    operands = _conv_operands(seed=23, device=COMPARE_DEVICE)
    results: dict[Backend, Any] = {}
    for backend in Backend:
        if not apply.is_registered(backend):
            continue
        try:
            results[backend] = deform_conv2d(
                operands["input"],
                operands["offset"],
                operands["weight"],
                operands["bias"],
                (1, 1),
                (0, 0),
                (1, 1),
                operands["mask"],
                backend=backend,
            )
        except NoBackendError as exc:
            # Skip only a refusal; a compile or launch failure is the defect.
            if "does not support" not in str(exc):
                raise
            continue

    if len(results) < 2:  # noqa: PLR2004
        pytest.skip(f"only {sorted(str(b) for b in results)} runs on this device")

    reference = results[Backend.EAGER]
    for backend, got in results.items():
        if backend is Backend.EAGER:
            continue
        torch.testing.assert_close(
            got.float(),
            reference.float(),
            rtol=1e-4,
            atol=1e-4,
            msg=lambda m, b=backend: (
                f"{b} disagrees with the reference on one tensor: {m}"
            ),
        )
