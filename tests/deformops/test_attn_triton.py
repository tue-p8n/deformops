"""The Triton single-scale deformable sampling kernel, against the reference.

Kernel-point order, the boundary convention and the gradient each have tests,
since an error in any of them raises nothing; see ``docs/conventions.md`` and
``docs/precision.md``.
"""

from __future__ import annotations

import pytest
import torch
from deformops.registry import Backend, attn

# Skipped without a device and marked ``gpu`` for ``-m gpu``.
if not torch.cuda.is_available():
    pytest.skip("CUDA required for the Triton kernel", allow_module_level=True)

pytestmark = pytest.mark.gpu

#: The two Triton configurations, as the ``fp32acc`` option selects them.
CONFIGURATIONS = [{"fp32acc": False}, {"fp32acc": True}]
CONFIGURATION_IDS = ["default", "fp32acc"]

#: The Triton kernel in its default configuration, and the eager reference.
TRITON = {"backend": Backend.TRITON, "fp32acc": False}
EAGER = {"backend": Backend.EAGER}

# Tolerances against a float32 reference: the kernel accumulates in float32 and the
# reference in the input dtype, so the reference rounds worse in half precision.
TOLERANCES = {
    torch.float32: (1e-5, 1e-4),
    torch.float16: (1e-2, 2e-2),
    torch.bfloat16: (5e-2, 2e-1),
}


def _operands(
    *,
    batch: int = 2,
    height: int = 9,
    width: int = 11,
    groups: int = 2,
    group_dims: int = 8,
    kernel_h: int = 3,
    kernel_w: int = 3,
    stride: int = 1,
    padding: int = 1,
    dilation: int = 1,
    drop_center: bool = False,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
    requires_grad: bool = False,
) -> dict:
    """A consistent operand set, shaped the way the operator expects."""
    points = kernel_h * kernel_w - (1 if drop_center else 0)
    out_h = (height + 2 * padding - (dilation * (kernel_h - 1) + 1)) // stride + 1
    out_w = (width + 2 * padding - (dilation * (kernel_w - 1) + 1)) // stride + 1
    value = torch.randn(
        batch,
        height,
        width,
        groups * group_dims,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )
    offset_mask = torch.randn(
        batch,
        out_h,
        out_w,
        groups * points * 3,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )
    return {
        "value": value,
        "offset_mask": offset_mask,
        "kernel_size": (kernel_h, kernel_w),
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "groups": groups,
        "group_dims": group_dims,
        "drop_center": drop_center,
    }


def _apply(operands: dict, selection: dict) -> torch.Tensor:
    """Run the operator on ``operands`` with ``selection``'s backend and options."""
    return attn.apply(
        operands["value"],
        operands["offset_mask"],
        operands["kernel_size"],
        stride=operands["stride"],
        padding=operands["padding"],
        dilation=operands["dilation"],
        groups=operands["groups"],
        group_dims=operands["group_dims"],
        drop_center=operands["drop_center"],
        **selection,
    )


# ---------------------------------------------------------------------------
# Forward parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("configuration", CONFIGURATIONS, ids=CONFIGURATION_IDS)
@pytest.mark.parametrize("dtype", list(TOLERANCES), ids=lambda d: str(d).split(".")[-1])
def test_forward_matches_the_reference(configuration: dict, dtype: torch.dtype) -> None:
    """Half-precision outputs match a float32 run on the same operand values."""
    torch.manual_seed(0)
    operands = _operands(dtype=torch.float32)
    rtol, atol = TOLERANCES[dtype]

    expected = _apply(operands, EAGER)
    cast = dict(operands)
    cast["value"] = operands["value"].to(dtype)
    cast["offset_mask"] = operands["offset_mask"].to(dtype)
    actual = _apply(cast, {"backend": Backend.TRITON, **configuration})

    torch.testing.assert_close(actual.float(), expected, rtol=rtol, atol=atol)


def test_half_precision_is_at_least_as_accurate_as_the_reference() -> None:
    """Against a float32 run, the float32-accumulating kernel is no worse.

    Losing the wide accumulator would otherwise show only as slow training drift.
    """
    torch.manual_seed(9)
    operands = _operands(dtype=torch.float32)
    oracle = _apply(operands, EAGER)

    for dtype in (torch.float16, torch.bfloat16):
        cast = dict(operands)
        cast["value"] = operands["value"].to(dtype)
        cast["offset_mask"] = operands["offset_mask"].to(dtype)
        kernel_error = (_apply(cast, TRITON).float() - oracle).abs().max()
        reference_error = (_apply(cast, EAGER).float() - oracle).abs().max()
        assert kernel_error <= reference_error, (
            f"{dtype}: kernel error {kernel_error:.4f} exceeds the same-dtype "
            f"reference's {reference_error:.4f}; the float32 accumulator is the "
            f"reason it should not"
        )


@pytest.mark.parametrize(
    ("kernel_h", "kernel_w"), [(3, 3), (5, 5), (1, 3), (3, 1), (5, 3)]
)
def test_forward_across_kernel_shapes(kernel_h: int, kernel_w: int) -> None:
    """Non-square and degenerate kernels exercise the grid mapping."""
    torch.manual_seed(1)
    operands = _operands(
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        padding=0,
        height=11,
        width=11,
    )
    expected = _apply(operands, EAGER)
    actual = _apply(operands, TRITON)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
def test_forward_across_geometry(stride: int, dilation: int, padding: int) -> None:
    """Stride, dilation and padding all enter the base-coordinate expression."""
    torch.manual_seed(2)
    operands = _operands(
        stride=stride, dilation=dilation, padding=padding, height=13, width=13
    )
    expected = _apply(operands, EAGER)
    actual = _apply(operands, TRITON)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# The kernel-point mapping, where an error raises nothing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kernel_h", "kernel_w"), [(3, 3), (5, 5), (5, 3), (1, 3), (3, 1)]
)
def test_remove_center_selects_the_same_points_as_the_reference(
    kernel_h: int, kernel_w: int
) -> None:
    """Dropping the center must compact the point index the same way.

    Row-major indexing or a ``(kernel - 1) // 2`` center still gives well-formed
    output, from the wrong offset triple after the center.
    """
    torch.manual_seed(3)
    operands = _operands(
        kernel_h=kernel_h, kernel_w=kernel_w, drop_center=True, padding=0
    )
    expected = _apply(operands, EAGER)
    actual = _apply(operands, TRITON)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_offset_scale_is_applied_to_the_grid_and_the_offset() -> None:
    """spread rescales the kernel grid as well as the learned offset."""
    torch.manual_seed(4)
    operands = _operands()
    for scale in (0.0, 0.5, 2.0):
        expected = attn.apply(
            operands["value"],
            operands["offset_mask"],
            operands["kernel_size"],
            padding=operands["padding"],
            groups=operands["groups"],
            group_dims=operands["group_dims"],
            spread=scale,
            **EAGER,
        )
        actual = attn.apply(
            operands["value"],
            operands["offset_mask"],
            operands["kernel_size"],
            padding=operands["padding"],
            groups=operands["groups"],
            group_dims=operands["group_dims"],
            spread=scale,
            **TRITON,
        )
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_samples_outside_the_input_contribute_zero() -> None:
    """Far-out offsets must zero the sample, not clamp it to the edge."""
    torch.manual_seed(5)
    operands = _operands(padding=0)
    offset_mask = operands["offset_mask"].clone()
    points = 9
    groups = operands["groups"]
    view = offset_mask.view(*offset_mask.shape[:3], groups, points, 3)
    view[..., :2] = 1e4  # push every sample far outside the input rectangle
    operands["offset_mask"] = view.reshape(offset_mask.shape)

    actual = _apply(operands, TRITON)
    assert torch.count_nonzero(actual) == 0, (
        "samples outside the input rectangle contributed a non-zero value; the "
        "boundary is an open interval and out-of-rectangle corners are zero, "
        "not clamped to the edge"
    )


def test_softmax_method_matches_the_reference() -> None:
    """The modulation mask is normalized over the kernel points, then applied."""
    torch.manual_seed(6)
    operands = _operands()
    expected = attn.apply(
        operands["value"],
        operands["offset_mask"],
        operands["kernel_size"],
        padding=operands["padding"],
        groups=operands["groups"],
        group_dims=operands["group_dims"],
        method="softmax",
        **EAGER,
    )
    actual = attn.apply(
        operands["value"],
        operands["offset_mask"],
        operands["kernel_size"],
        padding=operands["padding"],
        groups=operands["groups"],
        group_dims=operands["group_dims"],
        method="softmax",
        **TRITON,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("configuration", CONFIGURATIONS, ids=CONFIGURATION_IDS)
def test_backward_matches_the_reference(configuration: dict) -> None:
    """Gradients for every differentiable input."""
    torch.manual_seed(7)
    base = _operands(requires_grad=False)

    grads = {}
    for name, selected in (
        ("reference", EAGER),
        ("triton", {"backend": Backend.TRITON, **configuration}),
    ):
        value = base["value"].clone().requires_grad_(True)
        offset_mask = base["offset_mask"].clone().requires_grad_(True)
        out = attn.apply(
            value,
            offset_mask,
            base["kernel_size"],
            padding=base["padding"],
            groups=base["groups"],
            group_dims=base["group_dims"],
            **selected,
        )
        out.sum().backward()
        grads[name] = (value.grad, offset_mask.grad)

    for index, label in enumerate(("value", "offset_mask")):
        torch.testing.assert_close(
            grads["triton"][index],
            grads["reference"][index],
            rtol=1e-3,
            atol=1e-4,
            msg=lambda m, label=label: f"{label} gradient differs:\n{m}",
        )


def test_gradcheck() -> None:
    """Numerical gradient check, which the analytic comparison cannot replace.

    Float32 with the relaxed Triton tolerances, since Triton has no float64 path. No
    padding and mid-cell offsets keep the finite difference off the kinks of
    bilinear sampling, where it would fail against a correct analytic gradient.
    """
    torch.manual_seed(8)
    batch, height, width, groups, group_dims = 1, 8, 8, 1, 4
    kernel_h = kernel_w = 3
    points = kernel_h * kernel_w
    out_h, out_w = height - kernel_h + 1, width - kernel_w + 1

    value = torch.randn(
        batch, height, width, groups * group_dims, device="cuda", requires_grad=True
    )
    packed = torch.empty(batch, out_h, out_w, groups, points, 3, device="cuda")
    # Offsets sit a quarter-cell off the integer coordinates.
    packed[..., :2] = 0.25 + 0.05 * torch.randn_like(packed[..., :2])
    packed[..., 2] = torch.randn_like(packed[..., 2])
    offset_mask = (
        packed.reshape(batch, out_h, out_w, groups * points * 3)
        .clone()
        .requires_grad_(True)
    )

    def run(v: torch.Tensor, om: torch.Tensor) -> torch.Tensor:
        return attn.apply(
            v,
            om,
            (kernel_h, kernel_w),
            padding=0,
            groups=groups,
            group_dims=group_dims,
            **TRITON,
        )

    assert run(value, offset_mask).abs().max() > 1e-3, (
        "the forward is ~zero, so gradcheck would pass without testing anything"
    )
    torch.autograd.gradcheck(
        run,
        (value, offset_mask),
        eps=1e-3,
        atol=1e-2,
        nondet_tol=1e-3,
        fast_mode=False,
    )


def test_gradcheck_with_padded_offset_mask() -> None:
    """The backward writes through the right strides for a wide ``offset_mask``.

    ``DeformAttn2d`` rounds 27 channels up to a multiple of 8, so ``packed`` is a
    non-contiguous view. ``grad_packed`` must share its strides; otherwise every
    position after the first is written at the wrong offset.
    """
    torch.manual_seed(9)
    batch, height, width, groups, group_dims = 1, 8, 8, 1, 4
    kernel_h = kernel_w = 3
    points = kernel_h * kernel_w  # drop_center=False: an odd point count
    out_h, out_w = height - kernel_h + 1, width - kernel_w + 1
    used = groups * points * 3
    padded_width = ((used + 7) // 8) * 8
    assert padded_width > used, (
        "the padding this test exists to cover has to be nonzero"
    )

    value = torch.randn(
        batch, height, width, groups * group_dims, device="cuda", requires_grad=True
    )
    packed = torch.randn(batch, out_h, out_w, padded_width, device="cuda")
    triples = packed[..., :used].view(batch, out_h, out_w, groups, points, 3)
    # Offsets sit a quarter-cell off the integer coordinates, as in test_gradcheck.
    triples[..., :2] = 0.25 + 0.05 * torch.randn_like(triples[..., :2])
    offset_mask = packed.clone().requires_grad_(True)

    def run(v: torch.Tensor, om: torch.Tensor) -> torch.Tensor:
        return attn.apply(
            v,
            om,
            (kernel_h, kernel_w),
            padding=0,
            groups=groups,
            group_dims=group_dims,
            **TRITON,
        )

    assert run(value, offset_mask).abs().max() > 1e-3, (
        "the forward is ~zero, so gradcheck would pass without testing anything"
    )
    torch.autograd.gradcheck(
        run,
        (value, offset_mask),
        eps=1e-3,
        atol=1e-2,
        nondet_tol=1e-3,
        fast_mode=False,
    )
