"""Input validation for the operator entry points.

Constructor checks validate module integers and always run. Call checks validate
operands, because the kernels index ``value`` from the level table without
re-deriving it. Structural checks read metadata only and run on every call; value
checks synchronize, so they run only under ``DEFORMOPS_VALIDATE``. See
``docs/conventions.md``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

__all__ = [
    "VALIDATE_ENV_VAR",
    "check_2power",
    "check_attn_call",
    "check_conv_call",
    "check_conv_params",
    "check_divisible",
    "check_msda_call",
    "check_msda_params",
    "check_positive",
    "conv_output_size",
    "value_checks_enabled",
]

#: Largest flat address a signed 32-bit index holds; a device kernel may index at
#: that width, so larger inputs are rejected instead of wrapping.
_INT32_INDEX_LIMIT = (1 << 31) - 1


def _check_int32_addressable(batch: int, per_image_elements: int, name: str) -> None:
    """Reject a per-image extent that overflows int32 once multiplied by batch.

    The kernels base each image at ``batch_index * per_image_stride``, so the
    product, not the per-image size, is what has to fit.
    """
    total = batch * per_image_elements
    if total > _INT32_INDEX_LIMIT:
        msg = (
            f"{name} has {per_image_elements} elements per image and a batch "
            f"of {batch} ({total} total), which exceeds the int32 address "
            f"space the device kernel indexes with ({_INT32_INDEX_LIMIT}); "
            f"reduce the batch, spatial size, or channel count"
        )
        raise ValueError(msg)


def check_2power(n: int) -> None:
    """Check whether ``n`` is a positive power of 2.

    Parameters
    ----------
    n
        The integer to check.

    Raises
    ------
    ValueError
        If ``n`` is not positive, or is positive but not a power of 2.
    """
    if n <= 0:
        msg = f"Expected a positive integer, got: {n}"
        raise ValueError(msg)
    if n & (n - 1) != 0:
        msg = f"Expected a power of 2, got: {n}"
        raise ValueError(msg)


def check_divisible(a: int, b: int) -> None:
    """Check whether ``a`` is divisible by ``b``.

    Parameters
    ----------
    a
        The dividend.
    b
        The divisor.

    Raises
    ------
    ValueError
        If ``b`` is zero, or ``a`` is not evenly divisible by ``b``.
    """
    if b == 0:
        msg = "Expected a non-zero divisor, got: 0"
        raise ValueError(msg)
    if a % b != 0:
        msg = f"Expected {a} to be divisible by {b}"
        raise ValueError(msg)


def check_positive(n: int, name: str) -> None:
    """Check whether ``n`` is a positive integer.

    Parameters
    ----------
    n
        The integer to check.
    name
        The parameter's own name, used to name it in the error.

    Raises
    ------
    ValueError
        If ``n`` is not positive.
    """
    if n <= 0:
        msg = f"Expected {name} to be a positive integer, got: {n}"
        raise ValueError(msg)


def check_msda_params(
    dim: int, num_heads: int, num_levels: int, num_points: int
) -> None:
    """Check MSDA parameters for validity.

    Parameters
    ----------
    dim
        The model's embedding dimension.
    num_heads
        The number of attention heads ``dim`` is split across.
    num_levels
        The number of feature-map levels sampled.
    num_points
        The number of sampling points per level.

    Raises
    ------
    ValueError
        If any parameter is not positive, or ``dim`` is not divisible by
        ``num_heads``.
    """
    check_positive(dim, "dim")
    check_positive(num_heads, "num_heads")
    check_positive(num_levels, "num_levels")
    check_positive(num_points, "num_points")
    check_divisible(dim, num_heads)


def check_conv_params(in_channels: int, out_channels: int, groups: int) -> None:
    """Check classic deformable convolution's constructor parameters.

    Parameters
    ----------
    in_channels
        The number of input channels.
    out_channels
        The number of output channels.
    groups
        The number of convolution groups ``in_channels`` and ``out_channels``
        are each split across.

    Raises
    ------
    ValueError
        If any parameter is not positive, or ``in_channels``/``out_channels``
        is not divisible by ``groups``.
    """
    check_positive(in_channels, "in_channels")
    check_positive(out_channels, "out_channels")
    check_positive(groups, "groups")
    check_divisible(in_channels, groups)
    check_divisible(out_channels, groups)


# Call-time tensor validation.

#: Enables the value checks; any value but a falsy spelling below turns them on.
VALIDATE_ENV_VAR = "DEFORMOPS_VALIDATE"

_FALSY = frozenset({"0", "false", "no", "off", ""})


def value_checks_enabled() -> bool:
    """Whether to run the checks that read index tensors back to the host.

    Off by default: they synchronize and break ``torch.compile`` and CUDA-graph
    capture.
    """
    return os.getenv(VALIDATE_ENV_VAR, "0").strip().lower() not in _FALSY


def _has_values(*tensors: Tensor) -> bool:
    """Whether the tensors hold readable values: not fake, even wrapped, nor traced."""
    import torch  # noqa: PLC0415
    from torch._subclasses.fake_tensor import is_fake  # noqa: PLC0415

    if torch.compiler.is_compiling():
        return False
    return not any(is_fake(tensor) for tensor in tensors)


def _rank(name: str, tensor: Tensor, expected: int | tuple[int, ...]) -> None:
    """Require ``tensor`` to have one of the ``expected`` ranks."""
    allowed = (expected,) if isinstance(expected, int) else expected
    if tensor.dim() not in allowed:
        wanted = " or ".join(str(a) for a in allowed)
        msg = (
            f"{name} must be {wanted}-dimensional, got {tensor.dim()} dimensions "
            f"with shape {tuple(tensor.shape)}"
        )
        raise ValueError(msg)


def _axis(name: str, tensor: Tensor, axis: int, expected: int, why: str) -> None:
    """Require ``tensor.size(axis)`` to equal ``expected``."""
    actual = tensor.size(axis)
    if actual != expected:
        msg = (
            f"{name} has size {actual} on axis {axis}, expected {expected} ({why}). "
            f"Full shape: {tuple(tensor.shape)}"
        )
        raise ValueError(msg)


def _same_device(named: Sequence[tuple[str, Tensor]]) -> None:
    """Require every tensor to sit on the same device."""
    reference_name, reference = named[0]
    for name, tensor in named[1:]:
        if tensor.device != reference.device:
            msg = (
                f"{name} is on {tensor.device} but {reference_name} is on "
                f"{reference.device}; all operands must share one device"
            )
            raise ValueError(msg)


def _same_float_dtype(named: Sequence[tuple[str, Tensor]]) -> None:
    """Require every tensor to be floating point and share one dtype."""
    reference_name, reference = named[0]
    for name, tensor in named:
        if not tensor.is_floating_point():
            msg = f"{name} must be a floating-point tensor, got dtype {tensor.dtype}"
            raise ValueError(msg)
    for name, tensor in named[1:]:
        if tensor.dtype != reference.dtype:
            msg = (
                f"{name} has dtype {tensor.dtype} but {reference_name} has "
                f"{reference.dtype}; mixed input dtypes are not supported. Cast "
                f"the operands to a common dtype, or wrap the call in autocast"
            )
            raise ValueError(msg)


def _floating(name: str, tensor: Tensor) -> None:
    """Require a tensor to be floating point, without constraining its width."""
    if not tensor.is_floating_point():
        msg = f"{name} must be a floating-point tensor, got dtype {tensor.dtype}"
        raise ValueError(msg)


def _integral(name: str, tensor: Tensor) -> None:
    """Require an index tensor to hold integers."""
    if tensor.is_floating_point() or tensor.is_complex():
        msg = (
            f"{name} must be an integer tensor (it indexes the value tensor), "
            f"got dtype {tensor.dtype}"
        )
        raise ValueError(msg)


def _to_list(tensor: Tensor) -> Any:
    """Read a small index tensor back to the host. Synchronizes."""
    return tensor.detach().cpu().tolist()


def check_msda_call(
    value: Tensor,
    spatial_shapes: Tensor,
    level_start_index: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> None:
    r"""Validate the tensors passed to multi-scale deformable attention.

    With ``B`` batch, ``S`` positions over all levels, ``G`` heads, ``D`` channels
    per head, ``Q`` queries, ``L`` levels and ``K`` points per level.

    Parameters
    ----------
    value
        The sampled feature map, shape ``(B, S, G, D)``.
    spatial_shapes
        One ``(height, width)`` per level, shape ``(L, 2)``, integer.
    level_start_index
        The exclusive prefix sum of ``height * width`` over levels, shape
        ``(L,)``, integer.
    sampling_locations
        Normalized ``(x, y)`` sample points, shape ``(B, Q, G, L, K, 2)``.
    attention_weights
        Per-point modulation weights, shape ``(B, Q, G, L * K)`` (flat) or
        ``(B, Q, G, L, K)`` (split); both are accepted.

    Raises
    ------
    ValueError
        Describing the first mismatch. The checks that read the level table back
        to the host run only under ``DEFORMOPS_VALIDATE``.
    """
    _rank("value", value, 4)
    _rank("spatial_shapes", spatial_shapes, 2)
    _rank("level_start_index", level_start_index, 1)
    _rank("sampling_locations", sampling_locations, 6)
    _rank("attention_weights", attention_weights, (4, 5))

    _axis("spatial_shapes", spatial_shapes, 1, 2, "each row is (height, width)")
    _axis(
        "sampling_locations",
        sampling_locations,
        5,
        2,
        "each location is a normalized (x, y) pair",
    )

    batch, _, num_heads, _ = value.shape
    num_levels = spatial_shapes.size(0)
    _, num_queries, _, loc_levels, num_points, _ = sampling_locations.shape

    _axis("level_start_index", level_start_index, 0, num_levels, "one entry per level")
    _axis("sampling_locations", sampling_locations, 0, batch, "batch of value")
    _axis("sampling_locations", sampling_locations, 2, num_heads, "heads of value")
    _axis(
        "sampling_locations",
        sampling_locations,
        3,
        num_levels,
        "rows of spatial_shapes",
    )
    _axis("attention_weights", attention_weights, 0, batch, "batch of value")
    _axis(
        "attention_weights", attention_weights, 1, num_queries, "queries of locations"
    )
    _axis("attention_weights", attention_weights, 2, num_heads, "heads of value")

    # The kernels take flat weights; split ones are accepted because the softmax
    # normalizes over the joint (level, point) axis either way.
    if attention_weights.dim() == 4:  # noqa: PLR2004
        _axis(
            "attention_weights",
            attention_weights,
            3,
            loc_levels * num_points,
            "levels x points of sampling_locations",
        )
    else:
        _axis(
            "attention_weights",
            attention_weights,
            3,
            loc_levels,
            "levels of sampling_locations",
        )
        _axis(
            "attention_weights",
            attention_weights,
            4,
            num_points,
            "points of sampling_locations",
        )

    _integral("spatial_shapes", spatial_shapes)
    _integral("level_start_index", level_start_index)
    # The compiled kernels read the weights at the value's width, so a mismatch
    # misreads memory. Locations may differ under autocast; see docs/precision.md.
    _same_float_dtype(
        [
            ("value", value),
            ("attention_weights", attention_weights),
        ]
    )
    _floating("sampling_locations", sampling_locations)
    _same_device(
        [
            ("value", value),
            ("spatial_shapes", spatial_shapes),
            ("level_start_index", level_start_index),
            ("sampling_locations", sampling_locations),
            ("attention_weights", attention_weights),
        ]
    )

    if value_checks_enabled() and _has_values(spatial_shapes, level_start_index):
        _check_msda_level_layout(value, spatial_shapes, level_start_index)


def _check_msda_level_layout(
    value: Tensor, spatial_shapes: Tensor, level_start_index: Tensor
) -> None:
    """Check that the level table tiles the value tensor.

    Each level must start where the previous one ended, and the levels must cover
    ``value`` exactly. No kernel re-derives this, so a violation reads out of bounds.
    """
    shapes: list[list[int]] = _to_list(spatial_shapes)
    starts: list[int] = _to_list(level_start_index)

    for level, (height, width) in enumerate(shapes):
        if height <= 0 or width <= 0:
            msg = (
                f"spatial_shapes row {level} is ({height}, {width}); every level "
                f"must have a positive height and width"
            )
            raise ValueError(msg)

    expected: list[int] = []
    running = 0
    for height, width in shapes:
        expected.append(running)
        running += height * width

    if starts != expected:
        msg = (
            f"level_start_index {starts} is not the exclusive prefix sum of the "
            f"level sizes {[h * w for h, w in shapes]}; expected {expected}. The "
            f"kernels index the value tensor from this table without re-deriving "
            f"it, so a mismatch reads out of bounds"
        )
        raise ValueError(msg)

    spatial_size = value.size(1)
    if running != spatial_size:
        msg = (
            f"spatial_shapes describes {running} positions "
            f"({[(h, w) for h, w in shapes]}) but value has {spatial_size} on its "
            f"spatial axis; the levels must tile the value tensor exactly"
        )
        raise ValueError(msg)


def conv_output_size(
    extent: int, kernel: int, stride: int, padding: int, dilation: int
) -> int:
    """The output extent of a convolution along one axis.

    Shared so the reference, the device kernel and validation agree on the grid.

    Parameters
    ----------
    extent, kernel, stride, padding, dilation
        Along this axis; padding applies to each side.

    Returns
    -------
    The output extent along this axis.
    """
    return (extent + 2 * padding - (dilation * (kernel - 1) + 1)) // stride + 1


def check_attn_call(  # noqa: PLR0913
    value: Tensor,
    offset_mask: Tensor,
    kernel_h: int,
    kernel_w: int,
    groups: int,
    group_dims: int,
    *,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dilation_h: int,
    dilation_w: int,
    drop_center: bool,
) -> None:
    r"""Validate the tensors passed to single-scale deformable sampling.

    The geometry is checked too. The reference derives the output grid from the
    input, but the device kernel reads it from ``offset_mask``, so a mismatch would
    make the backends disagree.

    Parameters
    ----------
    value
        The input feature map, channels-last, shape ``(B, H, W, C)`` with
        ``C = groups * group_dims``.
    offset_mask
        Packed offsets and modulation, channels-last, shape
        ``(B, H_out, W_out, groups * K * 3)``, with ``K`` points per group
        (``kernel_h * kernel_w``, less the center under ``drop_center``) and
        two offset components and a mask per point.
    kernel_h, kernel_w
        The kernel extent.
    groups
        The number of groups ``value``'s channels are split across.
    group_dims
        The number of channels per group.
    stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w
        Convolution geometry; padding applies to each side.
    drop_center
        Whether the kernel's center point is excluded from ``K``.

    Raises
    ------
    ValueError
        Describing the first mismatch: geometry, rank or axis, ``offset_mask``
        width, int32 address space, or output grid.
    """
    check_positive(kernel_h, "kernel_h")
    check_positive(kernel_w, "kernel_w")
    check_positive(groups, "groups")
    check_positive(group_dims, "group_dims")
    check_positive(stride_h, "stride_h")
    check_positive(stride_w, "stride_w")
    check_positive(dilation_h, "dilation_h")
    check_positive(dilation_w, "dilation_w")

    _rank("value", value, 4)
    _rank("offset_mask", offset_mask, 4)

    _axis(
        "value",
        value,
        3,
        groups * group_dims,
        "channels-last layout with groups x group_dims channels",
    )
    _axis("offset_mask", offset_mask, 0, value.size(0), "batch of value")

    points = kernel_h * kernel_w - (1 if drop_center else 0)
    if points <= 0:
        msg = (
            f"kernel {kernel_h}x{kernel_w} with drop_center={drop_center} "
            f"leaves no sampling points"
        )
        raise ValueError(msg)
    required = groups * points * 3
    if offset_mask.size(3) < required:
        msg = (
            f"offset_mask has {offset_mask.size(3)} channels, need at least "
            f"{required} (groups={groups} x points={points} x 3 for "
            f"offset_x, offset_y, weight). Full shape: {tuple(offset_mask.shape)}"
        )
        raise ValueError(msg)

    _same_float_dtype([("value", value), ("offset_mask", offset_mask)])
    _same_device([("value", value), ("offset_mask", offset_mask)])

    _check_int32_addressable(
        value.size(0), value.size(1) * value.size(2) * value.size(3), "value"
    )

    expected_h = conv_output_size(value.size(1), kernel_h, stride_h, pad_h, dilation_h)
    expected_w = conv_output_size(value.size(2), kernel_w, stride_w, pad_w, dilation_w)

    _check_int32_addressable(
        offset_mask.size(0),
        expected_h * expected_w * offset_mask.size(3),
        "offset_mask",
    )

    if (offset_mask.size(1), offset_mask.size(2)) != (expected_h, expected_w):
        msg = (
            f"offset_mask has a {offset_mask.size(1)}x{offset_mask.size(2)} output "
            f"grid but the geometry implies {expected_h}x{expected_w} for a "
            f"{value.size(1)}x{value.size(2)} input with kernel "
            f"{kernel_h}x{kernel_w}, stride ({stride_h}, {stride_w}), padding "
            f"({pad_h}, {pad_w}) and dilation ({dilation_h}, {dilation_w}). The "
            f"reference derives this grid and the device kernel reads it from "
            f"offset_mask, so a mismatch changes the answer depending on which "
            f"backend runs"
        )
        raise ValueError(msg)


def check_conv_call(  # noqa: PLR0913, PLR0915
    value: Tensor,
    offset: Tensor,
    mask: Tensor | None,
    weight: Tensor,
    bias: Tensor | None,
    groups: int,
    offset_groups: int,
    *,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dilation_h: int,
    dilation_w: int,
) -> None:
    r"""Validate the tensors passed to classic deformable convolution.

    The geometry is checked for the reason :func:`check_attn_call` gives. Reads
    metadata only, so it is safe under ``torch.compile`` and runs on every call.

    Parameters
    ----------
    value
        The input feature map, channels-last, shape ``(B, H, W, C_in)``.
    offset
        Per-point ``(dy, dx)`` offsets, channels-last, shape
        ``(B, H_out, W_out, offset_groups, K, 2)``.
    mask
        Per-point modulation, channels-last, shape
        ``(B, H_out, W_out, offset_groups, K)``, or ``None`` for the
        unmodulated (v1) form.
    weight
        The convolution weight, shape ``(C_out, C_in // groups, kH, kW)``.
    bias
        Per-output-channel bias, shape ``(C_out,)``, or ``None``.
    groups
        Convolution groups of ``value`` and ``weight``.
    offset_groups
        Groups of ``offset`` and ``mask``, independent of ``groups``.
    stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w
        Convolution geometry; padding applies to each side.

    Raises
    ------
    ValueError
        Describing the first mismatch: geometry, rank or axis, group divisibility,
        int32 address space, or output grid.
    """
    check_positive(groups, "groups")
    check_positive(offset_groups, "offset_groups")
    check_positive(stride_h, "stride_h")
    check_positive(stride_w, "stride_w")
    check_positive(dilation_h, "dilation_h")
    check_positive(dilation_w, "dilation_w")

    _rank("value", value, 4)
    _rank("offset", offset, 6)
    _rank("weight", weight, 4)

    in_channels = value.size(3)
    out_channels, in_per_group, kernel_h, kernel_w = weight.shape
    points = kernel_h * kernel_w

    if in_per_group * groups != in_channels:
        msg = (
            f"weight expects {in_per_group} input channels per group and "
            f"{groups=}, which is {in_per_group * groups} channels, but value "
            f"has {in_channels}"
        )
        raise ValueError(msg)
    # The kernel writes an even share per group into an uninitialized buffer, so a
    # remainder leaves the last ``out_channels % groups`` channels unwritten.
    if out_channels % groups != 0:
        msg = (
            f"weight produces {out_channels} output channels, which is not "
            f"divisible by {groups=}; a grouped convolution partitions the "
            f"output channels evenly over its groups"
        )
        raise ValueError(msg)
    if in_channels % offset_groups != 0:
        msg = f"{in_channels=} must be divisible by {offset_groups=}"
        raise ValueError(msg)

    _axis("offset", offset, 0, value.size(0), "batch of value")
    _axis("offset", offset, 3, offset_groups, "one entry per offset group")
    _axis("offset", offset, 4, points, "one entry per kernel point")
    _axis("offset", offset, 5, 2, "a (dy, dx) pair per point")

    expected_h = conv_output_size(value.size(1), kernel_h, stride_h, pad_h, dilation_h)
    expected_w = conv_output_size(value.size(2), kernel_w, stride_w, pad_w, dilation_w)
    if (offset.size(1), offset.size(2)) != (expected_h, expected_w):
        msg = (
            f"offset has a {offset.size(1)}x{offset.size(2)} output grid but "
            f"the geometry implies {expected_h}x{expected_w} for a "
            f"{value.size(1)}x{value.size(2)} input with kernel "
            f"{kernel_h}x{kernel_w}, stride ({stride_h}, {stride_w}), padding "
            f"({pad_h}, {pad_w}) and dilation ({dilation_h}, {dilation_w}). The "
            f"reference derives this grid and the device kernel reads it from "
            f"offset, so a mismatch changes the answer depending on which "
            f"backend runs"
        )
        raise ValueError(msg)

    # The kernel bases each image at ``batch * stride`` for input and output; either
    # can be the larger.
    _check_int32_addressable(
        value.size(0), value.size(1) * value.size(2) * value.size(3), "value"
    )
    _check_int32_addressable(
        value.size(0), expected_h * expected_w * out_channels, "output"
    )

    if mask is not None:
        _rank("mask", mask, 5)
        _axis("mask", mask, 0, value.size(0), "batch of value")
        for axis in (1, 2):
            _axis("mask", mask, axis, offset.size(axis), "output grid of offset")
        _axis("mask", mask, 3, offset_groups, "one entry per offset group")
        _axis("mask", mask, 4, points, "one entry per kernel point")
        _floating("mask", mask)

    # The kernel reads bias by output channel without a length check, so a short one
    # reads past its end on the device.
    if bias is not None:
        _rank("bias", bias, 1)
        _axis("bias", bias, 0, out_channels, "one entry per output channel")

    # Autocast leaves offsets in float32 while value and weight are half, so offset
    # only has to be floating point.
    _floating("offset", offset)
    _same_float_dtype([("value", value), ("weight", weight)])
    named: list[tuple[str, Tensor]] = [
        ("value", value),
        ("offset", offset),
        ("weight", weight),
    ]
    if mask is not None:
        named.append(("mask", mask))
    if bias is not None:
        named.append(("bias", bias))
    _same_device(named)
