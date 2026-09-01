r"""Drop-in replacements for ``torchvision.ops`` deformable convolution.

    from deformops.integrations.torchvision import DeformConv2d, deform_conv2d

Same argument names, order and NCHW layout as ``torchvision.ops``. Results can
differ in the last bits, so a model trained on one may not reproduce bit for bit
on the other.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from deformops.nn.conv import DeformConv2d as _DeformConv2d
from deformops.registry import Backend, conv

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["DeformConv2d", "deform_conv2d"]


class DeformConv2d(_DeformConv2d):
    """:class:`deformops.nn.conv.DeformConv2d` at a torchvision-shaped path."""


def _to_channels_last(tensor: Tensor) -> Tensor:
    permuted = tensor.permute(0, 2, 3, 1)
    return permuted if permuted.is_contiguous() else permuted.contiguous()


def deform_conv2d(  # noqa: PLR0913
    input: Tensor,  # noqa: A002 - torchvision's parameter name
    offset: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
    mask: Tensor | None = None,
    *,
    backend: Backend | None = None,
) -> Tensor:
    """Drop-in replacement for ``torchvision.ops.deform_conv2d``."""
    batch = input.shape[0]
    _, in_per_group, kernel_h, kernel_w = weight.shape
    points = kernel_h * kernel_w
    out_h, out_w = offset.shape[2], offset.shape[3]
    groups = input.shape[1] // in_per_group
    offset_groups = offset.shape[1] // (2 * points)

    offset_native = (
        offset.reshape(batch, offset_groups, points, 2, out_h, out_w)
        .permute(0, 4, 5, 1, 2, 3)
        .contiguous()
    )
    mask_native = (
        mask.reshape(batch, offset_groups, points, out_h, out_w)
        .permute(0, 3, 4, 1, 2)
        .contiguous()
        if mask is not None
        else None
    )
    out = conv(
        _to_channels_last(input),
        offset_native,
        weight,
        bias=bias,
        mask=mask_native,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        offset_groups=offset_groups,
        backend=backend,
    )
    permuted = out.permute(0, 3, 1, 2)
    return permuted if permuted.is_contiguous() else permuted.contiguous()
