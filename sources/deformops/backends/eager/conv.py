r"""Registry adapter for the eager deformable convolution backend."""

from typing import Any

from ...registry import Backend, conv
from ...registry.conv import resolve_conv_call
from .conv_pytorch import forward as deform_conv2d_forward


def conv_eager(  # noqa: PLR0913
    input: Any,
    offset: Any,
    weight: Any,
    bias: Any = None,
    mask: Any = None,
    stride: Any = 1,
    padding: Any = 0,
    dilation: Any = 1,
    *,
    groups: int = 1,
    offset_groups: int = 1,
) -> Any:
    """Run the deformable convolution forward with PyTorch operators.

    Arguments are those of :func:`deformops.registry.conv.resolve_conv_call`.
    """
    call = resolve_conv_call(
        input,
        offset,
        weight,
        bias,
        mask,
        stride,
        padding,
        dilation,
        groups=groups,
        offset_groups=offset_groups,
    )
    return deform_conv2d_forward(
        input,
        offset,
        weight,
        bias,
        mask,
        call.stride_h,
        call.stride_w,
        call.pad_h,
        call.pad_w,
        call.dilation_h,
        call.dilation_w,
        call.groups,
        call.offset_groups,
    )


conv.register(Backend.EAGER, conv_eager, supports=lambda _context: True)
