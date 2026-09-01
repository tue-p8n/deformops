r"""DeformConv Triton kernel backend."""

from typing import Any

import torch

from deformops.core.capabilities import has_triton
from deformops.registry import Backend, conv


def conv_triton(  # noqa: PLR0913
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
    """Execute DeformConv2d forward using the Triton kernel.

    Arguments are those of :func:`deformops.registry.conv.resolve_conv_call`.
    """
    from deformops.registry.conv import resolve_conv_call  # noqa: PLC0415

    from .conv_deform2d import Backend as TritonConvBackend  # noqa: PLC0415

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
    with torch.cuda.device(input.device):
        return TritonConvBackend().forward(
            input,
            offset,
            mask,
            weight,
            bias,
            call.stride_h,
            call.stride_w,
            call.pad_h,
            call.pad_w,
            call.dilation_h,
            call.dilation_w,
            call.groups,
            call.offset_groups,
        )


def _triton_unavailable_reason() -> str:
    if not has_triton():
        return (
            "Triton is not installed; install deformops[triton] or a CUDA build "
            "of torch, which includes it"
        )
    return "no CUDA device is available"


if has_triton() and torch.cuda.is_available():

    def _supports(context: Any) -> bool:
        """A CUDA tensor, not float64."""
        return context.device.type == "cuda" and context.dtype != torch.float64

    conv.register(Backend.TRITON, conv_triton, supports=_supports)
else:
    conv.mark_unavailable(Backend.TRITON, _triton_unavailable_reason())
