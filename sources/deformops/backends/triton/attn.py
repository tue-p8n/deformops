r"""DeformAttn Triton kernel backend."""

from typing import Any

import torch

from deformops.core.capabilities import has_triton
from deformops.registry import Backend, attn


def attn_triton(
    v: Any,
    offset_mask: Any,
    *args: Any,
    fp32acc: bool = True,
    **kwargs: Any,
) -> Any:
    """Execute DeformAttn forward using the Triton kernel.

    Arguments are those of :func:`deformops.registry.attn.resolve_attn_call`.

    ``fp32acc`` picks the configuration. The selection policy sets it per
    architecture; an explicit call to this backend defaults to the float32
    accumulator.
    """
    from deformops.registry.attn import resolve_attn_call  # noqa: PLC0415

    if fp32acc:
        from .attn_query_tiled_fp32acc import Backend as TritonBackend  # noqa: PLC0415
    else:
        from .attn_query_tiled import Backend as TritonBackend  # noqa: PLC0415

    call = resolve_attn_call(v, offset_mask, *args, **kwargs)
    with torch.cuda.device(v.device):
        return TritonBackend().forward(
            v,
            offset_mask,
            call.kernel_h,
            call.kernel_w,
            call.stride_h,
            call.stride_w,
            call.pad_h,
            call.pad_w,
            call.dilation_h,
            call.dilation_w,
            call.groups,
            call.group_dims,
            call.spread,
            call.method,
            call.drop_center,
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
        """A CUDA tensor on SM 7.0 or newer, not float64."""
        return (
            context.device.type == "cuda"
            and context.dtype != torch.float64
            and context.sm_version >= (7, 0)
        )

    attn.register(Backend.TRITON, attn_triton, supports=_supports)
else:
    attn.mark_unavailable(Backend.TRITON, _triton_unavailable_reason())
