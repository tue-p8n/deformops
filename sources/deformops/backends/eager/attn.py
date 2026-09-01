r"""Registry adapter for the eager single-scale deformable sampling backend."""

from typing import Any

from ...registry import Backend, attn
from ...registry.attn import resolve_attn_call
from .attn_pytorch import Backend as PyTorchAttnBackend

_backend_impl = PyTorchAttnBackend()


def attn_eager(value: Any, offset_mask: Any, *args: Any, **kwargs: Any) -> Any:
    """Run the single-scale sampling forward with PyTorch operators.

    Arguments are those of :func:`deformops.registry.attn.resolve_attn_call`.
    """
    kwargs.pop("fp32acc", None)
    call = resolve_attn_call(value, offset_mask, *args, **kwargs)
    return _backend_impl.forward(
        value,
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


attn.register(Backend.EAGER, attn_eager, supports=lambda _context: True)
