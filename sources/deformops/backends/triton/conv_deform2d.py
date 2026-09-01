r"""Deformable convolution, v1 and v2: gather and contraction in one Triton pass.

One module, as the kernel specializations follow runtime state and requested
gradients. Registered as a ``torch.library`` Triton operator on import; kernels
in :mod:`._kernels.conv_deform2d`. See ``docs/kernels.md``.
"""

from __future__ import annotations

from ...core.capabilities import require_triton

require_triton(__name__.rsplit(".", 1)[-1])

from typing import Any, override

import torch

from ...registry import ConvBackend, ConvContext
from ._kernels.conv_deform2d import backward_launch, forward_launch
from ._vmap import make_vmap_rule

__all__ = ["Backend"]

#: Operator-describing positions. vmap over them raises, since it would contract
#: against the wrong filters or launch per element.
_SHARED_FORWARD = {3: "a convolution weight", 4: "a convolution bias"}
_SHARED_BACKWARD = {4: "a convolution weight"}

_OP_NS = "deformops_triton"
_FORWARD_OP = f"{_OP_NS}::conv_deform2d_forward"


@torch.library.triton_op(_FORWARD_OP, mutates_args=())  # pyright: ignore[reportUntypedFunctionDecorator]
def _forward_op(  # noqa: PLR0913
    value: torch.Tensor,
    offset: torch.Tensor,
    mask: torch.Tensor | None,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dilation_h: int,
    dilation_w: int,
    groups: int,
    offset_groups: int,
) -> torch.Tensor:
    return forward_launch(
        value.contiguous(),
        offset.contiguous(),
        mask.contiguous() if mask is not None else None,
        weight.contiguous(),
        bias.contiguous() if bias is not None else None,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups,
        offset_groups,
        offset.shape[1],
        offset.shape[2],
    )


_BACKWARD_OP = f"{_OP_NS}::conv_deform2d_backward"


@torch.library.triton_op(_BACKWARD_OP, mutates_args=())  # pyright: ignore[reportUntypedFunctionDecorator]
def _backward_op(  # noqa: PLR0913
    grad_output: torch.Tensor,
    value: torch.Tensor,
    offset: torch.Tensor,
    mask: torch.Tensor | None,
    weight: torch.Tensor,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dilation_h: int,
    dilation_w: int,
    groups: int,
    offset_groups: int,
    *,
    needs_value_grad: bool,
    needs_offset_grad: bool,
    needs_mask_grad: bool,
    needs_weight_grad: bool,
    bias_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    grad_value, grad_offset, grad_mask, grad_weight, grad_bias = backward_launch(
        grad_output.contiguous(),
        value.contiguous(),
        offset.contiguous(),
        mask.contiguous() if mask is not None else None,
        weight.contiguous(),
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups,
        offset_groups,
        needs_value_grad=needs_value_grad,
        needs_offset_grad=needs_offset_grad,
        needs_mask_grad=needs_mask_grad,
        needs_weight_grad=needs_weight_grad,
        bias_dtype=bias_dtype,
    )

    # Schemas have no ``Tensor?`` return: unrequested outputs are separate
    # zero-element placeholders, since returns must not alias.
    def placeholder() -> torch.Tensor:
        return value.new_empty(0)

    return (
        grad_value if grad_value is not None else placeholder(),
        grad_offset if grad_offset is not None else placeholder(),
        grad_mask if grad_mask is not None else placeholder(),
        grad_weight if grad_weight is not None else placeholder(),
        grad_bias if grad_bias is not None else placeholder(),
    )


def _setup_context(ctx: Any, inputs: tuple[Any, ...], output: torch.Tensor) -> None:  # noqa: ARG001
    value, offset, mask, weight, bias, *geometry = inputs
    ctx.save_for_backward(value, offset, mask, weight)
    ctx.geometry = geometry
    # Requested gradients, read here by input name instead of from
    # ``ctx.needs_input_grad``. A ``None`` bias dtype means no bias gradient.
    ctx.needs_value_grad = value.requires_grad
    ctx.needs_offset_grad = offset.requires_grad
    ctx.needs_mask_grad = mask is not None and mask.requires_grad
    ctx.needs_weight_grad = weight.requires_grad
    ctx.bias_dtype = bias.dtype if bias is not None and bias.requires_grad else None


def _backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
    value, offset, mask, weight = ctx.saved_tensors
    grad_value, grad_offset, grad_mask, grad_weight, grad_bias = (
        torch.ops.deformops_triton.conv_deform2d_backward(
            grad_output,
            value,
            offset,
            mask,
            weight,
            *ctx.geometry,
            needs_value_grad=ctx.needs_value_grad,
            needs_offset_grad=ctx.needs_offset_grad,
            needs_mask_grad=ctx.needs_mask_grad,
            needs_weight_grad=ctx.needs_weight_grad,
            bias_dtype=ctx.bias_dtype,
        )
    )
    # Restore ``None`` for the placeholders of unrequested outputs.
    return (
        grad_value if ctx.needs_value_grad else None,
        grad_offset if ctx.needs_offset_grad else None,
        grad_mask if ctx.needs_mask_grad else None,
        grad_weight if ctx.needs_weight_grad else None,
        grad_bias if ctx.bias_dtype is not None else None,
        *([None] * len(ctx.geometry)),
    )


torch.library.register_autograd(_FORWARD_OP, _backward, setup_context=_setup_context)
torch.library.register_vmap(
    _forward_op, make_vmap_rule(_forward_op, shared=_SHARED_FORWARD)
)
torch.library.register_vmap(
    _backward_op, make_vmap_rule(_backward_op, shared=_SHARED_BACKWARD)
)


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class Backend(ConvBackend):
    """Dispatch adapter for the fused Triton kernel."""

    @property
    @override
    def name(self) -> str:
        """The registry name, ``"triton_deform2d"``."""
        return "triton_deform2d"

    @override
    def supports(self, context: ConvContext) -> bool:
        """Whether ``context`` is CUDA and not float64, which Triton cannot run."""
        return context.device.type == "cuda" and context.dtype != torch.float64

    @override
    def forward(  # noqa: PLR0913
        self,
        value: torch.Tensor,
        offset: torch.Tensor,
        mask: torch.Tensor | None,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        stride_h: int,
        stride_w: int,
        pad_h: int,
        pad_w: int,
        dilation_h: int,
        dilation_w: int,
        groups: int,
        offset_groups: int,
    ) -> torch.Tensor:
        """Delegate to :func:`forward_launch`, which documents the operands."""
        return torch.ops.deformops_triton.conv_deform2d_forward(
            value,
            offset,
            mask,
            weight,
            bias,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            groups,
            offset_groups,
        )
