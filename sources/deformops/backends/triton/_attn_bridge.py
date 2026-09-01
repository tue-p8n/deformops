r"""Shared machinery behind the single-scale deformable attention modules.

A configuration states where its input gradient accumulates;
:func:`install_attn_family` registers the operator pair, fake kernels, autograd
and vmap. Softmax runs before the operator, so autograd owns its Jacobian. See
``docs/export.md``.
"""

from __future__ import annotations

from ...core.capabilities import require_triton

require_triton(__name__.rsplit(".", 1)[-1])

import typing
from typing import Any, override

import torch
from torch import Tensor

from ...registry import AttnBackend, AttnContext, AttnMethod
from ._kernels.attn_query_tiled import backward_launch, forward_launch
from ._vmap import make_vmap_rule

__all__ = ["OP_NAMESPACE", "AttnFamily", "install_attn_family"]

OP_NAMESPACE = "deformops_triton"


def softmax_weights(offset_mask: Tensor, groups: int, num_points: int) -> Tensor:
    """Softmax each packed ``(dx, dy, weight)`` weight over the kernel points."""
    batch, out_h, out_w, _ = offset_mask.shape
    packed = offset_mask[..., : groups * num_points * 3].reshape(
        batch, out_h, out_w, groups, num_points, 3
    )
    weights = packed[..., 2].softmax(dim=-1)
    normalized = torch.cat([packed[..., :2], weights.unsqueeze(-1)], dim=-1)
    return normalized.reshape(batch, out_h, out_w, groups * num_points * 3)


class AttnFamily(typing.NamedTuple):
    """Everything one configuration module re-exports."""

    backend: type[Any]


def install_attn_family(
    *,
    op_prefix: str,
    backend_name: str,
    fp32acc: bool,
    summary: str,
) -> AttnFamily:
    """Register one configuration's operator pair and build its adapter.

    Parameters
    ----------
    op_prefix
        Operator-name prefix, e.g. ``"attn_query_tiled"``. The registered names
        are ``{prefix}_forward`` and ``{prefix}_backward``.
    backend_name
        What this configuration answers to as a backend.
    fp32acc
        Whether the backward accumulates the input gradient in float32 rather
        than the input dtype.
    summary
        One line naming this configuration, for the adapter's docstring.
    """

    @torch.library.triton_op(f"{OP_NAMESPACE}::{op_prefix}_forward", mutates_args=())  # pyright: ignore[reportUntypedFunctionDecorator]
    def forward_op(  # noqa: PLR0913
        value: Tensor,
        offset_mask: Tensor,
        kernel_h: int,
        kernel_w: int,
        stride_h: int,
        stride_w: int,
        pad_h: int,
        pad_w: int,
        dilation_h: int,
        dilation_w: int,
        groups: int,
        group_dims: int,
        spread: float,
        drop_center: bool,
    ) -> Tensor:
        return forward_launch(
            value.contiguous(),
            offset_mask.contiguous(),
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            groups,
            group_dims,
            spread,
            drop_center=drop_center,
        )

    @torch.library.triton_op(f"{OP_NAMESPACE}::{op_prefix}_backward", mutates_args=())  # pyright: ignore[reportUntypedFunctionDecorator]
    def backward_op(  # noqa: PLR0913
        grad_output: Tensor,
        value: Tensor,
        offset_mask: Tensor,
        kernel_h: int,
        kernel_w: int,
        stride_h: int,
        stride_w: int,
        pad_h: int,
        pad_w: int,
        dilation_h: int,
        dilation_w: int,
        groups: int,
        group_dims: int,
        spread: float,
        drop_center: bool,
    ) -> tuple[Tensor, Tensor]:
        return backward_launch(
            grad_output.contiguous(),
            value.contiguous(),
            offset_mask.contiguous(),
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            groups,
            group_dims,
            spread,
            drop_center=drop_center,
            fp32acc=fp32acc,
        )

    torch.library.register_vmap(forward_op, make_vmap_rule(forward_op))
    torch.library.register_vmap(backward_op, make_vmap_rule(backward_op))

    def _setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:  # noqa: ARG001
        value, offset_mask, *geometry = inputs
        ctx.save_for_backward(value, offset_mask)
        ctx.geometry = geometry

    def _bridge(ctx: Any, grad_output: Tensor) -> tuple[Any, ...]:
        value, offset_mask = ctx.saved_tensors
        operator = getattr(getattr(torch.ops, OP_NAMESPACE), f"{op_prefix}_backward")
        grad_value, grad_offset_mask = operator(
            grad_output.contiguous(), value, offset_mask, *ctx.geometry
        )
        # One None per non-differentiable positional input.
        return (grad_value, grad_offset_mask, *([None] * len(ctx.geometry)))

    torch.library.register_autograd(
        f"{OP_NAMESPACE}::{op_prefix}_forward",
        _bridge,
        setup_context=_setup_context,
    )

    class Backend(AttnBackend):
        @property
        @override
        def name(self) -> str:
            return backend_name

        @override
        def supports(self, context: AttnContext) -> bool:
            """Whether ``context`` is CUDA and not float64, which Triton cannot run."""
            return context.device.type == "cuda" and context.dtype != torch.float64

        @override
        def forward(  # noqa: PLR0913
            self,
            value: Tensor,
            offset_mask: Tensor,
            kernel_h: int,
            kernel_w: int,
            stride_h: int,
            stride_w: int,
            pad_h: int,
            pad_w: int,
            dilation_h: int,
            dilation_w: int,
            groups: int,
            group_dims: int,
            spread: float,
            method: AttnMethod,
            drop_center: bool,
        ) -> Tensor:
            """See :meth:`AttnBackend.forward`.

            ``value`` is ``(B, H, W, group * group_dims)`` and ``offset_mask``
            ``(B, H_out, W_out, group * K * 3)`` with ``(dx, dy, weight)`` per
            point, both channels-last. Returns ``(B, H_out, W_out, group *
            group_dims)``.
            """
            num_points = kernel_h * kernel_w - (1 if drop_center else 0)
            if method == AttnMethod.SOFTMAX:
                offset_mask = softmax_weights(offset_mask, groups, num_points)
            operator = getattr(getattr(torch.ops, OP_NAMESPACE), f"{op_prefix}_forward")
            return operator(
                value,
                offset_mask,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                dilation_h,
                dilation_w,
                groups,
                group_dims,
                spread,
                drop_center,
            )

    Backend.__doc__ = f"Dispatch adapter for the Triton kernel: {summary}."
    Backend.__qualname__ = "Backend"

    return AttnFamily(backend=Backend)
