r"""Classic deformable convolution (DeformConv2d) dispatch."""

from __future__ import annotations

import abc
import dataclasses
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor

from ..core.sanitize import check_conv_call
from ._dispatch import Backend, Registry
from ._public_ops import define_public_op

__all__ = [
    "ConvBackend",
    "ConvCall",
    "ConvContext",
    "conv",
    "resolve_conv_call",
    "select_backend",
]


@dataclasses.dataclass(frozen=True)
class ConvContext:
    """What backend selection reads from a convolution call: device and dtype.

    Both are concrete under tracing, so a compiled graph guards on no extent.
    """

    device: torch.device
    dtype: torch.dtype


@runtime_checkable
class ConvBackend(Protocol):
    """The interface every classic deformable convolution backend implements."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """The unique name of the backend."""
        ...

    @abc.abstractmethod
    def supports(self, context: ConvContext) -> bool:
        """Whether this backend can handle the given context."""
        ...

    @abc.abstractmethod
    def forward(  # noqa: PLR0913
        self,
        value: Tensor,
        offset: Tensor,
        mask: Tensor | None,
        weight: Tensor,
        bias: Tensor | None,
        stride_h: int,
        stride_w: int,
        pad_h: int,
        pad_w: int,
        dilation_h: int,
        dilation_w: int,
        groups: int,
        offset_groups: int,
    ) -> Tensor:
        """Run the forward pass, differentiably, on normalized geometry.

        Operands and geometry are as in :func:`deformops.registry.conv`, split
        per axis by :func:`resolve_conv_call`.
        """
        ...


@dataclasses.dataclass(frozen=True)
class ConvCall:
    """The normalized geometry of one call, from :func:`resolve_conv_call`."""

    stride_h: int
    stride_w: int
    pad_h: int
    pad_w: int
    dilation_h: int
    dilation_w: int
    groups: int
    offset_groups: int


def _pair(name: str, value: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    """Expand an ``int`` or a two-element sequence to ``(h, w)``."""
    if isinstance(value, int):
        return value, value
    items = tuple(value)
    if len(items) != 2:  # noqa: PLR2004
        msg = f"{name} must be an int or a pair of ints, got {value!r}"
        raise ValueError(msg)
    return int(items[0]), int(items[1])


def resolve_conv_call(  # noqa: PLR0913
    value: Tensor,
    offset: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    mask: Tensor | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    *,
    groups: int = 1,
    offset_groups: int = 1,
) -> ConvCall:
    """Normalize and validate one convolution call; every backend calls this.

    Parameters
    ----------
    value
        Input feature map, channels-last, ``(B, H, W, C_in)``.
    offset
        Per-point ``(dy, dx)`` offsets, channels-last,
        ``(B, H_out, W_out, offset_groups, K, 2)``.
    weight
        Convolution weight, ``(C_out, C_in // groups, kH, kW)``.
    bias
        Optional ``(C_out,)`` bias.
    mask
        Per-point modulation ``(B, H_out, W_out, offset_groups, K)`` for the
        modulated (v2) form, or ``None`` for v1.
    stride, padding, dilation
        Convolution geometry, as an ``int`` or an ``(h, w)`` pair.
    groups
        Convolution groups; ``C_in`` and ``C_out`` must both be divisible.
    offset_groups
        Offset groups; ``C_in`` must be divisible.

    Returns
    -------
    The normalized call.

    Raises
    ------
    ValueError
        If the operands disagree with each other or with the geometry.
    """
    stride_h, stride_w = _pair("stride", stride)
    pad_h, pad_w = _pair("padding", padding)
    dilation_h, dilation_w = _pair("dilation", dilation)
    check_conv_call(
        value,
        offset,
        mask,
        weight,
        bias,
        groups,
        offset_groups,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        dilation_h=dilation_h,
        dilation_w=dilation_w,
    )
    return ConvCall(
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        dilation_h=dilation_h,
        dilation_w=dilation_w,
        groups=int(groups),
        offset_groups=int(offset_groups),
    )


def select_backend(context: ConvContext) -> list[Backend]:
    """The backends ``backend=None`` tries: eager for CPU and float64, else Triton."""
    if context.device.type != "cuda" or context.dtype == torch.float64:
        return [Backend.EAGER]
    return [Backend.TRITON]


def _context_of(
    value: Tensor,
    _offset: Tensor,
    _weight: Tensor,
    *_args: Any,
    **_options: Any,
) -> ConvContext:
    """Build the convolution context from the functional call arguments."""
    return ConvContext(device=value.device, dtype=value.dtype)


def _conv_call(  # noqa: PLR0913
    value: Tensor,
    offset: Tensor,
    weight: Tensor,
    /,
    bias: Tensor | None = None,
    mask: Tensor | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    *,
    groups: int = 1,
    offset_groups: int = 1,
    backend: Backend | None = None,
) -> Tensor:
    r"""Classic deformable convolution, dispatched to a backend.

    ``torchvision.ops.deform_conv2d`` over channels-last operands. Each kernel point
    is displaced by an offset and sampled bilinearly. A mask modulates the samples
    (v2; v1 has none), and the result is contracted against the weight. ``K`` is
    ``kH * kW``. ``value``, ``offset`` and ``weight`` are positional-only.

    Parameters
    ----------
    value
        The input feature map, channels-last, shape ``(B, H, W, C_in)``.
    offset
        ``(dy, dx)`` offsets in input pixels, channels-last,
        ``(B, H_out, W_out, offset_groups, K, 2)``, points row-major; the
        floating dtype may differ from ``value``.
    weight
        The convolution weight, shape ``(C_out, C_in // groups, kH, kW)``,
        with the dtype of ``value``.
    bias
        Per-output-channel bias, shape ``(C_out,)``, or ``None``.
    mask
        Per-point modulation, channels-last, shape
        ``(B, H_out, W_out, offset_groups, K)``, or ``None`` for v1.
    stride, padding, dilation
        The convolution geometry, each as an ``int`` or an ``(h, w)`` pair.
    groups
        Convolution groups; ``C_in`` and ``C_out`` must be divisible by it.
    offset_groups
        Offset groups over the input channels; ``C_in`` must be divisible by it.
    backend
        A :class:`~deformops.registry.Backend` member, or ``None`` to run the
        public operator ``torch.ops.deformops.conv``, which selects per
        :func:`deformops.registry.conv.select_backend`. ``Backend.CUDA`` runs
        the Triton kernel.

    Returns
    -------
    The convolution output, channels-last, shape ``(B, H_out, W_out, C_out)``,
    in the dtype of ``value``.

    Raises
    ------
    TypeError
        If ``backend`` is a raw string or a keyword is unknown.
    ValueError
        If the operands disagree with each other or with the geometry.
    NoBackendError
        If no requested or candidate backend can run the call; the message
        gives each reason.

    Notes
    -----
    Triton takes float16, bfloat16 and float32; float64 and non-CUDA tensors
    run eager.
    """
    raise NotImplementedError


conv: Registry[Any] = Registry(
    "conv", context_of=_context_of, select=select_backend, describe=_conv_call
)

# Public operator.

#: The public operator's argument list, with the geometry as integers per axis.
_PUBLIC_SCHEMA = (
    "(Tensor value, Tensor offset, Tensor? mask, Tensor weight, Tensor? bias, "
    "int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, "
    "int dilation_w, int groups, int offset_groups) -> Tensor"
)


def _public_kernel(  # noqa: PLR0913
    value: Tensor,
    offset: Tensor,
    mask: Tensor | None,
    weight: Tensor,
    bias: Tensor | None,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dilation_h: int,
    dilation_w: int,
    groups: int,
    offset_groups: int,
) -> Tensor:
    """The kernel of ``deformops::conv``, reordering ``mask`` after ``weight``."""
    return conv.run_selected(
        value,
        offset,
        weight,
        bias,
        mask,
        (stride_h, stride_w),
        (pad_h, pad_w),
        (dilation_h, dilation_w),
        groups=groups,
        offset_groups=offset_groups,
    )


define_public_op(
    "conv",
    _PUBLIC_SCHEMA,
    _public_kernel,
    shared={3: "a convolution weight", 4: "a convolution bias"},
)


def _route_selected(  # noqa: PLR0913
    value: Tensor,
    offset: Tensor,
    weight: Tensor,
    /,
    bias: Tensor | None = None,
    mask: Tensor | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    *,
    groups: int = 1,
    offset_groups: int = 1,
) -> Tensor:
    """Run a ``backend=None`` call through ``deformops::conv``; checks run inside."""
    stride_h, stride_w = _pair("stride", stride)
    pad_h, pad_w = _pair("padding", padding)
    dilation_h, dilation_w = _pair("dilation", dilation)
    return torch.ops.deformops.conv(
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
        int(groups),
        int(offset_groups),
    )


conv.route_selected_calls(_route_selected)
