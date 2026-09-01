r"""Single-scale deformable sampling (DeformAttn) dispatch."""

from __future__ import annotations

import abc
import dataclasses
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor

from ..core.runtime import Method, is_training_from, resolve_method, sm_version_for
from ..core.sanitize import check_attn_call
from ._dispatch import Backend, Registry
from ._public_ops import define_public_op

__all__ = [
    "AttnBackend",
    "AttnCall",
    "AttnContext",
    "AttnMethod",
    "attn",
    "resolve_attn_call",
    "select_backend",
]

AttnMethod = Method

#: Dtypes that train with the float32 accumulator before SM 9.0.
_REDUCED_PRECISION_DTYPES: frozenset[torch.dtype] = frozenset(
    {torch.bfloat16, torch.float16}
)

#: Keyword options a backend reads itself; they are not call geometry.
_BACKEND_OPTIONS: frozenset[str] = frozenset({"fp32acc"})


@dataclasses.dataclass(frozen=True)
class AttnContext:
    """The context of a single-scale sampling call, used as the memoization key.

    It reads no tensor extent, so a compiled graph does not specialize on a
    size to build it.
    """

    device: torch.device
    dtype: torch.dtype
    method: Method = Method.LINEAR
    sm_version: tuple[int, int] = (0, 0)
    is_training: bool = False


@dataclasses.dataclass(frozen=True)
class AttnCall:
    """The normalized geometry of one call, from :func:`resolve_attn_call`."""

    kernel_h: int
    kernel_w: int
    stride_h: int
    stride_w: int
    pad_h: int
    pad_w: int
    dilation_h: int
    dilation_w: int
    groups: int
    group_dims: int
    spread: float
    method: Method
    drop_center: bool


def _pair(name: str, value: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    """Expand an ``int`` or a two-element sequence to ``(h, w)``."""
    if isinstance(value, int):
        return value, value
    items = tuple(value)
    if len(items) != 2:  # noqa: PLR2004
        msg = f"{name} must be an int or a pair of ints, got {value!r}"
        raise ValueError(msg)
    return int(items[0]), int(items[1])


def _override_pair(
    pair: tuple[int, int], options: dict[str, Any], height: str, width: str
) -> tuple[int, int]:
    """Apply the separate ``*_h``/``*_w`` spellings, which must come together."""
    has_h, has_w = height in options, width in options
    if has_h != has_w:
        msg = f"{height} and {width} must be given together"
        raise TypeError(msg)
    if has_h:
        return int(options.pop(height)), int(options.pop(width))
    return pair


def resolve_attn_call(  # noqa: PLR0913
    value: Tensor,
    offset_mask: Tensor,
    kernel_size: int | tuple[int, int] = (3, 3),
    stride: int | tuple[int, int] = (1, 1),
    padding: int | tuple[int, int] = (1, 1),
    dilation: int | tuple[int, int] = (1, 1),
    groups: int = 4,
    group_dims: int = 16,
    spread: float = 1.0,
    method: str | Method | None = None,
    softmax: bool = False,
    drop_center: bool = False,
    **options: Any,
) -> AttnCall:
    """Normalize and validate one single-scale sampling call.

    Every backend calls this, so spellings, defaults and checks cannot drift.

    Parameters
    ----------
    value
        The sampled feature map, channels-last, ``(B, H, W, groups * group_dims)``.
    offset_mask
        Offsets and weights, channels-last, ``(B, H_out, W_out, C)`` with ``C``
        at least ``groups * K * 3``; extra channels are ignored.
    kernel_size, stride, padding, dilation
        Sampling geometry.
    groups
        Number of sampling groups.
    group_dims
        Channels per group.
    spread
        Scale applied to the predicted offsets.
    method
        Aggregation method; ``None`` defers to ``softmax``.
    softmax
        Selects :attr:`Method.SOFTMAX` when ``method`` is not given.
    drop_center
        Drop the center kernel point.
    **options
        ``kernel_h``/``kernel_w``, ``stride_h``/``stride_w``, ``pad_h``/``pad_w``
        and ``dilation_h``/``dilation_w`` (taking precedence),
        and ``fp32acc``. Any other keyword raises.

    Returns
    -------
    The normalized call.

    Raises
    ------
    TypeError
        On an unknown keyword, or a ``*_h`` keyword without its ``*_w`` pair.
    ValueError
        If the operands disagree with the geometry.
    """
    geometry = dict(options)
    kernel_h, kernel_w = _override_pair(
        _pair("kernel_size", kernel_size), geometry, "kernel_h", "kernel_w"
    )
    stride_h, stride_w = _override_pair(
        _pair("stride", stride), geometry, "stride_h", "stride_w"
    )
    pad_h, pad_w = _override_pair(_pair("padding", padding), geometry, "pad_h", "pad_w")
    dilation_h, dilation_w = _override_pair(
        _pair("dilation", dilation), geometry, "dilation_h", "dilation_w"
    )
    unknown = set(geometry) - _BACKEND_OPTIONS
    if unknown:
        msg = f"unexpected keyword argument(s): {', '.join(sorted(unknown))}"
        raise TypeError(msg)

    check_attn_call(
        value,
        offset_mask,
        kernel_h,
        kernel_w,
        groups,
        group_dims,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        dilation_h=dilation_h,
        dilation_w=dilation_w,
        drop_center=drop_center,
    )
    return AttnCall(
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        dilation_h=dilation_h,
        dilation_w=dilation_w,
        groups=groups,
        group_dims=group_dims,
        spread=float(spread),
        method=resolve_method(method, softmax=softmax),
        drop_center=bool(drop_center),
    )


@runtime_checkable
class AttnBackend(Protocol):
    """The interface that all single-scale sampling backends implement."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """The unique name of the backend."""
        ...

    @abc.abstractmethod
    def supports(self, context: AttnContext) -> bool:
        """Returns True if the backend can handle the given context."""
        ...

    @abc.abstractmethod
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
        method: Method,
        drop_center: bool,
    ) -> Tensor:
        """Run the forward pass, differentiably, on normalized geometry.

        Operands and geometry are as in :func:`deformops.registry.attn`,
        normalized by :func:`resolve_attn_call`.
        """
        ...


def select_backend(
    context: AttnContext,
) -> list[Backend | tuple[Backend, dict[str, Any]]]:
    """The backends ``backend=None`` tries for a single-scale call, in order.

    CPU tensors and float64 run eager; CUDA tensors run Triton only, with the
    accumulator chosen as in :func:`deformops.registry.msda.select_backend`.
    """
    if context.device.type != "cuda" or context.dtype == torch.float64:
        return [Backend.EAGER]
    fp32acc = (
        context.sm_version < (9, 0)
        and context.is_training
        and context.dtype in _REDUCED_PRECISION_DTYPES
    )
    return [(Backend.TRITON, {"fp32acc": fp32acc})]


def _context_of(
    value: Tensor,
    _offset_mask: Tensor,
    *_args: Any,
    method: str | Method | None = None,
    softmax: bool = False,
    **_options: Any,
) -> AttnContext:
    """Build the sampling context from the functional call arguments."""
    return AttnContext(
        device=value.device,
        dtype=value.dtype,
        method=resolve_method(method, softmax=softmax),
        sm_version=sm_version_for(value.device),
        is_training=is_training_from(value, _offset_mask),
    )


def _attn_call(  # noqa: PLR0913
    value: Tensor,
    offset_mask: Tensor,
    /,
    kernel_size: int | tuple[int, int] = (3, 3),
    stride: int | tuple[int, int] = (1, 1),
    padding: int | tuple[int, int] = (1, 1),
    dilation: int | tuple[int, int] = (1, 1),
    groups: int = 4,
    group_dims: int = 16,
    spread: float = 1.0,
    method: str | Method | None = None,
    softmax: bool = False,
    drop_center: bool = False,
    *,
    backend: Backend | None = None,
    **options: Any,
) -> Tensor:
    r"""Single-scale deformable sampling, dispatched to a backend.

    For every output location, samples ``value`` bilinearly at a kernel grid
    displaced by learned offsets. The samples are summed per group, weighted by a
    learned modulation. ``K`` is ``kernel_h * kernel_w``, less one with
    ``drop_center``; ``H_out`` and ``W_out`` follow the convolution geometry.
    ``value`` and ``offset_mask`` are positional-only.

    Parameters
    ----------
    value
        The sampled feature map, channels-last, shape
        ``(B, H, W, groups * group_dims)``.
    offset_mask
        Channels-last ``(B, H_out, W_out, C)``, ``C`` at least ``groups * K * 3``
        (extra channels ignored), holding ``(dx, dy, weight)`` per point as
        ``(groups, K, 3)``. Offsets are in input pixels; point order is in
        ``docs/conventions.md``.
    kernel_size, stride, padding, dilation
        The sampling geometry, each an ``int`` or an ``(h, w)`` pair.
    groups
        Channel groups, each with its own offsets and weights.
    group_dims
        Channels per group.
    spread
        Scale applied to each displaced grid point about the kernel center.
    method
        ``"linear"`` (default) uses the weights as given; ``"softmax"``
        normalizes them over each group's ``K`` points. ``None`` defers to
        ``softmax``.
    softmax
        ``True`` means ``method="softmax"`` when ``method`` is ``None``.
    drop_center
        Drop the center point, ``kernel // 2`` on each axis.
    backend
        A :class:`~deformops.registry.Backend` member, or ``None`` to run the
        public operator ``torch.ops.deformops.attn`` (or ``attn_softmax``),
        which selects per :func:`deformops.registry.attn.select_backend`.
        ``Backend.CUDA`` runs the Triton kernel.
    **options
        ``kernel_h``/``kernel_w``, ``stride_h``/``stride_w``, ``pad_h``/``pad_w``
        and ``dilation_h``/``dilation_w`` (each pair together, taking
        precedence), and ``fp32acc``, which
        accumulates the input gradient in float32. The policy sets it per
        architecture; an explicit Triton or CUDA call defaults to ``True``.

    Returns
    -------
    The aggregated samples, channels-last, shape
    ``(B, H_out, W_out, groups * group_dims)``, in the dtype of ``value``.

    Raises
    ------
    TypeError
        If ``backend`` is a raw string, a keyword is unknown, or a ``*_h``
        keyword lacks its ``*_w`` pair.
    ValueError
        If the operands disagree with each other or with the geometry.
    NoBackendError
        If no requested or candidate backend can run the call; the message
        gives each reason.

    Notes
    -----
    Triton takes float16, bfloat16 and float32; float64 and non-CUDA tensors
    run eager. Passing ``fp32acc`` with ``backend=None`` negotiates in
    Python, so tracing records the backend's operator.
    """
    raise NotImplementedError


attn: Registry[Any] = Registry(
    "attn", context_of=_context_of, select=select_backend, describe=_attn_call
)

# Public operators.

#: The public operators' argument list, shared by both aggregation methods.
_PUBLIC_SCHEMA = (
    "(Tensor value, Tensor offset_mask, int kernel_h, int kernel_w, "
    "int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, "
    "int dilation_w, int groups, int group_dims, float spread, "
    "bool drop_center) -> Tensor"
)


def _public_kernel(method: Method) -> Any:
    """The kernel of the public operator for one aggregation method."""

    def kernel(  # noqa: PLR0913
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
        return attn.run_selected(
            value,
            offset_mask,
            kernel_size=(kernel_h, kernel_w),
            stride=(stride_h, stride_w),
            padding=(pad_h, pad_w),
            dilation=(dilation_h, dilation_w),
            groups=groups,
            group_dims=group_dims,
            spread=spread,
            method=method,
            drop_center=drop_center,
        )

    return kernel


define_public_op("attn", _PUBLIC_SCHEMA, _public_kernel(Method.LINEAR))
define_public_op("attn_softmax", _PUBLIC_SCHEMA, _public_kernel(Method.SOFTMAX))


def _route_selected(
    value: Tensor, offset_mask: Tensor, *args: Any, **options: Any
) -> Tensor:
    """Run a ``backend=None`` call through ``deformops::attn[_softmax]``.

    A call passing a backend option negotiates in Python instead.
    """
    # Not ``options.keys() & ...``, which Dynamo cannot trace on every PyTorch.
    if any(name in options for name in _BACKEND_OPTIONS):
        return attn.run_selected(value, offset_mask, *args, **options)
    call = resolve_attn_call(value, offset_mask, *args, **options)
    softmax_op = call.method is Method.SOFTMAX
    op = torch.ops.deformops.attn_softmax if softmax_op else torch.ops.deformops.attn
    return op(
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
        call.drop_center,
    )


attn.route_selected_calls(_route_selected)
