r"""Multi-scale deformable attention (MSDA) dispatch: context, protocol, policy.

``msda`` is the registry every backend registers into and every caller uses.
"""

from __future__ import annotations

import abc
import dataclasses
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor

from ..core.runtime import Method, is_training_from, resolve_method, sm_version_for
from ..core.sanitize import check_msda_call
from ._dispatch import Backend, Registry
from ._public_ops import define_public_op

__all__ = [
    "MSDABackend",
    "MSDAContext",
    "MSDAMethod",
    "apply_msda_method",
    "msda_method_to_aggregate",
    "msda",
    "resolve_msda_attn",
    "resolve_method",
    "select_backend",
]

MSDAMethod = Method

#: Dtypes that train with the float32 accumulator before SM 9.0.
_REDUCED_PRECISION_DTYPES: frozenset[torch.dtype] = frozenset(
    {torch.bfloat16, torch.float16}
)


@dataclasses.dataclass(frozen=True)
class MSDAContext:
    """The context of one MSDA call: the hashable memoization key for selection.

    ``num_queries`` is excluded from equality, so varying resolutions do not
    grow the memo; a symbolic query count is recorded as ``0``. See
    ``docs/dispatch.md``.
    """

    device: torch.device
    num_levels: int
    num_heads: int
    num_points: int
    dtype: torch.dtype
    sm_version: tuple[int, int]
    is_training: bool
    num_queries: int = dataclasses.field(compare=False)
    method: Method = Method.LINEAR
    #: Channels per head; ``0`` when unknown, which no kernel constraint rejects.
    head_dim: int = 0

    @classmethod
    def from_call(
        cls,
        value: Tensor,
        spatial_shapes: Tensor,
        sampling_locations: Tensor,
        attention_weights: Tensor,
        method: Method,
    ) -> MSDAContext:
        """Build a context from the operator-call tensors.

        The tensors may be fake or functional, with symbolic sizes.
        """
        is_training = is_training_from(value, sampling_locations, attention_weights)
        # Symbolic under tracing, although torch annotates sizes as int.
        queries: int | torch.SymInt = sampling_locations.shape[1]
        return cls(
            device=value.device,
            num_levels=int(spatial_shapes.shape[0]),
            num_heads=int(value.shape[2]),
            num_points=int(sampling_locations.shape[4]),
            dtype=value.dtype,
            sm_version=sm_version_for(value.device),
            is_training=is_training,
            num_queries=0 if isinstance(queries, torch.SymInt) else queries,
            method=method,
            head_dim=int(value.shape[3]),
        )


@runtime_checkable
class MSDABackend(Protocol):
    """The interface that all backend adapters must implement."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """The unique name of the backend."""
        ...

    @abc.abstractmethod
    def supports(self, context: MSDAContext) -> bool:
        """Returns True if the backend can handle the given context."""
        ...

    @abc.abstractmethod
    def forward(
        self,
        value: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        sampling_locations: Tensor,
        attention_weights: Tensor,
        chunk_step: int,
        method: Method,
    ) -> Tensor:
        """Run the forward pass, differentiably, without operand checks.

        Shape symbols are as in :func:`deformops.registry.msda`.

        Parameters
        ----------
        value
            The flattened feature pyramid, shape ``(B, S, G, D)``.
        spatial_shapes
            One ``(height, width)`` row per level, shape ``(L, 2)``, integer.
        level_start_index
            Where each level starts on the ``S`` axis, shape ``(L,)``, integer.
        sampling_locations
            Normalized ``(x, y)`` sampling locations in ``[0, 1]``, shape
            ``(B, Q, G, L, K, 2)``; the dtype may differ from ``value``.
        attention_weights
            Raw attention weights, shape ``(B, Q, G, L * K)`` or
            ``(B, Q, G, L, K)``, with the dtype of ``value``.
        chunk_step
            The batch chunk size, for a backend that chunks; others ignore it.
        method
            :attr:`Method.SOFTMAX` normalizes ``attention_weights`` over the
            ``L * K`` points first.

        Returns
        -------
        The attended features, shape ``(B, Q, G * D)``, in the dtype of
        ``value``.
        """
        ...


def msda_method_to_aggregate(method: str | Method) -> int:
    """Convert a multi-scale method to the integer code the CUDA kernels use."""
    if not isinstance(method, Method):
        method = Method(method)
    match method:
        case Method.SOFTMAX:
            return 1
        case _:
            return 0


def apply_msda_method(attention_weights: Tensor, method: str | Method) -> Tensor:
    """Apply the normalization method to the raw attention weights."""
    if not isinstance(method, Method):
        method = Method(method)
    match method:
        case Method.LINEAR:
            return attention_weights
        case Method.SOFTMAX:
            _flat_layout, _split_layout = 4, 5
            if attention_weights.ndim == _split_layout:
                return torch.nn.functional.softmax(
                    attention_weights.flatten(-2), -1
                ).unflatten(-1, attention_weights.shape[-2:])
            if attention_weights.ndim != _flat_layout:
                msg = (
                    f"attention weights must be (B, Q, G, L * K) or (B, Q, G, L, K); "
                    f"got {attention_weights.ndim} dimensions."
                )
                raise ValueError(msg)
            return torch.nn.functional.softmax(attention_weights, -1)
        case _:
            msg = f"Unsupported attention mode {method!r}!"
            raise NotImplementedError(msg)


def resolve_msda_attn(
    attention_weights: Tensor, method: str | Method
) -> tuple[Tensor, int]:
    """Compute ``(attn_out, aggregate_code)`` for a multi-scale backend."""
    aggregate = msda_method_to_aggregate(method)
    if aggregate == 1:
        return attention_weights, 1
    return apply_msda_method(attention_weights, method), 0


def select_backend(
    context: MSDAContext,
) -> list[Backend | tuple[Backend, dict[str, Any]]]:
    """The backends ``backend=None`` tries for an MSDA call, in order.

    CPU tensors and float64 run eager. CUDA tensors try Triton then CUDA, never
    eager; half-precision training before SM 9.0 uses the float32 accumulator.
    See ``docs/dispatch.md`` and ``docs/precision.md``.
    """
    if context.device.type != "cuda" or context.dtype == torch.float64:
        return [Backend.EAGER]
    fp32acc = (
        context.sm_version < (9, 0)
        and context.is_training
        and context.dtype in _REDUCED_PRECISION_DTYPES
    )
    return [(Backend.TRITON, {"fp32acc": fp32acc}), Backend.CUDA]


#: Keywords a backend consumes besides ``method`` and ``softmax``: the batch
#: chunk and launch geometry of the CUDA kernel, and the Triton accumulator.
_CALL_OPTIONS: frozenset[str] = frozenset(
    {"chunk_step", "num_points", "channels_per_thread", "threads_per_block", "fp32acc"}
)


def _checked_arguments(
    level_start_index: Tensor | None,
    sampling_locations: Tensor | None,
    attention_weights: Tensor | None,
    options: dict[str, Any],
) -> tuple[Tensor, Tensor, Tensor]:
    """Require the three trailing operands and reject unknown keywords."""
    if (
        level_start_index is None
        or sampling_locations is None
        or attention_weights is None
    ):
        msg = (
            "msda takes (values, spatial_shapes, level_start_index, "
            "sampling_locations, attention_weights) positionally"
        )
        raise TypeError(msg)
    unknown = set(options) - _CALL_OPTIONS
    if unknown:
        msg = f"msda got unexpected keyword argument(s): {', '.join(sorted(unknown))}"
        raise TypeError(msg)
    return level_start_index, sampling_locations, attention_weights


def _context_of(  # noqa: PLR0913
    value: Tensor,
    spatial_shapes: Tensor,
    _start_index: Tensor | None = None,
    sampling_locations: Tensor | None = None,
    attention_weights: Tensor | None = None,
    *,
    method: str | Method | None = None,
    softmax: bool = False,
    **_options: Any,
) -> MSDAContext:
    """Build the MSDA context from the functional call arguments."""
    _start_index, sampling_locations, attention_weights = _checked_arguments(
        _start_index, sampling_locations, attention_weights, _options
    )
    # Before reading any extent, so every backend raises the same ValueError.
    check_msda_call(
        value, spatial_shapes, _start_index, sampling_locations, attention_weights
    )
    normalized = resolve_method(method, softmax=softmax)
    return MSDAContext.from_call(
        value,
        spatial_shapes,
        sampling_locations,
        attention_weights,
        normalized,
    )


def _msda_call(  # noqa: PLR0913
    value: Tensor,
    spatial_shapes: Tensor,
    level_start_index: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
    /,
    *,
    method: str | Method | None = None,
    softmax: bool = False,
    chunk_step: int = 128,
    backend: Backend | None = None,
) -> Tensor:
    r"""Multi-scale deformable attention, dispatched to a backend.

    Samples every level of a flattened feature pyramid bilinearly at per-query,
    per-head locations and returns the weighted sum. ``B`` batch, ``S`` spatial
    positions over all levels, ``G`` heads, ``D`` channels per head, ``Q``
    queries, ``L`` levels, ``K`` points per level. The five tensors are
    positional-only and may have any memory layout.

    Parameters
    ----------
    value
        The flattened feature pyramid, shape ``(B, S, G, D)``.
    spatial_shapes
        One ``(height, width)`` row per level, shape ``(L, 2)``, integer.
    level_start_index
        Where each level starts on the ``S`` axis, shape ``(L,)``, integer.
    sampling_locations
        Normalized ``(x, y)`` in ``[0, 1]``, shape ``(B, Q, G, L, K, 2)``; the
        floating dtype may differ from ``value``.
    attention_weights
        Per-point weights, shape ``(B, Q, G, L * K)`` or ``(B, Q, G, L, K)``,
        with the dtype of ``value``.
    method
        ``"linear"`` (default) uses the weights as given; ``"softmax"``
        normalizes them over the ``L * K`` points inside the kernel. ``None``
        defers to ``softmax``.
    softmax
        ``True`` means ``method="softmax"`` when ``method`` is ``None``.
    chunk_step
        The batch chunk size of an explicit ``Backend.CUDA`` call, ignored
        otherwise. A batch it does not divide runs in chunks of its largest
        divisor.
    backend
        A :class:`~deformops.registry.Backend` member, or ``None`` to run the
        public operator ``torch.ops.deformops.msda`` (or ``msda_softmax``),
        which selects per :func:`deformops.registry.msda.select_backend`.

    Returns
    -------
    The attended features, shape ``(B, Q, G * D)``, in the dtype of
    ``value``.

    Raises
    ------
    TypeError
        If ``backend`` is a raw string, an operand is missing, or a keyword is
        unknown.
    ValueError
        If the operands disagree in rank, size, dtype or device.
    NoBackendError
        If no requested or candidate backend can run the call; the message
        gives each reason.
    KernelUnavailableError
        If the CUDA kernel library cannot be downloaded or compiled.

    Notes
    -----
    CUDA and Triton take float16, bfloat16 and float32; float64 runs eager, or
    on a locally compiled CUDA library (``DEFORMOPS_COMPILE=always``).

    The CUDA backend needs ``D`` divisible by 8, 1 to 5 levels, 4 or 8 points
    per level, and at most 1024 threads per block (``G * D / 8`` in the
    forward, the backward's own stride when training).

    Backend-specific keywords (Triton ``fp32acc``; CUDA ``num_points``,
    ``channels_per_thread``, ``threads_per_block``) are ignored by other
    backends. Passing one with ``backend=None`` negotiates in Python, so
    tracing records the backend's operator; see ``docs/dispatch.md``.
    """
    raise NotImplementedError


msda: Registry[Any] = Registry(
    "msda", context_of=_context_of, select=select_backend, describe=_msda_call
)

# Public operators.

#: The public operators' argument list, shared by both aggregation methods.
_PUBLIC_SCHEMA = (
    "(Tensor value, Tensor spatial_shapes, Tensor level_start_index, "
    "Tensor sampling_locations, Tensor attention_weights) -> Tensor"
)

#: Where the level table sits in the public operators' argument list.
_LEVEL_TABLE = "the spatial shapes and level start offsets"

#: Keywords that override a policy or backend choice, so a call passing one
#: negotiates in Python. Not ``chunk_step``, which the layers pass every call.
_OVERRIDE_OPTIONS: frozenset[str] = _CALL_OPTIONS - {"chunk_step"}


def _public_kernel(method: Method) -> Any:
    """The kernel of the public operator for one aggregation method."""

    def kernel(
        value: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        sampling_locations: Tensor,
        attention_weights: Tensor,
    ) -> Tensor:
        return msda.run_selected(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            method=method,
        )

    return kernel


define_public_op(
    "msda",
    _PUBLIC_SCHEMA,
    _public_kernel(Method.LINEAR),
    shared={1: _LEVEL_TABLE, 2: _LEVEL_TABLE},
)
define_public_op(
    "msda_softmax",
    _PUBLIC_SCHEMA,
    _public_kernel(Method.SOFTMAX),
    shared={1: _LEVEL_TABLE, 2: _LEVEL_TABLE},
)


def _route_selected(  # noqa: PLR0913
    value: Tensor,
    spatial_shapes: Tensor,
    level_start_index: Tensor | None = None,
    sampling_locations: Tensor | None = None,
    attention_weights: Tensor | None = None,
    *,
    method: str | Method | None = None,
    softmax: bool = False,
    **options: Any,
) -> Tensor:
    """Run a ``backend=None`` call through ``deformops::msda[_softmax]``.

    ``chunk_step`` does not reach the backend on this path.
    """
    # Not ``options.keys() & ...``, which Dynamo cannot trace on every PyTorch.
    if any(name in options for name in _OVERRIDE_OPTIONS):
        return msda.run_selected(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            method=method,
            softmax=softmax,
            **options,
        )
    level_start_index, sampling_locations, attention_weights = _checked_arguments(
        level_start_index, sampling_locations, attention_weights, options
    )
    softmax_op = resolve_method(method, softmax=softmax) is Method.SOFTMAX
    op = torch.ops.deformops.msda_softmax if softmax_op else torch.ops.deformops.msda
    return op(
        value, spatial_shapes, level_start_index, sampling_locations, attention_weights
    )


msda.route_selected_calls(_route_selected)
