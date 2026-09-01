r"""Shared machinery behind every CUDA multi-scale deformable attention module.

A configuration differs only in its operator prefix, backend name and whether
its backward upcasts to fp32; :func:`install_msda_family` builds everything
else. The hooks implement what ``register_op_pair`` in ``core.build`` expects.
See ``docs/kernels.md``.
"""

import typing
from collections.abc import Callable
from typing import Any, override

import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx, once_differentiable

from ...core.build import lazy_extension, load_op, op_library, register_op_pair
from ...core.runtime import sm_version_for
from ...core.tuning import backward_thread_layout
from ...registry import (
    MSDABackend,
    MSDAContext,
    MSDAMethod,
    resolve_msda_attn,
)
from ..reference.msda import reference as _base_reference
from ._schema import define_schemas

__all__ = [
    "MSDAFamily",
    "backward_meta",
    "forward_meta",
    "install_msda_family",
    "make_backward",
    "make_setup_context",
    "reference",
]

#: Positional inputs of every ``msda`` forward op, as in ``_schema.py``;
#: :func:`make_setup_context` checks the arity.
_FORWARD_INPUTS = 9

#: Positions of the saved tensors and their gradients; the rest pad with ``None``.
_SAVED_TENSORS = 5
_VALUE_INDEX = 0
_LOCATION_INDEX = 3
_ATTENTION_INDEX = 4

#: Points per level the kernels have template instantiations for.
_SUPPORTED_POINTS = (4, 8)

#: The oldest architecture whose 16-bit atomic accumulator compiles, per dtype.
_NATIVE_ATOMIC_MINIMUM_SM: dict[torch.dtype, tuple[int, int]] = {
    torch.float16: (7, 0),
    torch.bfloat16: (8, 0),
}

#: Level counts the kernel templates are instantiated for.
_SUPPORTED_LEVELS = range(1, 6)

#: The forward launch stride, which the forward kernels are instantiated for.
_FORWARD_CHANNELS_PER_THREAD = 8

#: The hardware maximum on threads in one block.
_MAX_THREADS_PER_BLOCK = 1024


def supports_context(context: MSDAContext) -> bool:
    """Whether the CUDA MSDA kernels can run a call with this context.

    Repeats the launch checks, so dispatch falls back instead of raising. A call
    needs a CUDA device, 4 or 8 points, 1 to 5 levels and a per-head channel count
    divisible by the forward stride. Its block must fit the thread limit in the
    forward and, when training, the backward.
    """
    heads, head_dim = context.num_heads, context.head_dim
    fits_templates = (
        context.device.type == "cuda"
        and context.num_points in _SUPPORTED_POINTS
        and context.num_levels in _SUPPORTED_LEVELS
    )
    if not fits_templates or head_dim == 0:
        return fits_templates
    forward_threads = heads * head_dim // _FORWARD_CHANNELS_PER_THREAD
    fits_forward = (
        head_dim % _FORWARD_CHANNELS_PER_THREAD == 0
        and forward_threads <= _MAX_THREADS_PER_BLOCK
    )
    if not fits_forward or not context.is_training:
        return fits_forward
    backward_stride = backward_thread_layout(1, 1, heads, head_dim)[0]
    return heads * head_dim // backward_stride <= _MAX_THREADS_PER_BLOCK


# Reference


def reference(
    value: Tensor,
    spatial_shapes: Tensor | typing.Sequence[tuple[int, int]],
    sampling_locations: Tensor,
    attention_weights: Tensor,
    num_points: int = 8,
    softmax: bool = False,
) -> Tensor:
    """Run :func:`deformops.backends.reference.msda.reference` on flat weights.

    The correctness reference for every CUDA MSDA configuration.

    Parameters
    ----------
    value
        ``(B, S, G, D)`` flattened multi-level value tensor.
    spatial_shapes
        ``(L, 2)`` ``(height, width)`` per level.
    sampling_locations
        ``(B, Q, G, L, K, 2)`` normalized sampling locations in
        ``[0, 1]``.
    attention_weights
        ``(B, Q, G, L * K)`` attention weights.
    num_points
        Points per level, ``K``.
    softmax
        Normalize ``attention_weights`` over the last axis before use.

    Returns
    -------
    ``(B, Q, G * D)``.
    """
    if softmax:
        attention_weights = torch.nn.functional.softmax(attention_weights, -1)
    attention_weights = attention_weights.unflatten(
        -1, (len(spatial_shapes), num_points)
    )
    if not isinstance(spatial_shapes, Tensor):
        spatial_shapes = torch.tensor(
            spatial_shapes, dtype=torch.long, device=value.device
        )
    # An ignore, not cast(): softmax() types as partially unknown on some versions.
    return _base_reference(value, spatial_shapes, sampling_locations, attention_weights)  # type: ignore[arg-type]


# Autograd


def make_setup_context(
    op_prefix: str,
) -> Callable[[FunctionCtx, tuple[Any, ...], Tensor], None]:
    r"""Build the ``setup_context`` hook for one operator family.

    Saves the five input tensors, ``chunk_step``, ``K`` and a zero launch
    geometry, so the backward tunes its own layout.

    Parameters
    ----------
    op_prefix
        Names the module in an arity-mismatch error.

    Returns
    -------
    The ``setup_context(ctx, inputs, output)`` hook.
    """

    def setup_context(
        ctx: FunctionCtx,
        inputs: tuple[Any, ...],
        output: Tensor,  # noqa: ARG001
    ) -> None:
        if len(inputs) != _FORWARD_INPUTS:
            msg = (
                f"{op_prefix} setup_context received {len(inputs)} inputs, "
                f"expected {_FORWARD_INPUTS} (op schema drift?)"
            )
            raise RuntimeError(msg)
        cx: Any = ctx
        cx.chunk_step = inputs[5]
        cx.num_points = inputs[6]
        # Zero defers the layout to launch, so tracing never specializes on sizes.
        cx.channels_per_thread = 0
        cx.threads_per_block = 0
        cx.save_for_backward(*inputs[:_SAVED_TENSORS])

    return setup_context


def make_backward(
    backward_op: Callable[..., tuple[Tensor, Tensor, Tensor]],
    *,
    upcast_inputs: bool,
) -> Callable[..., tuple[Tensor | None, ...]]:
    r"""Build the backward closure around one registered backward op.

    Parameters
    ----------
    backward_op
        The registered backward op to call.
    upcast_inputs
        Hand the op fp32 copies of ``value``, ``sampling_locations``,
        ``attention_weights`` and ``grad_output``. Only the fp32-accumulator
        configuration upcasts; the others keep their dtype so their native
        16-bit atomic paths run.

    Returns
    -------
    The ``@once_differentiable`` backward. Its gradient tuple matches the
    forward inputs, ``None`` except for ``value``, ``sampling_locations`` and
    ``attention_weights``.
    """

    def cast(tensor: Tensor) -> Tensor:
        return tensor.float() if upcast_inputs else tensor

    def require_native_atomic(value: Tensor) -> None:
        # Below SM 7.0 (fp16) or 8.0 (bf16) the accumulator body is empty and the
        # value gradient would be zeros.
        if upcast_inputs:
            return
        minimum = _NATIVE_ATOMIC_MINIMUM_SM.get(value.dtype)
        if minimum is None:
            return
        sm = sm_version_for(value.device)
        if sm < minimum:
            msg = (
                f"This CUDA configuration accumulates {value.dtype} "
                f"gradients with native atomics, which need SM "
                f"{minimum[0]}.{minimum[1]} or newer; this device is SM "
                f"{sm[0]}.{sm[1]}. Use the default CUDA backend, which "
                f"accumulates in float32."
            )
            raise RuntimeError(msg)

    @once_differentiable
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> tuple[Tensor | None, ...]:
        """Run the backward op on the saved tensors and ``(B, Q, G * D)`` gradient."""
        cx: Any = ctx
        saved: tuple[Tensor, Tensor, Tensor, Tensor, Tensor] = cx.saved_tensors
        value, spatial_shapes, level_index, sampling_locations, attention_weights = (
            saved
        )
        require_native_atomic(value)

        with torch.autocast("cuda", enabled=False):
            grad_value, grad_locs, grad_attn = backward_op(
                cast(value),
                spatial_shapes,
                level_index,
                cast(sampling_locations),
                cast(attention_weights),
                cast(grad_output.contiguous()),
                cx.chunk_step,
                cx.num_points,
                cx.channels_per_thread,
                cx.threads_per_block,
            )
        gradients: list[Tensor | None] = [None] * _FORWARD_INPUTS
        gradients[_VALUE_INDEX] = grad_value.type_as(value)
        gradients[_LOCATION_INDEX] = grad_locs.type_as(sampling_locations)
        gradients[_ATTENTION_INDEX] = grad_attn.type_as(attention_weights)
        return tuple(gradients)

    return backward


# Fake tensors: one pair covers every configuration, which share one schema.


def forward_meta(
    value: Tensor,
    spatial_shapes: Tensor,  # noqa: ARG001
    level_start_index: Tensor,  # noqa: ARG001
    sampling_locations: Tensor,
    attention_weights: Tensor,  # noqa: ARG001
    chunk_step: int,  # noqa: ARG001
    K: int,  # noqa: ARG001, N803
    channels_per_thread: int,  # noqa: ARG001
    threads_per_block: int,  # noqa: ARG001
) -> Tensor:
    B, _, G, D = value.shape
    Q = sampling_locations.shape[1]
    return value.new_empty((B, Q, G * D))


def backward_meta(
    value: Tensor,
    spatial_shapes: Tensor,  # noqa: ARG001
    level_start_index: Tensor,  # noqa: ARG001
    sampling_locations: Tensor,
    attention_weights: Tensor,
    grad_output: Tensor,  # noqa: ARG001
    chunk_step: int,  # noqa: ARG001
    K: int,  # noqa: ARG001, N803
    channels_per_thread: int,  # noqa: ARG001
    threads_per_block: int,  # noqa: ARG001
) -> tuple[Tensor, Tensor, Tensor]:
    return (
        value.new_empty(value.shape),
        sampling_locations.new_empty(sampling_locations.shape),
        attention_weights.new_empty(attention_weights.shape),
    )


# Operator family


class MSDAFamily(typing.NamedTuple):
    """Everything one configuration module re-exports.

    ``load`` loads the kernel library once per process; its ``download`` and
    ``compile`` keywords override the environment for that first load.
    """

    forward: Callable[..., Tensor]
    backend: type[Any]
    load: Callable[..., Any]


def install_msda_family(
    *,
    op_prefix: str,
    backend_name: str,
    upcast_backward_inputs: bool,
    summary: str,
) -> MSDAFamily:
    r"""Register one configuration's operators and build its public surface.

    Runs at import. Schemas come first, because the ``.so`` registers against
    existing op names; everything but the library load is Python-level and eager.

    Parameters
    ----------
    op_prefix
        Operator prefix and extension name, a key of
        ``core.configs.EXTENSION_CONFIGS``, e.g. ``"msda_serialreduce"``.
    backend_name
        Backend name, e.g. ``"cuda_serialreduce"``.
    upcast_backward_inputs
        See :func:`make_backward`.
    summary
        One line for the adapter's docstring.

    Returns
    -------
    The functional entry point, the adapter class and the deferred load.
    """
    softmax_prefix = f"{op_prefix}_softmax"
    define_schemas()

    bind_extension = lazy_extension(op_prefix)
    forward_op_linear, backward_op_linear = load_op(op_prefix)
    forward_op_softmax, backward_op_softmax = load_op(softmax_prefix)

    backward_linear = make_backward(
        backward_op_linear, upcast_inputs=upcast_backward_inputs
    )
    backward_softmax = make_backward(
        backward_op_softmax, upcast_inputs=upcast_backward_inputs
    )
    register_op_pair(
        linear_op_library=op_library(op_prefix),
        softmax_op_library=op_library(softmax_prefix),
        setup_context=make_setup_context(op_prefix),
        backward_linear=backward_linear,
        backward_softmax=backward_softmax,
        forward_meta=forward_meta,
        backward_meta=backward_meta,
    )

    def forward(
        value: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        sampling_locations: Tensor,
        attention_weights: Tensor,
        chunk_step: int = 128,
        *,
        num_points: int,
        channels_per_thread: int = 0,
        threads_per_block: int = 0,
        softmax: bool = False,
    ) -> Tensor:
        r"""Multi-scale deformable sampling operator.

        Parameters
        ----------
        value
            ``(B, S, G, D)``.
        spatial_shapes
            ``(L, 2)``.
        level_start_index
            ``(L,)``.
        sampling_locations
            ``(B, Q, G, L, K, 2)``.
        attention_weights
            ``(B, Q, G, L * K)``.
        chunk_step
            Batch chunk size for the launch; also the fallback when the two
            launch-geometry arguments are not given explicitly.
        num_points
            Points per level, ``K``; selects the kernel template.
        channels_per_thread
            Explicit launch stride; derived from the shape when ``<= 0``.
        threads_per_block
            Explicit launch block size; derived from the shape when ``<= 0``.
        softmax
            Selects the in-kernel-softmax op over the linear one.

        Returns
        -------
        ``(B, Q, G * D)``.
        """
        if value.device.type != "cuda":
            msg = f"CUDA backend not available for {op_prefix}"
            raise RuntimeError(msg)

        # The kernels ignore strides and read the level tables as ``int64_t``;
        # the backward inherits these saved layouts.
        value = value.contiguous()
        spatial_shapes = spatial_shapes.to(torch.int64).contiguous()
        level_start_index = level_start_index.to(torch.int64).contiguous()
        sampling_locations = sampling_locations.contiguous()
        attention_weights = attention_weights.contiguous()

        num_query = sampling_locations.shape[1]
        batch = value.shape[0]
        if num_query == 0 or batch == 0:
            _, _, G, D = value.shape
            return value.new_zeros((batch, num_query, G * D))

        bind_extension()

        # Locations are float32 (float64 for float64 values) at any value dtype.
        location_dtype = (
            torch.float64 if value.dtype == torch.float64 else torch.float32
        )
        op = forward_op_softmax if softmax else forward_op_linear
        with torch.autocast("cuda", enabled=False):
            result = op(
                value,
                spatial_shapes,
                level_start_index,
                sampling_locations.to(location_dtype),
                attention_weights.type_as(value),
                chunk_step,
                num_points,
                channels_per_thread,
                threads_per_block,
            )
            return result.type_as(value)

    class Backend(MSDABackend):
        @property
        @override
        def name(self) -> str:
            return backend_name

        @override
        def supports(self, context: MSDAContext) -> bool:
            """See :func:`supports_context`."""
            return supports_context(context)

        @override
        def forward(
            self,
            value: Tensor,
            spatial_shapes: Tensor,
            level_start_index: Tensor,
            sampling_locations: Tensor,
            attention_weights: Tensor,
            chunk_step: int,
            method: MSDAMethod,
        ) -> Tensor:
            """Executes the forward pass for ``method``."""
            if method == MSDAMethod.SOFTMAX:
                normalized, use_softmax = attention_weights, True
            else:
                normalized, _ = resolve_msda_attn(attention_weights, method)
                use_softmax = False
            return forward(
                value,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                normalized,
                chunk_step,
                num_points=sampling_locations.shape[4],
                softmax=use_softmax,
            )

    Backend.__doc__ = f"Dispatch adapter for the split-layout CUDA kernel: {summary}."
    Backend.__qualname__ = "Backend"

    return MSDAFamily(forward=forward, backend=Backend, load=bind_extension)
