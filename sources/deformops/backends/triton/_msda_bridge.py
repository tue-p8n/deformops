r"""Shared machinery behind every Triton multi-scale deformable attention module.

A configuration module states its operator prefix, backend name and launches;
:func:`install_msda_family` registers the linear and softmax operators, their
backwards, autograd and vmap rules. ``torch.func.grad`` over these operators
raises; ``vmap`` then ordinary autograd works. See ``docs/export.md``.
"""

from ...core.capabilities import require_triton

require_triton(__name__.rsplit(".", 1)[-1])

import typing
from collections.abc import Callable
from typing import Any, override

import torch
from torch import Tensor

from ...registry import (
    MSDABackend,
    MSDAContext,
    MSDAMethod,
    resolve_msda_attn,
)
from ._vmap import make_vmap_rule

__all__ = ["OP_NAMESPACE", "MSDAFamily", "install_msda_family"]

#: Library namespace of every Triton operator, apart from the CUDA backend's.
OP_NAMESPACE = "deformops_triton"


# Softmax and dtype helpers.


def _cast_from_float64(
    value: Tensor, sampling_locations: Tensor, attention_weights: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Move a float64 call onto float32, which the kernels compile for.

    Triton has no float64 and the accumulators are float32; the caller casts the
    result back, which keeps ``gradcheck`` usable.
    """
    if value.dtype == torch.float64:
        return value.float(), sampling_locations.float(), attention_weights.float()
    return value, sampling_locations, attention_weights


def _softmax_probabilities(attention_weights: Tensor) -> Tensor:
    """Normalize ``attention_weights``, ``(B, Q, G, L, K)`` or ``(B, Q, G, L * K)``, over ``L * K``.

    Always float32: the Jacobian correction is a difference of similar
    magnitudes that loses its leading digits in 16-bit.
    """
    return torch.softmax(attention_weights.flatten(3).float(), dim=-1)


def _rebuilt_probabilities(
    attention_weights: Tensor, running_max: Tensor, running_sum: Tensor
) -> Tensor:
    """Rebuild the softmax weights from a caching forward's running max and sum.

    Caching replaces a max scan and a sum with one elementwise pass. Float32 and flattened, like
    :func:`_softmax_probabilities`.
    """
    flat = attention_weights.flatten(3).float()
    return (flat - running_max.unsqueeze(-1)).exp_().div_(running_sum.unsqueeze(-1))


def _softmax_jacobian(probabilities: Tensor, grad_probabilities: Tensor) -> Tensor:
    r"""Push a gradient back through the softmax that produced the weights.

    With :math:`p = \mathrm{softmax}(a)` over the flattened level-point axis,
    :math:`\partial p_i / \partial a_j = p_i (\delta_{ij} - p_j)`, so

    .. math:: \frac{\partial L}{\partial a_j}
              = p_j \left( \frac{\partial L}{\partial p_j}
                           - \sum_i p_i \frac{\partial L}{\partial p_i} \right).

    Both arguments are float32 and flattened; the result is too.
    """
    correction = (probabilities * grad_probabilities).sum(dim=-1, keepdim=True)
    return probabilities * (grad_probabilities - correction)


# Operator family.

#: A forward launch: ``(value, shapes, level_start_index, loc, attn,
#: aggregate)`` to the unflattened ``(B, Q, G, D)`` output in ``value.dtype``.
#: It may assume ``Q > 0`` and a non-float64 dtype. A configuration that caches
#: its online-softmax statistics returns ``(output, running_max, running_sum)``
#: instead, the two statistics ``(B, Q, G)`` float32.
ForwardLaunch = Callable[..., Any]

#: A backward launch: ``(grad_output, value, shapes, level_start_index, loc,
#: attn)`` to ``(grad_value, grad_loc, grad_attn)``, with ``grad_output`` given
#: as ``(B, Q, G, D)`` and ``grad_attn`` returned as ``(B, Q, G, L, K)``. Same
#: assumptions as the forward launch. This is always the *linear* backward;
#: the softmax operator wraps it.
BackwardLaunch = Callable[
    [Tensor, Tensor, Tensor, Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]
]


class MSDAFamily(typing.NamedTuple):
    """Everything one configuration module re-exports."""

    forward: Callable[..., Tensor]
    backend: type[Any]


def _is_empty(value: Tensor, sampling_locations: Tensor) -> bool:
    """Whether a call samples nothing, so no kernel should launch.

    A zero-sized grid fails, and the normalizing kernels divide by an empty sum.
    """
    return value.shape[0] == 0 or 0 in sampling_locations.shape[1:5]


#: The largest offset an int32-typed kernel argument can hold.
_INT32_LIMIT = (1 << 31) - 1

#: The largest launch grid extent along the batch axis.
_MAX_GRID_BATCH = 65535


def _launch_batch_chunk(
    value: Tensor, sampling_locations: Tensor, attention_weights: Tensor
) -> int:
    """The largest batch one launch can take.

    ``batch_index * stride`` is int32 in the kernels and wraps past ``2**31``,
    and the launch grid's batch axis holds at most 65535. A ``vmap`` folds into
    the batch, so both are reachable; larger calls split along the batch.

    Raises
    ------
    ValueError
        If a single sample is already too large to address.
    """
    per_sample = max(
        value[0].numel() if value.shape[0] else 0,
        sampling_locations[0].numel() if sampling_locations.shape[0] else 0,
        attention_weights[0].numel() if attention_weights.shape[0] else 0,
        1,
    )
    chunk = min(_INT32_LIMIT // per_sample, _MAX_GRID_BATCH)
    if chunk < 1:
        msg = (
            f"One sample holds {per_sample} elements, more than the int32 "
            f"address space the Triton kernels index with ({_INT32_LIMIT})."
        )
        raise ValueError(msg)
    return chunk


def install_msda_family(
    *,
    op_prefix: str,
    backend_name: str,
    forward_launch: ForwardLaunch,
    backward_launch: BackwardLaunch,
    summary: str,
    caches_softmax_statistics: bool = False,
) -> MSDAFamily:
    r"""Register one configuration's four operators and build its public surface.

    Runs at import of a configuration module.

    Parameters
    ----------
    op_prefix
        Operator-name prefix, e.g. ``"msda_query_tiled"``. The registered names
        are ``{prefix}_forward``, ``{prefix}_backward``,
        ``{prefix}_softmax_forward`` and ``{prefix}_softmax_backward``.
    backend_name
        What this configuration answers to as a backend, e.g.
        ``"triton_query_tiled"``.
    forward_launch
        See :data:`ForwardLaunch`.
    backward_launch
        See :data:`BackwardLaunch`.
    summary
        One line naming this configuration, for the adapter's docstring.
    caches_softmax_statistics
        Whether the softmax forward also returns its running max and sum-of-exp,
        which the softmax backward then takes to rebuild the weights.

    Returns
    -------
    The functional entry point and the dispatch adapter class.
    """
    #: Level-table positions: the backward has the output gradient in front.
    _TABLE = "the spatial shapes and level start offsets"
    forward_shared = {1: _TABLE, 2: _TABLE}
    backward_shared = {2: _TABLE, 3: _TABLE}

    def _empty_statistics(
        value: Tensor, batch: int, queries: int, heads: int
    ) -> tuple[Tensor, ...]:
        """Statistics for an empty call, so the schema holds without a launch."""
        if not caches_softmax_statistics:
            return ()
        shape = (batch, queries, heads)
        return (
            torch.zeros(shape, device=value.device, dtype=torch.float32),
            torch.zeros(shape, device=value.device, dtype=torch.float32),
        )

    def _forward_impl(
        value: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        sampling_locations: Tensor,
        attention_weights: Tensor,
        aggregate: int,
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        """The output, plus the statistics the launch cached, if any."""
        original_dtype = value.dtype
        value, sampling_locations, attention_weights = _cast_from_float64(
            value, sampling_locations, attention_weights
        )
        batch, _spatial, heads, channels = value.shape
        queries = sampling_locations.shape[1]
        if _is_empty(value, sampling_locations):
            empty = torch.zeros(
                (batch, queries, heads * channels),
                device=value.device,
                dtype=original_dtype,
            )
            return empty, _empty_statistics(value, batch, queries, heads)
        chunk = _launch_batch_chunk(value, sampling_locations, attention_weights)
        if chunk < batch:
            pieces = [
                _forward_impl(
                    value[start : start + chunk],
                    spatial_shapes,
                    level_start_index,
                    sampling_locations[start : start + chunk],
                    attention_weights[start : start + chunk],
                    aggregate,
                )
                for start in range(0, batch, chunk)
            ]
            output = torch.cat([piece[0] for piece in pieces]).to(original_dtype)
            statistics = tuple(
                torch.cat(parts)
                for parts in zip(*(piece[1] for piece in pieces), strict=True)
            )
            return output, statistics
        result = forward_launch(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            aggregate,
        )
        if caches_softmax_statistics:
            output, running_max, running_sum = result
            return output.flatten(2).to(original_dtype), (running_max, running_sum)
        return result.flatten(2).to(original_dtype), ()

    def _backward_impl(
        grad_output: Tensor,
        value: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        sampling_locations: Tensor,
        attention_weights: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        original_dtype = value.dtype
        value, sampling_locations, attention_weights = _cast_from_float64(
            value, sampling_locations, attention_weights
        )
        batch, _spatial, heads, channels = value.shape
        queries = sampling_locations.shape[1]
        if _is_empty(value, sampling_locations):
            return (
                torch.zeros_like(value).to(original_dtype),
                torch.zeros_like(sampling_locations).to(original_dtype),
                torch.zeros_like(attention_weights).to(original_dtype),
            )
        chunk = _launch_batch_chunk(value, sampling_locations, attention_weights)
        if chunk < batch:
            pieces = [
                _backward_impl(
                    grad_output[start : start + chunk],
                    value[start : start + chunk],
                    spatial_shapes,
                    level_start_index,
                    sampling_locations[start : start + chunk],
                    attention_weights[start : start + chunk],
                )
                for start in range(0, batch, chunk)
            ]
            grad_values, grad_locs, grad_attns = zip(*pieces, strict=True)
            return (
                torch.cat(grad_values).to(original_dtype),
                torch.cat(grad_locs).to(original_dtype),
                torch.cat(grad_attns).to(original_dtype),
            )
        # ``reshape``, not ``view``: an upstream operation can hand back a
        # non-contiguous gradient, which ``view`` rejects mid-training.
        grad_output = grad_output.reshape(batch, queries, heads, channels).to(
            value.dtype
        )
        grad_value, grad_loc, grad_attn = backward_launch(
            grad_output,
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
        )
        return (
            grad_value.to(original_dtype),
            grad_loc.to(original_dtype),
            grad_attn.view_as(attention_weights).to(original_dtype),
        )

    def _close_the_softmax(
        grad_output: Tensor,
        value: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        sampling_locations: Tensor,
        attention_weights: Tensor,
        probabilities: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Differentiate the softmax forward from the weights it sampled with.

        The linear backward on normalized weights gives exact value and location
        gradients and ``dL/dp``, which the host-side Jacobian turns into ``dL/da``.
        """
        weights = probabilities.view_as(attention_weights).to(attention_weights.dtype)
        grad_value, grad_loc, grad_weights = _backward_impl(
            grad_output,
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            weights,
        )
        grad_attn = _softmax_jacobian(probabilities, grad_weights.flatten(3).float())
        return (
            grad_value,
            grad_loc,
            grad_attn.view_as(attention_weights).to(attention_weights.dtype),
        )

    # Operator registration.

    def _register(op: Any, shared: dict[int, str]) -> Any:
        torch.library.register_vmap(op, make_vmap_rule(op, shared=shared))
        return op

    def _register_forward(name: str, aggregate: int) -> Any:
        @torch.library.triton_op(f"{OP_NAMESPACE}::{name}", mutates_args=())  # pyright: ignore[reportUntypedFunctionDecorator]
        def op(
            value: Tensor,
            spatial_shapes: Tensor,
            level_start_index: Tensor,
            sampling_locations: Tensor,
            attention_weights: Tensor,
        ) -> Tensor:
            output, _statistics = _forward_impl(
                value,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                attention_weights,
                aggregate,
            )
            return output

        return _register(op, forward_shared)

    def _register_caching_forward(name: str) -> Any:
        @torch.library.triton_op(f"{OP_NAMESPACE}::{name}", mutates_args=())  # pyright: ignore[reportUntypedFunctionDecorator]
        def op(
            value: Tensor,
            spatial_shapes: Tensor,
            level_start_index: Tensor,
            sampling_locations: Tensor,
            attention_weights: Tensor,
        ) -> tuple[Tensor, Tensor, Tensor]:
            output, statistics = _forward_impl(
                value,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                attention_weights,
                1,
            )
            return output, statistics[0], statistics[1]

        return _register(op, forward_shared)

    def _register_backward(name: str) -> Any:
        @torch.library.triton_op(f"{OP_NAMESPACE}::{name}", mutates_args=())  # pyright: ignore[reportUntypedFunctionDecorator]
        def op(
            grad_output: Tensor,
            value: Tensor,
            spatial_shapes: Tensor,
            level_start_index: Tensor,
            sampling_locations: Tensor,
            attention_weights: Tensor,
        ) -> tuple[Tensor, Tensor, Tensor]:
            return _backward_impl(
                grad_output,
                value,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                attention_weights,
            )

        return _register(op, backward_shared)

    def _register_softmax_backward(name: str) -> Any:
        @torch.library.triton_op(f"{OP_NAMESPACE}::{name}", mutates_args=())  # pyright: ignore[reportUntypedFunctionDecorator]
        def op(
            grad_output: Tensor,
            value: Tensor,
            spatial_shapes: Tensor,
            level_start_index: Tensor,
            sampling_locations: Tensor,
            attention_weights: Tensor,
        ) -> tuple[Tensor, Tensor, Tensor]:
            return _close_the_softmax(
                grad_output,
                value,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                attention_weights,
                _softmax_probabilities(attention_weights),
            )

        return _register(op, backward_shared)

    def _register_caching_softmax_backward(name: str) -> Any:
        @torch.library.triton_op(f"{OP_NAMESPACE}::{name}", mutates_args=())  # pyright: ignore[reportUntypedFunctionDecorator]
        def op(  # noqa: PLR0913
            grad_output: Tensor,
            value: Tensor,
            spatial_shapes: Tensor,
            level_start_index: Tensor,
            sampling_locations: Tensor,
            attention_weights: Tensor,
            running_max: Tensor,
            running_sum: Tensor,
        ) -> tuple[Tensor, Tensor, Tensor]:
            return _close_the_softmax(
                grad_output,
                value,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                attention_weights,
                _rebuilt_probabilities(attention_weights, running_max, running_sum),
            )

        return _register(op, backward_shared)

    linear_name = f"{op_prefix}_forward"
    softmax_name = f"{op_prefix}_softmax_forward"
    _register_forward(linear_name, 0)
    _register_backward(f"{op_prefix}_backward")
    if caches_softmax_statistics:
        _register_caching_forward(softmax_name)
        _register_caching_softmax_backward(f"{op_prefix}_softmax_backward")
    else:
        _register_forward(softmax_name, 1)
        _register_softmax_backward(f"{op_prefix}_softmax_backward")

    # Autograd.

    def _setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:  # noqa: ARG001
        ctx.save_for_backward(*inputs)

    def _setup_caching_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        _output, running_max, running_sum = output
        ctx.save_for_backward(*inputs, running_max, running_sum)

    def _make_autograd(backward_name: str) -> Callable[..., tuple[Any, ...]]:
        def bridge(ctx: Any, *grads: Tensor) -> tuple[Any, ...]:
            backward_op = getattr(getattr(torch.ops, OP_NAMESPACE), backward_name)
            # A caching forward's statistics are not differentiable.
            grad_value, grad_loc, grad_attn = backward_op(grads[0], *ctx.saved_tensors)
            # The level table is not differentiable.
            return grad_value, None, None, grad_loc, grad_attn

        return bridge

    torch.library.register_autograd(
        f"{OP_NAMESPACE}::{op_prefix}_forward",
        _make_autograd(f"{op_prefix}_backward"),
        setup_context=_setup_context,
    )
    torch.library.register_autograd(
        f"{OP_NAMESPACE}::{op_prefix}_softmax_forward",
        _make_autograd(f"{op_prefix}_softmax_backward"),
        setup_context=(
            _setup_caching_context if caches_softmax_statistics else _setup_context
        ),
    )

    # Public surface.

    def forward(
        value: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        sampling_locations: Tensor,
        attention_weights: Tensor,
        *,
        softmax: bool = False,
    ) -> Tensor:
        """Run this configuration's sampling kernel.

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
            ``(B, Q, G, L, K)`` or ``(B, Q, G, L * K)``.
        softmax
            Take ``attention_weights`` as logits and normalize in the kernel, instead of as
            weights.

        Returns
        -------
        ``(B, Q, G * D)``.
        """
        if _is_empty(value, sampling_locations):
            batch, _spatial, heads, channels = value.shape
            return value.new_zeros(
                (batch, sampling_locations.shape[1], heads * channels)
            )
        arguments = (
            value,
            spatial_shapes.to(torch.int32),
            level_start_index.to(torch.int32),
            sampling_locations,
            attention_weights,
        )
        # Triton launches on the current device, not the operands' device.
        operators = getattr(torch.ops, OP_NAMESPACE)
        with torch.cuda.device(value.device):
            if not softmax:
                return getattr(operators, linear_name)(*arguments)
            if caches_softmax_statistics:
                output, _running_max, _running_sum = getattr(operators, softmax_name)(
                    *arguments
                )
                return output
            return getattr(operators, softmax_name)(*arguments)

    class Backend(MSDABackend):
        @property
        @override
        def name(self) -> str:
            return backend_name

        @override
        def supports(self, context: MSDAContext) -> bool:
            """``True`` on CUDA devices only."""
            return context.device.type == "cuda"

        @override
        def forward(
            self,
            value: Tensor,
            spatial_shapes: Tensor,
            level_start_index: Tensor,
            sampling_locations: Tensor,
            attention_weights: Tensor,
            chunk_step: int,  # noqa: ARG002
            method: MSDAMethod,
        ) -> Tensor:
            """Run ``method``; ``chunk_step`` is unused, as nothing here chunks."""
            if method == MSDAMethod.SOFTMAX:
                return forward(
                    value,
                    spatial_shapes,
                    level_start_index,
                    sampling_locations,
                    attention_weights,
                    softmax=True,
                )
            normalized, _ = resolve_msda_attn(attention_weights, method)
            return forward(
                value,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                normalized,
                softmax=False,
            )

    forward.__doc__ = f"{forward.__doc__}\nConfiguration: {summary}.\n"
    Backend.__doc__ = f"Dispatch adapter for the Triton kernel: {summary}."
    Backend.__qualname__ = "Backend"

    return MSDAFamily(forward=forward, backend=Backend)
