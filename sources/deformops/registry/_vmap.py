r"""The vmap rule every deformops operator shares: fold the vmap axis into the batch.

A vmapped call becomes one call over ``size * B`` samples. Arguments that
describe the batch (level tables, weight, bias) raise when batched. Imports
only torch; see ``docs/export.md``.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping
from typing import Any

from torch import Tensor

__all__ = ["fold_into_batch", "make_vmap_rule", "unfold_from_batch"]


def fold_into_batch(tensor: Tensor, batch_dim: int | None, size: int) -> Tensor:
    """Merge a vmap dimension into a tensor's leading batch dimension.

    An argument without a vmap dimension is expanded, which materializes a copy.
    """
    if batch_dim is None:
        return tensor.unsqueeze(0).expand(size, *tensor.shape).flatten(0, 1)
    return tensor.movedim(batch_dim, 0).flatten(0, 1)


def unfold_from_batch(tensor: Tensor, size: int) -> Tensor:
    """Undo :func:`fold_into_batch` on a result, leaving the vmap axis at 0.

    A zero-element placeholder gradient becomes ``(size, 0)``.
    """
    if tensor.numel() == 0 and tensor.dim() <= 1:
        return tensor.new_empty((size, 0))
    return tensor.unflatten(0, (size, -1))


def _rule(
    info: Any,
    in_dims: tuple[int | None, ...],
    *args: Any,
    op: Callable[..., Any],
    shared: Mapping[int, str],
    **kwargs: Any,
) -> tuple[Any, Any]:
    for position, description in shared.items():
        if position < len(in_dims) and in_dims[position] is not None:
            msg = (
                f"vmap over {description} is not supported: it describes the "
                f"batch rather than varying across it. Pass it with in_dims=None."
            )
            raise RuntimeError(msg)
    size = info.batch_size
    folded = [
        fold_into_batch(argument, in_dims[position], size)
        if isinstance(argument, Tensor) and position not in shared
        else argument
        for position, argument in enumerate(args)
    ]
    result = op(*folded, **kwargs)
    if isinstance(result, Tensor):
        return unfold_from_batch(result, size), 0
    unfolded = tuple(unfold_from_batch(item, size) for item in result)
    return unfolded, tuple(0 for _ in unfolded)


def make_vmap_rule(
    op: Callable[..., Any], *, shared: Mapping[int, str] | None = None
) -> Callable[..., tuple[Any, Any]]:
    """Build the vmap rule for one operator.

    Parameters
    ----------
    op
        The operator to call once, with the vmapped axis folded in.
    shared
        Argument positions that describe the batch, mapped to a phrase naming
        them for the error. Other tensor positions are folded; non-tensors
        pass through.
    """
    return functools.partial(_rule, op=op, shared=shared or {})
