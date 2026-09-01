r"""Space-filling-curve query reordering for the CUDA MSDA configurations.

Permutes the queries before the production kernel for cache locality and
undoes the permutation on the output, so the result is unchanged. Host-side
only; see ``docs/kernels.md``.
"""

from collections.abc import Callable
from typing import Any, Literal, override

import torch
from torch import Tensor

from ....registry import (
    MSDABackend,
    MSDAContext,
    MSDAMethod,
    resolve_msda_attn,
)
from .._msda_bridge import supports_context

__all__ = ["SortCurve", "install_reordering_family"]

SortCurve = Literal["morton", "hilbert"]


def _morton_index(x: Tensor, y: Tensor, bits: int = 10) -> Tensor:
    r"""Z-order (Morton) index for spatial reordering (elementwise, any rank).

    Locations are clamped to [0, 1] for the bitwise ops. The clamp changes the
    order, not the result.
    """
    x = x.clamp(0.0, 1.0)
    y = y.clamp(0.0, 1.0)
    # Normalize to [0, 2^bits - 1]
    ix = (x * (2**bits - 1)).long()
    iy = (y * (2**bits - 1)).long()

    morton = torch.zeros_like(ix)
    for i in range(bits):
        morton |= (ix & (1 << i)) << i | (iy & (1 << i)) << (i + 1)
    return morton


def _hilbert_index(x: Tensor, y: Tensor, bits: int = 10) -> Tensor:
    r"""Hilbert-curve index for spatial reordering (elementwise, any rank).

    Iterative quadrant rotation over locations clamped to [0, 1], as in
    :func:`_morton_index`.
    """
    x = x.clamp(0.0, 1.0)
    y = y.clamp(0.0, 1.0)
    n = 1 << bits
    ix = (x * (n - 1)).long()
    iy = (y * (n - 1)).long()

    hilbert = torch.zeros_like(ix)
    s = n // 2
    while s > 0:
        rx = ((ix & s) > 0).long()
        ry = ((iy & s) > 0).long()
        hilbert = hilbert + s * s * ((3 * rx) ^ ry)
        swap = ry == 0
        flip = swap & (rx == 1)
        ix_flipped = torch.where(flip, s - 1 - ix, ix)
        iy_flipped = torch.where(flip, s - 1 - iy, iy)
        ix = torch.where(swap, iy_flipped, ix_flipped)
        iy = torch.where(swap, ix_flipped, iy_flipped)
        s //= 2
    return hilbert


_SORT_CURVE_FNS: dict[SortCurve, Callable[[Tensor, Tensor], Tensor]] = {
    "morton": _morton_index,
    "hilbert": _hilbert_index,
}


def _query_sort_indices(
    sampling_locations: Tensor, curve: SortCurve = "morton"
) -> Tensor:
    r"""Per-batch-row space-filling-curve sort indices, shape ``[B, Q]``.

    Sorts each row by the curve index of the location in ``sampling_locations``
    (B, Q, G, L, K, 2) averaged over head, level and point.
    """
    avg_loc = sampling_locations.mean(dim=(2, 3, 4))  # [B, Q, 2]
    curve_index = _SORT_CURVE_FNS[curve](avg_loc[..., 0], avg_loc[..., 1])  # [B, Q]
    return curve_index.sort(dim=-1)[1]


def _gather_along_dim1(value: Tensor, indices: Tensor) -> Tensor:
    r"""Batched ``values[i, indices[i]]`` for every row ``i``, in one call.

    ``indices`` is ``(B, Q)`` and broadcasts over the trailing dims of ``value``.
    """
    idx = indices.view(*indices.shape, *([1] * (value.ndim - indices.ndim)))
    idx = idx.expand(*indices.shape, *value.shape[indices.ndim :])
    return torch.take_along_dim(value, idx, dim=1)


def _scatter_along_dim1(sorted_values: Tensor, indices: Tensor) -> Tensor:
    r"""Inverse of :func:`_gather_along_dim1`: undo the batched reorder."""
    idx = indices.view(*indices.shape, *([1] * (sorted_values.ndim - indices.ndim)))
    idx = idx.expand_as(sorted_values)
    return torch.empty_like(sorted_values).scatter_(1, idx, sorted_values)


# Reordering family


def install_reordering_family(
    *,
    curve: SortCurve,
    backend_name: str,
    base_forward: Callable[..., Tensor],
    summary: str,
) -> tuple[Callable[..., Tensor], type[Any]]:
    r"""Build the entry point and adapter for one reordering configuration.

    Parameters
    ----------
    curve
        Which space-filling curve orders the queries.
    backend_name
        What this configuration answers to as a backend.
    base_forward
        The functional entry point to call between the gather and the scatter.
    summary
        One line describing the curve, used in the adapter's docstring.

    Returns
    -------
    The functional entry point and the dispatch adapter class.
    """
    if curve not in _SORT_CURVE_FNS:
        accepted = ", ".join(repr(k) for k in _SORT_CURVE_FNS)
        msg = f"curve must be one of {accepted}, got {curve!r}"
        raise ValueError(msg)

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
        r"""Reorder the queries, run the kernel, and scatter the output back.

        Same operands and result as
        :func:`deformops.backends.cuda.msda_fp32acc.forward`.
        """
        sort_indices = _query_sort_indices(sampling_locations, curve)
        result = base_forward(
            value,
            spatial_shapes,
            level_start_index,
            _gather_along_dim1(sampling_locations, sort_indices),
            _gather_along_dim1(attention_weights, sort_indices),
            chunk_step,
            num_points=num_points,
            channels_per_thread=channels_per_thread,
            threads_per_block=threads_per_block,
            softmax=softmax,
        )
        return _scatter_along_dim1(result, sort_indices)

    class Backend(MSDABackend):
        @property
        @override
        def name(self) -> str:
            return backend_name

        @override
        def supports(self, context: MSDAContext) -> bool:
            """See :func:`deformops.backends.cuda._msda_bridge.supports_context`."""
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

    return forward, Backend
