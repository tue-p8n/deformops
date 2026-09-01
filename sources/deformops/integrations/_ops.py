r"""The aggregation step every model integration shares.

Integrations compute their own sampling locations and weights. A concrete
backend object is accepted because ``torch.export`` needs one it can trace.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ..registry import (
    Backend,
    MSDABackend,
    MSDAMethod,
    msda,
)

__all__ = ["run_msda"]


def run_msda(
    value: Tensor,
    spatial_shapes: Tensor,
    level_start_index: Tensor | None,
    sampling_locations: Tensor,
    attention_weights: Tensor,
    *,
    chunk_step: int,
    backend: Backend | MSDABackend | None,
    method: MSDAMethod = MSDAMethod.LINEAR,
) -> Tensor:
    """Aggregate sampled values with attention weights.

    ``S`` is the spatial position count summed over the levels and ``D`` the
    channels per head.

    Parameters
    ----------
    value
        ``(B, S, G, D)``.
    spatial_shapes
        ``(L, 2)``, one ``(height, width)`` row per level.
    level_start_index
        ``(L,)``, the per-level offset into the flat ``value`` sequence, or
        ``None`` to derive it from ``spatial_shapes``.
    sampling_locations
        ``(B, Q, G, L, K, 2)`` normalized sampling locations.
    attention_weights
        ``(B, Q, G, L * K)`` (or ``(B, Q, G, L, K)``).
    chunk_step
        CUDA chunk size, passed only to a pinned backend; see
        ``docs/conventions.md``.
    backend
        A :class:`~deformops.registry.Backend` member, a backend object, or
        ``None`` to select one.
    method
        ``LINEAR`` for already-softmaxed weights, ``SOFTMAX`` for raw logits.

    Returns
    -------
    ``(B, Q, G * D)``.
    """
    if level_start_index is None:
        level_sizes = spatial_shapes[:, 0] * spatial_shapes[:, 1]
        level_start_index = torch.cat(
            [level_sizes.new_zeros(1), level_sizes.cumsum(0)[:-1]]
        )
    if isinstance(backend, MSDABackend):
        return backend.forward(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            chunk_step,
            method,
        )
    # The public operator's schema has no chunk size, so only an explicit backend
    # receives it.
    options: dict[str, Any] = {"method": method, "backend": backend}
    if backend is not None:
        options["chunk_step"] = chunk_step
    return msda(
        value,
        spatial_shapes,
        level_start_index,
        sampling_locations,
        attention_weights,
        **options,
    )
