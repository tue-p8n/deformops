r"""Eager MSDA: one bilinear gather with tensor-only indexing, runnable on CPU.

Uses no ``.item()``, so ``torch.compile`` traces it without a graph break. It
materializes a ``[N, Q, M, L, P, D]`` intermediate; the GPU kernels avoid it.
See ``docs/kernels.md``.
"""

from typing import override

import torch
from torch import Tensor

from ...registry import (
    MSDABackend,
    MSDAContext,
    MSDAMethod,
    apply_msda_method,
)

__all__ = ["Backend", "forward"]


def forward(
    value: Tensor,
    spatial_shapes: Tensor,
    level_start_index: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> Tensor:
    r"""Eager MSDA forward.

    Parameters
    ----------
    value
        ``[N, S, M, D]`` flattened multi-level value tensor.
    spatial_shapes
        ``[L, 2]`` ``(H, W)`` per level.
    level_start_index
        ``[L]`` per-level offset into the flat ``value`` spatial dim.
    sampling_locations
        ``[N, Q, M, L, P, 2]`` normalized sampling locations in ``[0, 1]``.
    attention_weights
        ``[N, Q, M, L, P]`` attention weights, already normalized.

    Returns
    -------
    Tensor
        ``[N, Q, M * D]``.

    Notes
    -----
    Pixel coordinates are ``locs * size - 0.5``, as in ``F.grid_sample`` with
    ``align_corners=False``; out-of-bounds corners contribute zero.
    """
    N, _S, M, D = value.shape
    _N, Q, _M, L, _P, _ = sampling_locations.shape

    # ``spatial_shapes`` rows are ``(H, W)``; int64 for indexing.
    h_l = spatial_shapes[:, 0].to(torch.long).view(1, 1, 1, L, 1)
    w_l = spatial_shapes[:, 1].to(torch.long).view(1, 1, 1, L, 1)
    start_l = level_start_index.to(torch.long).view(1, 1, 1, L, 1)

    dtype = value.dtype
    w_f = w_l.to(dtype)
    h_f = h_l.to(dtype)
    x = sampling_locations[..., 0] * w_f - 0.5  # [N, Q, M, L, P]
    y = sampling_locations[..., 1] * h_f - 0.5

    x0 = torch.floor(x).to(torch.long)
    y0 = torch.floor(y).to(torch.long)
    x1 = x0 + 1
    y1 = y0 + 1

    dx = x - x0.to(dtype)
    dy = y - y0.to(dtype)

    m_x0 = (x0 >= 0) & (x0 < w_l)
    m_x1 = (x1 >= 0) & (x1 < w_l)
    m_y0 = (y0 >= 0) & (y0 < h_l)
    m_y1 = (y1 >= 0) & (y1 < h_l)

    # Clamped for the gather; the masks zero out-of-bounds corners.
    zero = torch.zeros((), dtype=torch.long, device=value.device)
    x0c = torch.maximum(x0, zero).minimum(w_l - 1)
    x1c = torch.maximum(x1, zero).minimum(w_l - 1)
    y0c = torch.maximum(y0, zero).minimum(h_l - 1)
    y1c = torch.maximum(y1, zero).minimum(h_l - 1)

    # Flat indices into values' S dim: start + y * W + x.
    idx_00 = start_l + y0c * w_l + x0c  # [N, Q, M, L, P]
    idx_01 = start_l + y0c * w_l + x1c
    idx_10 = start_l + y1c * w_l + x0c
    idx_11 = start_l + y1c * w_l + x1c

    # Advanced-indexing gather: values[n, idx, m, :] per (n, q, m, l, p).
    n_idx = torch.arange(N, device=value.device).view(N, 1, 1, 1, 1)
    m_idx = torch.arange(M, device=value.device).view(1, 1, M, 1, 1)
    c00 = value[n_idx, idx_00, m_idx]  # [N, Q, M, L, P, D]
    c01 = value[n_idx, idx_01, m_idx]
    c10 = value[n_idx, idx_10, m_idx]
    c11 = value[n_idx, idx_11, m_idx]

    one = torch.ones((), dtype=dtype, device=value.device)
    w00 = ((one - dx) * (one - dy) * (m_x0 & m_y0).to(dtype)).unsqueeze(-1)
    w01 = (dx * (one - dy) * (m_x1 & m_y0).to(dtype)).unsqueeze(-1)
    w10 = ((one - dx) * dy * (m_x0 & m_y1).to(dtype)).unsqueeze(-1)
    w11 = (dx * dy * (m_x1 & m_y1).to(dtype)).unsqueeze(-1)

    sampled = w00 * c00 + w01 * c01 + w10 * c10 + w11 * c11  # [N, Q, M, L, P, D]

    out = (sampled * attention_weights.unsqueeze(-1)).sum(dim=(3, 4))  # [N, Q, M, D]
    return out.reshape(N, Q, M * D)


class Backend(MSDABackend):
    """Dispatch adapter for the eager implementation."""

    @property
    @override
    def name(self) -> str:
        """The registry name, ``"pytorch"``."""
        return "pytorch"

    @override
    def supports(self, context: MSDAContext) -> bool:  # noqa: ARG002
        """Always ``True``: pure PyTorch, runs anywhere the input tensors live."""
        return True

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
        """Normalize the weights per ``method`` and delegate to :func:`forward`.

        ``attention_weights`` is ``(N, Q, M, L * P)``; ``chunk_step`` is unused.
        """
        _attn = apply_msda_method(attention_weights, method)
        N, _, M, _ = value.shape
        _, Q, _, L, P, _ = sampling_locations.shape
        _attn = _attn.view(N, Q, M, L, P)
        return forward(
            value, spatial_shapes, level_start_index, sampling_locations, _attn
        )
