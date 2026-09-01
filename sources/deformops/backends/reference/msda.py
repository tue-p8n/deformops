r"""Reference MSDA forward on ``grid_sample``; the other backends are tested against it.

Simple, slow and not dispatchable. Its per-level ``.item()`` breaks the
``torch.compile`` graph once per level; the eager backend is the traceable one.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn.functional import grid_sample

__all__ = ["reference"]


def reference(
    value: Tensor,
    spatial_shapes: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> Tensor:
    """MSDA forward pass in base PyTorch operators.

    Parameters
    ----------
    value
        ``(N, S, M, D)`` flattened multi-level value tensor.
    spatial_shapes
        ``(L, 2)`` ``(height, width)`` per level.
    sampling_locations
        ``(N, Q, M, L, P, 2)`` normalized sampling locations in ``[0, 1]``.
    attention_weights
        ``(N, Q, M, L, P)`` attention weights, already softmaxed/rectified.

    Returns
    -------
    ``(N, Q, M * D)``.
    """

    N, _, M, D = value.shape
    _, Q, _, L, P, _ = sampling_locations.shape

    expected_loc_shape = (N, Q, M, L, P, 2)
    if sampling_locations.shape != expected_loc_shape:
        msg = (
            f"sampling_locations shape {tuple(sampling_locations.shape)} "
            f"!= expected {expected_loc_shape}"
        )
        raise ValueError(msg)
    expected_attn_shape = (N, Q, M, L, P)
    if attention_weights.shape != expected_attn_shape:
        msg = (
            f"attention_weights shape {tuple(attention_weights.shape)} "
            f"!= expected {expected_attn_shape}"
        )
        raise ValueError(msg)

    sampling_locations = 2 * sampling_locations - 1
    attention_weights = attention_weights.transpose(1, 2).reshape(N * M, 1, Q, L * P)

    level_sizes = spatial_shapes[:, 0] * spatial_shapes[:, 1]
    level_offsets = torch.cat(
        [
            torch.zeros(1, dtype=torch.long, device=spatial_shapes.device),
            level_sizes.cumsum(0)[:-1],
        ]
    )

    sampled_levels: list[Tensor] = []
    for i in range(L):
        h_i, w_i = int(spatial_shapes[i, 0].item()), int(spatial_shapes[i, 1].item())
        sz = int(level_sizes[i].item())
        off = int(level_offsets[i].item())

        v = value.narrow(1, off, sz)
        g = sampling_locations[:, :, :, i, :, :]

        v = v.flatten(2).transpose(1, 2).reshape(N * M, D, h_i, w_i)
        g = g.transpose(1, 2).flatten(0, 1)

        sampled = grid_sample(
            v,
            g,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled_levels.append(sampled)

    value = torch.stack(sampled_levels, dim=-2).flatten(-2)
    return (
        (value * attention_weights)
        .sum(-1)
        .view(N, M * D, Q)
        .transpose(1, 2)
        .contiguous()
    )
