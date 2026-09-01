# pyright: basic, reportMissingImports=false
r"""Launch shapes for the query-tiled sampling forward.

Configurations share the kernel and differ in the launch grid, since whether a
wider query block pays depends on the sampled shape.
"""

from __future__ import annotations

import typing

import torch

from ...core.capabilities import require_triton

require_triton(__name__.rsplit(".", 1)[-1])

import triton
from torch.library import wrap_triton

from ._kernels.msda_query_tiled import msda_forward_kernel

__all__ = ["DEFAULT_TILES", "WIDE_TILES", "TileShape", "query_tiled_forward_launch"]


class TileShape(typing.NamedTuple):
    """Query-block geometry for the sampling forward.

    ``num_warps`` and ``num_stages`` are left unset on the default shape so the
    launch falls back to Triton's own choice.
    """

    block_size_q: int
    num_warps: int | None = None
    num_stages: int | None = None


#: What the query-tiled backend runs.
DEFAULT_TILES = TileShape(block_size_q=32)

#: Four times the query block, twice the warps, one more pipeline stage.
WIDE_TILES = TileShape(block_size_q=128, num_warps=8, num_stages=3)


def query_tiled_forward_launch(  # noqa: PLR0913
    value: torch.Tensor,
    spatial_shapes: torch.Tensor,
    level_start_index: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    aggregate: int,
    *,
    tiles: TileShape = DEFAULT_TILES,
) -> torch.Tensor:
    """Run the sampling forward on ``tiles``, returning ``(B, Q, H, D)``.

    Assumes ``Q > 0`` and a non-float64 dtype. ``aggregate`` selects the
    in-kernel softmax and is a compile-time constant per operator.
    """
    B, _S, H, D = value.shape
    _, Q, _, L, P, _ = sampling_locations.shape
    # The kernel indexes weights as 5-D; callers may pass (B, Q, H, L * P).
    attention_weights = attention_weights.view(B, Q, H, L, P)
    output = torch.zeros((B, Q, H, D), device=value.device, dtype=value.dtype)
    metaparameters = {
        name: getattr(tiles, name)
        for name in ("num_warps", "num_stages")
        if getattr(tiles, name) is not None
    }
    grid = (triton.cdiv(Q, tiles.block_size_q), H, B)
    wrap_triton(msda_forward_kernel)[grid](
        value,
        spatial_shapes,
        level_start_index,
        sampling_locations,
        attention_weights,
        output,
        B,
        Q,
        H,
        L,
        P,
        D,
        *value.stride(),
        *sampling_locations.stride(),
        *attention_weights.stride(),
        *output.stride(),
        BLOCK_SIZE_Q=tiles.block_size_q,  # pyright: ignore[reportArgumentType]  # tl.constexpr takes the value at launch
        BLOCK_SIZE_D=triton.next_power_of_2(D),
        aggregate=aggregate,  # pyright: ignore[reportArgumentType]  # tl.constexpr takes the value at launch
        **metaparameters,
    )
    return output
