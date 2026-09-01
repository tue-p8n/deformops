r"""Reference single-scale deformable sampling; the Triton kernel is tested against it.

Pure PyTorch, traceable by ``torch.compile`` (no ``.item()``, ``.any()`` or
tensor-valued branch), and not dispatchable. See ``docs/conventions.md``.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["reference"]


def _gather(
    input_: Tensor,
    n_idx: Tensor,
    g_idx: Tensor,
    y: Tensor,
    x: Tensor,
    h_in: int,
    w_in: int,
) -> Tensor:
    """Gather ``input_`` at integer ``(y, x)``, zero outside [0, H) x [0, W)."""
    mask_valid = (y >= 0) & (y < h_in) & (x >= 0) & (x < w_in)
    y_safe = y.clamp(0, h_in - 1)
    x_safe = x.clamp(0, w_in - 1)
    res = input_[n_idx, y_safe, x_safe, g_idx]
    return res * mask_valid.unsqueeze(-1)


def reference(
    input: Tensor,
    offset: Tensor,
    mask: Tensor,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dilation_h: int,
    dilation_w: int,
    groups: int,
    group_channels: int,
    spread: float,
    drop_center: bool = False,
    softmax: bool = False,
) -> Tensor:
    """Reference implementation for single-scale deformable sampling.

    Parameters
    ----------
    input
        ``(B, H, W, groups * group_channels)``, channels-last.
    offset
        ``(B, H_out, W_out, groups, K, 2)``, holding ``(dx, dy)``.
    mask
        ``(B, H_out, W_out, groups, K)``.
    kernel_h, kernel_w
        Sampling grid extent.
    stride_h, stride_w
        Stride of the sampling grid.
    pad_h, pad_w
        Padding applied to ``input`` before the grid is laid out.
    dilation_h, dilation_w
        Dilation of the sampling grid.
    groups
        Number of deformable groups.
    group_channels
        Channels per group.
    spread
        Scale applied to the offset before it displaces a grid point.
    drop_center
        Whether to drop the kernel's center point from the sampling grid.
    softmax
        Normalize ``mask`` over the kernel points first.

    Returns
    -------
    ``(B, H_out, W_out, groups * group_channels)``, channels-last.
    """
    N, H_IN, W_IN, _C = input.shape
    H_OUT, W_OUT = offset.shape[1], offset.shape[2]
    K = kernel_h * kernel_w - (1 if drop_center else 0)

    if softmax:
        mask = torch.nn.functional.softmax(mask, dim=-1)

    qi_w = torch.arange(W_OUT, device=input.device)
    qi_h = torch.arange(H_OUT, device=input.device)
    qi_h, qi_w = torch.meshgrid(qi_h, qi_w, indexing="ij")

    kw_offset = (dilation_w * (kernel_w - 1)) // 2
    kh_offset = (dilation_h * (kernel_h - 1)) // 2

    # ``input.dtype``, not fp32: gradcheck's eps=1e-6 is below fp32 resolution.
    coord_dtype = input.dtype
    p0_w_ = (qi_w * stride_w - pad_w).to(coord_dtype) + kw_offset * (1.0 - spread)
    p0_h_ = (qi_h * stride_h - pad_h).to(coord_dtype) + kh_offset * (1.0 - spread)

    offset = offset.view(N, H_OUT, W_OUT, groups, K, 2).to(coord_dtype)
    mask = mask.view(N, H_OUT, W_OUT, groups, K).to(coord_dtype)
    input_ = input.view(N, H_IN, W_IN, groups, group_channels).to(coord_dtype)

    output = torch.zeros(
        (N, H_OUT, W_OUT, groups, group_channels),
        device=input.device,
        dtype=coord_dtype,
    )

    n_idx = torch.arange(N, device=input.device).view(N, 1, 1, 1)
    g_idx = torch.arange(groups, device=input.device).view(1, 1, 1, groups)

    # Floor division: an even kernel's "center" is one cell past the midpoint.
    center_h = kernel_h // 2
    center_w = kernel_w // 2

    k_idx = -1
    for kw in range(kernel_w):
        for kh in range(kernel_h):
            if drop_center and kw == center_w and kh == center_h:
                continue
            # Structural position (compacted for the center), never in-bounds count.
            k_idx += 1

            # offset[..., 0] is x, offset[..., 1] is y
            loc_w = (
                p0_w_.view(1, H_OUT, W_OUT, 1)
                + (kw * dilation_w + offset[..., k_idx, 0]) * spread
            )
            loc_h = (
                p0_h_.view(1, H_OUT, W_OUT, 1)
                + (kh * dilation_h + offset[..., k_idx, 1]) * spread
            )

            # A location in (-1, extent) has at least one corner inside the input.
            valid_loc = (
                (loc_h > -1.0) & (loc_h < H_IN) & (loc_w > -1.0) & (loc_w < W_IN)
            )
            # No ``valid_loc.any()`` short-circuit: it would break the compiled graph.

            h_low = loc_h.floor().long()
            w_low = loc_w.floor().long()
            h_high = h_low + 1
            w_high = w_low + 1

            lh = loc_h - h_low
            lw = loc_w - w_low
            hh = 1.0 - lh
            hw = 1.0 - lw

            w1, w2, w3, w4 = hh * hw, hh * lw, lh * hw, lh * lw

            v1 = _gather(input_, n_idx, g_idx, h_low, w_low, H_IN, W_IN)
            v2 = _gather(input_, n_idx, g_idx, h_low, w_high, H_IN, W_IN)
            v3 = _gather(input_, n_idx, g_idx, h_high, w_low, H_IN, W_IN)
            v4 = _gather(input_, n_idx, g_idx, h_high, w_high, H_IN, W_IN)

            sampled = (
                w1.unsqueeze(-1) * v1
                + w2.unsqueeze(-1) * v2
                + w3.unsqueeze(-1) * v3
                + w4.unsqueeze(-1) * v4
            )
            sampled = sampled * valid_loc.unsqueeze(-1)

            output += sampled * mask[..., k_idx].unsqueeze(-1)

    return (
        output.view(N, H_OUT, W_OUT, groups * group_channels)
        .to(input.dtype)
        .contiguous()
    )
