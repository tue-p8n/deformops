"""Eager deformable convolution, v1 and v2, over channels-last operands.

Dispatch runs it wherever the Triton kernel cannot, so it must trace under
``torch.compile``: no ``.item()``, no ``.any()``, no branch on a tensor value.
Kernel-point order, offset layout, boundaries and offset groups follow
torchvision; see ``docs/conventions.md``.
"""

from __future__ import annotations

from typing import override

import torch
from torch import Tensor

from ...registry import ConvBackend, ConvContext

__all__ = ["Backend", "forward"]


def _bilinear_gather(
    flat: Tensor, y: Tensor, x: Tensor, height: int, width: int
) -> Tensor:
    """Sample ``flat`` (B, H*W, C) at fractional ``(y, x)`` (B, P) into (B, P, C).

    Each corner outside the input contributes zero on its own.
    """
    y0 = torch.floor(y)
    x0 = torch.floor(x)
    dy = (y - y0).unsqueeze(-1)
    dx = (x - x0).unsqueeze(-1)
    y0i = y0.long()
    x0i = x0.long()

    out = torch.zeros(
        (flat.shape[0], y.shape[1], flat.shape[2]), dtype=flat.dtype, device=flat.device
    )
    for oy, ox, wy, wx in (
        (0, 0, 1.0 - dy, 1.0 - dx),
        (0, 1, 1.0 - dy, dx),
        (1, 0, dy, 1.0 - dx),
        (1, 1, dy, dx),
    ):
        yy = y0i + oy
        xx = x0i + ox
        inside = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
        index = (yy.clamp(0, height - 1) * width + xx.clamp(0, width - 1)).unsqueeze(-1)
        corner = torch.gather(flat, 1, index.expand(-1, -1, flat.shape[2]))
        out = out + torch.where(inside.unsqueeze(-1), corner * wy * wx, 0.0)
    return out


def forward(  # noqa: PLR0913
    value: Tensor,
    offset: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    mask: Tensor | None,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dilation_h: int,
    dilation_w: int,
    groups: int,
    offset_groups: int,
) -> Tensor:
    """Deformable convolution over channels-last operands.

    ``value`` is ``(B, H, W, C_in)``, ``offset`` is
    ``(B, H_out, W_out, offset_groups, K, 2)`` holding ``(dy, dx)``, ``mask``
    is ``(B, H_out, W_out, offset_groups, K)`` or ``None`` for v1, and
    ``weight`` is ``(C_out, C_in // groups, kH, kW)``. Returns
    ``(B, H_out, W_out, C_out)``.
    """
    batch, height, width, in_channels = value.shape
    out_channels, in_per_group, _kernel_h, kernel_w = weight.shape
    # ``divmod`` needs a Python int; the per-point loops specialize on it anyway.
    kernel_w = int(kernel_w)
    _, out_h, out_w, _, points, _ = offset.shape
    positions = out_h * out_w
    per_offset_group = in_channels // offset_groups
    out_per_group = out_channels // groups

    compute_dtype = torch.promote_types(value.dtype, torch.float32)
    flat = value.reshape(batch, height * width, in_channels).to(compute_dtype)

    rows = torch.arange(out_h, device=value.device)
    cols = torch.arange(out_w, device=value.device)
    grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")
    base_y = (grid_y.reshape(-1) * stride_h - pad_h).to(compute_dtype)
    base_x = (grid_x.reshape(-1) * stride_w - pad_w).to(compute_dtype)

    offset_flat = offset.reshape(batch, positions, offset_groups, points, 2).to(
        compute_dtype
    )
    mask_flat = (
        mask.reshape(batch, positions, offset_groups, points).to(compute_dtype)
        if mask is not None
        else None
    )

    # [B, P, C_in, K]: every sample the contraction will consume.
    sampled = torch.zeros(
        (batch, positions, in_channels, points),
        dtype=compute_dtype,
        device=value.device,
    )
    for k in range(points):
        # Row-major kernel numbering, matching torchvision.
        kh, kw = divmod(k, kernel_w)
        for group in range(offset_groups):
            lo = group * per_offset_group
            hi = lo + per_offset_group
            y = base_y[None, :] + kh * dilation_h + offset_flat[:, :, group, k, 0]
            x = base_x[None, :] + kw * dilation_w + offset_flat[:, :, group, k, 1]
            gathered = _bilinear_gather(flat[:, :, lo:hi], y, x, height, width)
            if mask_flat is not None:
                gathered = gathered * mask_flat[:, :, group, k].unsqueeze(-1)
            sampled[:, :, lo:hi, k] = gathered

    weight_c = weight.reshape(groups, out_per_group, in_per_group * points).to(
        compute_dtype
    )
    out = torch.zeros(
        (batch, positions, out_channels), dtype=compute_dtype, device=value.device
    )
    for group in range(groups):
        cin_lo = group * in_per_group
        block = sampled[:, :, cin_lo : cin_lo + in_per_group, :].reshape(
            batch * positions, in_per_group * points
        )
        # ``mm``, not ``matmul``: export lowers ``matmul`` to ``bmm``, which rounds
        # differently.
        out[:, :, group * out_per_group : (group + 1) * out_per_group] = torch.mm(
            block, weight_c[group].transpose(0, 1)
        ).reshape(batch, positions, out_per_group)

    if bias is not None:
        out = out + bias.to(compute_dtype)
    return out.reshape(batch, out_h, out_w, out_channels).to(value.dtype)


class Backend(ConvBackend):
    """Dispatch adapter for the eager implementation."""

    @property
    @override
    def name(self) -> str:
        return "pytorch"

    @override
    def supports(self, context: ConvContext) -> bool:  # noqa: ARG002
        return True

    @override
    def forward(  # noqa: PLR0913
        self,
        value: Tensor,
        offset: Tensor,
        mask: Tensor | None,
        weight: Tensor,
        bias: Tensor | None,
        stride_h: int,
        stride_w: int,
        pad_h: int,
        pad_w: int,
        dilation_h: int,
        dilation_w: int,
        groups: int,
        offset_groups: int,
    ) -> Tensor:
        """Delegate to :func:`forward`, which documents the operands."""
        return forward(
            value,
            offset,
            weight,
            bias,
            mask,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            groups,
            offset_groups,
        )
