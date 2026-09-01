# pyright: basic, reportMissingImports=false
r"""Output-pixel-tiled Triton kernels for single-scale deformable sampling.

The multi-scale query-tiled kernel with one feature map and output pixels as
queries. Its sampling coordinates come from the convolution geometry, whose grid
order, center and boundary must match the reference; see ``docs/conventions.md``.
"""

from __future__ import annotations

import torch

from ....core.capabilities import require_triton

require_triton(__name__.rsplit(".", 1)[-1])

import triton
import triton.language as tl
from torch.library import wrap_triton

__all__ = ["backward_launch", "forward_launch", "grid_points", "grid_table"]


#: Output positions per program, shared by the forward and the backward.
_BLOCK_SIZE_Q = 32

#: Forward tile elements per warp; a ratio, since the tile grows with channels.
_FORWARD_ELEMENTS_PER_WARP = 512

#: Backward tile elements per warp, fewer: the scattered accumulation wants many
#: warps in flight.
_BACKWARD_ELEMENTS_PER_WARP = 64


def _backward_num_warps(block_q: int, block_d: int) -> int:
    """Warps for a backward tile, clamped to what a launch can take."""
    warps = (block_q * block_d) // _BACKWARD_ELEMENTS_PER_WARP
    return max(4, min(warps, 32))


def _forward_num_warps(block_q: int, block_d: int) -> int:
    """Warps for a forward tile, clamped to what a launch can take."""
    warps = (block_q * block_d) // _FORWARD_ELEMENTS_PER_WARP
    return max(2, min(warps, 16))


def grid_points(
    kernel_h: int, kernel_w: int, *, drop_center: bool
) -> list[tuple[int, int]]:
    """The kernel grid in reference order: width-outer, center dropped on request.

    Public so tests can check it; a wrong order still gives well-formed output.

    Parameters
    ----------
    kernel_h, kernel_w
        Sampling grid extent.
    drop_center
        Whether to drop the kernel's center point from the sampling grid.

    Returns
    -------
    ``(kh, kw)`` per point, in walk order.
    """
    center_h, center_w = kernel_h // 2, kernel_w // 2
    points: list[tuple[int, int]] = []
    for kw in range(kernel_w):
        for kh in range(kernel_h):
            if drop_center and kw == center_w and kh == center_h:
                continue
            points.append((kh, kw))
    return points


#: Grid tables by geometry and device; a per-call build would be a host-to-device
#: copy that CUDA-graph capture cannot record.
_GRID_TABLES: dict[tuple[int, int, int, int, bool, torch.device], torch.Tensor] = {}


def grid_table(
    kernel_h: int,
    kernel_w: int,
    dilation_h: int,
    dilation_w: int,
    *,
    drop_center: bool,
    device: torch.device,
) -> torch.Tensor:
    """``(K, 2)`` of dilated grid displacements, indexed by point, cached.

    Keeps the grid order defined once and extents and dilation out of the
    kernel's constants. The kernel only reads the shared tensor.

    Parameters
    ----------
    kernel_h, kernel_w
        Sampling grid extent.
    dilation_h, dilation_w
        Dilation of the sampling grid.
    drop_center
        Whether to drop the kernel's center point from the sampling grid.
    device
        Where the returned table lives.

    Returns
    -------
    ``(K, 2)`` of dilated grid displacements, indexed by point.
    """
    key = (kernel_h, kernel_w, dilation_h, dilation_w, drop_center, device)
    cached = _GRID_TABLES.get(key)
    if cached is not None:
        return cached
    points = grid_points(kernel_h, kernel_w, drop_center=drop_center)
    table = torch.tensor(
        [(kh * dilation_h, kw * dilation_w) for kh, kw in points],
        dtype=torch.float32,
        device=device,
    )
    _GRID_TABLES[key] = table
    return table


@triton.jit
def _attn_forward_kernel(
    value_ptr,
    offset_mask_ptr,
    grid_ptr,
    output_ptr,
    height_in,
    width_in,
    width_out,
    num_positions,
    stride_vn,
    stride_vh,
    stride_vw,
    stride_vc,
    stride_on,
    stride_oq,
    stride_og,
    stride_ok,
    stride_oc,
    stride_yn,
    stride_yq,
    stride_yg,
    stride_yd,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    grid_center_h,
    grid_center_w,
    spread,
    group_dims,
    num_points: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """One block of output positions, for one group of one batch element."""
    q_block = tl.program_id(0)
    group_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)

    q_offsets = q_block * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    q_mask = q_offsets < num_positions
    d_offsets = tl.arange(0, BLOCK_SIZE_D)
    d_mask = d_offsets < group_dims

    out_h = q_offsets // width_out
    out_w = q_offsets % width_out

    # The centering term compensates for ``spread`` rescaling the grid.
    p0_h = (out_h * stride_h - pad_h).to(tl.float32) + grid_center_h * (1.0 - spread)
    p0_w = (out_w * stride_w - pad_w).to(tl.float32) + grid_center_w * (1.0 - spread)

    value_base = value_ptr + batch_idx * stride_vn + group_idx * group_dims * stride_vc
    acc = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_D), dtype=tl.float32)

    for point_idx in tl.static_range(num_points):  # pyright: ignore[reportGeneralTypeIssues]  # Triton loop
        grid_h = tl.load(grid_ptr + point_idx * 2 + 0)
        grid_w = tl.load(grid_ptr + point_idx * 2 + 1)
        base = (
            offset_mask_ptr
            + batch_idx * stride_on
            + q_offsets * stride_oq
            + group_idx * stride_og
            + point_idx * stride_ok
        )
        off_x = tl.load(base + 0 * stride_oc, mask=q_mask, other=0.0).to(tl.float32)
        off_y = tl.load(base + 1 * stride_oc, mask=q_mask, other=0.0).to(tl.float32)
        weight = tl.load(base + 2 * stride_oc, mask=q_mask, other=0.0).to(tl.float32)

        loc_w = p0_w + (grid_w + off_x) * spread
        loc_h = p0_h + (grid_h + off_y) * spread

        in_bounds = (
            (loc_h > -1.0)
            & (loc_h < height_in)
            & (loc_w > -1.0)
            & (loc_w < width_in)
            & q_mask
        )

        y0 = tl.math.floor(loc_h).to(tl.int32)
        x0 = tl.math.floor(loc_w).to(tl.int32)
        y1, x1 = y0 + 1, x0 + 1
        wy1 = loc_h - y0
        wx1 = loc_w - x0
        wy0 = 1.0 - wy1
        wx0 = 1.0 - wx1

        m00 = (y0 >= 0) & (y0 < height_in) & (x0 >= 0) & (x0 < width_in) & in_bounds
        m01 = (y0 >= 0) & (y0 < height_in) & (x1 >= 0) & (x1 < width_in) & in_bounds
        m10 = (y1 >= 0) & (y1 < height_in) & (x0 >= 0) & (x0 < width_in) & in_bounds
        m11 = (y1 >= 0) & (y1 < height_in) & (x1 >= 0) & (x1 < width_in) & in_bounds

        # Written out: Triton admits calls to jitted functions only.
        c_d = d_offsets[None, :] * stride_vc
        o00 = y0[:, None] * stride_vh + x0[:, None] * stride_vw + c_d
        o01 = y0[:, None] * stride_vh + x1[:, None] * stride_vw + c_d
        o10 = y1[:, None] * stride_vh + x0[:, None] * stride_vw + c_d
        o11 = y1[:, None] * stride_vh + x1[:, None] * stride_vw + c_d

        full00 = m00[:, None] & d_mask[None, :]
        full01 = m01[:, None] & d_mask[None, :]
        full10 = m10[:, None] & d_mask[None, :]
        full11 = m11[:, None] & d_mask[None, :]

        # No eviction hint on value loads: one slows the kernel on Ampere.
        v00 = tl.load(value_base + o00, mask=full00, other=0.0).to(tl.float32)
        v01 = tl.load(value_base + o01, mask=full01, other=0.0).to(tl.float32)
        v10 = tl.load(value_base + o10, mask=full10, other=0.0).to(tl.float32)
        v11 = tl.load(value_base + o11, mask=full11, other=0.0).to(tl.float32)

        sampled = (
            wy0[:, None] * wx0[:, None] * v00
            + wy0[:, None] * wx1[:, None] * v01
            + wy1[:, None] * wx0[:, None] * v10
            + wy1[:, None] * wx1[:, None] * v11
        )
        acc += tl.where(in_bounds, weight, 0.0)[:, None] * sampled

    out_offsets = (
        output_ptr
        + batch_idx * stride_yn
        + q_offsets[:, None] * stride_yq
        + group_idx * stride_yg
        + d_offsets[None, :] * stride_yd
    )
    tl.store(
        out_offsets,
        acc.to(output_ptr.dtype.element_ty),
        mask=q_mask[:, None] & d_mask[None, :],
    )


@triton.jit
def _attn_backward_kernel(
    value_ptr,
    offset_mask_ptr,
    grid_ptr,
    grad_output_ptr,
    grad_value_ptr,
    grad_offset_mask_ptr,
    height_in,
    width_in,
    width_out,
    num_positions,
    stride_vn,
    stride_vh,
    stride_vw,
    stride_vc,
    stride_on,
    stride_oq,
    stride_og,
    stride_ok,
    stride_oc,
    stride_gn,
    stride_gq,
    stride_gg,
    stride_gd,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    grid_center_h,
    grid_center_w,
    spread,
    group_dims,
    num_points: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """Gradients for the input, the offsets and the modulation mask.

    Only the input gradient scatters, atomically; offset and mask gradients are
    stores, one output position each.
    """
    q_block = tl.program_id(0)
    group_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)

    q_offsets = q_block * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    q_mask = q_offsets < num_positions
    d_offsets = tl.arange(0, BLOCK_SIZE_D)
    d_mask = d_offsets < group_dims

    out_h = q_offsets // width_out
    out_w = q_offsets % width_out

    p0_h = (out_h * stride_h - pad_h).to(tl.float32) + grid_center_h * (1.0 - spread)
    p0_w = (out_w * stride_w - pad_w).to(tl.float32) + grid_center_w * (1.0 - spread)

    value_base = value_ptr + batch_idx * stride_vn + group_idx * group_dims * stride_vc
    grad_value_base = (
        grad_value_ptr + batch_idx * stride_vn + group_idx * group_dims * stride_vc
    )

    grad_out = tl.load(
        grad_output_ptr
        + batch_idx * stride_gn
        + q_offsets[:, None] * stride_gq
        + group_idx * stride_gg
        + d_offsets[None, :] * stride_gd,
        mask=q_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    for point_idx in tl.static_range(num_points):  # pyright: ignore[reportGeneralTypeIssues]  # Triton loop
        grid_h = tl.load(grid_ptr + point_idx * 2 + 0)
        grid_w = tl.load(grid_ptr + point_idx * 2 + 1)
        base = (
            offset_mask_ptr
            + batch_idx * stride_on
            + q_offsets * stride_oq
            + group_idx * stride_og
            + point_idx * stride_ok
        )
        off_x = tl.load(base + 0 * stride_oc, mask=q_mask, other=0.0).to(tl.float32)
        off_y = tl.load(base + 1 * stride_oc, mask=q_mask, other=0.0).to(tl.float32)
        weight = tl.load(base + 2 * stride_oc, mask=q_mask, other=0.0).to(tl.float32)

        loc_w = p0_w + (grid_w + off_x) * spread
        loc_h = p0_h + (grid_h + off_y) * spread

        in_bounds = (
            (loc_h > -1.0)
            & (loc_h < height_in)
            & (loc_w > -1.0)
            & (loc_w < width_in)
            & q_mask
        )

        y0 = tl.math.floor(loc_h).to(tl.int32)
        x0 = tl.math.floor(loc_w).to(tl.int32)
        y1, x1 = y0 + 1, x0 + 1
        wy1 = loc_h - y0
        wx1 = loc_w - x0
        wy0 = 1.0 - wy1
        wx0 = 1.0 - wx1

        m00 = (y0 >= 0) & (y0 < height_in) & (x0 >= 0) & (x0 < width_in) & in_bounds
        m01 = (y0 >= 0) & (y0 < height_in) & (x1 >= 0) & (x1 < width_in) & in_bounds
        m10 = (y1 >= 0) & (y1 < height_in) & (x0 >= 0) & (x0 < width_in) & in_bounds
        m11 = (y1 >= 0) & (y1 < height_in) & (x1 >= 0) & (x1 < width_in) & in_bounds

        c_d = d_offsets[None, :] * stride_vc
        o00 = y0[:, None] * stride_vh + x0[:, None] * stride_vw + c_d
        o01 = y0[:, None] * stride_vh + x1[:, None] * stride_vw + c_d
        o10 = y1[:, None] * stride_vh + x0[:, None] * stride_vw + c_d
        o11 = y1[:, None] * stride_vh + x1[:, None] * stride_vw + c_d

        full00 = m00[:, None] & d_mask[None, :]
        full01 = m01[:, None] & d_mask[None, :]
        full10 = m10[:, None] & d_mask[None, :]
        full11 = m11[:, None] & d_mask[None, :]

        # No eviction hint, for the reason given in the forward.
        v00 = tl.load(value_base + o00, mask=full00, other=0.0).to(tl.float32)
        v01 = tl.load(value_base + o01, mask=full01, other=0.0).to(tl.float32)
        v10 = tl.load(value_base + o10, mask=full10, other=0.0).to(tl.float32)
        v11 = tl.load(value_base + o11, mask=full11, other=0.0).to(tl.float32)

        sampled = (
            wy0[:, None] * wx0[:, None] * v00
            + wy0[:, None] * wx1[:, None] * v01
            + wy1[:, None] * wx0[:, None] * v10
            + wy1[:, None] * wx1[:, None] * v11
        )

        gated_weight = tl.where(in_bounds, weight, 0.0)[:, None]
        scaled = gated_weight * grad_out

        # Input gradient over the four corners, with relaxed ordering: the
        # accumulations commute and nothing reads them until the kernel ends.
        tl.atomic_add(
            grad_value_base + o00,
            (wy0[:, None] * wx0[:, None] * scaled).to(grad_value_ptr.dtype.element_ty),
            mask=full00,
            sem="relaxed",
        )
        tl.atomic_add(
            grad_value_base + o01,
            (wy0[:, None] * wx1[:, None] * scaled).to(grad_value_ptr.dtype.element_ty),
            mask=full01,
            sem="relaxed",
        )
        tl.atomic_add(
            grad_value_base + o10,
            (wy1[:, None] * wx0[:, None] * scaled).to(grad_value_ptr.dtype.element_ty),
            mask=full10,
            sem="relaxed",
        )
        tl.atomic_add(
            grad_value_base + o11,
            (wy1[:, None] * wx1[:, None] * scaled).to(grad_value_ptr.dtype.element_ty),
            mask=full11,
            sem="relaxed",
        )

        # Coordinate gradients, reduced over channels and scaled by spread.
        d_dx = wy0[:, None] * (v01 - v00) + wy1[:, None] * (v11 - v10)
        d_dy = wx0[:, None] * (v10 - v00) + wx1[:, None] * (v11 - v01)
        grad_x = tl.sum(scaled * d_dx, axis=1) * spread
        grad_y = tl.sum(scaled * d_dy, axis=1) * spread
        grad_weight = tl.sum(grad_out * sampled, axis=1)

        grad_base = (
            grad_offset_mask_ptr
            + batch_idx * stride_on
            + q_offsets * stride_oq
            + group_idx * stride_og
            + point_idx * stride_ok
        )
        store_mask = q_mask & in_bounds
        tl.store(
            grad_base + 0 * stride_oc,
            grad_x.to(grad_offset_mask_ptr.dtype.element_ty),
            mask=store_mask,
        )
        tl.store(
            grad_base + 1 * stride_oc,
            grad_y.to(grad_offset_mask_ptr.dtype.element_ty),
            mask=store_mask,
        )
        tl.store(
            grad_base + 2 * stride_oc,
            grad_weight.to(grad_offset_mask_ptr.dtype.element_ty),
            mask=store_mask,
        )


# ---------------------------------------------------------------------------
# Launch helpers
# ---------------------------------------------------------------------------


def _geometry(
    value: torch.Tensor,
    offset_mask: torch.Tensor,
    kernel_h: int,
    kernel_w: int,
    dilation_h: int,
    dilation_w: int,
    groups: int,
    group_dims: int,
    *,
    drop_center: bool,
) -> tuple:
    """Shapes, the grid table and the launch grid, shared by both directions."""
    batch, height_in, width_in, _ = value.shape
    _, out_h, out_w, _ = offset_mask.shape
    num_points = kernel_h * kernel_w - (1 if drop_center else 0)
    positions = out_h * out_w
    table = grid_table(
        kernel_h,
        kernel_w,
        dilation_h,
        dilation_w,
        drop_center=drop_center,
        device=value.device,
    )
    # [N, Q, G, K, 3] of (offset_x, offset_y, weight), sliced from a wider tensor.
    packed = offset_mask[..., : groups * num_points * 3].reshape(
        batch, positions, groups, num_points, 3
    )
    grid = (triton.cdiv(positions, _BLOCK_SIZE_Q), groups, batch)
    return (
        batch,
        height_in,
        width_in,
        out_h,
        out_w,
        positions,
        num_points,
        table,
        packed,
        grid,
        triton.next_power_of_2(group_dims),
    )


def forward_launch(
    value: torch.Tensor,
    offset_mask: torch.Tensor,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dilation_h: int,
    dilation_w: int,
    groups: int,
    group_dims: int,
    spread: float,
    *,
    drop_center: bool,
) -> torch.Tensor:
    """Run the forward kernel and return ``(N, H_out, W_out, G * group_dims)``."""
    (
        batch,
        height_in,
        width_in,
        out_h,
        out_w,
        positions,
        num_points,
        table,
        packed,
        grid,
        block_d,
    ) = _geometry(
        value,
        offset_mask,
        kernel_h,
        kernel_w,
        dilation_h,
        dilation_w,
        groups,
        group_dims,
        drop_center=drop_center,
    )
    # ``empty``: the launch grid and its masks store every element once.
    output = torch.empty(
        (batch, positions, groups, group_dims),
        device=value.device,
        dtype=value.dtype,
    )
    wrap_triton(_attn_forward_kernel)[grid](
        value,
        packed,
        table,
        output,
        height_in,
        width_in,
        out_w,
        positions,
        *value.stride(),
        *packed.stride(),
        *output.stride(),
        # By keyword: after the unpacked stride tuples, a positional tail reads as
        # an argument-count mismatch.
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        grid_center_h=(dilation_h * (kernel_h - 1)) // 2,
        grid_center_w=(dilation_w * (kernel_w - 1)) // 2,
        spread=spread,
        group_dims=group_dims,
        num_points=num_points,  # pyright: ignore[reportArgumentType]  # Triton constexpr
        BLOCK_SIZE_Q=_BLOCK_SIZE_Q,  # pyright: ignore[reportArgumentType]
        BLOCK_SIZE_D=block_d,  # pyright: ignore[reportArgumentType]
        num_warps=_forward_num_warps(_BLOCK_SIZE_Q, block_d),  # pyright: ignore[reportCallIssue]
    )
    return output.reshape(batch, out_h, out_w, groups * group_dims)


def backward_launch(
    grad_output: torch.Tensor,
    value: torch.Tensor,
    offset_mask: torch.Tensor,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dilation_h: int,
    dilation_w: int,
    groups: int,
    group_dims: int,
    spread: float,
    *,
    drop_center: bool,
    fp32acc: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the backward kernel and return ``(grad_value, grad_offset_mask)``.

    ``fp32acc`` accumulates the input gradient in float32; see
    ``docs/precision.md``.
    """
    (
        batch,
        height_in,
        width_in,
        _out_h,
        _out_w,
        positions,
        num_points,
        table,
        packed,
        grid,
        block_d,
    ) = _geometry(
        value,
        offset_mask,
        kernel_h,
        kernel_w,
        dilation_h,
        dilation_w,
        groups,
        group_dims,
        drop_center=drop_center,
    )
    accum_dtype = torch.float32 if fp32acc else value.dtype
    grad_value = torch.zeros_like(value, dtype=accum_dtype)

    # The kernel indexes ``grad_packed`` with ``packed.stride()``. Slicing a zeroed
    # ``offset_mask`` like ``packed`` keeps the strides equal when it is wider;
    # ``zeros_like(packed)`` would be contiguous and overrun.
    used = groups * num_points * 3
    grad_offset_mask = torch.zeros_like(offset_mask)
    grad_packed = grad_offset_mask[..., :used].reshape(
        batch, positions, groups, num_points, 3
    )
    grad_output = grad_output.reshape(batch, positions, groups, group_dims)

    wrap_triton(_attn_backward_kernel)[grid](
        value,
        packed,
        table,
        grad_output,
        grad_value,
        grad_packed,
        height_in,
        width_in,
        _out_w,
        positions,
        *value.stride(),
        *packed.stride(),
        *grad_output.stride(),
        # By keyword: after the unpacked stride tuples, a positional tail reads as
        # an argument-count mismatch.
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        grid_center_h=(dilation_h * (kernel_h - 1)) // 2,
        grid_center_w=(dilation_w * (kernel_w - 1)) // 2,
        spread=spread,
        group_dims=group_dims,
        num_points=num_points,  # pyright: ignore[reportArgumentType]  # Triton constexpr
        BLOCK_SIZE_Q=_BLOCK_SIZE_Q,  # pyright: ignore[reportArgumentType]
        BLOCK_SIZE_D=block_d,  # pyright: ignore[reportArgumentType]
        num_warps=_backward_num_warps(_BLOCK_SIZE_Q, block_d),  # pyright: ignore[reportCallIssue]
    )

    # A no-op without the float32 accumulator.
    return grad_value.to(value.dtype), grad_offset_mask
