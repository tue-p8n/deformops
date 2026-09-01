# pyright: basic, reportMissingImports=false
r"""Point-parallel sampling kernels for multi-scale deformable attention.

One program per (query, point, head, batch), with no data reuse across points.
"""

from __future__ import annotations

import torch

from .....core.capabilities import require_triton

require_triton(__name__.rsplit(".", 1)[-1])

import triton
import triton.language as tl
from torch.library import wrap_triton

__all__ = ["backward_launch", "forward_launch"]


@triton.jit
def _msda_forward_kernel(
    value_ptr,
    shapes_ptr,
    level_start_index_ptr,
    loc_ptr,
    attn_ptr,
    output_ptr,
    batch_size,
    num_queries,
    num_heads,
    num_levels,
    num_points,
    dim_per_head,
    stride_vb,
    stride_vs,
    stride_vh,
    stride_vd,
    stride_lb,
    stride_lq,
    stride_lh,
    stride_ll,
    stride_lp,
    stride_lc,
    stride_ab,
    stride_aq,
    stride_ah,
    stride_al,
    stride_ap,
    stride_ob,
    stride_oq,
    stride_oh,
    stride_od,
    BLOCK_SIZE_D: tl.constexpr,
    aggregate: tl.constexpr,
):
    # Program IDs: Each program handles ONE (query, point, head, batch)
    qp_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)

    query_idx = qp_idx // num_points
    point_idx = qp_idx % num_points

    d_offsets = tl.arange(0, BLOCK_SIZE_D)
    d_mask = d_offsets < dim_per_head

    # Levels are few, so they loop inside the program instead of adding a launch axis.
    total_acc = tl.zeros([BLOCK_SIZE_D], dtype=tl.float32)

    if aggregate == 1:
        # Two passes: softmax over the L*P weights, then accumulate.
        attn_max = tl.full([], -float("inf"), dtype=tl.float32)
        for l in range(num_levels):
            for p in range(num_points):
                a_off_lp = (
                    batch_idx * stride_ab
                    + query_idx * stride_aq
                    + head_idx * stride_ah
                    + l * stride_al
                    + p * stride_ap
                )
                a_val = tl.load(attn_ptr + a_off_lp).to(tl.float32)
                attn_max = tl.maximum(attn_max, a_val)
        attn_sum = tl.zeros([], dtype=tl.float32)
        for l in range(num_levels):
            for p in range(num_points):
                a_off_lp = (
                    batch_idx * stride_ab
                    + query_idx * stride_aq
                    + head_idx * stride_ah
                    + l * stride_al
                    + p * stride_ap
                )
                a_val = tl.load(attn_ptr + a_off_lp).to(tl.float32)
                attn_sum += tl.exp(a_val - attn_max)

        for l in range(num_levels):
            h = tl.load(shapes_ptr + l * 2)
            w = tl.load(shapes_ptr + l * 2 + 1)
            start_idx = tl.load(level_start_index_ptr + l)

            l_off = (
                batch_idx * stride_lb
                + query_idx * stride_lq
                + head_idx * stride_lh
                + l * stride_ll
                + point_idx * stride_lp
            )
            lx = tl.load(loc_ptr + l_off + 0 * stride_lc).to(tl.float32)
            ly = tl.load(loc_ptr + l_off + 1 * stride_lc).to(tl.float32)

            a_off = (
                batch_idx * stride_ab
                + query_idx * stride_aq
                + head_idx * stride_ah
                + l * stride_al
                + point_idx * stride_ap
            )
            attn_raw = tl.load(attn_ptr + a_off).to(tl.float32)
            attention_weights = tl.exp(attn_raw - attn_max) / attn_sum

            px, py = lx * w - 0.5, ly * h - 0.5
            x0 = tl.math.floor(px.to(tl.float32)).to(tl.int32)
            y0 = tl.math.floor(py.to(tl.float32)).to(tl.int32)
            x1, y1 = x0 + 1, y0 + 1

            wx1, wy1 = (px - x0).to(tl.float32), (py - y0).to(tl.float32)
            wx0, wy0 = 1.0 - wx1, 1.0 - wy1

            v_base = value_ptr + batch_idx * stride_vb + head_idx * stride_vh

            m00 = (x0 >= 0) & (x0 < w) & (y0 >= 0) & (y0 < h)
            v00 = tl.load(
                v_base + (start_idx + y0 * w + x0) * stride_vs + d_offsets * stride_vd,
                mask=m00 & d_mask,
                other=0.0,
            )

            m01 = (x1 >= 0) & (x1 < w) & (y0 >= 0) & (y0 < h)
            v01 = tl.load(
                v_base + (start_idx + y0 * w + x1) * stride_vs + d_offsets * stride_vd,
                mask=m01 & d_mask,
                other=0.0,
            )

            m10 = (x0 >= 0) & (x0 < w) & (y1 >= 0) & (y1 < h)
            v10 = tl.load(
                v_base + (start_idx + y1 * w + x0) * stride_vs + d_offsets * stride_vd,
                mask=m10 & d_mask,
                other=0.0,
            )

            m11 = (x1 >= 0) & (x1 < w) & (y1 >= 0) & (y1 < h)
            v11 = tl.load(
                v_base + (start_idx + y1 * w + x1) * stride_vs + d_offsets * stride_vd,
                mask=m11 & d_mask,
                other=0.0,
            )

            total_acc += attention_weights * (
                wx0 * wy0 * v00 + wx1 * wy0 * v01 + wx0 * wy1 * v10 + wx1 * wy1 * v11
            )
    else:
        for l in range(num_levels):
            h = tl.load(shapes_ptr + l * 2)
            w = tl.load(shapes_ptr + l * 2 + 1)
            start_idx = tl.load(level_start_index_ptr + l)

            l_off = (
                batch_idx * stride_lb
                + query_idx * stride_lq
                + head_idx * stride_lh
                + l * stride_ll
                + point_idx * stride_lp
            )
            lx = tl.load(loc_ptr + l_off + 0 * stride_lc).to(tl.float32)
            ly = tl.load(loc_ptr + l_off + 1 * stride_lc).to(tl.float32)

            a_off = (
                batch_idx * stride_ab
                + query_idx * stride_aq
                + head_idx * stride_ah
                + l * stride_al
                + point_idx * stride_ap
            )
            attention_weights = tl.load(attn_ptr + a_off)

            px, py = lx * w - 0.5, ly * h - 0.5
            x0 = tl.math.floor(px.to(tl.float32)).to(tl.int32)
            y0 = tl.math.floor(py.to(tl.float32)).to(tl.int32)
            x1, y1 = x0 + 1, y0 + 1

            wx1, wy1 = (px - x0).to(tl.float32), (py - y0).to(tl.float32)
            wx0, wy0 = 1.0 - wx1, 1.0 - wy1

            v_base = value_ptr + batch_idx * stride_vb + head_idx * stride_vh

            m00 = (x0 >= 0) & (x0 < w) & (y0 >= 0) & (y0 < h)
            v00 = tl.load(
                v_base + (start_idx + y0 * w + x0) * stride_vs + d_offsets * stride_vd,
                mask=m00 & d_mask,
                other=0.0,
            )

            m01 = (x1 >= 0) & (x1 < w) & (y0 >= 0) & (y0 < h)
            v01 = tl.load(
                v_base + (start_idx + y0 * w + x1) * stride_vs + d_offsets * stride_vd,
                mask=m01 & d_mask,
                other=0.0,
            )

            m10 = (x0 >= 0) & (x0 < w) & (y1 >= 0) & (y1 < h)
            v10 = tl.load(
                v_base + (start_idx + y1 * w + x0) * stride_vs + d_offsets * stride_vd,
                mask=m10 & d_mask,
                other=0.0,
            )

            m11 = (x1 >= 0) & (x1 < w) & (y1 >= 0) & (y1 < h)
            v11 = tl.load(
                v_base + (start_idx + y1 * w + x1) * stride_vs + d_offsets * stride_vd,
                mask=m11 & d_mask,
                other=0.0,
            )

            total_acc += attention_weights * (
                wx0 * wy0 * v00 + wx1 * wy0 * v01 + wx0 * wy1 * v10 + wx1 * wy1 * v11
            )

    # Atomic add: several points contribute to the same output query-head.
    out_off = (
        batch_idx * stride_ob
        + query_idx * stride_oq
        + head_idx * stride_oh
        + d_offsets * stride_od
    )
    tl.atomic_add(
        output_ptr + out_off, total_acc.to(output_ptr.dtype.element_ty), mask=d_mask
    )


@triton.jit
def _msda_backward_kernel(
    value_ptr,
    shapes_ptr,
    level_start_index_ptr,
    loc_ptr,
    attn_ptr,
    grad_output_ptr,
    grad_value_ptr,
    grad_loc_ptr,
    grad_attn_ptr,
    batch_size,
    num_queries,
    num_heads,
    num_levels,
    num_points,
    dim_per_head,
    stride_vb,
    stride_vs,
    stride_vh,
    stride_vd,
    stride_lb,
    stride_lq,
    stride_lh,
    stride_ll,
    stride_lp,
    stride_lc,
    stride_ab,
    stride_aq,
    stride_ah,
    stride_al,
    stride_ap,
    stride_ob,
    stride_oq,
    stride_oh,
    stride_od,
    stride_gvb,
    stride_gvs,
    stride_gvh,
    stride_gvd,
    stride_glb,
    stride_glq,
    stride_glh,
    stride_gll,
    stride_glp,
    stride_glc,
    stride_gab,
    stride_gaq,
    stride_gah,
    stride_gal,
    stride_gap,
    BLOCK_SIZE_D: tl.constexpr,
):
    qp_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)

    query_idx = qp_idx // num_points
    point_idx = qp_idx % num_points

    d_offsets = tl.arange(0, BLOCK_SIZE_D)
    d_mask = d_offsets < dim_per_head

    g_out_off = (
        batch_idx * stride_ob
        + query_idx * stride_oq
        + head_idx * stride_oh
        + d_offsets * stride_od
    )
    grad_out = tl.load(grad_output_ptr + g_out_off, mask=d_mask, other=0.0)

    # Bilinear weights and corner reads, shared by all three gradients.
    for l in range(num_levels):
        h = tl.load(shapes_ptr + l * 2)
        w = tl.load(shapes_ptr + l * 2 + 1)
        start_idx = tl.load(level_start_index_ptr + l)

        l_off = (
            batch_idx * stride_lb
            + query_idx * stride_lq
            + head_idx * stride_lh
            + l * stride_ll
            + point_idx * stride_lp
        )
        lx = tl.load(loc_ptr + l_off + 0 * stride_lc).to(tl.float32)
        ly = tl.load(loc_ptr + l_off + 1 * stride_lc).to(tl.float32)

        a_off = (
            batch_idx * stride_ab
            + query_idx * stride_aq
            + head_idx * stride_ah
            + l * stride_al
            + point_idx * stride_ap
        )
        attention_weights = tl.load(attn_ptr + a_off)

        px, py = lx * w - 0.5, ly * h - 0.5
        x0 = tl.math.floor(px.to(tl.float32)).to(tl.int32)
        y0 = tl.math.floor(py.to(tl.float32)).to(tl.int32)
        x1, y1 = x0 + 1, y0 + 1

        wx1, wy1 = (px - x0).to(tl.float32), (py - y0).to(tl.float32)
        wx0, wy0 = 1.0 - wx1, 1.0 - wy1

        v_base = value_ptr + batch_idx * stride_vb + head_idx * stride_vh
        m00 = (x0 >= 0) & (x0 < w) & (y0 >= 0) & (y0 < h)
        m01 = (x1 >= 0) & (x1 < w) & (y0 >= 0) & (y0 < h)
        m10 = (x0 >= 0) & (x0 < w) & (y1 >= 0) & (y1 < h)
        m11 = (x1 >= 0) & (x1 < w) & (y1 >= 0) & (y1 < h)

        v00 = tl.load(
            v_base + (start_idx + y0 * w + x0) * stride_vs + d_offsets * stride_vd,
            mask=m00 & d_mask,
            other=0.0,
        )
        v01 = tl.load(
            v_base + (start_idx + y0 * w + x1) * stride_vs + d_offsets * stride_vd,
            mask=m01 & d_mask,
            other=0.0,
        )
        v10 = tl.load(
            v_base + (start_idx + y1 * w + x0) * stride_vs + d_offsets * stride_vd,
            mask=m10 & d_mask,
            other=0.0,
        )
        v11 = tl.load(
            v_base + (start_idx + y1 * w + x1) * stride_vs + d_offsets * stride_vd,
            mask=m11 & d_mask,
            other=0.0,
        )

        interp = wx0 * wy0 * v00 + wx1 * wy0 * v01 + wx0 * wy1 * v10 + wx1 * wy1 * v11
        d_attn = tl.sum(interp * grad_out)
        tl.atomic_add(
            grad_attn_ptr
            + batch_idx * stride_gab
            + query_idx * stride_gaq
            + head_idx * stride_gah
            + l * stride_gal
            + point_idx * stride_gap,
            d_attn,
        )

        weighted_grad = grad_out * attention_weights
        gv_base = grad_value_ptr + batch_idx * stride_gvb + head_idx * stride_gvh
        tl.atomic_add(
            gv_base + (start_idx + y0 * w + x0) * stride_gvs + d_offsets * stride_gvd,
            (weighted_grad * wx0 * wy0).to(grad_value_ptr.dtype.element_ty),
            mask=m00 & d_mask,
        )
        tl.atomic_add(
            gv_base + (start_idx + y0 * w + x1) * stride_gvs + d_offsets * stride_gvd,
            (weighted_grad * wx1 * wy0).to(grad_value_ptr.dtype.element_ty),
            mask=m01 & d_mask,
        )
        tl.atomic_add(
            gv_base + (start_idx + y1 * w + x0) * stride_gvs + d_offsets * stride_gvd,
            (weighted_grad * wx0 * wy1).to(grad_value_ptr.dtype.element_ty),
            mask=m10 & d_mask,
        )
        tl.atomic_add(
            gv_base + (start_idx + y1 * w + x1) * stride_gvs + d_offsets * stride_gvd,
            (weighted_grad * wx1 * wy1).to(grad_value_ptr.dtype.element_ty),
            mask=m11 & d_mask,
        )

        d_px = wy0 * (v01 - v00) + wy1 * (v11 - v10)
        d_py = wx0 * (v10 - v00) + wx1 * (v11 - v01)
        d_lx = tl.sum(d_px * grad_out) * attention_weights * w
        d_ly = tl.sum(d_py * grad_out) * attention_weights * h
        gloc_off = (
            batch_idx * stride_glb
            + query_idx * stride_glq
            + head_idx * stride_glh
            + l * stride_gll
            + point_idx * stride_glp
        )
        tl.atomic_add(
            grad_loc_ptr + gloc_off + 0 * stride_glc,
            d_lx.to(grad_loc_ptr.dtype.element_ty),
        )
        tl.atomic_add(
            grad_loc_ptr + gloc_off + 1 * stride_glc,
            d_ly.to(grad_loc_ptr.dtype.element_ty),
        )


def forward_launch(
    value: torch.Tensor,
    spatial_shapes: torch.Tensor,
    level_start_index: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    aggregate: int,
) -> torch.Tensor:
    """Forward launch returning ``(B, Q, H, D)``; assumes ``Q > 0``, not float64."""
    B, _S, H, D = value.shape
    _, Q, _, L, P, _ = sampling_locations.shape
    # The kernels index weights as 5-D; callers may pass (B, Q, H, L * P).
    attention_weights = attention_weights.view(B, Q, H, L, P)
    output = torch.zeros((B, Q, H, D), device=value.device, dtype=value.dtype)
    grid = (Q * P, H, B)
    wrap_triton(_msda_forward_kernel)[grid](
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
        BLOCK_SIZE_D=triton.next_power_of_2(D),
        aggregate=aggregate,  # pyright: ignore[reportArgumentType]  # tl.constexpr takes the value at launch
    )
    return output


def backward_launch(
    grad_output: torch.Tensor,
    value: torch.Tensor,
    spatial_shapes: torch.Tensor,
    level_start_index: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Linear backward; ``grad_output`` is ``(B, Q, H, D)``, ``grad_attn`` 5-D."""
    B, _S, H, D = value.shape
    _, Q, _, L, P, _ = sampling_locations.shape
    attention_weights = attention_weights.view(B, Q, H, L, P)
    grad_value = torch.zeros_like(value)
    grad_loc = torch.zeros_like(sampling_locations)
    grad_attn = torch.zeros_like(attention_weights)
    grid = (Q * P, H, B)
    wrap_triton(_msda_backward_kernel)[grid](
        value,
        spatial_shapes,
        level_start_index,
        sampling_locations,
        attention_weights,
        grad_output,
        grad_value,
        grad_loc,
        grad_attn,
        B,
        Q,
        H,
        L,
        P,
        D,
        *value.stride(),
        *sampling_locations.stride(),
        *attention_weights.stride(),
        *grad_output.stride(),
        *grad_value.stride(),
        *grad_loc.stride(),
        *grad_attn.stride(),
        BLOCK_SIZE_D=triton.next_power_of_2(D),
    )
    return grad_value, grad_loc, grad_attn
