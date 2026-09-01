# pyright: basic, reportMissingImports=false
r"""Query-tiled sampling kernels that cache their online-softmax statistics.

The forward also writes the running max and sum-of-exp, so the softmax backward
rebuilds the weights as ``exp(a - m) / z``. See ``docs/kernels.md``.
"""

from __future__ import annotations

import torch

from .....core.capabilities import require_triton

require_triton(__name__.rsplit(".", 1)[-1])

import triton
import triton.language as tl
from torch.library import wrap_triton

__all__ = ["atomic_backward_launch", "float32_backward_launch", "forward_launch"]


@triton.jit
def _msda_forward_kernel(
    value_ptr,
    shapes_ptr,
    level_start_index_ptr,
    loc_ptr,
    attn_ptr,
    output_ptr,
    m_run_ptr,
    z_run_ptr,
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
    stride_mb,
    stride_mq,
    stride_mh,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    aggregate: tl.constexpr,
):
    # Parallelize by Query Block, Head, and Batch
    q_block_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)

    q_offsets = q_block_idx * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    q_mask = q_offsets < num_queries

    d_offsets = tl.arange(0, BLOCK_SIZE_D)
    d_mask = d_offsets < dim_per_head

    # Initialize accumulator for the entire query block
    acc = tl.zeros([BLOCK_SIZE_Q, BLOCK_SIZE_D], dtype=tl.float32)

    # Online softmax (aggregate == 1): m_run is the running max over (l, p),
    # z_run the running sum of exp(a - m_run).
    m_run = tl.full([BLOCK_SIZE_Q], -float("inf"), dtype=tl.float32)
    z_run = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32)

    # Base pointer for values in this head
    v_base = value_ptr + batch_idx * stride_vb + head_idx * stride_vh

    # One pass over (l, p) builds the softmax denominator and ``acc`` together,
    # rescaling both when a new max appears.
    for l in range(num_levels):
        h = tl.load(shapes_ptr + l * 2)
        w = tl.load(shapes_ptr + l * 2 + 1)
        start_idx = tl.load(level_start_index_ptr + l)

        for p in range(num_points):
            # Locations for the block [BLOCK_SIZE_Q, 2]; loc[B, Q, M, L, P, 2]
            l_off = (
                batch_idx * stride_lb
                + q_offsets * stride_lq
                + head_idx * stride_lh
                + l * stride_ll
                + p * stride_lp
            )
            lx = tl.load(loc_ptr + l_off + 0 * stride_lc, mask=q_mask, other=0.0).to(
                tl.float32
            )
            ly = tl.load(loc_ptr + l_off + 1 * stride_lc, mask=q_mask, other=0.0).to(
                tl.float32
            )

            # Weights for the block [BLOCK_SIZE_Q]; attn[B, Q, M, L, P]. Softmax
            # masks out-of-range queries with -inf, linear with 0.
            a_off = (
                batch_idx * stride_ab
                + q_offsets * stride_aq
                + head_idx * stride_ah
                + l * stride_al
                + p * stride_ap
            )
            if aggregate == 1:
                attn_raw = tl.load(
                    attn_ptr + a_off, mask=q_mask, other=-float("inf")
                ).to(tl.float32)
                # Rescale acc and z_run to the new max's shifted exponent.
                m_new = tl.maximum(m_run, attn_raw)
                # Avoids ``-inf - -inf``, which would make the query NaN.
                scale = tl.where(m_new == m_run, 1.0, tl.exp(m_run - m_new))
                acc = acc * scale[:, None]
                z_run = z_run * scale
                attn_w = tl.where(
                    attn_raw == -float("inf"), 0.0, tl.exp(attn_raw - m_new)
                )
                m_run = m_new
                z_run = z_run + attn_w
            else:
                attn_w = tl.load(attn_ptr + a_off, mask=q_mask, other=0.0)

            # Map normalized [0, 1] to pixel space
            px = lx * w - 0.5
            py = ly * h - 0.5

            x0 = tl.math.floor(px.to(tl.float32)).to(tl.int32)
            y0 = tl.math.floor(py.to(tl.float32)).to(tl.int32)
            x1, y1 = x0 + 1, y0 + 1

            wx1 = (px - x0).to(tl.float32)
            wy1 = (py - y0).to(tl.float32)
            wx0 = 1.0 - wx1
            wy0 = 1.0 - wy1

            # Bilinear interpolation for the entire block
            p00_off = start_idx + y0 * w + x0
            p01_off = start_idx + y0 * w + x1
            p10_off = start_idx + y1 * w + x0
            p11_off = start_idx + y1 * w + x1

            m00 = (x0 >= 0) & (x0 < w) & (y0 >= 0) & (y0 < h) & q_mask
            m01 = (x1 >= 0) & (x1 < w) & (y0 >= 0) & (y0 < h) & q_mask
            m10 = (x0 >= 0) & (x0 < w) & (y1 >= 0) & (y1 < h) & q_mask
            m11 = (x1 >= 0) & (x1 < w) & (y1 >= 0) & (y1 < h) & q_mask

            v00 = tl.load(
                v_base + p00_off[:, None] * stride_vs + d_offsets[None, :] * stride_vd,
                mask=m00[:, None] & d_mask[None, :],
                other=0.0,
            )
            v01 = tl.load(
                v_base + p01_off[:, None] * stride_vs + d_offsets[None, :] * stride_vd,
                mask=m01[:, None] & d_mask[None, :],
                other=0.0,
            )
            v10 = tl.load(
                v_base + p10_off[:, None] * stride_vs + d_offsets[None, :] * stride_vd,
                mask=m10[:, None] & d_mask[None, :],
                other=0.0,
            )
            v11 = tl.load(
                v_base + p11_off[:, None] * stride_vs + d_offsets[None, :] * stride_vd,
                mask=m11[:, None] & d_mask[None, :],
                other=0.0,
            )

            acc += attn_w[:, None] * (
                wx0[:, None] * wy0[:, None] * v00
                + wx1[:, None] * wy0[:, None] * v01
                + wx0[:, None] * wy1[:, None] * v10
                + wx1[:, None] * wy1[:, None] * v11
            )

    if aggregate == 1:
        # z_run is positive for every valid query under q_mask.
        acc = acc / z_run[:, None]

        # Persist the statistics for the backward's elementwise rebuild.
        mz_off = batch_idx * stride_mb + q_offsets * stride_mq + head_idx * stride_mh
        tl.store(m_run_ptr + mz_off, m_run, mask=q_mask)
        tl.store(z_run_ptr + mz_off, z_run, mask=q_mask)

    # Store output block
    out_off = (
        batch_idx * stride_ob
        + q_offsets[:, None] * stride_oq
        + head_idx * stride_oh
        + d_offsets[None, :] * stride_od
    )
    tl.store(
        output_ptr + out_off,
        acc.to(output_ptr.dtype.element_ty),
        mask=q_mask[:, None] & d_mask[None, :],
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
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    q_block_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)

    q_offsets = q_block_idx * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    q_mask = q_offsets < num_queries
    d_offsets = tl.arange(0, BLOCK_SIZE_D)
    d_mask = d_offsets < dim_per_head

    g_out_off = (
        batch_idx * stride_ob
        + q_offsets[:, None] * stride_oq
        + head_idx * stride_oh
        + d_offsets[None, :] * stride_od
    )
    grad_out = tl.load(
        grad_output_ptr + g_out_off, mask=q_mask[:, None] & d_mask[None, :], other=0.0
    )

    v_base = value_ptr + batch_idx * stride_vb + head_idx * stride_vh
    gv_base = grad_value_ptr + batch_idx * stride_gvb + head_idx * stride_gvh

    for l in range(num_levels):
        h = tl.load(shapes_ptr + l * 2)
        w = tl.load(shapes_ptr + l * 2 + 1)
        start_idx = tl.load(level_start_index_ptr + l)

        for p in range(num_points):
            l_off = (
                batch_idx * stride_lb
                + q_offsets * stride_lq
                + head_idx * stride_lh
                + l * stride_ll
                + p * stride_lp
            )
            gl_off = (
                batch_idx * stride_glb
                + q_offsets * stride_glq
                + head_idx * stride_glh
                + l * stride_gll
                + p * stride_glp
            )
            lx = tl.load(loc_ptr + l_off + 0 * stride_lc, mask=q_mask, other=0.0).to(
                tl.float32
            )
            ly = tl.load(loc_ptr + l_off + 1 * stride_lc, mask=q_mask, other=0.0).to(
                tl.float32
            )

            a_off = (
                batch_idx * stride_ab
                + q_offsets * stride_aq
                + head_idx * stride_ah
                + l * stride_al
                + p * stride_ap
            )
            ga_off = (
                batch_idx * stride_gab
                + q_offsets * stride_gaq
                + head_idx * stride_gah
                + l * stride_gal
                + p * stride_gap
            )
            attention_weights = tl.load(attn_ptr + a_off, mask=q_mask, other=0.0)

            px, py = lx * w - 0.5, ly * h - 0.5
            x0 = tl.math.floor(px.to(tl.float32)).to(tl.int32)
            y0 = tl.math.floor(py.to(tl.float32)).to(tl.int32)
            x1, y1 = x0 + 1, y0 + 1

            wx1, wy1 = (px - x0).to(tl.float32), (py - y0).to(tl.float32)
            wx0, wy0 = 1.0 - wx1, 1.0 - wy1

            p00_off = start_idx + y0 * w + x0
            p01_off = start_idx + y0 * w + x1
            p10_off = start_idx + y1 * w + x0
            p11_off = start_idx + y1 * w + x1

            m00 = (x0 >= 0) & (x0 < w) & (y0 >= 0) & (y0 < h) & q_mask
            m01 = (x1 >= 0) & (x1 < w) & (y0 >= 0) & (y0 < h) & q_mask
            m10 = (x0 >= 0) & (x0 < w) & (y1 >= 0) & (y1 < h) & q_mask
            m11 = (x1 >= 0) & (x1 < w) & (y1 >= 0) & (y1 < h) & q_mask

            v00 = tl.load(
                v_base + p00_off[:, None] * stride_vs + d_offsets[None, :] * stride_vd,
                mask=m00[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            v01 = tl.load(
                v_base + p01_off[:, None] * stride_vs + d_offsets[None, :] * stride_vd,
                mask=m01[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            v10 = tl.load(
                v_base + p10_off[:, None] * stride_vs + d_offsets[None, :] * stride_vd,
                mask=m10[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            v11 = tl.load(
                v_base + p11_off[:, None] * stride_vs + d_offsets[None, :] * stride_vd,
                mask=m11[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            # Grad Attn
            interp = (
                (wx0 * wy0)[:, None] * v00
                + (wx1 * wy0)[:, None] * v01
                + (wx0 * wy1)[:, None] * v10
                + (wx1 * wy1)[:, None] * v11
            )
            d_attn = tl.sum(interp * grad_out, axis=1)
            tl.store(
                grad_attn_ptr + ga_off,
                d_attn.to(grad_attn_ptr.dtype.element_ty),
                mask=q_mask,
            )

            # Grad Loc
            d_px = wy0[:, None] * (v01 - v00) + wy1[:, None] * (v11 - v10)
            d_py = wx0[:, None] * (v10 - v00) + wx1[:, None] * (v11 - v01)
            d_lx = tl.sum(d_px * grad_out, axis=1) * attention_weights * w
            d_ly = tl.sum(d_py * grad_out, axis=1) * attention_weights * h
            tl.store(
                grad_loc_ptr + gl_off + 0 * stride_glc,
                d_lx.to(grad_loc_ptr.dtype.element_ty),
                mask=q_mask,
            )
            tl.store(
                grad_loc_ptr + gl_off + 1 * stride_glc,
                d_ly.to(grad_loc_ptr.dtype.element_ty),
                mask=q_mask,
            )

            # Relaxed atomics, avoiding serialized ordering.
            weighted_grad = grad_out * attention_weights[:, None]
            tl.atomic_add(
                gv_base
                + p00_off[:, None] * stride_gvs
                + d_offsets[None, :] * stride_gvd,
                (weighted_grad * (wx0 * wy0)[:, None]).to(
                    grad_value_ptr.dtype.element_ty
                ),
                mask=m00[:, None] & d_mask[None, :],
                sem="relaxed",
            )
            tl.atomic_add(
                gv_base
                + p01_off[:, None] * stride_gvs
                + d_offsets[None, :] * stride_gvd,
                (weighted_grad * (wx1 * wy0)[:, None]).to(
                    grad_value_ptr.dtype.element_ty
                ),
                mask=m01[:, None] & d_mask[None, :],
                sem="relaxed",
            )
            tl.atomic_add(
                gv_base
                + p10_off[:, None] * stride_gvs
                + d_offsets[None, :] * stride_gvd,
                (weighted_grad * (wx0 * wy1)[:, None]).to(
                    grad_value_ptr.dtype.element_ty
                ),
                mask=m10[:, None] & d_mask[None, :],
                sem="relaxed",
            )
            tl.atomic_add(
                gv_base
                + p11_off[:, None] * stride_gvs
                + d_offsets[None, :] * stride_gvd,
                (weighted_grad * (wx1 * wy1)[:, None]).to(
                    grad_value_ptr.dtype.element_ty
                ),
                mask=m11[:, None] & d_mask[None, :],
                sem="relaxed",
            )


def forward_launch(
    value: torch.Tensor,
    spatial_shapes: torch.Tensor,
    level_start_index: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    aggregate: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward launch returning ``(output, m_run, z_run)``.

    ``output`` is ``(B, Q, H, D)``; the float32 ``(B, Q, H)`` statistics are valid
    only for ``aggregate == 1`` and otherwise an uninitialized ``(1, 1, 1)``
    placeholder to discard. Assumes ``Q > 0`` and a non-float64 dtype.
    """
    B, _S, H, D = value.shape
    _, Q, _, L, P, _ = sampling_locations.shape
    attention_weights = attention_weights.view(B, Q, H, L, P)
    output = torch.zeros((B, Q, H, D), device=value.device, dtype=value.dtype)
    # The launch needs statistics pointers even when the writes fold away; the
    # linear path gets a ``(1, 1, 1)`` dummy instead of a full allocation.
    if aggregate == 1:
        m_run = torch.empty((B, Q, H), device=value.device, dtype=torch.float32)
        z_run = torch.empty((B, Q, H), device=value.device, dtype=torch.float32)
    else:
        m_run = torch.empty((1, 1, 1), device=value.device, dtype=torch.float32)
        z_run = torch.empty((1, 1, 1), device=value.device, dtype=torch.float32)
    BLOCK_SIZE_D = triton.next_power_of_2(D)
    BLOCK_SIZE_Q = 32  # Heuristic
    grid = (triton.cdiv(Q, BLOCK_SIZE_Q), H, B)
    wrap_triton(_msda_forward_kernel)[grid](
        value,
        spatial_shapes,
        level_start_index,
        sampling_locations,
        attention_weights,
        output,
        m_run,
        z_run,
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
        *m_run.stride(),
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,  # pyright: ignore[reportArgumentType]
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        aggregate=aggregate,  # pyright: ignore[reportArgumentType]
    )
    return output, m_run, z_run


def atomic_backward_launch(
    grad_output: torch.Tensor,
    value: torch.Tensor,
    spatial_shapes: torch.Tensor,
    level_start_index: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward launch returning ``(grad_value, grad_loc, grad_attn)``; ``Q > 0``."""
    B, _S, H, D = value.shape
    _, Q, _, L, P, _ = sampling_locations.shape
    attention_weights = attention_weights.view(B, Q, H, L, P)
    grad_value = torch.zeros_like(value)
    grad_loc = torch.zeros_like(sampling_locations)
    grad_attn = torch.zeros_like(attention_weights)
    BLOCK_SIZE_D = triton.next_power_of_2(D)
    BLOCK_SIZE_Q = 32
    grid = (triton.cdiv(Q, BLOCK_SIZE_Q), H, B)
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
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,  # pyright: ignore[reportArgumentType]
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )
    return grad_value, grad_loc, grad_attn


def float32_backward_launch(
    grad_output: torch.Tensor,
    value: torch.Tensor,
    spatial_shapes: torch.Tensor,
    level_start_index: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward kernel launch with float32 grad_value accumulator."""
    B, S, H, D = value.shape
    _, Q, _, L, P, _ = sampling_locations.shape
    # Kernel indexes attn as 5-D (B, Q, H, L, P); accept (B, Q, H, L*P) too.
    attention_weights = attention_weights.view(B, Q, H, L, P)
    grad_value = torch.zeros(B, S, H, D, device=value.device, dtype=torch.float32)
    grad_loc = torch.zeros_like(sampling_locations)
    grad_attn = torch.zeros_like(attention_weights)
    BLOCK_SIZE_D = triton.next_power_of_2(D)
    BLOCK_SIZE_Q = 32
    grid = (triton.cdiv(Q, BLOCK_SIZE_Q), H, B)
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
        sampling_locations.shape[3],
        sampling_locations.shape[4],
        D,
        *value.stride(),
        *sampling_locations.stride(),
        *attention_weights.stride(),
        *grad_output.stride(),
        *grad_value.stride(),
        *grad_loc.stride(),
        *grad_attn.stride(),
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,  # pyright: ignore[reportArgumentType]
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )
    return grad_value, grad_loc, grad_attn
