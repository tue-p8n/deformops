# pyright: basic, reportMissingImports=false
r"""Reduce-by-key backward kernels for multi-scale deformable attention.

Same gradient as the query-tiled backward, with colliding addresses combined
before the atomics: block- or warp-scoped deduplication, or sort plus segment
scan. All are slower than the scattered atomics; see ``docs/kernels.md``.
"""

from __future__ import annotations

import typing

import torch

from .....core.capabilities import require_triton

require_triton(__name__.rsplit(".", 1)[-1])

import triton
import triton.language as tl
from torch.library import wrap_triton

__all__ = [
    "BLOCK_SCOPE",
    "WARP_SCOPE",
    "DeduplicationScope",
    "reduce_backward_launch",
    "sort_segscan_backward_launch",
]


class DeduplicationScope(typing.NamedTuple):
    """Launch shape, and so scope, of the deduplicating backward."""

    block_size_q: int
    num_warps: int
    num_stages: int


#: Reduce across a 64-query program: the widest deduplication scope on offer.
BLOCK_SCOPE = DeduplicationScope(block_size_q=64, num_warps=4, num_stages=2)

#: One warp per program: a smaller scope, lower register pressure.
WARP_SCOPE = DeduplicationScope(block_size_q=32, num_warps=1, num_stages=2)


@triton.jit
def _msda_backward_reduce_kernel(  # noqa: PLR0913
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

    q_indices = tl.arange(0, BLOCK_SIZE_Q)

    g_out_off = (
        batch_idx * stride_ob
        + q_offsets[:, None] * stride_oq
        + head_idx * stride_oh
        + d_offsets[None, :] * stride_od
    )
    grad_out = tl.load(
        grad_output_ptr + g_out_off, mask=q_mask[:, None] & d_mask[None, :], other=0.0
    ).to(tl.float32)

    v_base = value_ptr + batch_idx * stride_vb + head_idx * stride_vh
    gv_base = grad_value_ptr + batch_idx * stride_gvb + head_idx * stride_gvh

    for level in range(num_levels):
        h = tl.load(shapes_ptr + level * 2)
        w = tl.load(shapes_ptr + level * 2 + 1)
        start_idx = tl.load(level_start_index_ptr + level)

        for p in range(num_points):
            l_off = (
                batch_idx * stride_lb
                + q_offsets * stride_lq
                + head_idx * stride_lh
                + level * stride_ll
                + p * stride_lp
            )
            gl_off = (
                batch_idx * stride_glb
                + q_offsets * stride_glq
                + head_idx * stride_glh
                + level * stride_gll
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
                + level * stride_al
                + p * stride_ap
            )
            ga_off = (
                batch_idx * stride_gab
                + q_offsets * stride_gaq
                + head_idx * stride_gah
                + level * stride_gal
                + p * stride_gap
            )
            # Source dtype, as in the query-tiled kernel.
            attention_weights = tl.load(attn_ptr + a_off, mask=q_mask, other=0.0)

            # The query-tiled casting order: upcasting first changes the rounding
            # of half-precision ``sampling_locations`` and breaks bit-exact parity.
            px = lx * w - 0.5
            py = ly * h - 0.5
            x0 = tl.math.floor(px.to(tl.float32)).to(tl.int32)
            y0 = tl.math.floor(py.to(tl.float32)).to(tl.int32)
            x1, y1 = x0 + 1, y0 + 1

            wx1 = (px - x0).to(tl.float32)
            wy1 = (py - y0).to(tl.float32)
            wx0 = 1.0 - wx1
            wy0 = 1.0 - wy1

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

            # grad_attn: per-query writes, no contention.
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

            # grad_loc: per-query writes, no contention.
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

            # grad_value: same-address contributions fold into one row, so at
            # most one atomic per unique address per corner. The ``[Q, Q, D]``
            # ``tl.where`` intermediate dominates register pressure.
            weighted_grad = grad_out * attention_weights[:, None]

            # --- corner (0, 0) ---
            contrib = weighted_grad * (wx0 * wy0)[:, None]
            contrib = tl.where(m00[:, None] & d_mask[None, :], contrib, 0.0)
            p_off_masked = tl.where(m00, p00_off, -1)
            match = (p_off_masked[:, None] == p_off_masked[None, :]) & (
                p_off_masked[:, None] >= 0
            )
            first_i = tl.min(tl.where(match, q_indices[:, None], BLOCK_SIZE_Q), axis=0)
            redirect = first_i[None, :] == q_indices[:, None]
            accumulated = tl.sum(
                tl.where(redirect[:, :, None], contrib[None, :, :], 0.0), axis=1
            )
            is_rep = (first_i == q_indices) & m00
            tl.atomic_add(
                gv_base
                + p00_off[:, None] * stride_gvs
                + d_offsets[None, :] * stride_gvd,
                accumulated.to(grad_value_ptr.dtype.element_ty),
                mask=is_rep[:, None] & d_mask[None, :],
                sem="relaxed",
            )

            # --- corner (0, 1) ---
            contrib = weighted_grad * (wx1 * wy0)[:, None]
            contrib = tl.where(m01[:, None] & d_mask[None, :], contrib, 0.0)
            p_off_masked = tl.where(m01, p01_off, -1)
            match = (p_off_masked[:, None] == p_off_masked[None, :]) & (
                p_off_masked[:, None] >= 0
            )
            first_i = tl.min(tl.where(match, q_indices[:, None], BLOCK_SIZE_Q), axis=0)
            redirect = first_i[None, :] == q_indices[:, None]
            accumulated = tl.sum(
                tl.where(redirect[:, :, None], contrib[None, :, :], 0.0), axis=1
            )
            is_rep = (first_i == q_indices) & m01
            tl.atomic_add(
                gv_base
                + p01_off[:, None] * stride_gvs
                + d_offsets[None, :] * stride_gvd,
                accumulated.to(grad_value_ptr.dtype.element_ty),
                mask=is_rep[:, None] & d_mask[None, :],
                sem="relaxed",
            )

            # --- corner (1, 0) ---
            contrib = weighted_grad * (wx0 * wy1)[:, None]
            contrib = tl.where(m10[:, None] & d_mask[None, :], contrib, 0.0)
            p_off_masked = tl.where(m10, p10_off, -1)
            match = (p_off_masked[:, None] == p_off_masked[None, :]) & (
                p_off_masked[:, None] >= 0
            )
            first_i = tl.min(tl.where(match, q_indices[:, None], BLOCK_SIZE_Q), axis=0)
            redirect = first_i[None, :] == q_indices[:, None]
            accumulated = tl.sum(
                tl.where(redirect[:, :, None], contrib[None, :, :], 0.0), axis=1
            )
            is_rep = (first_i == q_indices) & m10
            tl.atomic_add(
                gv_base
                + p10_off[:, None] * stride_gvs
                + d_offsets[None, :] * stride_gvd,
                accumulated.to(grad_value_ptr.dtype.element_ty),
                mask=is_rep[:, None] & d_mask[None, :],
                sem="relaxed",
            )

            # --- corner (1, 1) ---
            contrib = weighted_grad * (wx1 * wy1)[:, None]
            contrib = tl.where(m11[:, None] & d_mask[None, :], contrib, 0.0)
            p_off_masked = tl.where(m11, p11_off, -1)
            match = (p_off_masked[:, None] == p_off_masked[None, :]) & (
                p_off_masked[:, None] >= 0
            )
            first_i = tl.min(tl.where(match, q_indices[:, None], BLOCK_SIZE_Q), axis=0)
            redirect = first_i[None, :] == q_indices[:, None]
            accumulated = tl.sum(
                tl.where(redirect[:, :, None], contrib[None, :, :], 0.0), axis=1
            )
            is_rep = (first_i == q_indices) & m11
            tl.atomic_add(
                gv_base
                + p11_off[:, None] * stride_gvs
                + d_offsets[None, :] * stride_gvd,
                accumulated.to(grad_value_ptr.dtype.element_ty),
                mask=is_rep[:, None] & d_mask[None, :],
                sem="relaxed",
            )


def reduce_backward_launch(  # noqa: PLR0913
    grad_output: torch.Tensor,
    value: torch.Tensor,
    spatial_shapes: torch.Tensor,
    level_start_index: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    *,
    scope: DeduplicationScope,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deduplicating linear backward: collapse colliding addresses, then one atomic.

    ``grad_output`` is ``(B, Q, H, D)``; the weight gradient returns 5-D.
    """
    B, _S, H, D = value.shape
    _, Q, _, L, P, _ = sampling_locations.shape
    attention_weights = attention_weights.view(B, Q, H, L, P)
    grad_value = torch.zeros_like(value)
    grad_loc = torch.zeros_like(sampling_locations)
    grad_attn = torch.zeros_like(attention_weights)
    BLOCK_SIZE_D = triton.next_power_of_2(D)
    grid = (triton.cdiv(Q, scope.block_size_q), H, B)

    wrap_triton(_msda_backward_reduce_kernel)[grid](
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
        BLOCK_SIZE_Q=scope.block_size_q,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        num_warps=scope.num_warps,  # pyright: ignore[reportCallIssue]  # Triton launch metaparameter; not in its stubs
        num_stages=scope.num_stages,  # pyright: ignore[reportCallIssue]  # Triton launch metaparameter; not in its stubs
    )
    return grad_value, grad_loc, grad_attn


# --------------------------------------------------------------------------
# Sort + segment-scan
# --------------------------------------------------------------------------

_BLOCK_SIZE_Q = 32
_NUM_WARPS = 2
_NUM_STAGES = 2

# Sort key: address in the upper 32 bits, block-local query index in the lower.
_IDX_BITS = 32
_IDX_MASK = (1 << _IDX_BITS) - 1
# Exceeds every legal cell index once heads * channels >= 2, since the bridge caps a
# sample at 2^31 elements. It also keeps the signed packed key non-negative.
_SENTINEL_P_OFF = (1 << 30) - 1


@triton.jit
def _reduce_one_corner(  # noqa: PLR0913
    gv_base,
    stride_gvs,
    stride_gvd,
    p_off,
    corner_mask,
    contrib,
    d_offsets,
    d_mask,
    q_indices,
    grad_value_ptr,
    BLOCK_SIZE_Q: tl.constexpr,
    IDX_BITS: tl.constexpr,
    IDX_MASK: tl.constexpr,
    SENTINEL: tl.constexpr,
):
    """Sort-based dedup + segmented sum for a single corner of one (l, p)."""
    # Masked queries sort into a trailing sentinel segment, which ``valid`` excludes.
    p_off_masked = tl.where(corner_mask, p_off, SENTINEL)

    packed = (p_off_masked.to(tl.int64) << IDX_BITS) | (
        q_indices.to(tl.int64) & IDX_MASK
    )
    sorted_packed = tl.sort(packed)
    sorted_p_off = (sorted_packed >> IDX_BITS).to(tl.int32)
    sigma = (sorted_packed & IDX_MASK).to(tl.int32)

    # Gather contrib into sorted order: sorted_contrib[i] = contrib[sigma[i]].
    gather_mask = sigma[:, None] == q_indices[None, :]
    sorted_contrib = tl.sum(
        tl.where(gather_mask[:, :, None], contrib[None, :, :], 0.0), axis=1
    )

    # Equal keys are now contiguous.
    valid = sorted_p_off != SENTINEL
    seg_mask = (sorted_p_off[:, None] == sorted_p_off[None, :]) & valid[None, :]

    # Segment-scan: each seg-first row holds the sum over its segment.
    first_j_in_seg = tl.min(
        tl.where(seg_mask, q_indices[None, :], BLOCK_SIZE_Q), axis=1
    )
    is_seg_first = (first_j_in_seg == q_indices) & valid

    accumulated = tl.sum(
        tl.where(seg_mask[:, :, None], sorted_contrib[None, :, :], 0.0), axis=1
    )

    # One atomic per segment-first row, at the address held by that row.
    tl.atomic_add(
        gv_base + sorted_p_off[:, None] * stride_gvs + d_offsets[None, :] * stride_gvd,
        accumulated.to(grad_value_ptr.dtype.element_ty),
        mask=is_seg_first[:, None] & d_mask[None, :],
        sem="relaxed",
    )


@triton.jit
def _msda_backward_sort_segscan_kernel(  # noqa: PLR0913
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
    IDX_BITS: tl.constexpr,
    IDX_MASK: tl.constexpr,
    SENTINEL: tl.constexpr,
):
    q_block_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)

    q_offsets = q_block_idx * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    q_mask = q_offsets < num_queries
    d_offsets = tl.arange(0, BLOCK_SIZE_D)
    d_mask = d_offsets < dim_per_head
    q_indices = tl.arange(0, BLOCK_SIZE_Q)

    g_out_off = (
        batch_idx * stride_ob
        + q_offsets[:, None] * stride_oq
        + head_idx * stride_oh
        + d_offsets[None, :] * stride_od
    )
    grad_out = tl.load(
        grad_output_ptr + g_out_off, mask=q_mask[:, None] & d_mask[None, :], other=0.0
    ).to(tl.float32)

    v_base = value_ptr + batch_idx * stride_vb + head_idx * stride_vh
    gv_base = grad_value_ptr + batch_idx * stride_gvb + head_idx * stride_gvh

    for level in range(num_levels):
        h = tl.load(shapes_ptr + level * 2)
        w = tl.load(shapes_ptr + level * 2 + 1)
        start_idx = tl.load(level_start_index_ptr + level)

        for p in range(num_points):
            l_off = (
                batch_idx * stride_lb
                + q_offsets * stride_lq
                + head_idx * stride_lh
                + level * stride_ll
                + p * stride_lp
            )
            gl_off = (
                batch_idx * stride_glb
                + q_offsets * stride_glq
                + head_idx * stride_glh
                + level * stride_gll
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
                + level * stride_al
                + p * stride_ap
            )
            ga_off = (
                batch_idx * stride_gab
                + q_offsets * stride_gaq
                + head_idx * stride_gah
                + level * stride_gal
                + p * stride_gap
            )
            # Source dtype, as in the query-tiled kernel.
            attention_weights = tl.load(attn_ptr + a_off, mask=q_mask, other=0.0)

            # The query-tiled casting order: upcasting first changes the rounding
            # of half-precision ``sampling_locations`` and breaks bit-exact parity.
            px = lx * w - 0.5
            py = ly * h - 0.5
            x0 = tl.math.floor(px.to(tl.float32)).to(tl.int32)
            y0 = tl.math.floor(py.to(tl.float32)).to(tl.int32)
            x1, y1 = x0 + 1, y0 + 1

            wx1 = (px - x0).to(tl.float32)
            wy1 = (py - y0).to(tl.float32)
            wx0 = 1.0 - wx1
            wy0 = 1.0 - wy1

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

            # grad_attn: per-query writes, no contention.
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

            # grad_loc: per-query writes, no contention.
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

            # grad_value: sort + segscan per corner.
            weighted_grad = grad_out * attention_weights[:, None]

            contrib00 = weighted_grad * (wx0 * wy0)[:, None]
            contrib00 = tl.where(m00[:, None] & d_mask[None, :], contrib00, 0.0)
            _reduce_one_corner(
                gv_base,
                stride_gvs,
                stride_gvd,
                p00_off,
                m00,
                contrib00,
                d_offsets,
                d_mask,
                q_indices,
                grad_value_ptr,
                BLOCK_SIZE_Q=BLOCK_SIZE_Q,
                IDX_BITS=IDX_BITS,
                IDX_MASK=IDX_MASK,
                SENTINEL=SENTINEL,
            )

            contrib01 = weighted_grad * (wx1 * wy0)[:, None]
            contrib01 = tl.where(m01[:, None] & d_mask[None, :], contrib01, 0.0)
            _reduce_one_corner(
                gv_base,
                stride_gvs,
                stride_gvd,
                p01_off,
                m01,
                contrib01,
                d_offsets,
                d_mask,
                q_indices,
                grad_value_ptr,
                BLOCK_SIZE_Q=BLOCK_SIZE_Q,
                IDX_BITS=IDX_BITS,
                IDX_MASK=IDX_MASK,
                SENTINEL=SENTINEL,
            )

            contrib10 = weighted_grad * (wx0 * wy1)[:, None]
            contrib10 = tl.where(m10[:, None] & d_mask[None, :], contrib10, 0.0)
            _reduce_one_corner(
                gv_base,
                stride_gvs,
                stride_gvd,
                p10_off,
                m10,
                contrib10,
                d_offsets,
                d_mask,
                q_indices,
                grad_value_ptr,
                BLOCK_SIZE_Q=BLOCK_SIZE_Q,
                IDX_BITS=IDX_BITS,
                IDX_MASK=IDX_MASK,
                SENTINEL=SENTINEL,
            )

            contrib11 = weighted_grad * (wx1 * wy1)[:, None]
            contrib11 = tl.where(m11[:, None] & d_mask[None, :], contrib11, 0.0)
            _reduce_one_corner(
                gv_base,
                stride_gvs,
                stride_gvd,
                p11_off,
                m11,
                contrib11,
                d_offsets,
                d_mask,
                q_indices,
                grad_value_ptr,
                BLOCK_SIZE_Q=BLOCK_SIZE_Q,
                IDX_BITS=IDX_BITS,
                IDX_MASK=IDX_MASK,
                SENTINEL=SENTINEL,
            )


def sort_segscan_backward_launch(
    grad_output: torch.Tensor,
    value: torch.Tensor,
    spatial_shapes: torch.Tensor,
    level_start_index: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort-and-scan backward: exactly one atomic per distinct address.

    Same arguments and results as :func:`reduce_backward_launch`.
    """
    B, _S, H, D = value.shape
    _, Q, _, L, P, _ = sampling_locations.shape
    attention_weights = attention_weights.view(B, Q, H, L, P)
    grad_value = torch.zeros_like(value)
    grad_loc = torch.zeros_like(sampling_locations)
    grad_attn = torch.zeros_like(attention_weights)
    grid = (triton.cdiv(Q, _BLOCK_SIZE_Q), H, B)
    wrap_triton(_msda_backward_sort_segscan_kernel)[grid](
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
        BLOCK_SIZE_Q=_BLOCK_SIZE_Q,
        BLOCK_SIZE_D=triton.next_power_of_2(D),
        IDX_BITS=_IDX_BITS,
        IDX_MASK=_IDX_MASK,
        SENTINEL=_SENTINEL_P_OFF,
        num_warps=_NUM_WARPS,  # pyright: ignore[reportCallIssue]  # Triton launch metaparameter; not in its stubs
        num_stages=_NUM_STAGES,  # pyright: ignore[reportCallIssue]  # Triton launch metaparameter; not in its stubs
    )
    return grad_value, grad_loc, grad_attn
