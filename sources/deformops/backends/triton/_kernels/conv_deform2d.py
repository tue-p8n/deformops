# pyright: basic, reportMissingImports=false
"""Fused deformable convolution in Triton: gather and contraction in one pass.

Contracts the sampled values against the weight inside the kernel, so no
``(B, P, C_in, K)`` buffer exists. Accumulation is always float32. Conventions
follow torchvision; see ``docs/kernels.md`` and ``docs/conventions.md``.
"""

from __future__ import annotations

import torch

from ....core.capabilities import require_triton

require_triton(__name__.rsplit(".", 1)[-1])

import triton
import triton.language as tl
from torch.library import wrap_triton

__all__ = [
    "backward_launch",
    "forward_launch",
    "permute_weight",
    "tune_bucket",
]


def _allow_tf32() -> bool:
    """Whether the contraction may use TF32: never, see ``docs/precision.md``."""
    return False


def _configs(
    blocks_m: tuple[int, ...], blocks_n: tuple[int, ...], blocks_k: tuple[int, ...]
) -> list[triton.Config]:
    """Candidate launch shapes; Triton drops those a device cannot build.

    Few candidates, since autotuning recompiles the space per key. See
    ``docs/kernels.md``.
    """
    return [
        triton.Config(
            {"BLOCK_M": m, "BLOCK_N": n, "BLOCK_K": k},
            num_warps=w,
            num_stages=stages,
        )
        for m in blocks_m
        for n in blocks_n
        for k in blocks_k
        for w in (4, 8)
        for stages in (2,)
    ]


def tune_bucket(positions: int) -> int:
    """Round a position count to a power of two, for autotuner keying only.

    Keeps a varying input size from re-timing the configuration space per shape.

    Parameters
    ----------
    positions
        ``H_out * W_out`` for the call being launched.

    Returns
    -------
    The bucketed position count used as an autotuner cache key.
    """
    # Triton's stubs type this as ``constexpr``; at the host level it is an int.
    return int(triton.next_power_of_2(positions))


#: Autotuner key shared by both kernels; Triton skips a name a call lacks. The
#: flags are included so specializations never share a configuration that only
#: fits the cheaper one.
_TUNE_KEY = [
    "positions_bucket",
    "in_per_group",
    "out_per_group",
    "HAS_MASK",
    "HAS_BIAS",
    "NEEDS_VALUE_GRAD",
    "NEEDS_OFFSET_GRAD",
    "NEEDS_MASK_GRAD",
    "NEEDS_WEIGHT_GRAD",
]


# ``cache_results`` keeps each process from re-running a minutes-long sweep.
@triton.autotune(
    configs=_configs((32, 64, 128), (32, 64), (32, 64)),
    key=_TUNE_KEY,
    cache_results=True,
)
@triton.jit
def _conv_forward_kernel(
    x_ptr,
    off_ptr,
    msk_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    height,
    width,
    out_w,
    positions,
    # Read by the autotuner through _TUNE_KEY, not by the body below.
    positions_bucket,  # noqa: ARG001
    in_per_group,
    out_per_group,
    per_offset_group,
    sx_n,
    sx_h,
    sx_w,
    sx_c,
    so_n,
    so_p,
    so_g,
    so_k,
    so_c,
    sm_n,
    sm_p,
    sm_g,
    sm_k,
    sw_k,
    sw_i,
    sw_o,
    sy_n,
    sy_p,
    sy_o,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    offset_groups,
    kernel_w: tl.constexpr,
    num_points: tl.constexpr,
    HAS_MASK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One tile of output positions by output channels, for one batch element."""
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)
    batch = tl.program_id(2)

    # Grid axis 1 packs group and channel block; the autotuned block width
    # makes the split unknown at launch.
    blocks_per_group = tl.cdiv(out_per_group, BLOCK_N)
    group = pid_g // blocks_per_group
    pid_n = pid_g % blocks_per_group

    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_local = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m < positions
    n_mask = n_local < out_per_group
    n_global = group * out_per_group + n_local

    out_row = m // out_w
    out_col = m % out_w
    base_y = (out_row * stride_h - pad_h).to(tl.float32)
    base_x = (out_col * stride_w - pad_w).to(tl.float32)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in tl.static_range(num_points):  # pyright: ignore[reportGeneralTypeIssues]
        kh = k // kernel_w
        kw = k % kernel_w
        for og in range(offset_groups):
            # Input channels this offset group covers, in the coordinates of
            # the convolution group: the reduction walks the intersection.
            lo = og * per_offset_group - group * in_per_group
            hi = lo + per_offset_group
            lo = tl.maximum(lo, 0)
            hi = tl.minimum(hi, in_per_group)

            if hi > lo:
                obase = off_ptr + batch * so_n + m * so_p + og * so_g + k * so_k
                dy = tl.load(obase + 0 * so_c, mask=m_mask, other=0.0).to(tl.float32)
                dx = tl.load(obase + 1 * so_c, mask=m_mask, other=0.0).to(tl.float32)
                if HAS_MASK:
                    mv = tl.load(
                        msk_ptr + batch * sm_n + m * sm_p + og * sm_g + k * sm_k,
                        mask=m_mask,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    mv = tl.full((BLOCK_M,), 1.0, tl.float32)

                py = base_y + kh * dilation_h + dy
                px = base_x + kw * dilation_w + dx
                y0 = tl.math.floor(py).to(tl.int32)
                x0 = tl.math.floor(px).to(tl.int32)
                y1 = y0 + 1
                x1 = x0 + 1
                ly = py - y0
                lx = px - x0
                hy = 1.0 - ly
                hx = 1.0 - lx

                g00 = (y0 >= 0) & (y0 < height) & (x0 >= 0) & (x0 < width) & m_mask
                g01 = (y0 >= 0) & (y0 < height) & (x1 >= 0) & (x1 < width) & m_mask
                g10 = (y1 >= 0) & (y1 < height) & (x0 >= 0) & (x0 < width) & m_mask
                g11 = (y1 >= 0) & (y1 < height) & (x1 >= 0) & (x1 < width) & m_mask

                for c0 in range(lo, hi, BLOCK_K):
                    c_local = c0 + tl.arange(0, BLOCK_K)
                    c_mask = (c_local >= lo) & (c_local < hi)
                    c_global = group * in_per_group + c_local
                    xbase = x_ptr + batch * sx_n + c_global[None, :] * sx_c

                    v00 = tl.load(
                        xbase + y0[:, None] * sx_h + x0[:, None] * sx_w,
                        mask=g00[:, None] & c_mask[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    v01 = tl.load(
                        xbase + y0[:, None] * sx_h + x1[:, None] * sx_w,
                        mask=g01[:, None] & c_mask[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    v10 = tl.load(
                        xbase + y1[:, None] * sx_h + x0[:, None] * sx_w,
                        mask=g10[:, None] & c_mask[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    v11 = tl.load(
                        xbase + y1[:, None] * sx_h + x1[:, None] * sx_w,
                        mask=g11[:, None] & c_mask[None, :],
                        other=0.0,
                    ).to(tl.float32)

                    sampled = (
                        hy[:, None] * hx[:, None] * v00
                        + hy[:, None] * lx[:, None] * v01
                        + ly[:, None] * hx[:, None] * v10
                        + ly[:, None] * lx[:, None] * v11
                    ) * mv[:, None]

                    wt = tl.load(
                        w_ptr
                        + k * sw_k
                        + c_local[:, None] * sw_i
                        + n_global[None, :] * sw_o,
                        mask=c_mask[:, None] & n_mask[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    acc += tl.dot(sampled, wt, allow_tf32=ALLOW_TF32)

    if HAS_BIAS:
        acc += tl.load(bias_ptr + n_global, mask=n_mask, other=0.0).to(tl.float32)[
            None, :
        ]

    tl.store(
        y_ptr + batch * sy_n + m[:, None] * sy_p + n_global[None, :] * sy_o,
        acc.to(y_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


def permute_weight(weight: torch.Tensor) -> torch.Tensor:
    """Permute the weight so the kernel reads a ``[BLOCK_K, BLOCK_N]`` tile.

    Parameters
    ----------
    weight
        ``(C_out, C_in / groups, kH, kW)``.

    Returns
    -------
    ``(K, C_in / groups, C_out)``, where ``K = kH * kW``.
    """
    out_channels, in_per_group, kernel_h, kernel_w = weight.shape
    return (
        weight.permute(2, 3, 1, 0)
        .reshape(kernel_h * kernel_w, in_per_group, out_channels)
        .contiguous()
    )


def forward_launch(  # noqa: PLR0913
    value: torch.Tensor,
    offset: torch.Tensor,
    mask: torch.Tensor | None,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dilation_h: int,
    dilation_w: int,
    groups: int,
    offset_groups: int,
    out_h: int,
    out_w: int,
) -> torch.Tensor:
    """Run the forward kernel.

    Parameters
    ----------
    value
        ``(B, H, W, C_in)``, channels-last.
    offset
        ``(B, H_out, W_out, offset_groups, K, 2)``, holding ``(dy, dx)``.
    mask
        ``(B, H_out, W_out, offset_groups, K)``, or ``None`` for v1.
    weight
        ``(C_out, C_in / groups, kH, kW)``.
    bias
        ``(C_out,)``, or ``None``.
    stride_h, stride_w
        Stride of the convolution.
    pad_h, pad_w
        Padding applied to ``value``.
    dilation_h, dilation_w
        Dilation of the convolution.
    groups
        Convolution groups, over input and output channels.
    offset_groups
        Offset groups, over input channels only.
    out_h, out_w
        Output extent, taken from ``offset``.

    Returns
    -------
    ``(B, H_out, W_out, C_out)``.
    """
    batch, height, width, in_channels = value.shape
    out_channels, in_per_group, kernel_h, kernel_w = weight.shape
    points = kernel_h * kernel_w
    positions = out_h * out_w
    out_per_group = out_channels // groups

    weight_perm = permute_weight(weight)
    offset_flat = offset.reshape(batch, positions, offset_groups, points, 2)
    mask_flat = (
        mask.reshape(batch, positions, offset_groups, points)
        if mask is not None
        else None
    )
    out = torch.empty(
        (batch, positions, out_channels), device=value.device, dtype=value.dtype
    )

    def grid(meta: dict) -> tuple[int, int, int]:
        # ``cdiv`` is typed as returning ``constexpr``; a launch grid is ints.
        return (
            int(triton.cdiv(positions, meta["BLOCK_M"])),
            groups * int(triton.cdiv(out_per_group, meta["BLOCK_N"])),
            batch,
        )

    empty = value.new_zeros(1)
    wrap_triton(_conv_forward_kernel)[grid](  # pyright: ignore[reportArgumentType]
        value,
        offset_flat,
        mask_flat if mask_flat is not None else empty,
        weight_perm,
        bias if bias is not None else empty,
        out,
        height,
        width,
        out_w,
        positions,
        tune_bucket(positions),
        in_per_group,
        out_per_group,
        in_channels // offset_groups,
        *value.stride(),
        *offset_flat.stride(),
        *(mask_flat.stride() if mask_flat is not None else (0, 0, 0, 0)),
        *weight_perm.stride(),
        *out.stride(),
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        dilation_h=dilation_h,
        dilation_w=dilation_w,
        offset_groups=offset_groups,
        kernel_w=kernel_w,  # pyright: ignore[reportArgumentType]
        num_points=points,  # pyright: ignore[reportArgumentType]
        HAS_MASK=mask is not None,  # pyright: ignore[reportArgumentType]
        HAS_BIAS=bias is not None,  # pyright: ignore[reportArgumentType]
        ALLOW_TF32=_allow_tf32(),  # pyright: ignore[reportArgumentType]
    )
    return out.reshape(batch, out_h, out_w, out_channels)


# ``reset_to_zero`` is required: every output accumulates, so without it each
# autotuning run adds to the gradients of the previous one.
@triton.autotune(
    configs=_configs((32, 64), (32, 64), (32, 64)),
    key=_TUNE_KEY,
    reset_to_zero=["gx_ptr", "goff_ptr", "gmsk_ptr", "gw_ptr"],
    cache_results=True,
)
@triton.jit
def _conv_backward_kernel(
    x_ptr,
    off_ptr,
    msk_ptr,
    w_ptr,
    go_ptr,
    gx_ptr,
    goff_ptr,
    gmsk_ptr,
    gw_ptr,
    height,
    width,
    out_w,
    positions,
    # Read by the autotuner through _TUNE_KEY, not by the body below.
    positions_bucket,  # noqa: ARG001
    in_per_group,
    out_per_group,
    per_offset_group,
    sx_n,
    sx_h,
    sx_w,
    sx_c,
    so_n,
    so_p,
    so_g,
    so_k,
    so_c,
    sm_n,
    sm_p,
    sm_g,
    sm_k,
    sw_k,
    sw_i,
    sw_o,
    sg_n,
    sg_p,
    sg_o,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    offset_groups,
    kernel_w: tl.constexpr,
    num_points: tl.constexpr,
    HAS_MASK: tl.constexpr,
    NEEDS_VALUE_GRAD: tl.constexpr,
    NEEDS_OFFSET_GRAD: tl.constexpr,
    NEEDS_MASK_GRAD: tl.constexpr,
    NEEDS_WEIGHT_GRAD: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Gradients for input, offset, mask and weight, each through relaxed atomics.

    Each ``NEEDS_*`` flag is constexpr, so an unrequested output costs nothing.
    A shared intermediate is computed when any output that reads it is requested. See
    ``docs/kernels.md``.
    """
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)
    batch = tl.program_id(2)

    blocks_per_group = tl.cdiv(out_per_group, BLOCK_N)
    group = pid_g // blocks_per_group
    pid_n = pid_g % blocks_per_group

    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_local = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m < positions
    n_mask = n_local < out_per_group
    n_global = group * out_per_group + n_local

    out_row = m // out_w
    out_col = m % out_w
    base_y = (out_row * stride_h - pad_h).to(tl.float32)
    base_x = (out_col * stride_w - pad_w).to(tl.float32)

    grad_out = tl.load(
        go_ptr + batch * sg_n + m[:, None] * sg_p + n_global[None, :] * sg_o,
        mask=m_mask[:, None] & n_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    for k in tl.static_range(num_points):  # pyright: ignore[reportGeneralTypeIssues]
        kh = k // kernel_w
        kw = k % kernel_w
        for og in range(offset_groups):
            lo = og * per_offset_group - group * in_per_group
            hi = lo + per_offset_group
            lo = tl.maximum(lo, 0)
            hi = tl.minimum(hi, in_per_group)

            if hi > lo:
                obase = off_ptr + batch * so_n + m * so_p + og * so_g + k * so_k
                dy = tl.load(obase + 0 * so_c, mask=m_mask, other=0.0).to(tl.float32)
                dx = tl.load(obase + 1 * so_c, mask=m_mask, other=0.0).to(tl.float32)
                if HAS_MASK:
                    mv = tl.load(
                        msk_ptr + batch * sm_n + m * sm_p + og * sm_g + k * sm_k,
                        mask=m_mask,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    mv = tl.full((BLOCK_M,), 1.0, tl.float32)

                py = base_y + kh * dilation_h + dy
                px = base_x + kw * dilation_w + dx
                y0 = tl.math.floor(py).to(tl.int32)
                x0 = tl.math.floor(px).to(tl.int32)
                y1 = y0 + 1
                x1 = x0 + 1
                ly = py - y0
                lx = px - x0
                hy = 1.0 - ly
                hx = 1.0 - lx

                g00 = (y0 >= 0) & (y0 < height) & (x0 >= 0) & (x0 < width) & m_mask
                g01 = (y0 >= 0) & (y0 < height) & (x1 >= 0) & (x1 < width) & m_mask
                g10 = (y1 >= 0) & (y1 < height) & (x0 >= 0) & (x0 < width) & m_mask
                g11 = (y1 >= 0) & (y1 < height) & (x1 >= 0) & (x1 < width) & m_mask

                acc_dy = tl.zeros((BLOCK_M,), dtype=tl.float32)
                acc_dx = tl.zeros((BLOCK_M,), dtype=tl.float32)
                acc_m = tl.zeros((BLOCK_M,), dtype=tl.float32)

                for c0 in range(lo, hi, BLOCK_K):
                    c_local = c0 + tl.arange(0, BLOCK_K)
                    c_mask = (c_local >= lo) & (c_local < hi)
                    c_global = group * in_per_group + c_local
                    xbase = x_ptr + batch * sx_n + c_global[None, :] * sx_c
                    o00 = y0[:, None] * sx_h + x0[:, None] * sx_w
                    o01 = y0[:, None] * sx_h + x1[:, None] * sx_w
                    o10 = y1[:, None] * sx_h + x0[:, None] * sx_w
                    o11 = y1[:, None] * sx_h + x1[:, None] * sx_w
                    f00 = g00[:, None] & c_mask[None, :]
                    f01 = g01[:, None] & c_mask[None, :]
                    f10 = g10[:, None] & c_mask[None, :]
                    f11 = g11[:, None] & c_mask[None, :]

                    # Corner values, loaded only when a gradient reads them.
                    # Unannotated float placeholders, one per statement: Triton
                    # reads annotations as globals and rejects tuple assignment.
                    v00 = 0.0
                    v01 = 0.0
                    v10 = 0.0
                    v11 = 0.0
                    if NEEDS_WEIGHT_GRAD or NEEDS_OFFSET_GRAD or NEEDS_MASK_GRAD:
                        v00 = tl.load(xbase + o00, mask=f00, other=0.0).to(tl.float32)
                        v01 = tl.load(xbase + o01, mask=f01, other=0.0).to(tl.float32)
                        v10 = tl.load(xbase + o10, mask=f10, other=0.0).to(tl.float32)
                        v11 = tl.load(xbase + o11, mask=f11, other=0.0).to(tl.float32)

                    if NEEDS_WEIGHT_GRAD or NEEDS_MASK_GRAD:
                        bilinear = (
                            hy[:, None] * hx[:, None] * v00
                            + hy[:, None] * lx[:, None] * v01
                            + ly[:, None] * hx[:, None] * v10
                            + ly[:, None] * lx[:, None] * v11
                        )

                    if NEEDS_WEIGHT_GRAD:
                        # Every program contributes a partial weight-gradient tile.
                        tl.atomic_add(
                            gw_ptr
                            + k * sw_k
                            + c_local[:, None] * sw_i
                            + n_global[None, :] * sw_o,
                            tl.dot(
                                tl.trans(bilinear * mv[:, None]),
                                grad_out,
                                allow_tf32=ALLOW_TF32,
                            ),
                            mask=c_mask[:, None] & n_mask[None, :],
                            sem="relaxed",
                        )

                    if NEEDS_VALUE_GRAD or NEEDS_OFFSET_GRAD or NEEDS_MASK_GRAD:
                        wt = tl.load(
                            w_ptr
                            + k * sw_k
                            + c_local[:, None] * sw_i
                            + n_global[None, :] * sw_o,
                            mask=c_mask[:, None] & n_mask[None, :],
                            other=0.0,
                        ).to(tl.float32)

                        # Gradient arriving at each sampled value, summed over
                        # the output channels this program owns.
                        grad_sampled = tl.dot(
                            grad_out, tl.trans(wt), allow_tf32=ALLOW_TF32
                        )

                        if NEEDS_MASK_GRAD:
                            acc_m += tl.sum(grad_sampled * bilinear, axis=1)

                        grad_bilinear = grad_sampled * mv[:, None]

                        if NEEDS_OFFSET_GRAD:
                            # Interpolation slopes; outside corners contribute zero.
                            d_dy = hx[:, None] * (v10 - v00) + lx[:, None] * (v11 - v01)
                            d_dx = hy[:, None] * (v01 - v00) + ly[:, None] * (v11 - v10)
                            acc_dy += tl.sum(grad_bilinear * d_dy, axis=1)
                            acc_dx += tl.sum(grad_bilinear * d_dx, axis=1)

                        if NEEDS_VALUE_GRAD:
                            gxbase = gx_ptr + batch * sx_n + c_global[None, :] * sx_c
                            tl.atomic_add(
                                gxbase + o00,
                                hy[:, None] * hx[:, None] * grad_bilinear,
                                mask=f00,
                                sem="relaxed",
                            )
                            tl.atomic_add(
                                gxbase + o01,
                                hy[:, None] * lx[:, None] * grad_bilinear,
                                mask=f01,
                                sem="relaxed",
                            )
                            tl.atomic_add(
                                gxbase + o10,
                                ly[:, None] * hx[:, None] * grad_bilinear,
                                mask=f10,
                                sem="relaxed",
                            )
                            tl.atomic_add(
                                gxbase + o11,
                                ly[:, None] * lx[:, None] * grad_bilinear,
                                mask=f11,
                                sem="relaxed",
                            )

                if NEEDS_OFFSET_GRAD:
                    gobase = goff_ptr + batch * so_n + m * so_p + og * so_g + k * so_k
                    tl.atomic_add(gobase + 0 * so_c, acc_dy, mask=m_mask, sem="relaxed")
                    tl.atomic_add(gobase + 1 * so_c, acc_dx, mask=m_mask, sem="relaxed")
                if NEEDS_MASK_GRAD:
                    tl.atomic_add(
                        gmsk_ptr + batch * sm_n + m * sm_p + og * sm_g + k * sm_k,
                        acc_m,
                        mask=m_mask,
                        sem="relaxed",
                    )


def backward_launch(  # noqa: PLR0913
    grad_output: torch.Tensor,
    value: torch.Tensor,
    offset: torch.Tensor,
    mask: torch.Tensor | None,
    weight: torch.Tensor,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dilation_h: int,
    dilation_w: int,
    groups: int,
    offset_groups: int,
    *,
    needs_value_grad: bool,
    needs_offset_grad: bool,
    needs_mask_grad: bool,
    needs_weight_grad: bool,
    bias_dtype: torch.dtype | None,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Compute the gradients autograd asked for; the others are ``None`` and free.

    Accumulators are float32 for every operand dtype; see
    ``docs/precision.md``.

    Parameters
    ----------
    grad_output
        ``(B, H_out, W_out, C_out)``, channels-last.
    value
        ``(B, H, W, C_in)``, channels-last.
    offset
        ``(B, H_out, W_out, offset_groups, K, 2)``, holding ``(dy, dx)``.
    mask
        ``(B, H_out, W_out, offset_groups, K)``, or ``None`` for v1.
    weight
        ``(C_out, C_in / groups, kH, kW)``.
    stride_h, stride_w
        Stride of the convolution.
    pad_h, pad_w
        Padding applied to ``value``.
    dilation_h, dilation_w
        Dilation of the convolution.
    groups
        Convolution groups, over input and output channels.
    offset_groups
        Offset groups, over input channels only.
    needs_value_grad, needs_offset_grad, needs_mask_grad, needs_weight_grad
        Whether autograd asked for that output.
    bias_dtype
        The bias dtype if it requires a gradient, else ``None``.

    Returns
    -------
    ``(grad_value, grad_offset, grad_mask, grad_weight, grad_bias)``, each
    shaped like the operand it is the gradient of, or ``None`` for any
    output not requested.
    """
    # The bias gradient is a host-side sum; alone, it needs no launch.
    needs_kernel = (
        needs_value_grad or needs_offset_grad or needs_mask_grad or needs_weight_grad
    )
    batch, height, width, in_channels = value.shape
    out_channels, in_per_group, kernel_h, kernel_w = weight.shape
    points = kernel_h * kernel_w
    _, out_h, out_w, _, _, _ = offset.shape
    positions = out_h * out_w

    grad_flat = grad_output.reshape(batch, positions, out_channels)

    # The bias dtype, not the weight's: the two can differ in width.
    grad_bias = (
        grad_flat.sum(dim=(0, 1)).to(bias_dtype) if bias_dtype is not None else None
    )
    if not needs_kernel:
        return None, None, None, None, grad_bias

    # After the early return: the permutation copies, and a frozen weight skips it.
    weight_perm = permute_weight(weight)
    offset_flat = offset.reshape(batch, positions, offset_groups, points, 2)
    mask_flat = (
        mask.reshape(batch, positions, offset_groups, points)
        if mask is not None
        else None
    )

    # One placeholder dtype for untouched pointers avoids recompiles.
    empty = value.new_zeros(1, dtype=torch.float32)
    grad_value = (
        torch.zeros_like(value, dtype=torch.float32) if needs_value_grad else empty
    )
    grad_offset = (
        torch.zeros_like(offset_flat, dtype=torch.float32)
        if needs_offset_grad
        else empty
    )
    grad_mask = (
        torch.zeros_like(mask_flat, dtype=torch.float32)
        if needs_mask_grad and mask_flat is not None
        else empty
    )
    grad_weight_perm = (
        torch.zeros_like(weight_perm, dtype=torch.float32)
        if needs_weight_grad
        else empty
    )

    out_per_group = out_channels // groups

    def grid(meta: dict) -> tuple[int, int, int]:
        # ``cdiv`` is typed as returning ``constexpr``; a launch grid is ints.
        return (
            int(triton.cdiv(positions, meta["BLOCK_M"])),
            groups * int(triton.cdiv(out_per_group, meta["BLOCK_N"])),
            batch,
        )

    wrap_triton(_conv_backward_kernel)[grid](  # pyright: ignore[reportArgumentType]
        value,
        offset_flat,
        mask_flat if mask_flat is not None else empty,
        weight_perm,
        grad_flat,
        grad_value,
        grad_offset,
        grad_mask,
        grad_weight_perm,
        height,
        width,
        out_w,
        positions,
        tune_bucket(positions),
        in_per_group,
        out_per_group,
        in_channels // offset_groups,
        *value.stride(),
        *offset_flat.stride(),
        *(mask_flat.stride() if mask_flat is not None else (0, 0, 0, 0)),
        *weight_perm.stride(),
        *grad_flat.stride(),
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        dilation_h=dilation_h,
        dilation_w=dilation_w,
        offset_groups=offset_groups,
        kernel_w=kernel_w,  # pyright: ignore[reportArgumentType]
        num_points=points,  # pyright: ignore[reportArgumentType]
        HAS_MASK=mask is not None,  # pyright: ignore[reportArgumentType]
        NEEDS_VALUE_GRAD=needs_value_grad,  # pyright: ignore[reportArgumentType]
        NEEDS_OFFSET_GRAD=needs_offset_grad,  # pyright: ignore[reportArgumentType]
        NEEDS_MASK_GRAD=needs_mask_grad,  # pyright: ignore[reportArgumentType]
        NEEDS_WEIGHT_GRAD=needs_weight_grad,  # pyright: ignore[reportArgumentType]
        ALLOW_TF32=_allow_tf32(),  # pyright: ignore[reportArgumentType]
    )

    # Undo the host-side permutation the kernel contracts against.
    grad_weight = (
        grad_weight_perm.reshape(kernel_h, kernel_w, in_per_group, out_channels)
        .permute(3, 2, 0, 1)
        .contiguous()
        .to(weight.dtype)
        if needs_weight_grad
        else None
    )
    return (
        grad_value.to(value.dtype) if needs_value_grad else None,
        grad_offset.reshape(offset.shape).to(offset.dtype)
        if needs_offset_grad
        else None,
        grad_mask.reshape(mask.shape).to(mask.dtype)
        if needs_mask_grad and mask is not None
        else None,
        grad_weight,
        grad_bias,
    )
