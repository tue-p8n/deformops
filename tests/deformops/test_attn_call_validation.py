"""Operand validation at the single-scale deformable sampling entry point.

The kernel indexes the value tensor straight from the caller's geometry, so each
mismatch would otherwise be an out-of-bounds device read. Tests match the message
as well as the type.
"""

from __future__ import annotations

import pytest
import torch
from deformops.core.sanitize import check_attn_call
from deformops.registry import Backend, attn

#: A 3x3 kernel at stride one with a one-pixel pad: a 6x6 input gives a 6x6 output.
_GEOMETRY = {
    "stride_h": 1,
    "stride_w": 1,
    "pad_h": 1,
    "pad_w": 1,
    "dilation_h": 1,
    "dilation_w": 1,
}


def _operands() -> tuple[torch.Tensor, torch.Tensor]:
    """A consistent set of single-scale operands, channels-last."""
    groups, group_dims = 2, 4
    value = torch.randn(1, 6, 6, groups * group_dims)
    offset_mask = torch.randn(1, 6, 6, groups * 9 * 3)
    return value, offset_mask


def test_consistent_operands_are_accepted() -> None:
    value, offset_mask = _operands()
    check_attn_call(value, offset_mask, 3, 3, 2, 4, **_GEOMETRY, drop_center=False)


def test_channel_count_must_match_groups() -> None:
    value, offset_mask = _operands()
    with pytest.raises(ValueError, match=r"value has size 8 on axis 3"):
        check_attn_call(value, offset_mask, 3, 3, 4, 4, **_GEOMETRY, drop_center=False)


def test_offset_mask_must_be_wide_enough() -> None:
    value, _ = _operands()
    short = torch.randn(1, 6, 6, 6)
    with pytest.raises(ValueError, match=r"offset_mask has 6 channels, need at least"):
        check_attn_call(value, short, 3, 3, 2, 4, **_GEOMETRY, drop_center=False)


def test_batch_must_agree() -> None:
    value, _ = _operands()
    other_batch = torch.randn(3, 6, 6, 2 * 9 * 3)
    with pytest.raises(ValueError, match=r"offset_mask has size 3 on axis 0"):
        check_attn_call(value, other_batch, 3, 3, 2, 4, **_GEOMETRY, drop_center=False)


def test_kernel_without_sampling_points_is_rejected() -> None:
    value, offset_mask = _operands()
    with pytest.raises(ValueError, match=r"leaves no sampling points"):
        check_attn_call(value, offset_mask, 1, 1, 2, 4, **_GEOMETRY, drop_center=True)


def test_apply_rejects_before_reaching_a_backend() -> None:
    value, _ = _operands()
    short = torch.randn(1, 6, 6, 6)
    with pytest.raises(ValueError, match=r"need at least"):
        attn.apply(value, short, 3, groups=2, group_dims=4, backend=Backend.EAGER)


def test_rejects_a_batch_that_overflows_int32_although_per_image_size_does_not() -> (
    None
):
    """The address check includes the batch, as the kernel's address does.

    ``expand`` costs no memory: 2**30 elements per image fit int32, and a batch of 4
    pushes the kernel's pointer arithmetic past 2**31 - 1.
    """
    h = w = 1 << 15
    groups, group_dims = 1, 1
    value = torch.empty(1, 1, 1, groups * group_dims).expand(
        4, h, w, groups * group_dims
    )
    offset_mask = torch.empty(1, 1, 1, groups * 3).expand(4, h, w, groups * 3)
    with pytest.raises(ValueError, match=r"exceeds the int32 address space"):
        check_attn_call(
            value,
            offset_mask,
            1,
            1,
            groups,
            group_dims,
            stride_h=1,
            stride_w=1,
            pad_h=0,
            pad_w=0,
            dilation_h=1,
            dilation_w=1,
            drop_center=False,
        )
