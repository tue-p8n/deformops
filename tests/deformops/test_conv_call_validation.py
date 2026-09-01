"""Operand validation at the classic deformable convolution entry point.

The kernel indexes the value tensor straight from the caller's geometry, so each
mismatch would otherwise be an out-of-bounds device read. Tests match the message
as well as the type.
"""

from __future__ import annotations

import pytest
import torch
from deformops.core.sanitize import check_conv_call

#: A 3x3 kernel at stride one with a one-pixel pad: a 6x6 input gives a 6x6 output.
_GEOMETRY = {
    "stride_h": 1,
    "stride_w": 1,
    "pad_h": 1,
    "pad_w": 1,
    "dilation_h": 1,
    "dilation_w": 1,
}


def _operands() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A consistent set of classic-deform-conv operands, channels-last value."""
    groups, offset_groups = 1, 1
    in_channels = out_channels = 4
    value = torch.randn(1, 6, 6, in_channels)
    offset = torch.randn(1, 6, 6, offset_groups, 9, 2)
    weight = torch.randn(out_channels, in_channels // groups, 3, 3)
    return value, offset, weight


def test_consistent_operands_are_accepted() -> None:
    value, offset, weight = _operands()
    check_conv_call(value, offset, None, weight, None, 1, 1, **_GEOMETRY)


def test_rejects_a_batch_that_overflows_int32() -> None:
    """The batch-multiplied address check of the sampling operator, input and output."""
    h = w = 1 << 15
    value = torch.empty(1, 1, 1, 1).expand(4, h, w, 1)
    offset = torch.empty(1, 1, 1, 1, 1, 2).expand(4, h, w, 1, 1, 2)
    weight = torch.randn(1, 1, 1, 1)
    with pytest.raises(ValueError, match=r"exceeds the int32 address space"):
        check_conv_call(
            value,
            offset,
            None,
            weight,
            None,
            1,
            1,
            stride_h=1,
            stride_w=1,
            pad_h=0,
            pad_w=0,
            dilation_h=1,
            dilation_w=1,
        )
