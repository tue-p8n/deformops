r"""Layer interfaces onto classic deformable convolution."""

import math
from collections.abc import Sequence
from typing import override

import torch
from torch import Tensor, nn

from ..core.sanitize import check_conv_params
from ..registry import Backend, conv

__all__ = ["DeformConv2d"]


def _pair(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    first, second = value
    return (int(first), int(second))


class DeformConv2d(nn.Module):
    """Deformable convolution, matching ``torchvision.ops.DeformConv2d``.

    Constructor, parameters, initialization, state dict and NCHW ``forward``
    are interchangeable with torchvision's. Without a mask it is v1, with one
    the modulated v2; the operator runs through :data:`deformops.registry.conv`.

    Introduced by Deformable Convolutional Networks (Dai et al., ICCV 2017); the
    mask comes from Deformable ConvNets v2 (Zhu et al., CVPR 2019). See the
    Citation section of the README.

    Parameters
    ----------
    in_channels, out_channels
        Channel counts ``C_in`` and ``C_out``, both divisible by ``groups``.
    kernel_size, stride, padding, dilation
        Convolution geometry, each an ``int`` or an ``(h, w)`` pair.
    groups
        Convolution groups.
    bias
        Whether to learn a per-output-channel bias.
    backend
        A :class:`~deformops.registry.Backend` member, or ``None`` for
        :func:`deformops.registry.conv.select_backend`.

    Raises
    ------
    ValueError
        If a channel or group count is not positive, or ``in_channels`` or
        ``out_channels`` is not divisible by ``groups``.
    """

    backend: Backend | None

    def __init__(  # noqa: PLR0913
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        bias: bool = True,  # noqa: FBT001, FBT002 - torchvision's signature
        backend: Backend | None = None,
    ) -> None:
        super().__init__()
        check_conv_params(in_channels, out_channels, groups)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.backend = backend

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Kaiming-uniform weight and matching bias, as torchvision does."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
            fan_in = self.in_channels // self.groups * math.prod(self.kernel_size)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @override
    def forward(
        self,
        input: Tensor,
        offset: Tensor,
        mask: Tensor | None = None,  # noqa: A002
    ) -> Tensor:
        """Deformable convolution over NCHW operands.

        ``K = kH * kW``; ``offset_groups`` is inferred from ``offset``.

        Parameters
        ----------
        input
            The input, shape ``(B, C_in, H, W)``.
        offset
            Offsets in input pixels, ``(B, 2 * offset_groups * K, H_out,
            W_out)``, channels grouped ``(offset_groups, K, 2)`` as ``(dy, dx)``
            over row-major points, as in torchvision. ``in_channels`` must be
            divisible by ``offset_groups``.
        mask
            Per-point modulation for v2, shape
            ``(B, offset_groups * K, H_out, W_out)``, or ``None`` for v1.

        Returns
        -------
        The output, shape ``(B, C_out, H_out, W_out)``.

        Raises
        ------
        ValueError
            If ``input`` does not have ``in_channels`` channels, or the
            operands disagree with each other or with the geometry.
        """
        if input.shape[1] != self.in_channels:
            msg = (
                f"input has {input.shape[1]} channels, but this layer was "
                f"built for in_channels={self.in_channels}"
            )
            raise ValueError(msg)

        batch = input.shape[0]
        _, _, kernel_h, kernel_w = self.weight.shape
        points = kernel_h * kernel_w
        out_h, out_w = offset.shape[2], offset.shape[3]
        offset_groups = offset.shape[1] // (2 * points)

        offset_native = (
            offset.reshape(batch, offset_groups, points, 2, out_h, out_w)
            .permute(0, 4, 5, 1, 2, 3)
            .contiguous()
        )
        mask_native = (
            mask.reshape(batch, offset_groups, points, out_h, out_w)
            .permute(0, 3, 4, 1, 2)
            .contiguous()
            if mask is not None
            else None
        )
        input_cl = input.permute(0, 2, 3, 1)
        if not input_cl.is_contiguous():
            input_cl = input_cl.contiguous()
        out = conv(
            input_cl,
            offset_native,
            self.weight,
            bias=self.bias,
            mask=mask_native,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            offset_groups=offset_groups,
            backend=self.backend,
        )
        permuted = out.permute(0, 3, 1, 2)
        return permuted if permuted.is_contiguous() else permuted.contiguous()

    @override
    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, "
            f"groups={self.groups}, bias={self.bias is not None})"  # pyright: ignore[reportUnnecessaryComparison]
        )
