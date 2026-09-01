r"""Layer interfaces onto single-scale deformable sampling."""

import math
from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    override,
)

import torch
from torch import Tensor, nn

from ..core.runtime import Method as AttnMethod
from ..core.sanitize import conv_output_size
from ..registry import Backend

__all__ = ["DeformAttn2d", "DeformSample2d"]


def _same_padding(kernel_size: int, dilation: int) -> int:
    """The padding that keeps a stride-1 output grid the size of its input."""
    return dilation * (kernel_size - 1) // 2


def _center_start(kernel_size: int, padding: int, dilation: int) -> int:
    """Input index of the first output location's kernel center."""
    return dilation * (kernel_size // 2) - padding


class _ScaleLinear(nn.Module):
    r"""Per-group sigmoid gate."""

    scale_weight: nn.Parameter
    scale_bias: nn.Parameter

    def __init__(self, dim: int, groups: int):
        super().__init__()

        self.scale_weight = nn.Parameter(
            torch.zeros((groups, dim), dtype=torch.float32)
        )
        self.scale_bias = nn.Parameter(torch.zeros((groups,), dtype=torch.float32))

    @override
    def forward(self, query: Tensor) -> Tensor:
        """Per-group gate values."""
        return nn.functional.linear(
            query,
            weight=self.scale_weight,
            bias=self.scale_bias,
        ).sigmoid()


class DeformAttn2d[Project: nn.Module](nn.Module):
    r"""Single-scale 2D deformable sampling layer (DCNv3, InternImage).

    Per output location and group, a linear map of the query predicts an offset and
    a weight for each point of a square kernel. The layer samples the projected
    value there through :data:`deformops.registry.attn`, sums the weighted samples
    and projects the result out. The offset and weight
    projection starts at zero: a new layer samples the undisplaced grid with
    zero weights. Citation: see the README.

    Parameters
    ----------
    dim
        Channels of the query, the value and the output. ``dim // groups``
        must be a whole multiple of 16.
    kernel_size, stride, dilation
        The square sampling kernel and its geometry over the value.
    padding
        Padding per side, or ``"same"`` for ``dilation * (kernel_size - 1) //
        2``, which keeps a stride-1 output the size of the input.
    groups
        Channel groups, each with its own offsets and weights.
    spread
        Scale of each displaced grid point about the kernel center,
        applied to the base grid and the predicted offset alike.
    center_gate
        Blend the output with the projected value at each kernel center,
        ``out * (1 - gate) + center * gate``, with ``gate`` a per-group sigmoid
        of a linear map of the output. Needs an odd ``kernel_size`` and
        ``padding`` at most ``dilation * (kernel_size - 1) // 2``.
    drop_center
        Drop the center point, leaving
        ``kernel_size ** 2 - 1`` points per group.
    project
        Module class called as ``project(dim, dim)`` for the value and output
        projections, or ``None`` for neither.
    method
        Weight normalization: ``"linear"`` uses the
        weights as predicted, ``"softmax"`` normalizes over each group's points.
    backend
        A :class:`~deformops.registry.Backend` member, or ``None`` for
        :func:`deformops.registry.attn.select_backend`.
    norm
        Module applied to the output after the output projection, or ``None``.
    activation
        Module applied after ``norm``, or ``None``.

    Raises
    ------
    ValueError
        If ``dim`` is not divisible by ``groups``, ``dim // groups`` by 16, or
        ``center_gate`` meets an even ``kernel_size`` or too much ``padding``.
    """

    dim: Final[int]
    kernel_size: Final[int]
    stride: Final[int]
    dilation: Final[int]
    padding: Final[int]
    groups: Final[int]
    group_dims: Final[int]
    spread: Final[float]
    center_gate: Final[bool]
    drop_center: Final[bool]
    method: Final[AttnMethod]
    backend: Final[Backend | None]

    proj_offset_mask: nn.Linear
    proj_value: Project | None
    proj_output: Project | None
    center_scale: _ScaleLinear | None
    norm: nn.Module | None
    activation: nn.Module | None

    def __init__(  # noqa: PLR0913
        self,
        dim: int,
        kernel_size: int = 3,
        *,
        stride: int = 1,
        padding: int | Literal["same"] = "same",
        dilation: int = 1,
        groups: int = 4,
        spread: float = 1.0,
        center_gate: bool = False,
        drop_center: bool = False,
        project: type[Project] | None = nn.Linear,
        method: AttnMethod | Literal["softmax", "linear"] = AttnMethod.LINEAR,
        backend: Backend | None = None,
        norm: nn.Module | None = None,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()

        if dim % groups != 0:
            msg = f"{dim=} must be divisible by {groups=}"
            raise ValueError(msg)
        if (dim_per_group := dim // groups) % 16 != 0:  # noqa: PLR2004
            msg = f"channels per group ({dim_per_group}) must be divisible by 16"
            raise ValueError(msg)

        self.dim = dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        if padding == "same":
            padding = _same_padding(kernel_size, dilation)
        self.padding = padding
        if center_gate and kernel_size % 2 == 0:
            msg = f"center_gate needs an odd kernel_size, got {kernel_size}"
            raise ValueError(msg)
        if center_gate and padding > _same_padding(kernel_size, dilation):
            msg = (
                f"center_gate needs every kernel center inside the input, which "
                f"padding={padding} exceeds for kernel_size={kernel_size} and "
                f"dilation={dilation}; use at most {_same_padding(kernel_size, dilation)}"
            )
            raise ValueError(msg)
        self.groups = groups
        self.group_dims = dim // groups
        self.spread = spread
        self.center_gate = center_gate
        self.drop_center = drop_center
        self.method = AttnMethod(method)
        self.backend = backend

        kernel_numel = self.groups * (
            self.kernel_size * self.kernel_size - int(self.drop_center)
        )
        self.proj_offset_mask = nn.Linear(
            dim, int(math.ceil((kernel_numel * 3) / 8) * 8)
        )

        if project is not None:
            self.proj_value = project(dim, dim)
            self.proj_output = project(dim, dim)
        else:
            self.register_module("proj_value", None)
            self.register_module("proj_output", None)

        if center_gate:
            self.center_scale = _ScaleLinear(dim, groups)
        else:
            self.register_module("center_scale", None)

        self.norm = norm
        self.activation = activation

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        bias = self.proj_offset_mask.bias
        if bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
            _ = nn.init.zeros_(bias.data)
        _ = nn.init.zeros_(self.proj_offset_mask.weight.data)

        for mod in [self.proj_value, self.proj_output]:
            if mod is None:
                continue
            reset_fn = getattr(mod, "reset_parameters", None)
            if callable(reset_fn):
                reset_fn()
                continue
            assert isinstance(mod, nn.Module)
            mod_bias = getattr(mod, "bias", None)
            if isinstance(mod_bias, Tensor):
                _ = nn.init.zeros_(mod_bias.data)
            mod_weight = getattr(mod, "weight", None)
            if isinstance(mod_weight, Tensor):
                _ = nn.init.xavier_uniform_(mod_weight.data)

    @override
    def forward(  # noqa: C901
        self,
        q: Tensor,
        v: Tensor,
        shape: torch.Size | tuple[int, int] | None = None,
    ) -> Tensor:
        """Sample ``v`` at query-derived offsets and aggregate.

        The query and the value each come as NCHW or flattened channels-last
        tokens, independently; ``H_out``, ``W_out`` is the output grid.

        Parameters
        ----------
        q
            The query, ``(N, dim, H_out, W_out)`` or
            ``(N, H_out * W_out, dim)``.
        v
            The value, ``(N, dim, H, W)`` or ``(N, H * W, dim)``.
        shape
            The value's ``(H, W)``. Required when ``v`` is flattened;
            otherwise defaults to ``v``'s spatial size.

        Returns
        -------
        The output in the query's layout: ``(N, dim, H_out, W_out)`` for a
        4D ``q``, ``(N, H_out * W_out, dim)`` for a flattened one.

        Raises
        ------
        ValueError
            If ``v`` is flattened and ``shape`` is not given, or the operands
            disagree with the layer's geometry.
        """
        ndim_q = q.ndim
        H_q, W_q = None, None
        if ndim_q == 4:  # noqa: PLR2004
            H_q, W_q = q.shape[-2:]
            q = q.flatten(2).permute(0, 2, 1).contiguous()

        ndim_v = v.ndim
        if ndim_v == 4:  # noqa: PLR2004
            if shape is None:
                shape = v.shape[-2:]
            v = v.flatten(2).permute(0, 2, 1).contiguous()

        if shape is None:
            msg = "shape must be given when v is a flattened (N, H * W, C) tensor"
            raise ValueError(msg)
        H, W = shape
        N, _L, _C = v.shape

        if self.proj_value is not None:
            v = self.proj_value(v)

        if H_q is None:
            H_q = conv_output_size(
                H, self.kernel_size, self.stride, self.padding, self.dilation
            )
            W_q = conv_output_size(
                W, self.kernel_size, self.stride, self.padding, self.dilation
            )

        offset_mask = self.proj_offset_mask(q).reshape(N, H_q, W_q, -1)
        v_grid = v.reshape(N, H, W, -1)

        from ..registry import (  # noqa: PLC0415
            attn as _attn_forward,
        )

        out = _attn_forward(
            v_grid,
            offset_mask,
            kernel_size=(self.kernel_size, self.kernel_size),
            stride=(self.stride, self.stride),
            padding=(self.padding, self.padding),
            dilation=(self.dilation, self.dilation),
            groups=self.groups,
            group_dims=self.group_dims,
            spread=self.spread,
            method=self.method,
            drop_center=self.drop_center,
            backend=self.backend,
        )

        H_OUT, W_OUT = out.shape[1:3]

        if self.center_scale is not None:
            gate = self.center_scale(out)
            gate = (
                gate[..., None].repeat(1, 1, 1, 1, self.dim // self.groups).flatten(-2)
            )
            # The value at each window's kernel center.
            start = _center_start(self.kernel_size, self.padding, self.dilation)
            center = v_grid[:, start :: self.stride, start :: self.stride][
                :, :H_OUT, :W_OUT
            ]
            out = out * (1 - gate) + center * gate

        out = out.flatten(1, 2)
        if self.proj_output is not None:
            out = self.proj_output(out)

        if ndim_q == 4:  # noqa: PLR2004
            out = out.permute(0, 2, 1).unflatten(2, (H_OUT, W_OUT)).contiguous()

        if self.norm is not None:
            out = self.norm(out)
        if self.activation is not None:
            out = self.activation(out)

        return out

    if TYPE_CHECKING:
        __call__ = forward


class DeformSample2d[Project: nn.Module](nn.Module):
    r"""Deformable sampling of an input at offsets predicted from itself.

    A depthwise convolution with the sampler's geometry derives the query from the
    input. A :class:`DeformAttn2d` built as ``sample`` samples the input at the
    offsets it predicts, on the same output grid. Parameters,
    constraints, defaults and raised errors are those of :class:`DeformAttn2d`.
    """

    offset: nn.Conv2d
    sample: DeformAttn2d[Project]

    def __init__(  # noqa: PLR0913
        self,
        dim: int,
        kernel_size: int = 3,
        *,
        stride: int = 1,
        padding: int | Literal["same"] = "same",
        dilation: int = 1,
        groups: int = 4,
        spread: float = 1.0,
        center_gate: bool = False,
        drop_center: bool = False,
        project: type[Project] | None = nn.Linear,
        method: Literal["softmax", "linear"] = "linear",
        backend: Backend | None = None,
        norm: nn.Module | None = None,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()

        if padding == "same":
            padding = _same_padding(kernel_size, dilation)

        self.offset = nn.Conv2d(
            dim,
            dim,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            padding_mode="zeros",
            groups=dim,
        )

        self.sample = DeformAttn2d(
            dim,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            spread=spread,
            center_gate=center_gate,
            drop_center=drop_center,
            project=project,
            method=method,
            backend=backend,
            norm=norm,
            activation=activation,
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        if self.offset.bias is not None:
            _ = nn.init.zeros_(self.offset.bias.data)
        _ = nn.init.kaiming_normal_(self.offset.weight.data, a=1, mode="fan_in")

    @override
    def forward(
        self,
        input: Tensor,
        shape: torch.Size | tuple[int, int] | None = None,
    ) -> Tensor:
        """Compute per-pixel offsets from ``input`` and sample it at them.

        Parameters
        ----------
        input
            ``(N, dim, H, W)``, or flattened channels-last tokens
            ``(N, H * W, dim)``.
        shape
            The input's ``(H, W)``. Required for flattened tokens and rejected
            for a 4D input.

        Returns
        -------
        ``(N, dim, H_out, W_out)`` for a 4D input,
        ``(N, H_out * W_out, dim)`` for flattened tokens.

        Raises
        ------
        ValueError
            If ``shape`` is missing for flattened tokens or given for a 4D
            input.
        """
        ndim_input = input.ndim
        if ndim_input == 4:  # noqa: PLR2004
            if shape is not None:
                msg = "shape must not be given for a (N, C, H, W) input"
                raise ValueError(msg)
            shape = input.shape[-2:]
            x_4d = input
        else:
            if shape is None:
                msg = "shape must be given for a flattened (N, H * W, C) input"
                raise ValueError(msg)
            H, W = shape
            N, _L, C = input.shape
            x_4d = input.view(N, H, W, C).permute(0, 3, 1, 2).contiguous()

        q = self.offset(x_4d)

        if ndim_input == 3:  # noqa: PLR2004
            q = q.permute(0, 2, 3, 1).flatten(1, 2)

        return self.sample(q, input, shape=shape)

    if TYPE_CHECKING:
        __call__ = forward
