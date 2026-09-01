r"""Multi-scale deformable attention (MSDA) layer.

``MSDAMethod`` is :class:`deformops.core.runtime.Method` under this layer's name.
"""

import math
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    override,
)

import torch
from torch import Tensor, nn

from ..core.runtime import Method as MSDAMethod
from ..core.sanitize import check_2power, check_msda_params
from ..registry import Backend, MSDABackend

__all__ = ["MSDAMethod", "MultiScaleDeformAttn2d"]


class MultiScaleDeformAttn2d(nn.Module):
    r"""Multi-scale deformable attention layer.

    Each query predicts, per head, ``num_levels * num_points`` offsets around its
    reference point and a weight per sample. The layer samples the projected value
    pyramid at those offsets through :data:`deformops.registry.msda`, sums the
    weighted samples and projects the result out.

    Introduced by Deformable DETR (Zhu et al., ICLR 2021). The kernels are from
    "Efficient Multi-Scale Deformable Attention on GPUs" (Stolle, TMLR 2026). See
    the Citation section of the README.

    Parameters
    ----------
    dim
        Query and hidden dimension, split evenly across the heads.
    dim_value
        Value token dimension, projected to ``dim``; ``None`` means ``dim``.
    dim_output
        Output dimension; ``None`` means ``dim``.
    num_heads
        Heads ``G``; ``dim // num_heads`` must be a power of two.
    num_levels
        Feature-map levels ``L``.
    num_points
        Sampling points ``K`` per head and level.
    method
        Weight normalization, as an :data:`MSDAMethod` member or its value.
        ``"softmax"`` (the default) normalizes the logits jointly over the
        ``L * K`` samples of each head; ``"linear"`` uses them as predicted.
    backend
        A :class:`~deformops.registry.Backend` member; a
        :class:`~deformops.registry.MSDABackend` instance, called directly
        without the registry's selection and operand checks; or ``None`` for
        :func:`deformops.registry.msda.select_backend`.
    chunk_step
        Batch chunk size, read only by the CUDA backend and passed only to a
        pinned backend; with ``backend=None`` the CUDA kernel uses its default
        of 64. See ``docs/conventions.md``.

    Raises
    ------
    ValueError
        If ``dim``, ``num_heads``, ``num_levels`` or ``num_points`` is not
        positive, ``dim`` is not divisible by ``num_heads``, or
        ``dim // num_heads`` is not a power of two.

    Notes
    -----
    ``load_state_dict`` renames Deformable DETR's parameter names
    (``sampling_offsets``, ``attention_weights``, ``value_proj``,
    ``output_proj``) to ``proj_offset``, ``proj_weights``, ``proj_value`` and
    ``proj_output`` for checkpoints with a metadata version below 2.
    """

    _version = 2

    dim: Final[int]
    num_heads: Final[int]
    num_levels: Final[int]
    num_points: Final[int]

    proj_offset: nn.Linear
    proj_weights: nn.Linear
    proj_value: nn.Linear
    proj_output: nn.Linear

    method: Final[MSDAMethod]
    backend: Final[Backend | MSDABackend | None]
    chunk_step: Final[int]

    def __init__(
        self,
        dim: int,
        dim_value: int | None = None,
        dim_output: int | None = None,
        *,
        num_heads: int,
        num_levels: int,
        num_points: int = 4,
        method: MSDAMethod | str = MSDAMethod.SOFTMAX,
        backend: Backend | MSDABackend | None = None,
        chunk_step: int = 128,
    ):
        super().__init__()

        check_msda_params(dim, num_heads, num_levels, num_points)
        check_2power(dim // num_heads)

        self.dim = dim
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.method = MSDAMethod(method)
        self.backend = backend
        self.chunk_step = chunk_step

        self.proj_offset = nn.Linear(dim, num_heads * num_levels * num_points * 2)
        self.proj_weights = nn.Linear(dim, num_heads * num_levels * num_points)
        self.proj_value = nn.Linear(dim_value or dim, dim)
        self.proj_output = nn.Linear(dim, dim_output or dim)

        self._register_load_state_dict_pre_hook(self._load_detr_state_dict)
        self._reset_parameters()

    def _load_detr_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        *args: object,  # noqa: ARG002
        **kwargs: object,  # noqa: ARG002
    ) -> None:
        version = local_metadata.get("version", 1)
        if isinstance(version, int) and version >= self._version:
            return
        mapping = {
            "sampling_offsets": "proj_offset",
            "attention_weights": "proj_weights",
            "value_proj": "proj_value",
            "output_proj": "proj_output",
        }
        for detr_name, ops_name in mapping.items():
            for suffix in [".weight", ".bias"]:
                key = f"{prefix}{detr_name}{suffix}"
                if key in state_dict:
                    state_dict[f"{prefix}{ops_name}{suffix}"] = state_dict.pop(key)
        # Unmapped keys are kept, so ``load_state_dict`` reports them under ``strict``.

    def _reset_parameters(self):
        H, P, L = self.num_heads, self.num_points, self.num_levels
        nn.init.constant_(self.proj_offset.weight.data, 0.0)
        t = torch.arange(H, dtype=torch.float32) * (2.0 * math.pi / H)
        grid = torch.stack([t.cos(), t.sin()], -1)
        grid = (
            (grid / grid.abs().max(-1, keepdim=True)[0])
            .view(H, 1, 1, 2)
            .repeat(1, L, P, 1)
        )
        for i in range(P):
            grid[:, :, i, :] *= i + 1
        self.proj_offset.bias.data.copy_(grid.reshape(-1))
        nn.init.constant_(self.proj_weights.weight.data, 0.0)
        nn.init.constant_(self.proj_weights.bias.data, 0.0)
        nn.init.xavier_uniform_(self.proj_value.weight.data)
        nn.init.constant_(self.proj_value.bias.data, 0.0)
        nn.init.xavier_uniform_(self.proj_output.weight.data)
        nn.init.constant_(self.proj_output.bias.data, 0.0)

    @override
    def forward(
        self,
        q: Tensor,
        p: Tensor,
        v: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Sample and aggregate ``v`` at offsets predicted from ``q``.

        ``N`` is the value token count summed over the levels.

        Parameters
        ----------
        q
            The queries, shape ``(B, Q, dim)``.
        p
            Reference points per query and level, ``(B, Q, L, 2)`` or
            ``(B, Q, L, 4)``, normalized to ``[0, 1]``. An ``(x, y)`` point takes
            offsets in pixels of each level, divided by the level's
            ``(width, height)``. A ``(cx, cy, w, h)`` box places each sample at
            ``(cx, cy) + offset / num_points * (w, h) / 2``.
        v
            Flattened value pyramid ``(B, N, dim_value)``, levels in the order
            of ``spatial_shapes``.
        spatial_shapes
            Integer ``(height, width)`` per level, ``(L, 2)``.
        level_start_index
            Integer start of each level in ``v``, the exclusive prefix sum of
            ``height * width``, ``(L,)``.
        padding_mask
            Boolean ``(B, N)``, ``True`` at padded tokens, whose projected
            values are zeroed before sampling; or ``None``.

        Returns
        -------
        The attended queries, shape ``(B, Q, dim_output)``.
        """
        B, Q, _ = q.shape
        _, _, _L, P = p.shape
        _, N, _ = v.shape

        v = self.proj_value(v)
        if padding_mask is not None:
            v = v.masked_fill(padding_mask[..., None], 0.0)
        v = v.view(B, N, self.num_heads, self.dim // self.num_heads)

        attention_weights = self.proj_weights(q).view(
            B, Q, self.num_heads, self.num_levels * self.num_points
        )

        loc_off = self.proj_offset(q).view(
            B, Q, self.num_heads, self.num_levels, self.num_points, 2
        )
        if P == 2:  # noqa: PLR2004
            loc_norm = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                p[:, :, None, :, None, :]
                + loc_off / loc_norm[None, None, None, :, None, :]
            )
        else:
            sampling_locations = (
                p[:, :, None, :, None, :2]
                + loc_off / self.num_points * p[:, :, None, :, None, 2:] * 0.5
            )

        if isinstance(self.backend, MSDABackend):
            out = self.backend.forward(
                v,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                attention_weights,
                self.chunk_step,
                self.method,
            )
        else:
            from ..registry import (  # noqa: PLC0415
                msda as _msda_forward,
            )

            # The public operator's schema has no chunk size, so only an explicit
            # backend receives it.
            options: dict[str, Any] = {
                "method": self.method,
                "backend": self.backend,
            }
            if self.backend is not None:
                options["chunk_step"] = self.chunk_step
            out = _msda_forward(
                v,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                attention_weights,
                **options,
            )

        return self.proj_output(out)

    if TYPE_CHECKING:
        __call__ = forward
