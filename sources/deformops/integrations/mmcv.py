r"""MMCV multi-scale deformable attention on deformops.

``mmcv-lite`` ships without MMCV's compiled extension, so its module-level
``multi_scale_deformable_attn_pytorch`` is the live path. This module replaces
that function, the attention module, or both (:func:`patch`). Needs
``deformops[mmcv]``.
"""

from __future__ import annotations

import importlib
import importlib.util
import types
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from ._ops import run_msda
from ._patch import patch_modules

if TYPE_CHECKING:
    from deformops.registry import Backend, MSDABackend


def _import_mmcv_ops() -> Any:
    """Import ``mmcv.ops``, loading an empty namespace for a missing ``mmcv._ext``.

    ``mmcv.ops`` loads the extension at import even under ``mmcv-lite``, which
    lacks it; deformops replaces the op that would have used it.
    """
    ext_loader: Any = importlib.import_module("mmcv.utils.ext_loader")  # noqa: PLC0415

    real_load_ext = ext_loader.load_ext

    def _load_ext_or_empty(*args: Any, **kwargs: Any) -> Any:
        try:
            return real_load_ext(*args, **kwargs)
        except (ImportError, ModuleNotFoundError):
            return types.SimpleNamespace()

    ext_loader.load_ext = _load_ext_or_empty
    return importlib.import_module("mmcv.ops.multi_scale_deform_attn")


if importlib.util.find_spec("mmcv") is None:  # pragma: no cover
    msg = (
        "deformops.integrations.mmcv requires the optional 'mmcv-lite' "
        "dependency. Install it with `pip install deformops[mmcv]`."
    )
    raise ModuleNotFoundError(msg)

try:
    _mmcv_module: Any = _import_mmcv_ops()
    _MMCVMultiScaleDeformableAttention: Any = _mmcv_module.MultiScaleDeformableAttention
except (ImportError, AttributeError) as exc:  # pragma: no cover
    import importlib  # noqa: PLC0415

    mmcv: Any = importlib.import_module("mmcv")
    msg = (
        f"The installed mmcv {getattr(mmcv, '__version__', 'unknown')} does not "
        f"provide the multi-scale deformable attention module deformops "
        f"replaces ({type(exc).__name__}: {exc})."
    )
    raise ImportError(msg) from exc

__all__ = [
    "MultiScaleDeformableAttention",
    "patch",
    "patch_multi_scale_deformable_attn_pytorch",
]

#: ``reference_points`` last-dim choices, per the mmcv operator.
_REFERENCE_POINTS_DIM = 2
_REFERENCE_POINTS_DIM_WITH_SCALE = 4


class MultiScaleDeformableAttention(_MMCVMultiScaleDeformableAttention):
    """MMCV's MSDA, backed by deformops.

    Reuses the module's projections; only the aggregation changes.
    """

    batch_first: bool
    embed_dims: int
    num_heads: int
    num_levels: int
    num_points: int
    chunk_step: int
    backend: Backend | MSDABackend | None

    def __init__(
        self,
        other: Any,
        *,
        backend: Backend | MSDABackend | None = None,
    ) -> None:
        nn.Module.__init__(self)
        self.batch_first = other.batch_first
        self.embed_dims = other.embed_dims
        self.num_heads = other.num_heads
        self.num_levels = other.num_levels
        self.num_points = other.num_points
        self.chunk_step = other.im2col_step
        self.backend = backend

        self.value_proj = other.value_proj
        self.sampling_offsets = other.sampling_offsets
        self.attention_weights = other.attention_weights
        self.output_proj = other.output_proj
        self.dropout = other.dropout

    def forward(  # noqa: PLR0913
        self,
        query: Tensor,
        key: Tensor | None = None,
        value: Tensor | None = None,
        identity: Tensor | None = None,
        query_pos: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        reference_points: Tensor | None = None,
        spatial_shapes: Tensor | None = None,
        level_start_index: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Sample and aggregate with deformops, then project and residual."""
        del key, level_start_index, kwargs
        value_t: Tensor = query if value is None else value
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value_t = value_t.permute(1, 0, 2)

        batch_size, num_query, _ = query.shape
        num_value = value_t.shape[1]
        if reference_points is None or spatial_shapes is None:
            msg = "reference_points and spatial_shapes are required"
            raise ValueError(msg)

        value_t = self.value_proj(value_t)
        if key_padding_mask is not None:
            value_t = value_t.masked_fill(key_padding_mask[..., None], 0.0)
        value_t = value_t.view(
            batch_size, num_value, self.num_heads, self.embed_dims // self.num_heads
        )

        sampling_offsets = self.sampling_offsets(query).view(
            batch_size, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            batch_size, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1)

        if reference_points.shape[-1] == _REFERENCE_POINTS_DIM:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == _REFERENCE_POINTS_DIM_WITH_SCALE:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.num_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            msg = (
                f"Last dim of reference_points must be 2 or 4, but got "
                f"{reference_points.shape[-1]}"
            )
            raise ValueError(msg)

        output = run_msda(
            value_t,
            spatial_shapes,
            None,
            sampling_locations,
            attention_weights,
            chunk_step=self.chunk_step,
            backend=self.backend,
        )
        output = self.output_proj(output)
        if not self.batch_first:
            output = output.permute(1, 0, 2)
        return self.dropout(output) + identity


def patch_multi_scale_deformable_attn_pytorch(
    *,
    backend: Backend | MSDABackend | None = None,
) -> Any:
    """Replace mmcv's PyTorch deformable-attention function with deformops."""

    def _deformops_multi_scale_deformable_attn_pytorch(
        value: Tensor,
        value_spatial_shapes: Tensor,
        sampling_locations: Tensor,
        attention_weights: Tensor,
    ) -> Tensor:
        return run_msda(
            value,
            value_spatial_shapes.to(torch.long),
            None,
            sampling_locations,
            attention_weights,
            chunk_step=128,
            backend=backend,
        )

    _mmcv_module.multi_scale_deformable_attn_pytorch = (
        _deformops_multi_scale_deformable_attn_pytorch
    )
    return _deformops_multi_scale_deformable_attn_pytorch


def patch(
    model: nn.Module,
    *,
    backend: Backend | MSDABackend | None = None,
    strict: bool = True,
) -> nn.Module:
    """Route a model's MMCV attention instances and function through deformops.

    Parameters
    ----------
    model
        The model, patched in place.
    backend
        Backend for every replacement; ``None`` selects one per call.
    strict
        Raise when the model contains no MMCV deformable attention module.

    Returns
    -------
    ``model``, for chaining.

    Raises
    ------
    RuntimeError
        If ``strict`` is set and no attention module was replaced.
    """
    patch_modules(
        model,
        [
            (
                _MMCVMultiScaleDeformableAttention,
                lambda child: MultiScaleDeformableAttention(child, backend=backend),
            )
        ],
        strict=strict,
    )
    patch_multi_scale_deformable_attn_pytorch(backend=backend)
    return model
