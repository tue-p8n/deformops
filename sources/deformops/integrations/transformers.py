r"""Hugging Face multi-scale deformable attention on deformops.

Drop-in replacements for Deformable DETR's parameter-free attention core and for
Mask2Former's attention, which keeps its pretrained projections. :func:`patch`
replaces the ones a model contains. Needs ``deformops[deformable-detr]`` or
``deformops[mask2former]``.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, cast, override

import torch
from torch import Tensor, nn

from ._ops import run_msda
from ._patch import patch_modules

if TYPE_CHECKING:
    from deformops.registry import Backend, MSDABackend

if importlib.util.find_spec("transformers") is None:  # pragma: no cover
    msg = (
        "deformops.integrations.transformers requires the optional "
        "'transformers' dependency. Install it with "
        "`pip install deformops[deformable-detr]` or "
        "`pip install deformops[mask2former]`."
    )
    raise ModuleNotFoundError(msg)

try:
    import transformers.models.deformable_detr.modeling_deformable_detr as _hf_detr
    import transformers.models.mask2former.modeling_mask2former as _hf_mask2former

    _HF_MSDA = _hf_detr.MultiScaleDeformableAttention
    _HF_MSDA_M2F = (
        _hf_mask2former.Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention
    )
except (ImportError, AttributeError) as exc:  # pragma: no cover
    import transformers  # noqa: PLC0415

    msg = (
        f"The installed transformers {transformers.__version__} does not define "
        f"the multi-scale deformable attention classes deformops replaces "
        f"({type(exc).__name__}: {exc}). This integration is tested against "
        f"transformers 4.40 through 5.x."
    )
    raise ImportError(msg) from exc

__all__ = [
    "Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention",
    "MultiScaleDeformableAttention",
    "patch",
]

#: ``reference_points`` last-dim choices, per the Hugging Face operator.
_REFERENCE_POINTS_DIM = 2
_REFERENCE_POINTS_DIM_WITH_SCALE = 4


class MultiScaleDeformableAttention(_HF_MSDA):
    """Hugging Face's parameter-free MSDA core, backed by deformops."""

    #: Opts out of upstream's Hub-kernel tag, which ``kernelize`` would use to
    #: replace this forward; without a name it keeps this forward and warns.
    kernel_layer_name = None

    def __init__(
        self,
        *,
        backend: Backend | MSDABackend | None = None,
    ) -> None:
        super().__init__()
        self.backend = backend

    @override
    def forward(  # noqa: PLR0913
        self,
        value: Tensor,
        value_spatial_shapes: Tensor,
        value_spatial_shapes_list: list[tuple[int, int]],
        level_start_index: Tensor,
        sampling_locations: Tensor,
        attention_weights: Tensor,
        im2col_step: int,
    ) -> Tensor:
        """Aggregate with deformops; upstream's pre-split list is unused."""
        del value_spatial_shapes_list
        return run_msda(
            value,
            value_spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            chunk_step=im2col_step,
            backend=self.backend,
        )


class Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention(_HF_MSDA_M2F):
    """Mask2Former's MSDA, backed by deformops, reusing its pretrained projections."""

    #: Opts out of Hub-kernel replacement; see
    #: :attr:`MultiScaleDeformableAttention.kernel_layer_name`.
    kernel_layer_name = None

    chunk_step: int
    d_model: int
    n_levels: int
    n_heads: int
    n_points: int

    def __init__(
        self,
        other: _HF_MSDA_M2F,
        *,
        backend: Backend | MSDABackend | None = None,
    ) -> None:
        super().__init__(
            embed_dim=other.d_model,
            num_heads=other.n_heads,
            n_levels=other.n_levels,
            n_points=other.n_points,
        )
        self.chunk_step = other.im2col_step
        self.backend = backend

        self.sampling_offsets = other.sampling_offsets
        self.attention_weights = other.attention_weights
        self.value_proj = other.value_proj
        self.output_proj = other.output_proj

    @override
    def with_pos_embed(
        self, tensor: Tensor, position_embeddings: Tensor | None
    ) -> Tensor:
        """Add position embeddings to ``tensor`` when they are given."""
        return tensor if position_embeddings is None else tensor + position_embeddings

    @override
    def forward(  # noqa: PLR0913
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        encoder_hidden_states: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,  # noqa: ARG002
        position_embeddings: Tensor | None = None,
        reference_points: Tensor | None = None,
        spatial_shapes_list: list[tuple[int, int]] | None = None,
        level_start_index: Tensor | None = None,
        output_attentions: bool = False,  # noqa: ARG002 - part of the HF signature
        spatial_shapes: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Multi-scale deformable self-attention over the feature pyramid."""
        if spatial_shapes_list is None:
            if spatial_shapes is None:
                msg = "one of spatial_shapes_list or spatial_shapes is required"
                raise ValueError(msg)
            rows = cast(list[list[int]], spatial_shapes.tolist())
            spatial_shapes_list = [(int(h), int(w)) for h, w in rows]

        if position_embeddings is not None:
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)

        if encoder_hidden_states is None or reference_points is None:
            msg = "encoder_hidden_states and reference_points are required"
            raise ValueError(msg)
        batch_size, num_queries, _ = hidden_states.shape
        sequence_length = encoder_hidden_states.shape[1]
        if sum(h * w for h, w in spatial_shapes_list) != sequence_length:
            msg = (
                "Make sure to align the spatial shapes with the sequence length "
                "of the encoder hidden states"
            )
            raise ValueError(msg)

        value = self.value_proj(encoder_hidden_states)
        if attention_mask is not None:
            value = value.masked_fill(attention_mask[..., None], 0)
        value = value.view(
            batch_size, sequence_length, self.n_heads, self.d_model // self.n_heads
        )

        sampling_offsets = self.sampling_offsets(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = nn.functional.softmax(
            self.attention_weights(hidden_states).view(
                batch_size, num_queries, self.n_heads, self.n_levels * self.n_points
            ),
            -1,
        )

        if reference_points.shape[-1] == _REFERENCE_POINTS_DIM:
            offset_normalizer = torch.tensor(
                [[width, height] for height, width in spatial_shapes_list],
                dtype=reference_points.dtype,
                device=reference_points.device,
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == _REFERENCE_POINTS_DIM_WITH_SCALE:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.n_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            msg = (
                f"Last dim of reference_points must be 2 or 4, but got "
                f"{reference_points.shape[-1]}"
            )
            raise ValueError(msg)

        shapes = torch.as_tensor(
            spatial_shapes_list, dtype=torch.long, device=value.device
        )
        output = run_msda(
            value,
            shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            chunk_step=self.chunk_step,
            backend=self.backend,
        )
        output = self.output_proj(output)
        return output, attention_weights.view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points
        )


def patch(
    model: nn.Module,
    *,
    backend: Backend | MSDABackend | None = None,
    strict: bool = True,
) -> nn.Module:
    """Replace a model's Hugging Face MSDA modules, both kinds in one traversal.

    Parameters
    ----------
    model
        The Hugging Face model, patched in place.
    backend
        Backend for every replacement; ``None`` selects one per call.
    strict
        Raise when nothing matches, which usually means the installed
        ``transformers`` names the modules differently.

    Returns
    -------
    ``model``, for chaining.

    Raises
    ------
    RuntimeError
        If ``strict`` is set and nothing was replaced.
    """
    return patch_modules(
        model,
        [
            (
                _HF_MSDA,
                lambda _child: MultiScaleDeformableAttention(backend=backend),
            ),
            (
                _HF_MSDA_M2F,
                lambda child: (
                    Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention(
                        child, backend=backend
                    )
                ),
            ),
        ],
        strict=strict,
    )
