r"""Mask2Former's deformops overrides.

The attention comes from :mod:`deformops.integrations.transformers`. This module
adds a mask predictor that uses ``matmul`` for ``einsum``, so the graph exports,
and a loss that assigns with ``torchmatch`` instead of scipy. Needs
``deformops[mask2former]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast, override

import torch
from torch import Tensor, nn

from deformops.integrations import transformers as _transformers
from deformops.integrations._patch import patch_modules
from deformops.registry import Backend, MSDABackend

if TYPE_CHECKING:
    from transformers.models.mask2former.modeling_mask2former import (
        Mask2FormerMaskPredictor,
    )

try:
    from torchmatch.assignment import solve as _assignment_solve
    from transformers.models.mask2former.modeling_mask2former import (
        Mask2FormerHungarianMatcher as _HFHungarianMatcher,
    )
    from transformers.models.mask2former.modeling_mask2former import (
        Mask2FormerLoss as _HFLoss,
    )
    from transformers.models.mask2former.modeling_mask2former import (
        Mask2FormerMaskPredictor as _HFMaskPredictor,
    )
    from transformers.models.mask2former.modeling_mask2former import (
        pair_wise_dice_loss,
        pair_wise_sigmoid_cross_entropy_loss,
        sample_point,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - depends on install extras
    msg = (
        "deformops.models.mask2former requires the optional 'transformers' and "
        "'torchmatch' dependencies. Install them with "
        "`pip install deformops[mask2former]`."
    )
    raise ModuleNotFoundError(msg) from exc

#: The transformers integration's attention. Needs ``deformops[mask2former]``.
PatchedMask2FormerPixelDecoderEncoderMultiscaleDeformableAttention = (
    _transformers.Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention
)

__all__ = [
    "PatchedMask2FormerHungarianMatcher",
    "PatchedMask2FormerLoss",
    "PatchedMask2FormerMaskPredictor",
    "PatchedMask2FormerPixelDecoderEncoderMultiscaleDeformableAttention",
    "patch_mask2former",
]

#: Probability below which a predicted mask pixel is treated as masked out.
_MASK_THRESHOLD = 0.5


class PatchedMask2FormerMaskPredictor(nn.Module):
    """Drop-in :class:`~transformers.Mask2FormerMaskPredictor` using ``matmul``.

    Same parameters and return tuple; ``matmul`` replaces ``einsum`` for export.
    """

    hidden_size: int
    num_heads: int
    mask_embedder: nn.Module

    def __init__(self, other: Mask2FormerMaskPredictor) -> None:
        super().__init__()
        self.hidden_size = other.hidden_size
        self.num_heads = other.num_heads
        self.mask_embedder = other.mask_embedder

    @override
    def forward(
        self,
        outputs: Tensor,
        pixel_embeddings: Tensor,
        attention_mask_target_size: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Predict per-query mask logits and the attention mask they imply."""
        mask_embeddings = self.mask_embedder(outputs.transpose(0, 1))

        # Equivalent to einsum("bqc, bchw -> bqhw"), but export-friendly.
        batch_size, num_queries, num_channels = mask_embeddings.shape
        _, _, height, width = pixel_embeddings.shape
        pixel_embeddings = pixel_embeddings.reshape(batch_size, num_channels, -1)
        outputs_mask = torch.matmul(mask_embeddings, pixel_embeddings).reshape(
            batch_size, num_queries, height, width
        )

        attention_mask = nn.functional.interpolate(
            outputs_mask,
            size=attention_mask_target_size,
            mode="bilinear",
            align_corners=False,
        )
        attention_mask = (
            attention_mask.sigmoid()
            .flatten(2)
            .unsqueeze(1)
            .repeat(1, self.num_heads, 1, 1)
        )
        attention_mask = (
            (attention_mask.flatten(0, 1) < _MASK_THRESHOLD).bool().detach()
        )
        return outputs_mask, attention_mask


class PatchedMask2FormerHungarianMatcher(_HFHungarianMatcher):
    """Hugging Face's matcher, assigning with :func:`torchmatch.assignment.solve`.

    The cost matrix is built as upstream, then moved to CPU float32, as the scipy
    path does, for torchmatch's CPU kernel.
    """

    @torch.no_grad()
    @override
    def forward(  # noqa: PLR0913  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        masks_queries_logits: Tensor,
        class_queries_logits: Tensor,
        mask_labels: Tensor,
        class_labels: Tensor,
    ) -> list[tuple[Tensor, Tensor]]:
        """Match predicted masks to labels, per batch element."""
        indices: list[tuple[Tensor, Tensor]] = []

        batch_size = masks_queries_logits.shape[0]
        for i in range(batch_size):
            pred_probs = class_queries_logits[i].softmax(-1)
            pred_mask = masks_queries_logits[i]

            cost_class = -pred_probs[:, class_labels[i]]
            target_mask = mask_labels[i].to(pred_mask)
            target_mask = target_mask[:, None]
            pred_mask = pred_mask[:, None]

            point_coordinates = torch.rand(
                1, self.num_points, 2, device=pred_mask.device
            )
            target_coordinates = point_coordinates.repeat(target_mask.shape[0], 1, 1)
            target_mask = sample_point(
                target_mask, target_coordinates, align_corners=False
            ).squeeze(1)

            pred_coordinates = point_coordinates.repeat(pred_mask.shape[0], 1, 1)
            pred_mask = sample_point(
                pred_mask, pred_coordinates, align_corners=False
            ).squeeze(1)

            cost_mask = pair_wise_sigmoid_cross_entropy_loss(pred_mask, target_mask)
            cost_dice = pair_wise_dice_loss(pred_mask, target_mask)
            cost_matrix = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            )
            cost_matrix = torch.minimum(cost_matrix, torch.tensor(1e10))
            cost_matrix = torch.maximum(cost_matrix, torch.tensor(-1e10))
            cost_matrix = torch.nan_to_num(cost_matrix, 0)

            matches, _unmatched_rows, _unmatched_cols = cast(
                tuple[Tensor, Tensor, Tensor],
                _assignment_solve(
                    cost_matrix.detach().to(torch.float32).cpu(), unpack=True
                ),
            )
            indices.append((matches[:, 0], matches[:, 1]))

        return [
            (
                torch.as_tensor(row, dtype=torch.int64),
                torch.as_tensor(col, dtype=torch.int64),
            )
            for row, col in indices
        ]


class PatchedMask2FormerLoss(_HFLoss):
    """Hugging Face's loss with :class:`PatchedMask2FormerHungarianMatcher`, no scipy.

    Weights, buffers and loss math are inherited unchanged.
    """

    def __init__(self, config: Any, weight_dict: dict[str, float]) -> None:
        # Not ``super().__init__``: upstream's constructor calls
        # ``requires_backends(self, ["scipy"])``.
        nn.Module.__init__(self)
        self.num_labels = config.num_labels
        self.weight_dict = weight_dict

        self.eos_coef = config.no_object_weight
        empty_weight = torch.ones(self.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        self.num_points = config.train_num_points
        self.oversample_ratio = config.oversample_ratio
        self.importance_sample_ratio = config.importance_sample_ratio

        self.matcher = PatchedMask2FormerHungarianMatcher(
            cost_class=config.class_weight,
            cost_dice=config.dice_weight,
            cost_mask=config.mask_weight,
            num_points=self.num_points,
        )


def _install_loss_patch() -> None:
    """Point ``Mask2FormerForUniversalSegmentation`` at the torchmatch loss.

    Idempotent; applied at import so a model built before wrapping avoids scipy.
    """
    import transformers.models.mask2former.modeling_mask2former as mm  # noqa: PLC0415

    if mm.Mask2FormerLoss is not PatchedMask2FormerLoss:
        mm.Mask2FormerLoss = PatchedMask2FormerLoss


_install_loss_patch()


def patch_mask2former(
    model: nn.Module,
    *,
    backend: Backend | MSDABackend | None = None,
    strict: bool = True,
) -> nn.Module:
    """Apply the Mask2Former overrides to ``model``, in place.

    Parameters
    ----------
    model
        A :class:`~transformers.Mask2FormerForUniversalSegmentation`.
    backend
        Backend forwarded to the deformable attention layers.
    strict
        Raise when the model lacks the attention or mask-predictor modules.

    Returns
    -------
    ``model``, for chaining.
    """
    _install_loss_patch()
    _transformers.patch(model, backend=backend, strict=strict)
    patch_modules(
        model,
        [(_HFMaskPredictor, PatchedMask2FormerMaskPredictor)],
        strict=strict,
    )
    return model
