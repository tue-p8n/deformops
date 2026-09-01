"""Small random-weight Mask2Former and Deformable DETR for GPU model tests.

:data:`MODELS` maps each to its factory, wrapper, image size and output picker.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import pytest

pytest.importorskip("transformers", reason="the models need transformers")

try:
    from transformers import (
        DeformableDetrConfig,
        DeformableDetrForObjectDetection,
        Mask2FormerConfig,
        Mask2FormerForUniversalSegmentation,
        ResNetConfig,
    )
    from transformers.models.swin.configuration_swin import SwinConfig
except ImportError as exc:
    pytest.skip(f"model classes unavailable: {exc}", allow_module_level=True)

import torch  # noqa: E402
from deformops.models import DeformableDETR, Mask2Former  # noqa: E402
from deformops.registry import Backend  # noqa: E402
from torch import nn  # noqa: E402

__all__ = ["MODELS", "TinyModel", "flat_model"]


class TinyModel(NamedTuple):
    """One small model configuration."""

    network: Callable[[], nn.Module]
    wrapper: type[nn.Module]
    image_size: int
    outputs_of: Callable[[Any], tuple[torch.Tensor, ...]]


def _mask2former_net() -> nn.Module:
    backbone = SwinConfig(
        image_size=64,
        patch_size=4,
        embed_dim=8,
        depths=[1, 1, 1, 1],
        num_heads=[1, 1, 1, 1],
        window_size=2,
        out_indices=[1, 2, 3, 4],
    )
    config = Mask2FormerConfig(
        feature_size=32,
        mask_feature_size=32,
        hidden_dim=32,
        num_attention_heads=4,
        encoder_layers=1,
        decoder_layers=1,
        encoder_feedforward_dim=64,
        dim_feedforward=64,
        num_queries=10,
        backbone_config=backbone,
    )
    return Mask2FormerForUniversalSegmentation(config)


def _deformable_detr_net() -> nn.Module:
    config = DeformableDetrConfig(
        backbone_config=ResNetConfig(
            embedding_size=16,
            hidden_sizes=[16, 24, 32, 48],
            depths=[1, 1, 1, 1],
            out_features=["stage2", "stage3", "stage4"],
        ),
        use_pretrained_backbone=False,
        d_model=32,
        encoder_layers=1,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        num_queries=10,
        num_feature_levels=4,
    )
    return DeformableDetrForObjectDetection(config)


MODELS = {
    "mask2former": TinyModel(
        _mask2former_net,
        Mask2Former,
        64,
        lambda out: (out.class_queries_logits, out.masks_queries_logits),
    ),
    "deformable_detr": TinyModel(
        _deformable_detr_net,
        DeformableDETR,
        128,
        lambda out: (out.logits, out.pred_boxes),
    ),
}


class _Flat(nn.Module):
    """A model whose forward returns a tuple of tensors, as export requires."""

    def __init__(self, model: nn.Module, outputs_of: Callable[[Any], Any]) -> None:
        super().__init__()
        self.model = model
        self.outputs_of = outputs_of

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(self.outputs_of(self.model(pixel_values)))


def flat_model(
    name: str, backend: Backend | None, device: str, network: nn.Module | None = None
) -> nn.Module:
    """Wrap a fresh or given network for ``backend`` with a tuple output."""
    spec = MODELS[name]
    net = network if network is not None else spec.network()
    return _Flat(spec.wrapper(net.to(device), backend=backend), spec.outputs_of)
