"""Mask2Former's deformops overrides, against the Hugging Face originals.

Skips without ``transformers`` or ``torchmatch``; the real-weight test is marked
``integration``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torchmatch", reason="Mask2Former's loss needs torchmatch")
pytest.importorskip("transformers", reason="Mask2Former needs transformers")

# A torchvision built against another torch raises ImportError on the model class.
# That failure is unrelated to deformops, so the module skips.
try:
    from transformers import Mask2FormerForUniversalSegmentation  # noqa: F401
except ImportError as exc:
    pytest.skip(f"Mask2Former model unavailable: {exc}", allow_module_level=True)

import torch  # noqa: E402
from deformops.backends.eager.msda_pytorch import (  # noqa: E402
    Backend as ExportBackend,
)
from deformops.models import Mask2Former  # noqa: E402
from deformops.models.mask2former import (  # noqa: E402
    PatchedMask2FormerHungarianMatcher,
    PatchedMask2FormerLoss,
    PatchedMask2FormerMaskPredictor,
    PatchedMask2FormerPixelDecoderEncoderMultiscaleDeformableAttention,
)
from torch import nn  # noqa: E402


def test_mask2former_overrides_are_modules() -> None:
    assert issubclass(Mask2Former, nn.Module)
    assert issubclass(PatchedMask2FormerMaskPredictor, nn.Module)
    assert issubclass(
        PatchedMask2FormerPixelDecoderEncoderMultiscaleDeformableAttention, nn.Module
    )


def test_patched_loss_uses_the_torchmatch_matcher() -> None:
    """The loss constructs without scipy and installs the torchmatch matcher."""
    from transformers import Mask2FormerConfig

    loss = PatchedMask2FormerLoss(Mask2FormerConfig(), {"loss_ce": 1.0})
    assert isinstance(loss.matcher, PatchedMask2FormerHungarianMatcher)


def test_patched_matcher_returns_a_valid_assignment() -> None:
    """torchmatch's assignment produces in-range, one-to-one indices."""
    torch.manual_seed(0)
    matcher = PatchedMask2FormerHungarianMatcher(
        cost_class=1.0, cost_mask=1.0, cost_dice=1.0, num_points=16
    )
    num_queries = 5
    matches = matcher(
        masks_queries_logits=torch.randn(1, num_queries, 8, 8),
        class_queries_logits=torch.randn(1, num_queries, 4),
        mask_labels=[torch.rand(2, 8, 8)],
        class_labels=[torch.tensor([0, 2])],
    )
    assert len(matches) == 1
    rows, cols = matches[0]
    assert rows.shape == cols.shape == (2,)
    assert rows.unique().numel() == rows.numel()
    assert cols.unique().numel() == cols.numel()
    assert int(rows.max()) < num_queries
    assert int(cols.max()) < 2


def test_patched_attention_matches_hugging_face() -> None:
    """The deformops MSDA override is numerically identical to HF's."""
    from deformops.registry import Backend
    from transformers.models.mask2former.modeling_mask2former import (
        Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention as HFAttention,
    )

    torch.manual_seed(0)
    embed, heads, levels, points = 32, 4, 3, 4
    shapes = [(8, 8), (4, 4), (2, 2)]
    sequence = sum(h * w for h, w in shapes)

    hf = HFAttention(embed_dim=embed, num_heads=heads, n_levels=levels, n_points=points)
    patched = PatchedMask2FormerPixelDecoderEncoderMultiscaleDeformableAttention(
        hf, backend=Backend.EAGER
    )
    hidden = torch.randn(2, sequence, embed)
    kwargs = {
        "encoder_hidden_states": hidden,
        "position_embeddings": torch.randn(2, sequence, embed),
        "reference_points": torch.rand(2, sequence, levels, 2) * 0.8 + 0.1,
        "spatial_shapes_list": shapes,
        "level_start_index": torch.tensor([0, 64, 80]),
    }

    with torch.no_grad():
        out_hf, weights_hf = hf(hidden, **kwargs)
        out_patched, weights_patched = patched(hidden, **kwargs)

    torch.testing.assert_close(out_hf, out_patched, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(weights_hf, weights_patched, atol=1e-6, rtol=1e-6)


def _tiny_mask2former() -> Mask2Former:
    """A random-weight Mask2Former small enough to run and export on CPU."""
    from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation
    from transformers.models.swin.configuration_swin import SwinConfig

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
    net = Mask2FormerForUniversalSegmentation(config).eval()
    return Mask2Former(net, backend=ExportBackend()).eval()


def test_tiny_mask2former_forwards() -> None:
    model = _tiny_mask2former()
    image = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        output = model(image)
    assert output.class_queries_logits.shape == (1, 10, 3)
    assert output.masks_queries_logits.shape == (1, 10, 16, 16)


def test_tiny_mask2former_exports() -> None:
    """The patched model exports as a whole through the traceable eager backend."""
    model = _tiny_mask2former()
    image = torch.randn(1, 3, 64, 64)
    exported = torch.export.export(model, args=(image,))
    assert exported is not None
    assert len(list(exported.graph.nodes)) > 0


@pytest.mark.integration
def test_mask2former_from_pretrained_real_weights() -> None:
    """Load the published base checkpoint, run it, and export it."""
    checkpoint = "facebook/mask2former-swin-base-coco-instance"
    try:
        model = Mask2Former.from_pretrained(checkpoint, backend=ExportBackend()).eval()
    except OSError as exc:  # pragma: no cover - network dependent
        pytest.skip(f"checkpoint {checkpoint!r} unavailable: {exc}")

    patched = [
        module
        for module in model.modules()
        if isinstance(
            module, PatchedMask2FormerPixelDecoderEncoderMultiscaleDeformableAttention
        )
    ]
    assert patched, "no deformops MSDA layer was installed"

    image = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        output = model(image)
    assert output.class_queries_logits.shape[0] == 1
    assert output.masks_queries_logits.shape[0] == 1

    torch.export.export(model, args=(image,))
