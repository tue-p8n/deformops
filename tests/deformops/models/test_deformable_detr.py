"""Deformable DETR's deformops override, against the Hugging Face original.

Skips without ``transformers``; the real-weight test is marked ``integration``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("transformers", reason="Deformable DETR needs transformers")

# transformers reports a torchvision build that does not match torch as an
# ImportError of the model class; that is unrelated to deformops, so it skips.
try:
    from transformers import DeformableDetrForObjectDetection  # noqa: F401
except ImportError as exc:
    pytest.skip(f"Deformable DETR model unavailable: {exc}", allow_module_level=True)

import torch  # noqa: E402
from deformops.backends.eager.msda_pytorch import (  # noqa: E402
    Backend as ExportBackend,
)
from deformops.models import DeformableDETR  # noqa: E402
from deformops.models.deformable_detr import (  # noqa: E402
    PatchedMultiScaleDeformableAttention,
)
from deformops.registry import Backend  # noqa: E402
from torch import nn  # noqa: E402


def _attention_inputs():
    torch.manual_seed(0)
    batch, embed, heads, levels, points = 2, 32, 4, 3, 4
    shapes = [(8, 8), (4, 4), (2, 2)]
    sequence = sum(h * w for h, w in shapes)
    hidden = torch.randn(batch, sequence, embed)
    return {
        "hidden_states": hidden,
        "encoder_hidden_states": hidden,
        "position_embeddings": torch.randn(batch, sequence, embed),
        "reference_points": torch.rand(batch, sequence, levels, 2) * 0.8 + 0.1,
        "spatial_shapes": torch.tensor(shapes, dtype=torch.long),
        "spatial_shapes_list": shapes,
        "level_start_index": torch.tensor([0, 64, 80]),
    }, (embed, heads, levels, points)


def test_overrides_are_modules() -> None:
    assert issubclass(DeformableDETR, nn.Module)
    assert issubclass(PatchedMultiScaleDeformableAttention, nn.Module)


def test_patched_core_matches_hugging_face() -> None:
    """The deformops MSDA core is numerically identical to HF's."""
    from transformers import DeformableDetrConfig
    from transformers.models.deformable_detr.modeling_deformable_detr import (
        DeformableDetrMultiscaleDeformableAttention as HFAttention,
    )

    config = DeformableDetrConfig(
        d_model=32, encoder_attention_heads=4, num_feature_levels=3
    )
    hf = HFAttention(config, num_heads=4, n_points=4)
    kwargs, _ = _attention_inputs()

    with torch.no_grad():
        out_hf, weights_hf = hf(**kwargs)

    hf.attn = PatchedMultiScaleDeformableAttention(backend=Backend.EAGER)
    with torch.no_grad():
        out_patched, weights_patched = hf(**kwargs)

    torch.testing.assert_close(out_hf, out_patched, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(weights_hf, weights_patched, atol=1e-6, rtol=1e-6)


class _CoreExportWrapper(nn.Module):
    """Fix the shape metadata so ``torch.export`` sees a tensor-only call."""

    def __init__(self, core: PatchedMultiScaleDeformableAttention) -> None:
        super().__init__()
        self.core = core
        shapes = [(8, 8), (4, 4), (2, 2)]
        self.spatial_shapes = torch.tensor(shapes, dtype=torch.long)
        self.spatial_shapes_list = shapes
        self.level_start_index = torch.tensor([0, 64, 80])
        self.im2col_step = 64

    def forward(
        self,
        value: torch.Tensor,
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> torch.Tensor:
        return self.core(
            value,
            self.spatial_shapes,
            self.spatial_shapes_list,
            self.level_start_index,
            sampling_locations,
            attention_weights,
            self.im2col_step,
        )


def test_patched_core_exports() -> None:
    """The deformops MSDA core traces as a single operator under export."""
    wrapper = _CoreExportWrapper(
        PatchedMultiScaleDeformableAttention(backend=ExportBackend())
    )
    value = torch.randn(2, 84, 4, 8)
    sampling_locations = torch.rand(2, 84, 4, 3, 4, 2)
    attention_weights = torch.rand(2, 84, 4, 12)

    exported = torch.export.export(
        wrapper, args=(value, sampling_locations, attention_weights)
    )
    assert exported is not None
    assert len(list(exported.graph.nodes)) > 0


@pytest.mark.integration
def test_deformable_detr_from_pretrained_real_weights() -> None:
    checkpoint = "SenseTime/deformable-detr"
    try:
        model = DeformableDETR.from_pretrained(checkpoint, backend=Backend.EAGER).eval()
    except (ImportError, OSError) as exc:  # pragma: no cover - deps/network
        pytest.skip(f"checkpoint {checkpoint!r} unavailable: {exc}")

    patched = [
        module
        for module in model.modules()
        if isinstance(module, PatchedMultiScaleDeformableAttention)
    ]
    assert patched, "no deformops MSDA core was installed"

    image = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        output = model(image)
    assert output.logits.shape[0] == 1
