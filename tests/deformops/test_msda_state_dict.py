"""CPU-runnable tests for the DETR-state-dict load hook on MultiScaleDeformAttn2d."""

import pytest
import torch
from deformops.nn import MultiScaleDeformAttn2d


def test_load_detr_state_dict_reports_unknown_keys_when_strict():
    """A strict load reports an unmapped key in a DETR state dict."""
    m = MultiScaleDeformAttn2d(dim=16, num_heads=2, num_levels=1, num_points=1)
    bad_state = {
        **{f"{k}": v for k, v in m.state_dict().items()},
        "unknown_legacy_thing.weight": torch.zeros(1),
    }
    with pytest.raises(RuntimeError, match="unknown_legacy_thing"):
        m.load_state_dict(bad_state, strict=True)


def test_load_detr_state_dict_ignores_unknown_keys_when_not_strict():
    """``strict=False`` loads the mapped keys and ignores the rest."""
    m = MultiScaleDeformAttn2d(dim=16, num_heads=2, num_levels=1, num_points=1)
    bad_state = {
        "sampling_offsets.weight": torch.ones_like(m.proj_offset.weight),
        "sampling_offsets.bias": torch.ones_like(m.proj_offset.bias),
        "unknown_legacy_thing.weight": torch.zeros(1),
    }
    result = m.load_state_dict(bad_state, strict=False)
    assert "unknown_legacy_thing.weight" in result.unexpected_keys
    assert torch.equal(m.proj_offset.weight, torch.ones_like(m.proj_offset.weight))


def test_load_detr_state_dict_renames_known_legacy_keys():
    """Loading renames the known legacy DETR keys without raising."""
    m = MultiScaleDeformAttn2d(dim=16, num_heads=2, num_levels=1, num_points=1)
    legacy_state = {
        "sampling_offsets.weight": torch.zeros_like(m.proj_offset.weight),
        "sampling_offsets.bias": torch.zeros_like(m.proj_offset.bias),
        "attention_weights.weight": torch.zeros_like(m.proj_weights.weight),
        "attention_weights.bias": torch.zeros_like(m.proj_weights.bias),
        "value_proj.weight": torch.zeros_like(m.proj_value.weight),
        "value_proj.bias": torch.zeros_like(m.proj_value.bias),
        "output_proj.weight": torch.zeros_like(m.proj_output.weight),
        "output_proj.bias": torch.zeros_like(m.proj_output.bias),
    }
    # Should not raise: all keys map.
    m.load_state_dict(legacy_state, strict=False)
