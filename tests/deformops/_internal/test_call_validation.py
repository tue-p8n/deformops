"""Operand validation at the multi-scale deformable attention entry point.

The kernels index the value tensor straight from the level table, so each mismatch
would otherwise be an out-of-bounds device read. Tests match the message as well as
the type. Also covers the variable that enables value checks for every operator.
"""

from __future__ import annotations

import pytest
import torch
from deformops.core.sanitize import (
    VALIDATE_ENV_VAR,
    check_msda_call,
    value_checks_enabled,
)
from deformops.nn import MultiScaleDeformAttn2d
from deformops.registry import Backend, msda

BATCH = 2
HEADS = 2
CHANNELS = 4
QUERIES = 5
POINTS = 3
SHAPES = ((4, 4), (2, 2))
LEVELS = len(SHAPES)
SPATIAL = sum(h * w for h, w in SHAPES)


def _operands(
    *, flat_attention: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """A consistent set of MSDA operands on the CPU."""
    value = torch.randn(BATCH, SPATIAL, HEADS, CHANNELS)
    spatial_shapes = torch.tensor(SHAPES, dtype=torch.long)
    sizes = [h * w for h, w in SHAPES]
    starts = torch.tensor([sum(sizes[:i]) for i in range(LEVELS)], dtype=torch.long)
    locations = torch.rand(BATCH, QUERIES, HEADS, LEVELS, POINTS, 2)
    attention_shape = (
        (BATCH, QUERIES, HEADS, LEVELS * POINTS)
        if flat_attention
        else (BATCH, QUERIES, HEADS, LEVELS, POINTS)
    )
    attention = torch.rand(*attention_shape)
    return value, spatial_shapes, starts, locations, attention


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flat_attention", [True, False])
def test_consistent_operands_are_accepted(flat_attention: bool) -> None:
    """Both accepted attention-weight layouts pass validation."""
    check_msda_call(*_operands(flat_attention=flat_attention))


def test_the_reference_backend_still_runs_end_to_end() -> None:
    """Validation does not reject what the reference backend accepts."""
    value, shapes, starts, locations, attention = _operands()
    out = msda.apply(value, shapes, starts, locations, attention, backend=Backend.EAGER)
    assert out.shape == (BATCH, QUERIES, HEADS * CHANNELS)


# ---------------------------------------------------------------------------
# Structural checks, which run on every call
# ---------------------------------------------------------------------------


def test_rank_mismatch_names_the_operand() -> None:
    value, shapes, starts, locations, attention = _operands()
    with pytest.raises(ValueError, match=r"value must be 4-dimensional"):
        check_msda_call(value[0], shapes, starts, locations, attention)


def test_locations_must_carry_coordinate_pairs() -> None:
    value, shapes, starts, locations, attention = _operands()
    broken = torch.rand(BATCH, QUERIES, HEADS, LEVELS, POINTS, 3)
    with pytest.raises(ValueError, match=r"normalized \(x, y\) pair"):
        check_msda_call(value, shapes, starts, broken, attention)


def test_head_count_must_agree_between_value_and_locations() -> None:
    value, shapes, starts, locations, attention = _operands()
    broken = torch.rand(BATCH, QUERIES, HEADS + 1, LEVELS, POINTS, 2)
    with pytest.raises(ValueError, match=r"sampling_locations has size 3 on axis 2"):
        check_msda_call(value, shapes, starts, broken, attention)


def test_level_count_must_agree_between_shapes_and_locations() -> None:
    value, shapes, starts, locations, attention = _operands()
    broken = torch.rand(BATCH, QUERIES, HEADS, LEVELS + 1, POINTS, 2)
    with pytest.raises(ValueError, match=r"rows of spatial_shapes"):
        check_msda_call(value, shapes, starts, broken, attention)


def test_flat_attention_width_must_match_levels_times_points() -> None:
    value, shapes, starts, locations, _ = _operands()
    broken = torch.rand(BATCH, QUERIES, HEADS, LEVELS * POINTS + 1)
    with pytest.raises(ValueError, match=r"levels x points of sampling_locations"):
        check_msda_call(value, shapes, starts, locations, broken)


def test_start_index_needs_one_entry_per_level() -> None:
    value, shapes, _, locations, attention = _operands()
    broken = torch.zeros(LEVELS + 1, dtype=torch.long)
    with pytest.raises(ValueError, match=r"level_start_index has size 3"):
        check_msda_call(value, shapes, broken, locations, attention)


def test_index_tensors_must_be_integral() -> None:
    value, shapes, starts, locations, attention = _operands()
    with pytest.raises(ValueError, match=r"spatial_shapes must be an integer tensor"):
        check_msda_call(value, shapes.float(), starts, locations, attention)


def test_value_and_weights_must_share_a_dtype() -> None:
    """The kernels template on the value dtype and read the weights at it."""
    value, shapes, starts, locations, attention = _operands()
    with pytest.raises(ValueError, match=r"mixed input dtypes are not supported"):
        check_msda_call(value, shapes, starts, locations, attention.half())


def test_locations_may_differ_in_width_from_the_value() -> None:
    """Autocast promotes locations to fp32 and leaves value and weights in half.

    That is the ordinary Deformable DETR and Mask2Former call.
    """
    value, shapes, starts, locations, attention = _operands()
    check_msda_call(value.half(), shapes, starts, locations, attention.half())


def test_locations_must_still_be_floating_point() -> None:
    value, shapes, starts, locations, attention = _operands()
    with pytest.raises(ValueError, match=r"sampling_locations must be a floating"):
        check_msda_call(value, shapes, starts, locations.long(), attention)


def test_autocast_module_call_is_not_rejected() -> None:
    """A module call under autocast, with mixed-width locations, passes validation."""
    layer = MultiScaleDeformAttn2d(
        dim=64, num_heads=4, num_levels=2, num_points=4, backend=Backend.EAGER
    )
    shapes = torch.tensor([[4, 4], [2, 2]])
    starts = torch.tensor([0, 16])
    query = torch.randn(1, 8, 64)
    reference = torch.rand(1, 8, 2, 2)
    value = torch.randn(1, 20, 64)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = layer(query, reference, value, shapes, starts)
    assert out.shape == (1, 8, 64)


def test_meta_device_mismatch_is_rejected_without_a_gpu() -> None:
    """The structural checks cover device agreement, so ``meta`` exercises it."""
    value, shapes, starts, locations, attention = _operands()
    with pytest.raises(ValueError, match=r"all operands must share one device"):
        check_msda_call(value, shapes, starts, locations.to("meta"), attention)


def test_apply_rejects_before_reaching_a_backend() -> None:
    """The check sits at the entry point, ahead of backend dispatch."""
    value, shapes, starts, locations, attention = _operands()
    with pytest.raises(ValueError, match=r"value must be 4-dimensional"):
        msda.apply(
            value[0], shapes, starts, locations, attention, backend=Backend.EAGER
        )


# ---------------------------------------------------------------------------
# Value checks, opt-in because they synchronize
# ---------------------------------------------------------------------------


def test_value_checks_are_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(VALIDATE_ENV_VAR, raising=False)
    assert value_checks_enabled() is False


@pytest.mark.parametrize("setting", ["1", "true", "on", "yes"])
def test_value_checks_turn_on_by_environment(
    monkeypatch: pytest.MonkeyPatch, setting: str
) -> None:
    monkeypatch.setenv(VALIDATE_ENV_VAR, setting)
    assert value_checks_enabled() is True


def test_inconsistent_start_index_passes_structurally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dangerous case: right shape, wrong contents, no complaint by default."""
    monkeypatch.delenv(VALIDATE_ENV_VAR, raising=False)
    value, shapes, _, locations, attention = _operands()
    wrong = torch.tensor([0, 999], dtype=torch.long)
    check_msda_call(value, shapes, wrong, locations, attention)


def test_inconsistent_start_index_is_caught_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(VALIDATE_ENV_VAR, "1")
    value, shapes, _, locations, attention = _operands()
    wrong = torch.tensor([0, 999], dtype=torch.long)
    with pytest.raises(ValueError, match=r"not the exclusive prefix sum"):
        check_msda_call(value, shapes, wrong, locations, attention)


def test_levels_must_tile_the_value_tensor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(VALIDATE_ENV_VAR, "1")
    value, shapes, starts, locations, attention = _operands()
    padded = torch.randn(BATCH, SPATIAL + 7, HEADS, CHANNELS)
    with pytest.raises(ValueError, match=r"must tile the value tensor exactly"):
        check_msda_call(padded, shapes, starts, locations, attention)


def test_degenerate_level_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(VALIDATE_ENV_VAR, "1")
    value, _, starts, locations, attention = _operands()
    shapes = torch.tensor([[4, 4], [0, 2]], dtype=torch.long)
    with pytest.raises(ValueError, match=r"positive height and width"):
        check_msda_call(value, shapes, starts, locations, attention)
