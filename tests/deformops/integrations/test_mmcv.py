"""mmcv's multi-scale deformable attention, routed through deformops.

Needs ``mmcv-lite`` (the ``mmcv`` extra). The integration installs mmcv's loader
for the absent compiled extension, so mmcv is reached through it.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

pytest.importorskip("mmcv", reason="needs the optional mmcv-lite extra")

from deformops.integrations import mmcv as _mmcv  # noqa: E402
from deformops.registry import Backend  # noqa: E402


def _operands(batch: int = 2, queries: int = 6, dims: int = 32):
    generator = torch.Generator().manual_seed(0)
    shapes = torch.tensor([[8, 8], [4, 4]])
    starts = torch.tensor([0, 64])
    value = torch.randn(batch, 80, dims, generator=generator)
    query = torch.randn(batch, queries, dims, generator=generator)
    reference = torch.rand(batch, queries, 2, 2, generator=generator)
    return query, value, reference, shapes, starts


def _upstream() -> nn.Module:
    torch.manual_seed(0)
    module = _mmcv._MMCVMultiScaleDeformableAttention(
        embed_dims=32, num_heads=4, num_levels=2, num_points=4, batch_first=True
    )
    # Non-zero offsets and weights, so the comparison exercises real sampling.
    for parameter in module.parameters():
        nn.init.normal_(parameter, std=0.1)
    return module.eval()


def test_patch_replaces_attention_and_matches_upstream() -> None:
    upstream = _upstream()
    query, value, reference, shapes, starts = _operands()
    with torch.no_grad():
        expected = upstream(
            query,
            value=value,
            reference_points=reference,
            spatial_shapes=shapes,
            level_start_index=starts,
        )

    model = nn.Sequential(upstream)
    _mmcv.patch(model, backend=Backend.EAGER)
    assert isinstance(model[0], _mmcv.MultiScaleDeformableAttention)
    with torch.no_grad():
        got = model[0](
            query,
            value=value,
            reference_points=reference,
            spatial_shapes=shapes,
            level_start_index=starts,
        )
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)


def test_patch_raises_when_nothing_matches() -> None:
    with pytest.raises(RuntimeError, match="No module"):
        _mmcv.patch(nn.Sequential(nn.Linear(2, 2)))
    _mmcv.patch(nn.Sequential(nn.Linear(2, 2)), strict=False)


def test_patch_redirects_the_mmcv_pytorch_function() -> None:
    replacement = _mmcv.patch_multi_scale_deformable_attn_pytorch()
    assert _mmcv._mmcv_module.multi_scale_deformable_attn_pytorch is replacement
