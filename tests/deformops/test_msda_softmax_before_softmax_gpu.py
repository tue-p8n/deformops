"""The in-kernel-softmax backward with out-of-bounds samples and masked logits.

Gradients must be finite and match the reference: an unwritten out-of-bounds tile
slot, or a softmax seeded at ``-inf``, produces NaN. The softmax spans all ``L * K``
points of a query head. See ``docs/kernels.md``.
"""

import pytest
import torch
from deformops.registry import Backend, msda

pytestmark = pytest.mark.gpu

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available"
)

_REFERENCE = Backend.EAGER
_BACKENDS = [
    Backend.CUDA,
    Backend.TRITON,
]


def _run(
    backend: Backend,
    head_dim: int,
    num_points: int,
    seed: int,
    *,
    mask_first_point: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device("cuda")
    batch, num_heads, num_levels, num_queries = 2, 4, 3, 200
    shapes = torch.tensor([[32, 32], [16, 16], [8, 8]], dtype=torch.long, device=device)
    start_index = torch.cat(
        [
            torch.zeros(1, dtype=torch.long, device=device),
            shapes[:-1, 0] * shapes[:-1, 1],
        ]
    ).cumsum(0)
    spatial = int((shapes[:, 0] * shapes[:, 1]).sum())

    torch.manual_seed(seed)
    value = torch.randn(batch, spatial, num_heads, head_dim, device=device)
    # Wide locations exercise the tile slots out-of-bounds points never write.
    loc = (
        torch.rand(
            batch, num_queries, num_heads, num_levels, num_points, 2, device=device
        )
        * 2.0
        - 0.5
    ).requires_grad_(True)
    attn = torch.randn(
        batch, num_queries, num_heads, num_levels, num_points, device=device
    )
    if mask_first_point:
        # Masking the first visited point NaNs a softmax whose maximum starts at -inf.
        attn[..., 0, 0] = float("-inf")
    attn = attn.requires_grad_(True)
    v = value.clone().requires_grad_(True)
    upstream = torch.randn(batch, num_queries, num_heads * head_dim, device=device)
    out = msda(
        v,
        shapes,
        start_index,
        loc,
        attn,
        method="softmax",
        backend=backend,
    )
    out.backward(upstream)
    assert v.grad is not None
    assert loc.grad is not None
    assert attn.grad is not None
    return out.detach(), v.grad, loc.grad, attn.grad


@requires_cuda
@pytest.mark.parametrize("backend", _BACKENDS, ids=str)
@pytest.mark.parametrize("head_dim", [32, 128])  # 32 -> warp path, 128 -> shared path
@pytest.mark.parametrize("num_points", [4, 8])
def test_softmax_out_of_bounds_no_nan_and_parity(
    backend: Backend, head_dim: int, num_points: int
) -> None:
    for seed in range(3):
        ref = _run(_REFERENCE, head_dim, num_points, seed)
        got = _run(backend, head_dim, num_points, seed)
        for name, a, b in zip(
            ["output", "grad_value", "grad_loc", "grad_attn"], got, ref, strict=True
        ):
            assert not torch.isnan(a).any(), (
                f"{backend.name} {name}: NaN with out-of-bounds locations"
            )
            max_abs = (a - b).abs().max().item()
            rel = max_abs / (b.abs().max().item() + 1e-8)
            assert rel < 1e-4 or max_abs < 1e-4, (
                f"{backend.name} {name}: rel={rel:.2e} vs {_REFERENCE.name}"
            )


@requires_cuda
@pytest.mark.parametrize("backend", _BACKENDS, ids=str)
@pytest.mark.parametrize("head_dim", [32, 128])  # 32 -> warp path, 128 -> shared path
@pytest.mark.parametrize("num_points", [4, 8])
def test_softmax_masked_logit_matches_reference(
    backend: Backend, head_dim: int, num_points: int
) -> None:
    """A ``-inf`` logit at the first sampled point does not NaN the query.

    Locations are in bounds, so only the softmax accumulator's seed is tested.
    """
    for seed in range(3):
        ref = _run(_REFERENCE, head_dim, num_points, seed, mask_first_point=True)
        got = _run(backend, head_dim, num_points, seed, mask_first_point=True)
        for name, a, b in zip(
            ["output", "grad_value", "grad_loc", "grad_attn"], got, ref, strict=True
        ):
            assert not torch.isnan(a).any(), (
                f"{backend.name} {name}: NaN with a masked (-inf) attention logit"
            )
            max_abs = (a - b).abs().max().item()
            rel = max_abs / (b.abs().max().item() + 1e-8)
            assert rel < 1e-4 or max_abs < 1e-4, (
                f"{backend.name} {name}: rel={rel:.2e} vs {_REFERENCE.name}"
            )
