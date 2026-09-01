"""The backward shared-memory reduction, against the reference on the linear path.

``D = 128`` gives more lanes per head than one warp holds, so the kernel takes
the shared path; see ``docs/kernels.md``.
"""

import pytest
import torch
from deformops.registry import Backend, msda

pytestmark = pytest.mark.gpu

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available"
)


@requires_cuda
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(Backend.CUDA, id="cuda"),
    ],
)
def test_backward_shared_reduction_parity(backend: Backend) -> None:
    device = torch.device("cuda")
    batch, num_heads, head_dim = 2, 4, 128  # 128 -> 64 lanes/head -> shared path
    num_levels, num_points, num_queries = 3, 4, 200

    shapes = torch.tensor([[32, 32], [16, 16], [8, 8]], dtype=torch.long, device=device)
    start_index = torch.cat(
        [
            torch.zeros(1, dtype=torch.long, device=device),
            shapes[:-1, 0] * shapes[:-1, 1],
        ]
    ).cumsum(0)
    spatial = int((shapes[:, 0] * shapes[:, 1]).sum())

    torch.manual_seed(0)
    value = torch.randn(batch, spatial, num_heads, head_dim, device=device)
    locations = torch.rand(
        batch, num_queries, num_heads, num_levels, num_points, 2, device=device
    )
    attention = torch.randn(
        batch, num_queries, num_heads, num_levels, num_points, device=device
    )
    upstream = torch.randn(batch, num_queries, num_heads * head_dim, device=device)

    def run(
        be: Backend,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        v = value.clone().requires_grad_(True)
        loc = locations.clone().requires_grad_(True)
        attn = attention.clone().requires_grad_(True)
        out = msda(
            v,
            shapes,
            start_index,
            loc,
            attn,
            method="linear",
            backend=be,
        )
        out.backward(upstream)
        assert v.grad is not None
        assert loc.grad is not None
        assert attn.grad is not None
        return out.detach(), v.grad, loc.grad, attn.grad

    got = run(backend)
    ref = run(Backend.EAGER)
    for name, a, b in zip(
        ["output", "grad_value", "grad_loc", "grad_attn"], got, ref, strict=True
    ):
        max_abs = (a - b).abs().max().item()
        rel = max_abs / (b.abs().max().item() + 1e-8)
        assert rel < 1e-3 or max_abs < 1e-3, f"{backend.name} {name}: rel={rel:.2e}"
