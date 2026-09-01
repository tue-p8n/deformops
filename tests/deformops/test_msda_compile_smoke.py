import pytest
import torch
from deformops.nn import MultiScaleDeformAttn2d
from deformops.registry import Backend


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "backend",
    [Backend.EAGER, Backend.CUDA, Backend.TRITON],
)
def test_compile_smoke(backend):
    B, Q, dim = 1, 4, 16
    num_heads, num_levels, num_points = 1, 1, 4

    device = torch.device("cuda")

    m = MultiScaleDeformAttn2d(
        dim=dim,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
        backend=backend,
    ).to(device)

    # Conditionally compile the module
    m_to_run = m if backend == Backend.EAGER else torch.compile(m)

    S = 4 * num_levels
    q = torch.randn(B, Q, dim, device=device)
    p = torch.rand(B, Q, num_levels, 2, device=device)
    v = torch.randn(B, S, dim, device=device)
    shapes = torch.tensor([[2, 2]] * num_levels, dtype=torch.long, device=device)
    level_index = torch.zeros(num_levels, dtype=torch.long, device=device)

    # Execution
    out1 = m_to_run(q, p, v, shapes, level_index)
    out2 = m_to_run(
        q, p, v, shapes, level_index
    )  # Run again to ensure it executes compiled

    assert out1.shape == (B, Q, dim)
    torch.testing.assert_close(out1, out2)
