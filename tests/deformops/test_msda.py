import pytest
import torch
from deformops.nn import MultiScaleDeformAttn2d
from deformops.registry import Backend

if not torch.cuda.is_available():
    pytest.skip("CUDA is not available", allow_module_level=True)

pytestmark = pytest.mark.gpu

DEVICE = torch.device("cuda")


@pytest.mark.parametrize("dim", [32, 64])
@pytest.mark.parametrize("num_heads", [2, 4])
@pytest.mark.parametrize("num_levels", [1, 2])
def test_msda_forward_shape(dim, num_heads, num_levels):
    B, Q, S = 2, 8, 32
    num_points = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    m = MultiScaleDeformAttn2d(
        dim=dim,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
        method="softmax",
        backend=Backend.CUDA,
    ).to(DEVICE)

    q = torch.randn(B, Q, dim, device=DEVICE)
    p = torch.rand(B, Q, num_levels, 2, device=DEVICE)
    v = torch.randn(B, S, dim, device=DEVICE)

    # Valid shapes and level_index for S=32 at this num_levels
    if num_levels == 1:
        shapes = torch.tensor([[8, 4]], dtype=torch.long, device=DEVICE)
        level_index = torch.tensor([0], dtype=torch.long, device=DEVICE)
    else:
        shapes = torch.tensor([[4, 4], [4, 4]], dtype=torch.long, device=DEVICE)
        level_index = torch.tensor([0, 16], dtype=torch.long, device=DEVICE)

    out = m(q, p, v, shapes, level_index)

    assert out.shape == (B, Q, dim)


def test_msda_methods():
    B, Q, S, dim = 1, 4, 16, 16
    num_heads, num_levels, num_points = 2, 1, 2
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    q = torch.randn(B, Q, dim, device=DEVICE)
    p = torch.rand(B, Q, num_levels, 2, device=DEVICE)
    v = torch.randn(B, S, dim, device=DEVICE)
    shapes = torch.tensor([[4, 4]], dtype=torch.long, device=DEVICE)
    level_index = torch.tensor([0], dtype=torch.long, device=DEVICE)

    for method in ["linear", "softmax"]:
        m = MultiScaleDeformAttn2d(
            dim=dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            method=method,
        ).to(DEVICE)
        out = m(q, p, v, shapes, level_index)
        assert out.shape == (B, Q, dim)


def test_msda_padding_mask():
    B, Q, S, dim = 2, 4, 16, 16
    num_heads, num_levels, num_points = 2, 1, 2
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    m = MultiScaleDeformAttn2d(
        dim=dim, num_heads=num_heads, num_levels=num_levels, num_points=num_points
    ).to(DEVICE)

    q = torch.randn(B, Q, dim, device=DEVICE)
    p = torch.rand(B, Q, num_levels, 2, device=DEVICE)
    v = torch.randn(B, S, dim, device=DEVICE)
    shapes = torch.tensor([[4, 4]], dtype=torch.long, device=DEVICE)
    level_index = torch.tensor([0], dtype=torch.long, device=DEVICE)

    padding_mask = torch.zeros(B, S, dtype=torch.bool, device=DEVICE)
    padding_mask[:, 8:] = True  # Mask out half of the values

    out = m(q, p, v, shapes, level_index, padding_mask=padding_mask)
    assert out.shape == (B, Q, dim)


def test_msda_parity():
    B, Q, S, dim = 2, 4, 16, 32
    num_heads, num_levels, num_points = 2, 2, 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    m = MultiScaleDeformAttn2d(
        dim=dim, num_heads=num_heads, num_levels=num_levels, num_points=num_points
    ).to(DEVICE)

    # Same weights for both
    q = torch.randn(B, Q, dim, device=DEVICE)
    p = torch.rand(B, Q, num_levels, 2, device=DEVICE)
    v = torch.randn(B, S, dim, device=DEVICE)
    shapes = torch.tensor([[4, 2], [4, 2]], dtype=torch.long, device=DEVICE)
    level_index = torch.tensor([0, 8], dtype=torch.long, device=DEVICE)

    # 1. Forward (CUDA if available)
    out_cuda = m(q, p, v, shapes, level_index)

    # 2. Eager (CPU): moving everything to the CPU routes dispatch to eager.
    m_cpu = m.to("cpu")
    out_cpu = m_cpu(q.cpu(), p.cpu(), v.cpu(), shapes.cpu(), level_index.cpu())

    torch.testing.assert_close(out_cuda.cpu(), out_cpu, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("backend", [pytest.param(Backend.TRITON, id="triton")])
def test_msda_triton_backend_end_to_end(backend):
    B, Q, S, dim = 2, 4, 16, 32
    num_heads, num_levels, num_points = 2, 2, 4
    device = torch.device("cuda")

    m = MultiScaleDeformAttn2d(
        dim=dim,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
        backend=backend,
    ).to(device)

    q = torch.randn(B, Q, dim, device=device, requires_grad=True)
    p = torch.rand(B, Q, num_levels, 2, device=device, requires_grad=True)
    v = torch.randn(B, S, dim, device=device, requires_grad=True)
    shapes = torch.tensor([[4, 2], [4, 2]], dtype=torch.long, device=device)
    level_index = torch.tensor([0, 8], dtype=torch.long, device=device)

    out = m(q, p, v, shapes, level_index)
    assert out.shape == (B, Q, dim)

    out.sum().backward()
    assert q.grad is not None
    assert p.grad is not None
    assert v.grad is not None


def test_msda_backward():
    B, Q, S, dim = 2, 4, 16, 32
    num_heads, num_levels, num_points = 2, 1, 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    m = MultiScaleDeformAttn2d(
        dim=dim,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
        method="softmax",
        backend=Backend.CUDA,
    ).to(DEVICE)

    q = torch.randn(B, Q, dim, device=DEVICE, requires_grad=True)
    p = torch.rand(B, Q, num_levels, 2, device=DEVICE, requires_grad=True)
    v = torch.randn(B, S, dim, device=DEVICE, requires_grad=True)
    shapes = torch.tensor([[4, 4]], dtype=torch.long, device=DEVICE)
    level_index = torch.tensor([0], dtype=torch.long, device=DEVICE)

    out = m(q, p, v, shapes, level_index)
    out.sum().backward()

    assert q.grad is not None
    assert p.grad is not None
    assert v.grad is not None


def test_msda_invalid_params():
    with pytest.raises(ValueError, match="Expected dim to be a positive integer"):
        MultiScaleDeformAttn2d(dim=0, num_heads=1, num_levels=1, num_points=1)

    with pytest.raises(ValueError, match="Expected num_heads to be a positive integer"):
        MultiScaleDeformAttn2d(dim=16, num_heads=0, num_levels=1, num_points=1)

    with pytest.raises(ValueError, match="Expected 16 to be divisible by 3"):
        MultiScaleDeformAttn2d(dim=16, num_heads=3, num_levels=1, num_points=1)

    with pytest.raises(ValueError, match="Expected a power of 2, got: 6"):
        # dim=12, num_heads=2 => d_per_head=6 (not power of 2)
        MultiScaleDeformAttn2d(dim=12, num_heads=2, num_levels=1, num_points=1)


@pytest.mark.gpu
def test_msda_kernel_aggregate_parity():
    """The CUDA in-kernel softmax matches the eager backend's softmax in Python."""
    B, Q, dim, num_heads, num_levels, num_points = 2, 300, 256, 8, 4, 4
    # shapes: [[100, 167], [50, 84], [25, 42], [13, 21]]
    # S = 100*167 + 50*84 + 25*42 + 13*21 = 16700 + 4200 + 1050 + 273 = 22223
    S = 22223

    shapes = torch.tensor(
        [[100, 167], [50, 84], [25, 42], [13, 21]], device="cuda", dtype=torch.long
    )
    level_index = torch.tensor(
        [0, 16700, 20900, 21950], device="cuda", dtype=torch.long
    )

    torch.manual_seed(42)
    q = torch.randn(B, Q, dim, device="cuda", dtype=torch.float32)
    p = torch.rand(B, Q, num_levels, 2, device="cuda", dtype=torch.float32).clamp(
        0.01, 0.99
    )
    v = torch.randn(B, S, dim, device="cuda", dtype=torch.float32)

    # Softmax applied in Python by the eager backend before the op.
    m_python = MultiScaleDeformAttn2d(
        dim=dim,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
        method="softmax",
        backend=Backend.EAGER,
    ).to("cuda", dtype=torch.float32)

    # Softmax applied inside the CUDA kernel.
    m_kernel = MultiScaleDeformAttn2d(
        dim=dim,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
        method="softmax",
        backend=Backend.CUDA,
    ).to("cuda", dtype=torch.float32)

    # Copy weights so both modules produce the same projection outputs.
    m_kernel.load_state_dict(m_python.state_dict())

    with torch.no_grad():
        out_python = m_python(q, p, v, shapes, level_index)
        out_kernel = m_kernel(q, p, v, shapes, level_index)

    torch.testing.assert_close(out_python, out_kernel, rtol=1e-4, atol=1e-4)
