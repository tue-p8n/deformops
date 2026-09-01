import deformops  # noqa: F401
import pytest
import torch
from deformops.nn import MultiScaleDeformAttn2d
from deformops.registry import Backend

# Imported to register their ``deformops_triton`` ops; skipped on hosts without triton.
try:
    import deformops.backends.triton.experimental.msda_query_tiled_autotuned  # noqa: F401
    import deformops.backends.triton.experimental.msda_query_tiled_autotuned_fp32acc  # noqa: F401
    import deformops.backends.triton.experimental.msda_query_tiled_cached  # noqa: F401
    import deformops.backends.triton.experimental.msda_query_tiled_cached_fp32acc  # noqa: F401
    from deformops.backends.triton.msda_query_tiled_fp32acc import (
        Backend as TritonFp32AccBackend,
    )
except ImportError:
    TritonFp32AccBackend = None


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_triton_forward_parity():
    # Setup
    B, Q, dim = 2, 8, 32
    num_heads, num_levels, num_points = 4, 2, 4
    device = torch.device("cuda")

    # Initialize with same weights/inputs
    m_pt = MultiScaleDeformAttn2d(
        dim=dim,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
        backend=Backend.EAGER,
    ).to(device)

    m_triton = MultiScaleDeformAttn2d(
        dim=dim,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
        backend=Backend.TRITON,
    ).to(device)

    m_triton.load_state_dict(m_pt.state_dict())

    q = torch.randn(B, Q, dim, device=device)
    p = torch.rand(B, Q, num_levels, 2, device=device)
    v = torch.randn(B, 32, dim, device=device)
    shapes = torch.tensor([[4, 4], [4, 4]], dtype=torch.long, device=device)
    level_index = torch.tensor([0, 16], dtype=torch.long, device=device)

    # Forward passes
    out_pt = m_pt(q, p, v, shapes, level_index)
    out_triton = m_triton(q, p, v, shapes, level_index)

    torch.testing.assert_close(out_pt, out_triton, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_triton_forward_parity_dtypes(dtype):
    # Setup
    B, Q, dim = 2, 8, 32
    num_heads, num_levels, num_points = 4, 2, 4
    device = torch.device("cuda")

    # Initialize with same weights/inputs
    m_pt = MultiScaleDeformAttn2d(
        dim=dim,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
        backend=Backend.EAGER,
    ).to(device, dtype=dtype)

    m_triton = MultiScaleDeformAttn2d(
        dim=dim,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
        backend=Backend.TRITON,
    ).to(device, dtype=dtype)

    m_triton.load_state_dict(m_pt.state_dict())

    q = torch.randn(B, Q, dim, device=device, dtype=dtype)
    p = torch.rand(B, Q, num_levels, 2, device=device, dtype=dtype)
    v = torch.randn(B, 32, dim, device=device, dtype=dtype)
    shapes = torch.tensor([[4, 4], [4, 4]], dtype=torch.long, device=device)
    level_index = torch.tensor([0, 16], dtype=torch.long, device=device)

    # Forward passes
    out_pt = m_pt(q, p, v, shapes, level_index)
    out_triton = m_triton(q, p, v, shapes, level_index)

    # Relax tolerances for half-precision
    atol, rtol = (1e-5, 1e-5) if dtype == torch.float32 else (1e-2, 1e-2)
    torch.testing.assert_close(out_pt, out_triton, atol=atol, rtol=rtol)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_triton_backward_gradcheck():
    # Gradcheck diverges at the image edge, so samples are kept inside. H = W is large
    # against the DETR grid offsets, and p is drawn from [0.25, 0.75].
    B, Q, dim = 1, 2, 8
    num_heads, num_levels, num_points = 1, 1, 4
    device = torch.device("cuda")
    dtype = torch.float64  # Use float64 for gradcheck
    H = W = 16

    q = torch.randn(B, Q, dim, device=device, dtype=dtype, requires_grad=True)
    p = (
        torch.rand(B, Q, num_levels, 2, device=device, dtype=dtype) * 0.5 + 0.25
    ).requires_grad_(True)
    v = torch.randn(B, H * W, dim, device=device, dtype=dtype, requires_grad=True)
    shapes = torch.tensor([[H, W]], dtype=torch.long, device=device)
    level_index = torch.tensor([0], dtype=torch.long, device=device)

    m = MultiScaleDeformAttn2d(
        dim=dim,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
        # The kernel object: dispatch refuses float64 for Triton; the kernel casts it.
        backend=TritonFp32AccBackend(),
    ).to(device, dtype=dtype)
    # Zero the grid bias so each sample starts at p, off the boundary.
    torch.nn.init.zeros_(m.proj_offset.bias)

    def func(q_, p_, v_):
        return m(q_, p_, v_, shapes, level_index)

    # Float32 inside the kernel: eps is well above its quantization noise.
    torch.autograd.gradcheck(func, (q, p, v), eps=1e-3, atol=1e-2, fast_mode=False)


# The wrappers discard the (m_run, z_run) statistics, so these tests check them at the
# registered op. A regression would otherwise show only as wrong gradients.


def _build_op_inputs(B: int, Q: int, H: int, D: int, L: int, K: int, dtype):
    device = torch.device("cuda")
    spatial = [(8, 8)] * L
    sizes = [h * w for h, w in spatial]
    N = sum(sizes)
    value = torch.randn(B, N, H, D, device=device, dtype=dtype)
    loc = torch.rand(B, Q, H, L, K, 2, device=device, dtype=dtype) * 0.5 + 0.25
    attn_raw = torch.randn(B, Q, H, L * K, device=device, dtype=dtype)
    shapes = torch.tensor(spatial, dtype=torch.int32, device=device)
    lsi = torch.tensor([0] + sizes[:-1], dtype=torch.int32, device=device).cumsum(0)
    return value, shapes, lsi, loc, attn_raw


def _softmax_matches_stats(raw_attn, m_run, z_run, *, atol: float, rtol: float) -> None:
    """Assert the kernel's saved stats rebuild softmax(raw_attn)."""
    rebuilt = (raw_attn.float() - m_run.unsqueeze(-1)).exp_().div_(z_run.unsqueeze(-1))
    ref = torch.softmax(raw_attn.float(), dim=-1)
    torch.testing.assert_close(rebuilt, ref, atol=atol, rtol=rtol)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_msda_query_tiled_cached_softmax_forward_op_returns_three_tensors():
    """The op returns ``(output, m_run, z_run)`` with the shapes the bridge expects.

    ``exp(raw_attn - m_run) / z_run`` must equal ``softmax(raw_attn)``.
    """
    B, Q, H, D, L, K = 1, 4, 2, 8, 1, 4
    value, shapes, lsi, loc, attn = _build_op_inputs(B, Q, H, D, L, K, torch.float32)
    result = torch.ops.deformops_triton.msda_query_tiled_cached_softmax_forward(
        value,
        shapes,
        lsi,
        loc,
        attn,
    )
    assert isinstance(result, tuple)
    assert len(result) == 3
    output, m_run, z_run = result
    assert output.shape == (B, Q, H * D)
    assert m_run.shape == (B, Q, H)
    assert z_run.shape == (B, Q, H)
    assert m_run.dtype == torch.float32
    assert z_run.dtype == torch.float32
    _softmax_matches_stats(attn, m_run, z_run, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_triton_cached_fp32acc_softmax_forward_op_returns_three_tensors():
    """The same assertions for the fp32acc variant under the softmax method."""
    B, Q, H, D, L, K = 1, 4, 2, 8, 1, 4
    value, shapes, lsi, loc, attn = _build_op_inputs(B, Q, H, D, L, K, torch.float32)
    result = torch.ops.deformops_triton.msda_query_tiled_cached_fp32acc_softmax_forward(
        value,
        shapes,
        lsi,
        loc,
        attn,
    )
    assert isinstance(result, tuple)
    assert len(result) == 3
    output, m_run, z_run = result
    assert output.shape == (B, Q, H * D)
    assert m_run.shape == (B, Q, H)
    assert z_run.shape == (B, Q, H)
    assert m_run.dtype == torch.float32
    assert z_run.dtype == torch.float32
    _softmax_matches_stats(attn, m_run, z_run, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_triton_cached_fp32acc_linear_forward_op_returns_one_tensor():
    """The linear operator returns no statistics; only softmax caches them."""
    B, Q, H, D, L, K = 1, 4, 2, 8, 1, 4
    value, shapes, lsi, loc, attn = _build_op_inputs(B, Q, H, D, L, K, torch.float32)
    output = torch.ops.deformops_triton.msda_query_tiled_cached_fp32acc_forward(
        value,
        shapes,
        lsi,
        loc,
        attn,
    )
    assert isinstance(output, torch.Tensor)
    assert output.shape == (B, Q, H * D)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("backend", [pytest.param(Backend.TRITON, id="triton")])
def test_triton_cached_softmax_handles_zero_queries(backend: Backend):
    """With ``Q == 0``, the ops return ``(B, 0, *)`` tensors and backward succeeds.

    The bridge unpacks a 3-tuple on this path too.
    """
    B, H, dim = 2, 2, 16
    num_levels, num_points = 1, 4
    device = torch.device("cuda")
    dtype = torch.float32
    q = torch.randn(B, 0, dim, device=device, dtype=dtype, requires_grad=True)
    p = torch.rand(B, 0, num_levels, 2, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, 64, dim, device=device, dtype=dtype, requires_grad=True)
    shapes = torch.tensor([[8, 8]], dtype=torch.long, device=device)
    level_index = torch.tensor([0], dtype=torch.long, device=device)
    m = MultiScaleDeformAttn2d(
        dim=dim,
        num_heads=H,
        num_levels=num_levels,
        num_points=num_points,
        backend=backend,
    ).to(device, dtype=dtype)
    out = m(q, p, v, shapes, level_index)
    assert out.shape == (B, 0, dim)
    out.sum().backward()  # must not raise


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_msda_query_tiled_autotuned_softmax_forward_op_returns_three_tensors():
    """The autotuned softmax forward returns what the cached kernel returns."""
    B, Q, H, D, L, K = 1, 4, 2, 8, 1, 4
    value, shapes, lsi, loc, attn = _build_op_inputs(B, Q, H, D, L, K, torch.float32)
    result = torch.ops.deformops_triton.msda_query_tiled_autotuned_softmax_forward(
        value,
        shapes,
        lsi,
        loc,
        attn,
    )
    assert isinstance(result, tuple)
    assert len(result) == 3
    output, m_run, z_run = result
    assert output.shape == (B, Q, H * D)
    assert m_run.shape == (B, Q, H)
    assert z_run.shape == (B, Q, H)
    assert m_run.dtype == torch.float32
    assert z_run.dtype == torch.float32
    _softmax_matches_stats(attn, m_run, z_run, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_triton_autotuned_fp32acc_softmax_forward_op_returns_three_tensors():
    """The same assertions for the fp32acc variant under the softmax method."""
    B, Q, H, D, L, K = 1, 4, 2, 8, 1, 4
    value, shapes, lsi, loc, attn = _build_op_inputs(B, Q, H, D, L, K, torch.float32)
    result = (
        torch.ops.deformops_triton.msda_query_tiled_autotuned_fp32acc_softmax_forward(
            value,
            shapes,
            lsi,
            loc,
            attn,
        )
    )
    assert isinstance(result, tuple)
    assert len(result) == 3
    output, m_run, z_run = result
    assert output.shape == (B, Q, H * D)
    assert m_run.shape == (B, Q, H)
    assert z_run.shape == (B, Q, H)
    assert m_run.dtype == torch.float32
    assert z_run.dtype == torch.float32
    _softmax_matches_stats(attn, m_run, z_run, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_triton_autotuned_fp32acc_linear_forward_op_returns_one_tensor():
    """Linear method: the operator returns only the output."""
    B, Q, H, D, L, K = 1, 4, 2, 8, 1, 4
    value, shapes, lsi, loc, attn = _build_op_inputs(B, Q, H, D, L, K, torch.float32)
    output = torch.ops.deformops_triton.msda_query_tiled_autotuned_fp32acc_forward(
        value,
        shapes,
        lsi,
        loc,
        attn,
    )
    assert isinstance(output, torch.Tensor)
    assert output.shape == (B, Q, H * D)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("backend", [pytest.param(Backend.TRITON, id="triton")])
def test_triton_autotuned_softmax_handles_zero_queries(backend: Backend):
    """Q == 0 short-circuit must produce ``(B, 0, *)`` tensors without crashing."""
    B, H, dim = 2, 2, 16
    num_levels, num_points = 1, 4
    device = torch.device("cuda")
    dtype = torch.float32
    q = torch.randn(B, 0, dim, device=device, dtype=dtype, requires_grad=True)
    p = torch.rand(B, 0, num_levels, 2, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, 64, dim, device=device, dtype=dtype, requires_grad=True)
    shapes = torch.tensor([[8, 8]], dtype=torch.long, device=device)
    level_index = torch.tensor([0], dtype=torch.long, device=device)
    m = MultiScaleDeformAttn2d(
        dim=dim,
        num_heads=H,
        num_levels=num_levels,
        num_points=num_points,
        backend=backend,
    ).to(device, dtype=dtype)
    out = m(q, p, v, shapes, level_index)
    assert out.shape == (B, 0, dim)
    out.sum().backward()  # must not raise
