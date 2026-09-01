import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA is not available", allow_module_level=True)

pytestmark = pytest.mark.gpu

from deformops.backends.cuda import msda_fp32acc  # noqa: E402, F401

# Imported for their side effect: each registers its ``deformops_triton`` operators.
pytest.importorskip("triton")
import deformops.backends.triton.experimental.msda_query_tiled_autotuned  # noqa: E402, F401
import deformops.backends.triton.experimental.msda_query_tiled_autotuned_fp32acc  # noqa: E402, F401
import deformops.backends.triton.experimental.msda_query_tiled_cached  # noqa: E402, F401
import deformops.backends.triton.experimental.msda_query_tiled_cached_fp32acc  # noqa: E402, F401
from torch._subclasses.fake_tensor import FakeTensorMode  # noqa: E402


def test_msda_cuda_softmax_fake_tensors():
    """In-kernel softmax via the production ``_softmax_forward`` op with fake tensors."""
    N, M, D, L, P, Q = 1, 2, 8, 1, 4, 8
    d_stride, block_thread = 8, 4  # minimal tuning params (not executed)

    with FakeTensorMode():
        shapes = torch.tensor([[4, 4]], dtype=torch.long, device="cuda")
        start_index = torch.tensor([0], dtype=torch.long, device="cuda")
        values = torch.randn((N, 16, M, D), device="cuda", requires_grad=True)
        locs = torch.rand((N, Q, M, L, P, 2), device="cuda", requires_grad=True)
        # Raw logits; the softmax variant normalizes them in-kernel.
        attn = torch.randn((N, Q, M, L, P), device="cuda", requires_grad=True)

        output = torch.ops.deformops_cuda.msda_fp32acc_softmax_forward(
            values,
            shapes,
            start_index,
            locs,
            attn,
            64,
            P,
            d_stride,
            block_thread,
        )
        assert output.shape == (N, Q, M * D)

        output.sum().backward()


# Triton fakes must match the real ops' arity, dtype and shape, or compiled models
# fail at run time instead of trace time.


def test_msda_query_tiled_cached_softmax_forward_fake_tensors():
    """The fake returns a tuple with float32 ``(B, Q, H)`` statistics, like the op."""
    N, M, D, L, P, Q = 1, 2, 8, 1, 4, 8

    with FakeTensorMode():
        shapes = torch.tensor([[4, 4]], dtype=torch.int32, device="cuda")
        start_index = torch.tensor([0], dtype=torch.int32, device="cuda")
        values = torch.randn((N, 16, M, D), device="cuda", requires_grad=True)
        locs = torch.rand((N, Q, M, L, P, 2), device="cuda", requires_grad=True)
        attn = torch.randn((N, Q, M, L * P), device="cuda", requires_grad=True)

        result = torch.ops.deformops_triton.msda_query_tiled_cached_softmax_forward(
            values,
            shapes,
            start_index,
            locs,
            attn,
        )
        assert isinstance(result, tuple)
        assert len(result) == 3
        output, m_run, z_run = result
        assert output.shape == (N, Q, M * D)
        assert m_run.shape == (N, Q, M)
        assert z_run.shape == (N, Q, M)
        assert m_run.dtype == torch.float32
        assert z_run.dtype == torch.float32

        output.sum().backward()


def test_triton_cached_fp32acc_softmax_fake_tensors():
    """The same outputs for the fp32acc variant with the softmax method."""
    N, M, D, L, P, Q = 1, 2, 8, 1, 4, 8

    with FakeTensorMode():
        shapes = torch.tensor([[4, 4]], dtype=torch.int32, device="cuda")
        start_index = torch.tensor([0], dtype=torch.int32, device="cuda")
        values = torch.randn((N, 16, M, D), device="cuda", requires_grad=True)
        locs = torch.rand((N, Q, M, L, P, 2), device="cuda", requires_grad=True)
        attn = torch.randn((N, Q, M, L * P), device="cuda", requires_grad=True)

        output, m_run, z_run = (
            torch.ops.deformops_triton.msda_query_tiled_cached_fp32acc_softmax_forward(
                values,
                shapes,
                start_index,
                locs,
                attn,
            )
        )
        assert output.shape == (N, Q, M * D)
        assert m_run.shape == (N, Q, M)
        assert z_run.shape == (N, Q, M)
        assert m_run.dtype == torch.float32
        assert z_run.dtype == torch.float32

        output.sum().backward()


def test_triton_cached_fp32acc_linear_fake_tensors():
    """The linear fake returns one tensor; only softmax caches statistics."""
    N, M, D, L, P, Q = 1, 2, 8, 1, 4, 8

    with FakeTensorMode():
        shapes = torch.tensor([[4, 4]], dtype=torch.int32, device="cuda")
        start_index = torch.tensor([0], dtype=torch.int32, device="cuda")
        values = torch.randn((N, 16, M, D), device="cuda", requires_grad=True)
        locs = torch.rand((N, Q, M, L, P, 2), device="cuda", requires_grad=True)
        attn = torch.randn((N, Q, M, L * P), device="cuda", requires_grad=True)

        output = torch.ops.deformops_triton.msda_query_tiled_cached_fp32acc_forward(
            values,
            shapes,
            start_index,
            locs,
            attn,
        )
        assert isinstance(output, torch.Tensor)
        assert output.shape == (N, Q, M * D)

        output.sum().backward()


def test_msda_query_tiled_autotuned_softmax_forward_fake_tensors():
    """The autotuned fake registration returns the same (output, m_run, z_run)."""
    N, M, D, L, P, Q = 1, 2, 8, 1, 4, 8

    with FakeTensorMode():
        shapes = torch.tensor([[4, 4]], dtype=torch.int32, device="cuda")
        start_index = torch.tensor([0], dtype=torch.int32, device="cuda")
        values = torch.randn((N, 16, M, D), device="cuda", requires_grad=True)
        locs = torch.rand((N, Q, M, L, P, 2), device="cuda", requires_grad=True)
        attn = torch.randn((N, Q, M, L * P), device="cuda", requires_grad=True)

        result = torch.ops.deformops_triton.msda_query_tiled_autotuned_softmax_forward(
            values,
            shapes,
            start_index,
            locs,
            attn,
        )
        assert isinstance(result, tuple)
        assert len(result) == 3
        output, m_run, z_run = result
        assert output.shape == (N, Q, M * D)
        assert m_run.shape == (N, Q, M)
        assert z_run.shape == (N, Q, M)
        assert m_run.dtype == torch.float32
        assert z_run.dtype == torch.float32

        output.sum().backward()


def test_triton_autotuned_fp32acc_softmax_fake_tensors():
    """fp32acc variant with the softmax method: stats shape must be ``(B, Q, H)``."""
    N, M, D, L, P, Q = 1, 2, 8, 1, 4, 8

    with FakeTensorMode():
        shapes = torch.tensor([[4, 4]], dtype=torch.int32, device="cuda")
        start_index = torch.tensor([0], dtype=torch.int32, device="cuda")
        values = torch.randn((N, 16, M, D), device="cuda", requires_grad=True)
        locs = torch.rand((N, Q, M, L, P, 2), device="cuda", requires_grad=True)
        attn = torch.randn((N, Q, M, L * P), device="cuda", requires_grad=True)

        output, m_run, z_run = (
            torch.ops.deformops_triton.msda_query_tiled_autotuned_fp32acc_softmax_forward(
                values,
                shapes,
                start_index,
                locs,
                attn,
            )
        )
        assert output.shape == (N, Q, M * D)
        assert m_run.shape == (N, Q, M)
        assert z_run.shape == (N, Q, M)
        assert m_run.dtype == torch.float32
        assert z_run.dtype == torch.float32

        output.sum().backward()


def test_triton_autotuned_fp32acc_linear_fake_tensors():
    """Linear method: one tensor out, and no statistics."""
    N, M, D, L, P, Q = 1, 2, 8, 1, 4, 8

    with FakeTensorMode():
        shapes = torch.tensor([[4, 4]], dtype=torch.int32, device="cuda")
        start_index = torch.tensor([0], dtype=torch.int32, device="cuda")
        values = torch.randn((N, 16, M, D), device="cuda", requires_grad=True)
        locs = torch.rand((N, Q, M, L, P, 2), device="cuda", requires_grad=True)
        attn = torch.randn((N, Q, M, L * P), device="cuda", requires_grad=True)

        output = torch.ops.deformops_triton.msda_query_tiled_autotuned_fp32acc_forward(
            values,
            shapes,
            start_index,
            locs,
            attn,
        )
        assert isinstance(output, torch.Tensor)
        assert output.shape == (N, Q, M * D)

        output.sum().backward()
