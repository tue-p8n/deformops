import pytest
import torch
from deformops.backends.reference.msda import reference as msda_reference
from deformops.registry import Backend
from deformops.registry import msda as _msda_op
from hypothesis import given, settings
from hypothesis import strategies as st


class _MSDAShim:
    @staticmethod
    def forward(*args, **kwargs):
        kwargs.setdefault("method", "linear")
        return _msda_op(*args, **kwargs)

    reference = staticmethod(msda_reference)


msda = _MSDAShim()

# Device configuration
if not torch.cuda.is_available():
    pytest.skip("CUDA is not available", allow_module_level=True)

pytestmark = pytest.mark.gpu

DEVICE = torch.device("cuda")


@st.composite
def msda_inputs(draw):
    N = draw(st.integers(min_value=1, max_value=2))
    M = draw(st.sampled_from([1, 2, 4]))
    D = draw(st.sampled_from([16, 32]))
    L = draw(st.integers(min_value=1, max_value=4))
    P = draw(st.sampled_from([4, 8]))  # CUDA kernel specialized for 4, 8
    Q = draw(st.integers(min_value=1, max_value=32))

    # Spatial shapes for each level
    shapes = []
    for _ in range(L):
        H = draw(st.integers(min_value=4, max_value=32))
        W = draw(st.integers(min_value=4, max_value=32))
        shapes.append((H, W))

    shapes_tensor = torch.tensor(shapes, dtype=torch.long, device=DEVICE)
    start_index = torch.cat(
        [
            torch.tensor([0], device=DEVICE),
            shapes_tensor[:-1, 0] * shapes_tensor[:-1, 1],
        ]
    ).cumsum(0)
    S = (shapes_tensor[:, 0] * shapes_tensor[:, 1]).sum().item()

    # Seeded by hypothesis, so a failing example replays and shrinks.
    generator = torch.Generator(device=DEVICE).manual_seed(
        draw(st.integers(min_value=0, max_value=2**31 - 1))
    )
    values = torch.randn(
        (N, S, M, D), device=DEVICE, generator=generator, requires_grad=True
    )
    locs = torch.rand(
        (N, Q, M, L, P, 2), device=DEVICE, generator=generator, requires_grad=True
    )
    attn = torch.randn(
        (N, Q, M, L, P), device=DEVICE, generator=generator, requires_grad=True
    )
    # Softmax is applied, but the gradient is checked on the raw attn
    attn_fwd = attn.softmax(-1)

    return values, shapes_tensor, start_index, locs, attn, attn_fwd


@pytest.mark.parametrize("dtype", [torch.float32])
@given(inputs=msda_inputs())
@settings(deadline=None, max_examples=10)
def test_msda_cuda_parity(dtype, inputs):
    values, shapes, start_index, locs, attn, attn_fwd = inputs
    values = values.to(dtype)
    locs = locs.to(dtype)
    attn = attn.to(dtype)
    attn_fwd = attn_fwd.to(dtype)

    # 1. CPU Reference
    expected = msda.reference(
        values.detach().cpu(),
        shapes.cpu(),
        locs.detach().cpu(),
        attn_fwd.detach().cpu(),
    )

    # 2. CUDA Forward
    actual = msda.forward(
        values, shapes, start_index, locs, attn_fwd, backend=Backend.CUDA
    )

    torch.testing.assert_close(actual.cpu(), expected, atol=1e-3, rtol=1e-3)

    # 3. CUDA Backward
    actual.sum().backward()

    # Compare against CPU autograd of the reference
    values_ref = values.detach().cpu().requires_grad_(True)
    locs_ref = locs.detach().cpu().requires_grad_(True)
    attn_ref = attn_fwd.detach().cpu().requires_grad_(True)

    expected_ref = msda.reference(values_ref, shapes.cpu(), locs_ref, attn_ref)
    expected_ref.sum().backward()

    torch.testing.assert_close(values.grad.cpu(), values_ref.grad, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(locs.grad.cpu(), locs_ref.grad, atol=1e-3, rtol=1e-3)
    # attn_fwd is not a leaf: compare at attn, through softmax, on both graphs.
    attn_ref_leaf = attn.detach().cpu().requires_grad_(True)
    expected_ref_full = msda.reference(
        values_ref, shapes.cpu(), locs_ref, attn_ref_leaf.softmax(-1)
    )
    expected_ref_full.sum().backward()

    torch.testing.assert_close(
        attn.grad.cpu(), attn_ref_leaf.grad, atol=1e-3, rtol=1e-3
    )


def test_msda_cuda_fake_tensors():
    from torch._subclasses.fake_tensor import FakeTensorMode

    # The CUDA kernels need the per-head channel count divisible by 8.
    N, M, D, L, P, Q = 1, 2, 8, 1, 4, 8

    with FakeTensorMode():
        shapes = torch.tensor([[4, 4]], dtype=torch.long, device="cuda")
        start_index = torch.tensor([0], dtype=torch.long, device="cuda")
        values = torch.randn((N, 16, M, D), device="cuda", requires_grad=True)
        locs = torch.rand((N, Q, M, L, P, 2), device="cuda", requires_grad=True)
        attn = torch.randn((N, Q, M, L, P), device="cuda", requires_grad=True)

        # Real callers pass softmaxed attn
        attn_fwd = attn.softmax(-1)

        output = msda.forward(
            values, shapes, start_index, locs, attn_fwd, backend=Backend.CUDA
        )
        assert output.shape == (N, Q, M * D)

        output.sum().backward()
        assert values.grad is not None
        assert locs.grad is not None
        assert attn.grad is not None
