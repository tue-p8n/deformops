import pytest
import torch
from deformops.backends.reference.msda import reference as msda_reference
from deformops.registry import msda as _msda_op
from hypothesis import given, settings
from hypothesis import strategies as st


class _MSDAShim:
    """The MSDA registry paired with its reference, linear unless overridden."""

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
    # Batch size
    N = draw(st.integers(min_value=1, max_value=4))
    # Number of heads
    M = draw(st.sampled_from([1, 2, 4, 8]))
    # Head dimension (a power of 2, which some kernels require)
    D = draw(st.sampled_from([16, 32]))
    # Number of levels
    L = draw(st.integers(min_value=1, max_value=4))
    # Number of points
    P = draw(st.integers(min_value=1, max_value=8))
    # Number of queries
    Q = draw(st.integers(min_value=1, max_value=64))

    # Spatial shapes for each level
    shapes = []
    for _ in range(L):
        H = draw(st.integers(min_value=1, max_value=64))
        W = draw(st.integers(min_value=1, max_value=64))
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
    values = torch.randn((N, S, M, D), device=DEVICE, generator=generator)
    locs = torch.rand((N, Q, M, L, P, 2), device=DEVICE, generator=generator)
    attn = torch.randn((N, Q, M, L, P), device=DEVICE, generator=generator).softmax(-1)

    return values, shapes_tensor, start_index, locs, attn


@pytest.mark.parametrize("dtype", [torch.float32])
@given(inputs=msda_inputs())
@settings(deadline=None, max_examples=20)
def test_msda_parity(dtype, inputs):
    values, shapes, start_index, locs, attn = inputs
    values = values.to(dtype)
    locs = locs.to(dtype)
    attn = attn.to(dtype)

    # 1. CPU Reference (Always available)
    expected = msda.reference(values.cpu(), shapes.cpu(), locs.cpu(), attn.cpu())

    # 2. Forward implementation (Switch between CPU reference and CUDA)
    actual = msda.forward(values, shapes, start_index, locs, attn)

    torch.testing.assert_close(actual.cpu(), expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float32])
@given(inputs=msda_inputs())
@settings(deadline=None, max_examples=10)
def test_msda_resolutions(dtype, inputs):
    # Also force one level to be tiny.
    values, shapes, start_index, locs, attn = inputs

    # Make the first level 1x1.
    shapes[0] = torch.tensor([1, 1], device=DEVICE)

    output = msda.forward(values, shapes, start_index, locs, attn)
    assert output.shape == (
        values.shape[0],
        locs.shape[1],
        values.shape[2] * values.shape[3],
    )


@pytest.mark.parametrize("n_heads", [1, 3, 5])  # Non-standard head counts
@pytest.mark.parametrize("n_levels", [1, 2, 5])
def test_msda_configs(n_heads, n_levels):
    N, Q, P, D = 2, 8, 4, 16
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    shapes = torch.tensor([(16, 16)] * n_levels, dtype=torch.long, device=DEVICE)
    start_index = torch.zeros(n_levels, dtype=torch.long, device=DEVICE)
    if n_levels > 1:
        start_index = torch.cat(
            [torch.tensor([0], device=DEVICE), (shapes[:-1, 0] * shapes[:-1, 1])]
        ).cumsum(0)

    S = (shapes[:, 0] * shapes[:, 1]).sum().item()

    values = torch.randn((N, S, n_heads, D), device=DEVICE)
    locs = torch.rand((N, Q, n_heads, n_levels, P, 2), device=DEVICE)
    attn = torch.randn((N, Q, n_heads, n_levels, P), device=DEVICE).softmax(-1)

    output = msda.forward(values, shapes, start_index, locs, attn)
    assert output.shape == (N, Q, n_heads * D)


def test_msda_out_of_bounds():
    N, Q, M, L, P, D = 1, 1, 1, 1, 1, 16
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    shapes = torch.tensor([(4, 4)], dtype=torch.long, device=DEVICE)
    start_index = torch.tensor([0], dtype=torch.long, device=DEVICE)
    values = torch.ones((N, 16, M, D), device=DEVICE)

    # Locs far outside [0, 1]
    locs = torch.tensor([[[[[[2.0, 2.0]]]]]], device=DEVICE)
    attn = torch.ones((N, Q, M, L, P), device=DEVICE)

    output = msda.forward(values, shapes, start_index, locs, attn)

    # Should be zero due to padding_mode="zeros"
    assert torch.all(output == 0)


def test_msda_backward():
    N, Q, M, L, P, D = 2, 4, 2, 2, 2, 8
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    shapes = torch.tensor([(4, 4), (4, 4)], dtype=torch.long, device=DEVICE)
    start_index = torch.tensor([0, 16], dtype=torch.long, device=DEVICE)

    values = torch.randn((N, 32, M, D), device=DEVICE, requires_grad=True)
    locs = torch.rand((N, Q, M, L, P, 2), device=DEVICE, requires_grad=True)
    attn = torch.randn((N, Q, M, L, P), device=DEVICE, requires_grad=True).softmax(-1)
    attn.retain_grad()

    output = msda.forward(values, shapes, start_index, locs, attn)
    output.sum().backward()

    assert values.grad is not None
    assert locs.grad is not None
    assert attn.grad is not None


def test_msda_backward_cpu():
    N, Q, M, L, P, D = 2, 4, 2, 2, 2, 8
    # Force CPU
    DEVICE = torch.device("cpu")

    shapes = torch.tensor([(4, 4), (4, 4)], dtype=torch.long, device=DEVICE)
    torch.tensor([0, 16], dtype=torch.long, device=DEVICE)

    values = torch.randn((N, 32, M, D), device=DEVICE, requires_grad=True)
    locs = torch.rand((N, Q, M, L, P, 2), device=DEVICE, requires_grad=True)
    attn = torch.randn((N, Q, M, L, P), device=DEVICE, requires_grad=True).softmax(-1)
    attn.retain_grad()

    # The reference uses only torch ops, so autograd runs through it on CPU.
    output = msda.reference(values, shapes, locs, attn)
    output.sum().backward()

    assert values.grad is not None
    assert locs.grad is not None
    assert attn.grad is not None


def test_msda_fake():
    from torch._subclasses.fake_tensor import FakeTensorMode

    N, Q, M, L, P, D = 2, 4, 2, 2, 2, 8
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with FakeTensorMode():
        shapes = torch.tensor([(4, 4), (4, 4)], dtype=torch.long, device=DEVICE)
        start_index = torch.tensor([0, 16], dtype=torch.long, device=DEVICE)
        values = torch.randn((N, 32, M, D), device=DEVICE)
        locs = torch.rand((N, Q, M, L, P, 2), device=DEVICE)
        attn = torch.rand((N, Q, M, L, P), device=DEVICE)

        output = msda.forward(values, shapes, start_index, locs, attn)
        assert output.shape == (N, Q, M * D)
