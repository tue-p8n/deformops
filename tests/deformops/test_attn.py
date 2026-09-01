import pytest
import torch
from deformops.nn import DeformAttn2d, DeformSample2d


@pytest.fixture(scope="module")
def device(any_device):
    """The CPU always, CUDA when present.

    Off CUDA the modules run the eager backend, so every shape and dtype check here
    also runs on the CPU.
    """
    return any_device


# =========================================================================
# DeformAttn2d (cross-attention op)
# =========================================================================


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "input_shape", [(1, 128, 16, 16), (1, 256, 16, 16)]
)  # [N, C, H, W]
def test_deform_attn2d_forward_4d(dtype, input_shape, device):
    N, C, H, W = input_shape
    groups = 4 if C == 128 else 8

    model = DeformAttn2d(dim=C, kernel_size=3, groups=groups).to(device).to(dtype)
    x = torch.randn(input_shape, device=device, dtype=dtype)

    # In self-attention mode (q=x, v=x)
    output = model(x, x)

    assert output.shape == x.shape
    assert output.dtype == dtype


def test_deform_attn2d_cross_attention(device):
    N, C, H, W = 1, 64, 16, 16
    model = DeformAttn2d(dim=C, kernel_size=3).to(device)

    q = torch.randn((N, C, H, W), device=device)
    v = torch.randn((N, C, H, W), device=device)

    output = model(q, v)
    assert output.shape == (N, C, H, W)


@pytest.mark.parametrize("dtype", [torch.float32])
def test_deform_attn2d_forward_3d(dtype, device):
    N, C, H, W = 1, 128, 16, 16
    L = H * W
    groups = 4

    model = DeformAttn2d(dim=C, kernel_size=3, groups=groups).to(device).to(dtype)
    x = torch.randn((N, L, C), device=device, dtype=dtype)

    output = model(x, x, shape=(H, W))

    assert output.shape == (N, L, C)
    assert output.dtype == dtype


def test_deform_attn2d_stride(device):
    C = 64
    model = DeformAttn2d(dim=C, kernel_size=3, stride=2, padding=1).to(device)
    torch.randn((1, C, 16, 16), device=device)

    # With stride 2, a 16x16 input gives an 8x8 output, so q is 8x8.
    q = torch.randn((1, C, 8, 8), device=device)
    v = torch.randn((1, C, 16, 16), device=device)

    output = model(q, v)
    assert output.shape == (1, C, 8, 8)


# =========================================================================
# DeformSample2d (self-attention variant)
# =========================================================================


@pytest.mark.parametrize("drop_center", [True, False])
@pytest.mark.parametrize("center_gate", [True, False])
def test_deform_sample2d_variants(drop_center, center_gate, device):
    C = 64
    model = DeformSample2d(
        dim=C,
        kernel_size=3,
        drop_center=drop_center,
        center_gate=center_gate,
    ).to(device)

    x = torch.randn((1, C, 16, 16), device=device)
    output = model(x)
    assert output.shape == x.shape


def test_deform_sample2d_aggregate_softmax(device):
    C = 64
    model = DeformSample2d(
        dim=C,
        kernel_size=3,
        method="softmax",
    ).to(device)
    x = torch.randn((1, C, 16, 16), device=device)

    output = model(x)
    assert output.shape == x.shape


def test_deform_sample2d_no_projection(device):
    C = 64
    model = DeformSample2d(
        dim=C,
        kernel_size=3,
        project=None,
    ).to(device)
    x = torch.randn((1, C, 16, 16), device=device)

    output = model(x)
    assert output.shape == x.shape


def test_deform_sample2d_stride(device):
    C = 64
    model = DeformSample2d(dim=C, kernel_size=3, stride=2, padding=1).to(device)
    x = torch.randn((1, C, 16, 16), device=device)

    output = model(x)
    assert output.shape == (1, C, 8, 8)
