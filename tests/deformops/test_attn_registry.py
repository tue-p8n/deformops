from __future__ import annotations

import pytest
import torch
from deformops.registry import Backend, attn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def test_attn_registry_eager_registered():
    """Eager backend is registered in deformops.registry.attn."""
    assert attn.is_registered(Backend.EAGER)
    assert attn.get(Backend.EAGER) is not None


def test_attn_eager_execution():
    """attn dispatches to EAGER backend when explicitly requested."""
    N, H, W, C = 1, 4, 4, 8
    G = 1

    value = torch.randn(N, H, W, C, device="cpu")
    offset_mask = torch.randn(N, H, W, 32, device="cpu")

    out = attn(
        value,
        offset_mask,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        groups=G,
        group_dims=C // G,
        spread=1.0,
        softmax=False,
        backend=Backend.EAGER,
    )
    assert out is not None
    assert out.device.type == "cpu"
    assert out.shape == (N, H, W, C)


def test_attn_raw_string_backend_rejected():
    """Raw strings for backend parameter raise TypeError in Registry."""
    N, H, W, C = 1, 4, 4, 8
    value = torch.randn(N, H, W, C, device="cpu")
    offset_mask = torch.randn(N, H, W, 32, device="cpu")

    with pytest.raises(TypeError, match="Backend StrEnum"):
        attn(
            value,
            offset_mask,
            backend="eager",  # type: ignore[arg-type]
        )


@pytest.mark.gpu
@requires_cuda
def test_attn_cuda_or_triton_execution():
    """Dispatches on GPU host if CUDA or Triton backends are registered."""
    N, H, W, C = 1, 8, 8, 8
    G = 1

    value = torch.randn(N, H, W, C, device=DEVICE)
    offset_mask = torch.randn(N, H, W, 32, device=DEVICE)

    out = attn(
        value,
        offset_mask,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        groups=G,
        group_dims=C // G,
        spread=1.0,
        softmax=False,
    )
    assert out is not None
    assert out.shape == (N, H, W, C)
