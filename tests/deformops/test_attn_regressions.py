import pytest
import torch
from deformops.backends.reference.attn import reference as attn_reference
from deformops.registry import attn as attn_op


class _AttnShim:
    @staticmethod
    def forward(value, offset_mask, kh, kw, sh, sw, ph, pw, dh, dw, g, gd, scale):
        return attn_op(
            value,
            offset_mask,
            kernel_size=(kh, kw),
            stride=(sh, sw),
            padding=(ph, pw),
            dilation=(dh, dw),
            groups=g,
            group_dims=gd,
            spread=scale,
        )

    reference = staticmethod(attn_reference)


attn_shim = _AttnShim()

if not torch.cuda.is_available():
    pytest.skip("CUDA is not available", allow_module_level=True)

pytestmark = pytest.mark.gpu


def test_attn_large_padding_parity():
    """With padding larger than the kernel radius, CUDA matches the reference."""
    device = torch.device("cuda")
    dtype = torch.float32

    KH, KW = 3, 3
    stride_h, stride_w = 1, 1
    pad_h, pad_w = 2, 2
    dilation_h, dilation_w = 1, 1
    group = 16
    group_dims = 32
    spread = 0.5

    # Dimensions
    HO, WO = 10, 10
    N = 2
    C = group * group_dims

    # Input size H=8, W=8
    H, W = 8, 8

    torch.manual_seed(42)
    value = torch.randn((N, H, W, C), device=device, dtype=dtype)
    offset_mask = torch.randn(
        (N, HO, WO, group * KH * KW * 3), device=device, dtype=dtype
    )

    K = KH * KW
    # Interleaved (x, y, modulation) per point, matching the kernels.
    triples = offset_mask.view(N, HO, WO, group, K, 3)
    offset = triples[..., :2].reshape(N, HO, WO, group, K * 2)
    mask = triples[..., 2].reshape(N, HO, WO, group, K)

    # 1. Forward (CUDA)
    actual = attn_shim.forward(
        value,
        offset_mask,
        KH,
        KW,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        group,
        group_dims,
        spread,
    )

    # 2. Reference (CPU)
    expected = attn_shim.reference(
        value.float().cpu(),
        offset.float().cpu(),
        mask.float().cpu(),
        KH,
        KW,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        group,
        group_dims,
        spread,
    )

    # Large padding puts samples on the input boundary, where the kernel's
    # bounds check has to agree with the reference.
    torch.testing.assert_close(actual.float().cpu(), expected, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    test_attn_large_padding_parity()
