import pytest
import torch
from deformops.backends.reference.attn import reference as attn_reference
from deformops.registry import attn as attn_op
from hypothesis import given, settings
from hypothesis import strategies as st


@pytest.fixture(scope="module")
def device(any_device):
    """The CPU always, CUDA when present.

    The eager backend and the reference both run on the CPU, so the parity test also
    runs in the CPU job.
    """
    return any_device


class _AttnShim:
    """Positional spelling of the single-scale registry call, paired with its
    reference, so the tests below read as one argument per geometry field."""

    @staticmethod
    def forward(
        value,
        offset_mask,
        kh,
        kw,
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
        g,
        gd,
        scale,
        drop_center=False,
    ):
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
            drop_center=drop_center,
        )

    reference = staticmethod(attn_reference)


attn_shim = _AttnShim()


@st.composite
def deform2d_inputs(draw):
    G = draw(st.sampled_from([8, 16]))
    C_per_group = draw(st.sampled_from([16, 32]))
    C = G * C_per_group

    # Kernel size: Fixed to 3x3 (K=9, or K=8 with the center point excluded)
    KH, KW = 3, 3
    drop_center = draw(st.booleans())
    K = KH * KW - int(drop_center)

    stride = draw(st.sampled_from([1, 2]))
    pad = draw(st.sampled_from([0, 1, 2]))
    dilation = draw(st.sampled_from([1, 2]))

    # N * HO * WO must be a multiple of block_multiplier, at most 64 here.
    # HO * WO = 64 is a multiple of every such value.
    HO, WO = 8, 8
    N = draw(st.sampled_from([1, 2]))

    # Calculate input H, W based on HO, WO
    H = (HO - 1) * stride + (dilation * (KH - 1) + 1) - 2 * pad
    W = (WO - 1) * stride + (dilation * (KW - 1) + 1) - 2 * pad

    # Keep H and W at least 8.
    H = max(H, 8)
    W = max(W, 8)

    # Precise HO, WO
    HO = (H + 2 * pad - (dilation * (KH - 1) + 1)) // stride + 1
    WO = (W + 2 * pad - (dilation * (KW - 1) + 1)) // stride + 1

    # Tensor values come from a generator seeded by hypothesis, so a failing
    # example replays and shrinks with the exact operands that failed.
    generator = torch.Generator().manual_seed(
        draw(st.integers(min_value=0, max_value=2**31 - 1))
    )
    value = torch.randn((N, H, W, C), generator=generator)
    # offset_mask: [N, HO, WO, G * K * 3] (K accounts for drop_center)
    offset_mask = torch.randn((N, HO, WO, G * K * 3), generator=generator)

    return {
        "value": value,
        "offset_mask": offset_mask,
        "kernel_h": KH,
        "kernel_w": KW,
        "drop_center": drop_center,
        "stride_h": stride,
        "stride_w": stride,
        "pad_h": pad,
        "pad_w": pad,
        "dilation_h": dilation,
        "dilation_w": dilation,
        "groups": G,
        "group_dims": C_per_group,
        # No 0.0: every point would sample the center and the offsets drop out;
        # test_attn_triton covers that case.
        "spread": draw(st.sampled_from([0.5, 1.0, 2.0])),
    }


_CUDA = (
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available"),
)


# Half precision only on CUDA: the eager backend accumulates in the input dtype and
# exceeds these float32-accumulator tolerances by design.
@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        pytest.param(torch.device("cpu"), torch.float32, id="cpu-float32"),
        *(
            pytest.param(
                torch.device("cuda"), dtype, id=f"cuda-{str(dtype)[6:]}", marks=_CUDA
            )
            for dtype in (torch.float32, torch.float16, torch.bfloat16)
        ),
    ],
)
@given(inputs=deform2d_inputs())
@settings(deadline=None, max_examples=10)
def test_attn_parity(dtype, inputs, device):
    # Convert inputs to target device and dtype
    value = inputs["value"].to(device).to(dtype)
    offset_mask = inputs["offset_mask"].to(device).to(dtype)

    # Split offset_mask for reference implementation
    N, HO, WO, _ = offset_mask.shape
    G = inputs["groups"]
    drop_center = inputs["drop_center"]
    K = inputs["kernel_h"] * inputs["kernel_w"] - int(drop_center)

    # (x, y, modulation) per sampling point: the kernels interleave the offset
    # pair and the weight, so the split for the reference does too.
    triples = offset_mask.view(N, HO, WO, G, K, 3)
    offset = triples[..., :2].reshape(N, HO, WO, G, K * 2)
    mask = triples[..., 2].reshape(N, HO, WO, G, K)

    # 1. Forward (registry dispatch)
    actual = attn_shim.forward(
        value,
        offset_mask,
        inputs["kernel_h"],
        inputs["kernel_w"],
        inputs["stride_h"],
        inputs["stride_w"],
        inputs["pad_h"],
        inputs["pad_w"],
        inputs["dilation_h"],
        inputs["dilation_w"],
        inputs["groups"],
        inputs["group_dims"],
        inputs["spread"],
        drop_center,
    )

    # 2. Reference (CPU)
    expected = attn_shim.reference(
        value.float().cpu(),
        offset.float().cpu(),
        mask.float().cpu(),
        inputs["kernel_h"],
        inputs["kernel_w"],
        inputs["stride_h"],
        inputs["stride_w"],
        inputs["pad_h"],
        inputs["pad_w"],
        inputs["dilation_h"],
        inputs["dilation_w"],
        inputs["groups"],
        inputs["group_dims"],
        inputs["spread"],
        drop_center=drop_center,
    )

    # Compare
    atol, rtol = 1e-3, 1e-3
    if dtype == torch.float16:
        atol, rtol = 1e-2, 1e-2
    elif dtype == torch.bfloat16:
        atol, rtol = 5e-2, 5e-2

    torch.testing.assert_close(actual.float().cpu(), expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32])
def test_attn_backward(dtype, device):
    N, H, W, C = 1, 16, 16, 128
    G, C_per_group = 8, 16
    KH, KW = 3, 3
    stride, pad, dilation = 1, 1, 1
    HO, WO = 16, 16

    value = torch.randn((N, H, W, C), device=device, dtype=dtype, requires_grad=True)
    offset_mask = torch.randn(
        (N, HO, WO, G * KH * KW * 3), device=device, dtype=dtype, requires_grad=True
    )

    output = attn_shim.forward(
        value,
        offset_mask,
        KH,
        KW,
        stride,
        stride,
        pad,
        pad,
        dilation,
        dilation,
        G,
        C_per_group,
        1.0,
    )

    output.sum().backward()

    assert value.grad is not None
    assert offset_mask.grad is not None
    assert not torch.all(value.grad == 0)
    assert not torch.all(offset_mask.grad == 0)


def test_attn_out_of_bounds(device):
    N, H, W, C = 1, 16, 16, 128
    G, C_per_group = 8, 16
    KH, KW = 3, 3
    stride, pad, dilation = 1, 1, 1
    HO, WO = 16, 16

    value = torch.ones((N, H, W, C), device=device)
    offset_mask = torch.zeros((N, HO, WO, G * KH * KW * 3), device=device)
    triples = offset_mask.view(N, HO, WO, G, KH * KW, 3)
    triples[..., :2] = 100.0  # push every sample far outside the input
    triples[..., 2] = 1.0

    output = attn_shim.forward(
        value,
        offset_mask,
        KH,
        KW,
        stride,
        stride,
        pad,
        pad,
        dilation,
        dilation,
        G,
        C_per_group,
        1.0,
    )

    assert torch.all(output == 0)
