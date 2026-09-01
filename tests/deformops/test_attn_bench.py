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

# Fixed shapes for benchmarking
N, H, W, C = 4, 64, 64, 128
G, group_dims = 8, 16
KH, KW = 3, 3
stride, pad, dilation = 1, 1, 1
HO, WO = 64, 64
K = KH * KW


@pytest.fixture
def bench_inputs(device):
    def _create(dtype):
        value = torch.randn(
            (N, H, W, C), device=device, dtype=dtype, requires_grad=True
        )
        offset_mask = torch.randn(
            (N, HO, WO, G * K * 3), device=device, dtype=dtype, requires_grad=True
        )
        return value, offset_mask

    return _create


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_bench_attn_forward(benchmark, bench_inputs, dtype):
    value, offset_mask = bench_inputs(dtype)

    benchmark(
        attn_shim.forward,
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
        group_dims,
        1.0,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_bench_attn_backward(benchmark, bench_inputs, dtype):
    value, offset_mask = bench_inputs(dtype)
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
        group_dims,
        1.0,
    )
    grad_output = torch.randn_like(output)

    def run_backward():
        output.backward(grad_output, retain_graph=True)
        value.grad = None
        offset_mask.grad = None

    benchmark(run_backward)


def test_bench_attn_reference_cpu(benchmark):
    # Reference only on CPU, use smaller shape to keep it reasonable
    N_ref, H_ref, W_ref, C_ref = 1, 32, 32, 64
    G_ref, D_ref = 8, 8
    HO_ref, WO_ref = 32, 32

    value = torch.randn((N_ref, H_ref, W_ref, C_ref))
    offset = torch.randn((N_ref, HO_ref, WO_ref, G_ref, K, 2))
    mask = torch.randn((N_ref, HO_ref, WO_ref, G_ref, K))

    benchmark(
        attn_shim.reference,
        value,
        offset,
        mask,
        KH,
        KW,
        stride,
        stride,
        pad,
        pad,
        dilation,
        dilation,
        G_ref,
        D_ref,
        1.0,
    )
