import pytest
import torch
from deformops.nn import DeformSample2d

# Fixed shapes for benchmarking
N, C, H, W = 4, 128, 64, 64
groups = 4


@pytest.fixture
def bench_inputs(device):
    def _create(dtype):
        x = torch.randn((N, C, H, W), device=device, dtype=dtype, requires_grad=True)
        model = DeformSample2d(dim=C, kernel_size=3, groups=groups).to(device).to(dtype)
        return model, x

    return _create


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_bench_attn_module_forward(benchmark, bench_inputs, dtype):
    model, x = bench_inputs(dtype)
    benchmark(model, x)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_bench_attn_module_backward(benchmark, bench_inputs, dtype):
    model, x = bench_inputs(dtype)
    output = model(x)
    grad_output = torch.randn_like(output)

    def run_backward():
        output.backward(grad_output, retain_graph=True)
        for p in model.parameters():
            p.grad = None
        x.grad = None

    benchmark(run_backward)
