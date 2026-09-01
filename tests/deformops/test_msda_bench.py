"""MSDA module benchmarks: the device kernel, and the eager reference beside it.

Every device benchmark takes ``device`` and skips without CUDA. The reference
benchmarks run on the CPU as the baseline for the device kernel.
"""

import pytest
import torch
from deformops.nn import MultiScaleDeformAttn2d

B, Q, S, DIM = 2, 256, 4096, 256
NUM_HEADS, NUM_LEVELS, NUM_POINTS = 8, 4, 4


def _module_and_operands(target: torch.device):
    m = MultiScaleDeformAttn2d(
        dim=DIM, num_heads=NUM_HEADS, num_levels=NUM_LEVELS, num_points=NUM_POINTS
    ).to(target)
    q = torch.randn(B, Q, DIM, device=target)
    p = torch.rand(B, Q, NUM_LEVELS, 2, device=target)
    v = torch.randn(B, S, DIM, device=target)
    shapes = torch.tensor([[32, 32]] * NUM_LEVELS, dtype=torch.long, device=target)
    level_index = torch.arange(NUM_LEVELS, device=target) * 1024
    return m, (q, p, v, shapes, level_index)


def test_msda_benchmark(benchmark, device):
    m, operands = _module_and_operands(device)

    for _ in range(5):
        _ = m(*operands)

    benchmark(m, *operands)


@pytest.mark.usefixtures("device")
def test_msda_reference_benchmark(benchmark):
    # CPU operands route dispatch to the eager backend.
    m, operands = _module_and_operands(torch.device("cpu"))

    for _ in range(2):
        _ = m(*operands)

    benchmark(m, *operands)


@pytest.mark.usefixtures("device")
def test_msda_reference_compiled_benchmark(benchmark):
    m, operands = _module_and_operands(torch.device("cpu"))
    m_compiled = torch.compile(m, backend="eager")

    for _ in range(3):
        _ = m_compiled(*operands)

    benchmark(m_compiled, *operands)
