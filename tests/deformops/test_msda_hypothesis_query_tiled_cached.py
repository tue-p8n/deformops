import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA required for query-tiled-cached tests", allow_module_level=True)

pytestmark = pytest.mark.gpu

from deformops.backends.triton.experimental.msda_query_tiled_cached import (
    forward as msda_query_tiled_cached,
)

DEVICE = torch.device("cuda")


@pytest.mark.gpu
def test_query_tiled_cached_softmax_backward_5d_vs_4d_attn():
    """Softmax gradients agree for ``(B, Q, H, L*P)`` and ``(B, Q, H, L, P)`` attn.

    The bridge reduces over all ``L*P`` points.
    """
    N, M, D, L, P, Q = 1, 2, 16, 2, 4, 8
    dtype = torch.float32
    value = torch.randn(N, 32, M, D, device=DEVICE, dtype=dtype)
    shapes = torch.tensor([[4, 4], [4, 4]], device=DEVICE, dtype=torch.int32)
    start_index = torch.tensor([0, 16], device=DEVICE, dtype=torch.int32)
    locs = torch.rand(N, Q, M, L, P, 2, device=DEVICE, dtype=dtype) * 0.6 + 0.2
    attn_5d = torch.randn(N, Q, M, L, P, device=DEVICE, dtype=dtype)
    attn_4d = attn_5d.reshape(N, Q, M, L * P)

    def _run(attn_input):
        v = value.clone().requires_grad_(True)
        l = locs.clone().requires_grad_(True)
        a = attn_input.clone().requires_grad_(True)
        out = msda_query_tiled_cached(v, shapes, start_index, l, a, softmax=True)
        out.sum().backward()
        return v.grad, l.grad, a.grad.reshape(N, Q, M, L * P)

    g_v_5d, g_l_5d, g_a_5d = _run(attn_5d)
    g_v_4d, g_l_4d, g_a_4d = _run(attn_4d)
    torch.testing.assert_close(g_v_5d, g_v_4d, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(g_l_5d, g_l_4d, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(g_a_5d, g_a_4d, atol=1e-5, rtol=1e-5)
