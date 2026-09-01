import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "CUDA required for query-tiled-autotuned tests", allow_module_level=True
    )

pytestmark = pytest.mark.gpu

from deformops.backends.triton.experimental.msda_query_tiled_autotuned import (
    forward as msda_query_tiled_autotuned,
)
from deformops.backends.triton.experimental.msda_query_tiled_cached import (
    forward as msda_query_tiled_cached,
)

DEVICE = torch.device("cuda")


@pytest.mark.gpu
def test_autotuned_softmax_backward_5d_vs_4d_attn():
    """Grads from msda_query_tiled_autotuned softmax must agree whether attn is (B,Q,H,L*P) or (B,Q,H,L,P)."""
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
        out = msda_query_tiled_autotuned(v, shapes, start_index, l, a, softmax=True)
        out.sum().backward()
        return v.grad, l.grad, a.grad.reshape(N, Q, M, L * P)

    g_v_5d, g_l_5d, g_a_5d = _run(attn_5d)
    g_v_4d, g_l_4d, g_a_4d = _run(attn_4d)
    torch.testing.assert_close(g_v_5d, g_v_4d, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(g_l_5d, g_l_4d, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(g_a_5d, g_a_4d, atol=1e-5, rtol=1e-5)


# Parity with the cached configuration, the validated baseline.

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

_ALL_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


@st.composite
def _msda_inputs(draw, dtypes=_ALL_DTYPES):
    N = draw(st.integers(min_value=1, max_value=2))
    M = draw(st.sampled_from([1, 2, 4, 8]))
    D = draw(st.sampled_from([16, 32, 64]))
    L = draw(st.integers(min_value=1, max_value=4))
    P = draw(st.sampled_from([4, 8]))
    Q = draw(st.integers(min_value=1, max_value=256))
    dtype = draw(st.sampled_from(dtypes))
    # Seeded by hypothesis, so a failing example replays and shrinks.
    generator = torch.Generator(device=DEVICE).manual_seed(
        draw(st.integers(min_value=0, max_value=2**31 - 1))
    )

    value_list, shapes, start_index = [], [], []
    offset = 0
    for _ in range(L):
        h = draw(st.integers(min_value=1, max_value=32))
        w = draw(st.integers(min_value=1, max_value=32))
        shapes.append([h, w])
        start_index.append(offset)
        value_list.append(
            torch.randn(N, h * w, M, D, device=DEVICE, dtype=dtype, generator=generator)
        )
        offset += h * w

    value = torch.cat(value_list, dim=1)
    shapes_t = torch.tensor(shapes, device=DEVICE, dtype=torch.long)
    start_t = torch.tensor(start_index, device=DEVICE, dtype=torch.long)
    locs = torch.rand(N, Q, M, L, P, 2, device=DEVICE, dtype=dtype, generator=generator)
    attn = torch.randn(N, Q, M, L, P, device=DEVICE, dtype=dtype, generator=generator)
    return value, shapes_t, start_t, locs, attn, dtype


@pytest.mark.gpu
@settings(
    deadline=None,
    suppress_health_check=[HealthCheck.data_too_large],
    max_examples=20,
)
@given(inputs=_msda_inputs())
def test_autotuned_forward_parity_vs_cached(inputs):
    """Autotuned forward must match the cached-kernel output within dtype tolerances."""
    value, shapes, start_index, locs, attn, _dtype = inputs

    with torch.no_grad():
        out_cached = msda_query_tiled_cached(
            value, shapes, start_index, locs, attn, softmax=True
        )
        out_autotuned = msda_query_tiled_autotuned(
            value, shapes, start_index, locs, attn, softmax=True
        )

    if _dtype in (torch.float16, torch.bfloat16):
        atol, rtol = 1e-2, 1e-2
    else:
        atol, rtol = 1e-5, 1e-5
    torch.testing.assert_close(out_autotuned, out_cached, atol=atol, rtol=rtol)


@pytest.mark.gpu
@settings(
    deadline=None,
    suppress_health_check=[HealthCheck.data_too_large],
    max_examples=20,
)
@given(inputs=_msda_inputs(dtypes=(torch.float32,)))
def test_autotuned_backward_parity_vs_cached(inputs):
    """Autotuned gradients match cached ones in float32; see test_msda_bias for half."""
    value, shapes, start_index, locs, attn, _dtype = inputs

    def _run(kern):
        v = value.clone().requires_grad_(True)
        l = locs.clone().requires_grad_(True)
        a = attn.clone().requires_grad_(True)
        out = kern(v, shapes, start_index, l, a, softmax=True)
        out.sum().backward()
        return v.grad, l.grad, a.grad

    gv8, gl8, ga8 = _run(msda_query_tiled_cached)
    gv9, gl9, ga9 = _run(msda_query_tiled_autotuned)

    torch.testing.assert_close(gv9, gv8, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(gl9, gl8, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(ga9, ga8, atol=1e-5, rtol=1e-5)
