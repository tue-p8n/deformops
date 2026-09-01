import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA required for query-tiled tests", allow_module_level=True)

pytestmark = pytest.mark.gpu

from deformops.backends.cuda import msda_fp32acc as msda_cuda
from deformops.backends.triton.msda_query_tiled import forward as msda_query_tiled
from deformops.registry import Backend
from deformops.registry import msda as _msda_op


class _MSDAShim:
    @staticmethod
    def forward(*args, **kwargs):
        kwargs.setdefault("method", "linear")
        return _msda_op(*args, **kwargs)


msda = _MSDAShim()
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

DEVICE = torch.device("cuda")


_ALL_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


@st.composite
def msda_query_tiled_inputs(draw, dtypes=_ALL_DTYPES):
    # Random but valid configurations
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

    value_list = []
    shapes = []
    start_index = []
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
    shapes = torch.tensor(shapes, device=DEVICE, dtype=torch.long)
    start_index = torch.tensor(start_index, device=DEVICE, dtype=torch.long)

    locs = torch.rand(N, Q, M, L, P, 2, device=DEVICE, dtype=dtype, generator=generator)
    attn = torch.randn(N, Q, M, L, P, device=DEVICE, dtype=dtype, generator=generator)

    return (
        value,
        shapes,
        start_index,
        locs,
        attn,
        dtype,
    )


@settings(
    deadline=None,
    suppress_health_check=[HealthCheck.data_too_large],
    max_examples=20,
)
@given(inputs=msda_query_tiled_inputs())
def test_query_tiled_cuda_triton_parity(inputs):
    value, shapes, start_index, locs, attn, _dtype = inputs

    # Forward
    with torch.no_grad():
        out_cuda = msda.forward(
            value, shapes, start_index, locs, attn, backend=Backend.CUDA
        )
        out_triton = msda.forward(
            value,
            shapes,
            start_index,
            locs,
            attn,
            backend=Backend.TRITON,
            fp32acc=False,
        )

    # CUDA and Triton sum in different orders; half precision diverges on large calls.
    if _dtype == torch.float16:
        atol, rtol = 2e-1, 2e-1
    elif _dtype == torch.bfloat16:
        atol, rtol = 5e-3, 5e-3
    else:
        atol, rtol = 1e-4, 1e-4
    torch.testing.assert_close(out_cuda, out_triton, atol=atol, rtol=rtol)


@settings(
    deadline=None,
    suppress_health_check=[HealthCheck.data_too_large],
    max_examples=20,
)
# No bfloat16: its CUDA and Triton gradients diverge on SM 8.0 (see test_msda_bias).
@given(inputs=msda_query_tiled_inputs(dtypes=(torch.float32, torch.float16)))
def test_query_tiled_backward_parity(inputs):
    value, shapes, start_index, locs, attn, _dtype = inputs

    v_cu = value.clone().requires_grad_(True)
    l_cu = locs.clone().requires_grad_(True)
    a_cu = attn.clone().requires_grad_(True)

    v_tr = value.clone().requires_grad_(True)
    l_tr = locs.clone().requires_grad_(True)
    a_tr = attn.clone().requires_grad_(True)

    # Fwd + Bwd CUDA
    out_cuda = msda.forward(v_cu, shapes, start_index, l_cu, a_cu, backend=Backend.CUDA)
    out_cuda.sum().backward()

    # Fwd + Bwd Triton
    out_tr = msda.forward(
        v_tr,
        shapes,
        start_index,
        l_tr,
        a_tr,
        backend=Backend.TRITON,
        fp32acc=False,
    )
    out_tr.sum().backward()

    # Compare gradients
    atol = 5e-2 if _dtype == torch.float16 else 1e-3

    torch.testing.assert_close(v_cu.grad, v_tr.grad, atol=atol, rtol=atol)
    torch.testing.assert_close(l_cu.grad, l_tr.grad, atol=atol, rtol=atol)
    torch.testing.assert_close(a_cu.grad, a_tr.grad, atol=atol, rtol=atol)


@settings(
    deadline=None,
    suppress_health_check=[HealthCheck.data_too_large],
    max_examples=10,
)
@given(inputs=msda_query_tiled_inputs())
def test_query_tiled_softmax_aggregate_parity(inputs):
    value, shapes, start_index, locs, attn, _dtype = inputs

    # ``points`` picks the kernel template (default 8); a mismatch reads past locs.
    points = locs.shape[4]

    # The fused softmax spans the flattened (L, P) axis.
    attn_sm = attn.flatten(-2).softmax(dim=-1).view_as(attn)

    with torch.no_grad():
        # Manual Softmax (CUDA linear op)
        out_manual = msda_cuda.forward(
            value, shapes, start_index, locs, attn_sm, num_points=points, softmax=False
        )

        # Fused Softmax (CUDA softmax op): pass raw logits.
        out_fused = msda_cuda.forward(
            value, shapes, start_index, locs, attn, num_points=points, softmax=True
        )

        # Fused Softmax (Triton)
        out_triton = msda_query_tiled(
            value, shapes, start_index.to(torch.int32), locs, attn, softmax=True
        )

    if _dtype == torch.float16:
        atol, rtol = 5e-2, 5e-2
    elif _dtype == torch.bfloat16:
        atol, rtol = 1e-2, 1e-2
    else:
        atol, rtol = 1e-3, 1e-3
    torch.testing.assert_close(out_fused, out_manual, atol=atol, rtol=rtol)
    torch.testing.assert_close(
        out_triton, out_manual.view_as(out_triton), atol=atol, rtol=rtol
    )
