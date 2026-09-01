"""Operand validation, dispatch and build behavior that a user can observe."""

from __future__ import annotations

import dataclasses
import threading

import pytest
import torch
from deformops.core import build as _build
from deformops.core.errors import KernelBuildError, NoBackendError
from deformops.nn import DeformAttn2d, DeformSample2d
from deformops.registry import Backend, attn, conv, msda
from deformops.registry.attn import AttnContext
from deformops.registry.attn import select_backend as select_attn_backend
from deformops.registry.msda import MSDAContext

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _msda_operands(batch=1, queries=5, heads=4, dim=8, points=4, device="cpu"):
    shapes = torch.tensor([[8, 8], [4, 4]], device=device)
    starts = torch.tensor([0, 64], device=device)
    value = torch.randn(batch, 80, heads, dim, device=device)
    loc = torch.rand(batch, queries, heads, 2, points, 2, device=device)
    weights = torch.rand(batch, queries, heads, 2 * points, device=device)
    return value, shapes, starts, loc, weights


# Call validation


def test_msda_rejects_a_misspelled_keyword():
    with pytest.raises(TypeError, match="mehtod"):
        msda(*_msda_operands(), mehtod="softmax")


def test_msda_rejects_a_wrong_rank_before_reading_extents():
    value, shapes, starts, loc, weights = _msda_operands()
    with pytest.raises(ValueError, match="value must be 4-dimensional"):
        msda(value[0], shapes, starts, loc, weights)


def test_attn_rejects_a_misspelled_keyword():
    value = torch.randn(1, 6, 6, 32)
    offset_mask = torch.zeros(1, 6, 6, 2 * 9 * 3)
    with pytest.raises(TypeError, match="kernal_size"):
        attn(value, offset_mask, groups=2, group_dims=16, kernal_size=3)


def test_attn_accepts_int_and_pair_geometry_alike():
    value = torch.randn(1, 6, 6, 32)
    offset_mask = torch.randn(1, 6, 6, 2 * 9 * 3) * 0.1
    as_int = attn(value, offset_mask, 3, 1, 1, 1, groups=2, group_dims=16)
    as_pair = attn(
        value,
        offset_mask,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        groups=2,
        group_dims=16,
    )
    torch.testing.assert_close(as_int, as_pair)


def test_conv_rejects_a_misspelled_keyword():
    value = torch.randn(1, 6, 6, 4)
    offset = torch.zeros(1, 6, 6, 1, 9, 2)
    weight = torch.randn(4, 4, 3, 3)
    with pytest.raises(TypeError):
        conv(value, offset, weight, padding=1, group=2)


def test_conv_rejects_out_channels_not_divisible_by_groups():
    value = torch.randn(1, 6, 6, 4)
    offset = torch.zeros(1, 6, 6, 1, 9, 2)
    weight = torch.randn(5, 2, 3, 3)
    with pytest.raises(ValueError, match="divisible"):
        conv(value, offset, weight, padding=1, groups=2)


# Layers


@pytest.mark.parametrize(
    "geometry",
    [{}, {"stride": 2}, {"dilation": 2}, {"stride": 2, "dilation": 2}, {"padding": 0}],
    ids=["default", "stride", "dilation", "stride-dilation", "valid"],
)
def test_center_gate_runs_for_every_admitted_geometry(geometry):
    layer = DeformSample2d(32, 3, groups=2, center_gate=True, **geometry)
    out = layer(torch.randn(2, 32, 13, 15))
    assert out.shape[:2] == (2, 32)


@pytest.mark.parametrize(
    ("kernel_size", "padding"), [(4, 1), (3, 2)], ids=["even-kernel", "wide-padding"]
)
def test_center_gate_rejects_geometries_without_a_center(kernel_size, padding):
    with pytest.raises(ValueError, match="center_gate"):
        DeformAttn2d(32, kernel_size, groups=2, padding=padding, center_gate=True)


def test_deform_sample_honors_dilation():
    layer = DeformSample2d(32, 3, groups=2, dilation=2)
    assert layer.offset.dilation == (2, 2)
    assert layer(torch.randn(1, 32, 11, 11)).shape == (1, 32, 11, 11)


# Dispatch


@pytest.mark.parametrize(
    ("sm", "training", "dtype", "fp32acc"),
    [
        ((8, 6), True, torch.bfloat16, True),
        ((8, 6), False, torch.bfloat16, False),
        ((8, 6), True, torch.float32, False),
        ((9, 0), True, torch.bfloat16, False),
    ],
)
def test_attn_policy_picks_the_accumulator_like_msda(sm, training, dtype, fp32acc):
    context = AttnContext(
        device=torch.device("cuda"), dtype=dtype, sm_version=sm, is_training=training
    )
    assert select_attn_backend(context) == [(Backend.TRITON, {"fp32acc": fp32acc})]


def test_msda_context_equality_ignores_the_query_count():
    base = MSDAContext(
        device=torch.device("cpu"),
        num_levels=2,
        num_heads=4,
        num_points=4,
        dtype=torch.float32,
        sm_version=(0, 0),
        is_training=False,
        num_queries=10,
        head_dim=8,
    )
    other = dataclasses.replace(base, num_queries=20)
    assert base == other
    assert hash(base) == hash(other)


@pytest.mark.parametrize(
    ("changes", "supported"),
    [
        ({}, True),
        ({"num_points": 3}, False),
        ({"num_levels": 6}, False),
        ({"head_dim": 12}, False),
        ({"num_heads": 64, "head_dim": 256}, False),
        ({"num_heads": 8, "head_dim": 512, "is_training": False}, True),
        ({"num_heads": 8, "head_dim": 512, "is_training": True}, False),
        ({"device": torch.device("cpu")}, False),
    ],
)
def test_cuda_capability_predicate_encodes_the_kernel_limits(changes, supported):
    pytest.importorskip("deformops.backends.cuda._msda_bridge")
    from deformops.backends.cuda._msda_bridge import supports_context

    context = MSDAContext(
        device=torch.device("cuda"),
        num_levels=4,
        num_heads=8,
        num_points=4,
        dtype=torch.float32,
        sm_version=(8, 6),
        is_training=True,
        num_queries=100,
        head_dim=32,
    )
    assert supports_context(dataclasses.replace(context, **changes)) is supported


def test_backend_loading_reports_a_broken_adapter(monkeypatch):
    from deformops.registry._dispatch import Registry

    registry = Registry("no_such_operator", context_of=lambda *_: None, select=None)
    with pytest.warns(RuntimeWarning, match="failed to import"):
        registry.available()
    assert "failed to import" in registry.unavailable_reason(Backend.EAGER)


def test_backend_loading_is_complete_before_any_thread_sees_it(monkeypatch):
    import time
    import types

    from deformops.registry import _dispatch

    registry = _dispatch.Registry("race_probe", context_of=lambda *_: None, select=None)

    def slow_import(name):
        # The delay opens the window where another thread could see an empty registry.
        time.sleep(0.05)
        if name.endswith(".eager.race_probe"):
            registry.register(Backend.EAGER, lambda *_: None)

    monkeypatch.setattr(
        _dispatch, "importlib", types.SimpleNamespace(import_module=slow_import)
    )
    seen: list[list[str]] = []
    threads = [
        threading.Thread(target=lambda: seen.append(registry.available()))
        for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert seen == [["eager"]] * 8


# Build


def test_missing_cuda_toolkit_raises_instead_of_building_an_empty_library(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("DEFORMOPS_COMPILE", "1")
    monkeypatch.setenv("DEFORMOPS_DOWNLOAD", "0")
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path))
    monkeypatch.setenv("CUDA_HOME", str(tmp_path / "missing"))
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setattr(_build, "_try_load_prebuilt", lambda _name: None)
    with pytest.raises(KernelBuildError, match="needs a CUDA toolkit"):
        _build.define_library("msda_fp32acc", with_cuda=True)


def test_an_unloadable_cached_library_is_discarded(tmp_path):
    lib_file = tmp_path / "msda_fp32acc.so"
    lib_file.write_bytes(b"truncated")
    with pytest.warns(RuntimeWarning, match="Discarding"):
        assert _build._load_cached_library("msda_fp32acc", lib_file) is False
    assert not lib_file.exists()


@pytest.mark.parametrize("value", ["", "  "])
def test_an_empty_opt_level_means_the_default(monkeypatch, value):
    monkeypatch.setenv("DEFORMOPS_OPT_LEVEL", value)
    assert _build._resolve_opt_level(2) == 2


@pytest.mark.parametrize("value", ["", "  "])
def test_an_empty_compile_variable_means_never(monkeypatch, value):
    monkeypatch.setenv("DEFORMOPS_COMPILE", value)
    assert _build.compile_mode() is _build.CompileMode.NEVER


# Triton kernels on a GPU


@pytest.mark.gpu
@requires_cuda
def test_triton_msda_splits_a_batch_past_the_launch_limit(monkeypatch):
    pytest.importorskip("triton")
    from deformops.backends.triton import _msda_bridge

    monkeypatch.setattr(_msda_bridge, "_MAX_GRID_BATCH", 3)
    value, shapes, starts, loc, weights = _msda_operands(batch=7, device="cuda")
    for tensor in (value, loc, weights):
        tensor.requires_grad_()
    out = msda(value, shapes, starts, loc, weights, backend=Backend.TRITON)
    out.square().sum().backward()
    got = (out.detach(), value.grad, loc.grad, weights.grad)

    reference = [t.detach().clone().requires_grad_() for t in (value, loc, weights)]
    ref_out = msda(
        reference[0], shapes, starts, reference[1], reference[2], backend=Backend.EAGER
    )
    ref_out.square().sum().backward()
    expected = (ref_out.detach(), *(t.grad for t in reference))
    for a, b in zip(got, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize("batch", [70, 96, 7])
def test_cuda_msda_splits_a_batch_into_launch_chunks(batch):
    """A batch larger than, or not divisible by, ``chunk_step`` matches eager."""
    value, shapes, starts, loc, weights = _msda_operands(batch=batch, device="cuda")
    grads = {}
    for backend in (Backend.CUDA, Backend.EAGER):
        operands = [t.detach().clone().requires_grad_() for t in (value, loc, weights)]
        out = msda(
            operands[0],
            shapes,
            starts,
            operands[1],
            operands[2],
            chunk_step=64,
            backend=backend,
        )
        out.square().sum().backward()
        grads[backend] = (out.detach(), *(t.grad for t in operands))
    for got, expected in zip(grads[Backend.CUDA], grads[Backend.EAGER], strict=True):
        torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.gpu
@requires_cuda
def test_unsupported_explicit_cuda_call_raises_no_backend_error():
    value, shapes, starts, loc, weights = _msda_operands(dim=12, device="cuda")
    with pytest.raises(NoBackendError, match="does not support"):
        msda(value, shapes, starts, loc, weights, backend=Backend.CUDA)


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize("backend", [Backend.TRITON, Backend.CUDA, Backend.EAGER])
@pytest.mark.parametrize("empty_axis", ["batch", "queries"])
def test_msda_empty_calls_return_zeros(backend, empty_axis):
    sizes = {"batch": 0} if empty_axis == "batch" else {"queries": 0}
    value, shapes, starts, loc, weights = _msda_operands(device="cuda", **sizes)
    out = msda(value, shapes, starts, loc, weights, backend=backend)
    assert out.shape == (loc.shape[0], loc.shape[1], value.shape[2] * value.shape[3])


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fp32acc", [False, True])
def test_triton_half_precision_location_gradient_matches_cuda(dtype, fp32acc):
    """Half-precision locations map to pixels in float32 on every backend.

    Mapping at half precision moves samples near a cell boundary into the next cell.
    CUDA widens first, so it is the reference; see ``docs/precision.md``.
    """
    pytest.importorskip("triton")
    generator = torch.Generator(device="cuda").manual_seed(0)
    shapes = torch.tensor([[31, 29], [13, 17]], device="cuda")
    starts = torch.tensor([0, 31 * 29], device="cuda")
    spatial = 31 * 29 + 13 * 17
    operands = (
        torch.randn(1, spatial, 4, 16, device="cuda", generator=generator),
        torch.rand(1, 64, 4, 2, 4, 2, device="cuda", generator=generator),
        torch.randn(1, 64, 4, 8, device="cuda", generator=generator),
    )
    grads = {}
    for name, options in (
        ("cuda", {"backend": Backend.CUDA}),
        ("triton", {"backend": Backend.TRITON, "fp32acc": fp32acc}),
    ):
        value, loc, weights = (t.to(dtype).requires_grad_() for t in operands)
        msda(value, shapes, starts, loc, weights, **options).float().sum().backward()
        grads[name] = loc.grad.float()
    # At the gradient's storage resolution; a sample in the wrong cell is off by the
    # gradient itself.
    torch.testing.assert_close(grads["triton"], grads["cuda"], rtol=1e-2, atol=5e-2)
