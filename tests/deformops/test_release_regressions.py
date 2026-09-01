"""Operator and integration behavior callers rely on.

Method defaults, strides, point counts, boundary gradients and module patching.
"""

from __future__ import annotations

import pytest
import torch
from deformops.core.runtime import Method
from deformops.integrations._patch import patch_modules
from deformops.registry import Backend, msda
from deformops.registry.msda import resolve_method
from torch import nn

_CUDA_MARKS = (
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available"),
)

# Method resolution


@pytest.mark.parametrize("softmax", [False, True])
@pytest.mark.parametrize("method", [Method.LINEAR, Method.SOFTMAX, "linear", "softmax"])
def test_resolve_method_explicit_method_wins(method, softmax: bool) -> None:
    assert resolve_method(method, softmax=softmax) is Method(method)


def test_resolve_method_softmax_flag_selects_softmax() -> None:
    assert resolve_method(None, softmax=True) is Method.SOFTMAX


def test_resolve_method_defaults_to_linear() -> None:
    assert resolve_method(None) is Method.LINEAR
    assert resolve_method(None, softmax=False) is Method.LINEAR


def test_resolve_method_rejects_an_unknown_name() -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        resolve_method("not-a-method")


# MSDA operands

_BATCH, _QUERIES, _HEADS, _CHANNELS = 2, 10, 2, 16
_LEVEL_SHAPES = ((8, 8), (4, 4))


def _level_tables(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    shapes = torch.tensor(_LEVEL_SHAPES, dtype=torch.long, device=device)
    sizes = [h * w for h, w in _LEVEL_SHAPES]
    starts = torch.tensor([0, *sizes[:-1]], dtype=torch.long, device=device).cumsum(0)
    return shapes, starts


def _operands(
    device: torch.device, *, points: int = 4, seed: int = 0
) -> dict[str, torch.Tensor]:
    """Contiguous MSDA operands with every location inside the feature map."""
    generator = torch.Generator().manual_seed(seed)
    levels = len(_LEVEL_SHAPES)
    spatial = sum(h * w for h, w in _LEVEL_SHAPES)
    shapes, starts = _level_tables(device)
    value = torch.randn(_BATCH, spatial, _HEADS, _CHANNELS, generator=generator)
    # Kept inside [0.1, 0.9] so no sample sits on the border clamp.
    locs = torch.rand(_BATCH, _QUERIES, _HEADS, levels, points, 2, generator=generator)
    locs = locs * 0.8 + 0.1
    attn = torch.randn(_BATCH, _QUERIES, _HEADS, levels * points, generator=generator)
    return {
        "value": value.to(device),
        "shapes": shapes,
        "starts": starts,
        "locs": locs.to(device),
        "attn": attn.to(device),
    }


def _call(ops: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
    return msda(
        ops["value"], ops["shapes"], ops["starts"], ops["locs"], ops["attn"], **kwargs
    )


# Functional default method


def test_functional_default_is_linear_on_eager() -> None:
    ops = _operands(torch.device("cpu"))
    default = _call(ops, backend=Backend.EAGER)
    linear = _call(ops, backend=Backend.EAGER, method="linear")
    softmax = _call(ops, backend=Backend.EAGER, method="softmax")
    torch.testing.assert_close(default, linear, rtol=0, atol=0)
    assert not torch.allclose(default, softmax), (
        "the operands do not tell linear from softmax, so the default is untested"
    )


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(Backend.TRITON, id="triton", marks=_CUDA_MARKS),
        pytest.param(Backend.CUDA, id="cuda", marks=_CUDA_MARKS),
    ],
)
def test_functional_default_matches_eager_on_device_backends(backend: Backend) -> None:
    """No method means the same thing on every backend, and so does softmax."""
    ops = _operands(torch.device("cuda"))
    for method in (None, "softmax"):
        options = {} if method is None else {"method": method}
        expected = _call(ops, backend=Backend.EAGER, **options)
        actual = _call(ops, backend=backend, **options)
        torch.testing.assert_close(
            actual,
            expected,
            rtol=1e-4,
            atol=1e-5,
            msg=lambda m, method=method: f"method={method!r}: {m}",
        )


# Non-contiguous operands


def _forward_and_grads(
    value_leaf: torch.Tensor,
    value: torch.Tensor,
    locs_leaf: torch.Tensor,
    locs: torch.Tensor,
    attn_leaf: torch.Tensor,
    attn: torch.Tensor,
    shapes: torch.Tensor,
    starts: torch.Tensor,
    backend: Backend,
    method: str,
) -> tuple[torch.Tensor, ...]:
    output = msda(value, shapes, starts, locs, attn, backend=backend, method=method)
    # A non-uniform upstream gradient exposes a stride mix-up in the backward.
    weights = torch.linspace(-1.0, 1.0, output.numel(), device=output.device)
    (output * weights.view_as(output)).sum().backward()
    return output.detach(), value_leaf.grad, locs_leaf.grad, attn_leaf.grad


@pytest.mark.parametrize(
    ("backend", "device"),
    [
        pytest.param(Backend.EAGER, "cpu", id="eager-cpu"),
        pytest.param(Backend.EAGER, "cuda", id="eager-cuda", marks=_CUDA_MARKS),
        pytest.param(Backend.TRITON, "cuda", id="triton", marks=_CUDA_MARKS),
        pytest.param(Backend.CUDA, "cuda", id="cuda", marks=_CUDA_MARKS),
    ],
)
@pytest.mark.parametrize("method", ["linear", "softmax"])
def test_non_contiguous_operands_match_contiguous_copies(
    backend: Backend, device: str, method: str
) -> None:
    """Sliced locations and weights and a transposed value tensor are ordinary input.

    Their strides break a kernel that assumes a packed layout.
    """
    ops = _operands(torch.device(device))
    shapes, starts = ops["shapes"], ops["starts"]

    # Non-contiguous: the operator receives views; the gradients reach the leaves.
    value_leaf = ops["value"].transpose(1, 2).contiguous().requires_grad_(True)
    value_view = value_leaf.transpose(1, 2)
    packed_leaf = torch.cat([ops["locs"], torch.rand_like(ops["locs"])], dim=-1)
    packed_leaf.requires_grad_(True)
    locs_view = packed_leaf[..., :2]
    attn_packed = torch.cat([ops["attn"], torch.rand_like(ops["attn"])], dim=-1)
    attn_packed.requires_grad_(True)
    attn_view = attn_packed[..., : ops["attn"].shape[-1]]
    assert not value_view.is_contiguous()
    assert not locs_view.is_contiguous()
    assert not attn_view.is_contiguous()

    out_view, g_value_leaf, g_packed, g_attn_packed = _forward_and_grads(
        value_leaf,
        value_view,
        packed_leaf,
        locs_view,
        attn_packed,
        attn_view,
        shapes,
        starts,
        backend,
        method,
    )
    g_attn_view = g_attn_packed[..., : ops["attn"].shape[-1]]

    # Contiguous copies of the same values.
    value = ops["value"].clone().requires_grad_(True)
    locs = ops["locs"].clone().requires_grad_(True)
    attn_copy = ops["attn"].clone().requires_grad_(True)
    out, g_value, g_locs, g_attn = _forward_and_grads(
        value, value, locs, locs, attn_copy, attn_copy, shapes, starts, backend, method
    )

    close = {"rtol": 1e-5, "atol": 1e-6}
    torch.testing.assert_close(out_view, out, **close)
    torch.testing.assert_close(g_value_leaf.transpose(1, 2), g_value, **close)
    torch.testing.assert_close(g_packed[..., :2], g_locs, **close)
    torch.testing.assert_close(
        g_packed[..., 2:], torch.zeros_like(g_packed[..., 2:]), rtol=0, atol=0
    )
    torch.testing.assert_close(g_attn_view, g_attn, **close)
    unused = g_attn_packed[..., ops["attn"].shape[-1] :]
    torch.testing.assert_close(unused, torch.zeros_like(unused), rtol=0, atol=0)


# Point count on the CUDA kernel


@pytest.mark.parametrize("method", ["linear", "softmax"])
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_eight_point_msda_on_cuda_matches_eager(method: str) -> None:
    """The CUDA backend runs the eight-point template for eight-point operands."""
    ops = _operands(torch.device("cuda"), points=8, seed=3)

    results = {}
    for backend in (Backend.EAGER, Backend.CUDA):
        leaves = {
            name: ops[name].clone().requires_grad_(True)
            for name in ("value", "locs", "attn")
        }
        output = msda(
            leaves["value"],
            ops["shapes"],
            ops["starts"],
            leaves["locs"],
            leaves["attn"],
            backend=backend,
            method=method,
        )
        output.sum().backward()
        results[backend] = (
            output.detach(),
            *(leaves[name].grad for name in ("value", "locs", "attn")),
        )

    for label, actual, expected in zip(
        ("output", "value gradient", "location gradient", "attention gradient"),
        results[Backend.CUDA],
        results[Backend.EAGER],
        strict=True,
    ):
        torch.testing.assert_close(
            actual,
            expected,
            rtol=1e-4,
            atol=1e-5,
            msg=lambda m, label=label: f"{label} differs: {m}",
        )


# Samples on pixel boundaries


@pytest.mark.parametrize("backend", [Backend.TRITON, Backend.CUDA], ids=str)
@pytest.mark.parametrize("method", ["linear", "softmax"])
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_grid_aligned_locations_match_eager(method: str, backend: Backend) -> None:
    """Samples at integer pixels, including -1 and the far edge, match eager's gradient.

    Grid-initialized detection models sample there on their first step.
    """
    generator = torch.Generator().manual_seed(5)
    levels = ((8, 8), (4, 4), (2, 2))
    shapes = torch.tensor(levels)
    areas = shapes.prod(1)
    starts = torch.cat([areas.new_zeros(1), areas.cumsum(0)[:-1]])
    batch, queries, heads, channels, points = 2, 11, 4, 8, 4
    value = torch.randn(batch, int(areas.sum()), heads, channels, generator=generator)
    # Pixel coordinate p maps to the location (p + 0.5) / size.
    pixels = torch.randint(
        -2, 10, (batch, queries, heads, len(levels), points, 2), generator=generator
    ).float()
    locs = (pixels + 0.5) / shapes.flip(1).view(1, 1, 1, -1, 1, 2)
    attn = torch.randn(batch, queries, heads, len(levels) * points, generator=generator)

    results = {}
    for run in (Backend.EAGER, backend):
        leaves = [t.cuda().requires_grad_(True) for t in (value, locs, attn)]
        output = msda(
            leaves[0],
            shapes.cuda(),
            starts.cuda(),
            leaves[1],
            leaves[2],
            backend=run,
            method=method,
        )
        upstream = torch.linspace(-1, 1, output.numel(), device="cuda")
        (output * upstream.view_as(output)).sum().backward()
        results[run] = (output.detach(), *(leaf.grad for leaf in leaves))

    for label, actual, expected in zip(
        ("output", "value gradient", "location gradient", "attention gradient"),
        results[backend],
        results[Backend.EAGER],
        strict=True,
    ):
        torch.testing.assert_close(
            actual,
            expected,
            rtol=1e-4,
            atol=1e-5,
            msg=lambda m, label=label: f"{label} differs: {m}",
        )


# Module patching


class _Leaf(nn.Module):
    """An upstream class nested inside another upstream class."""


class _Outer(nn.Module):
    """An upstream class whose replacement keeps its children."""

    def __init__(self) -> None:
        super().__init__()
        self.inner = _Leaf()


class _OuterReplacement(nn.Module):
    def __init__(self, original: _Outer) -> None:
        super().__init__()
        self.inner = original.inner


class _LeafReplacement(nn.Module):
    pass


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.Linear(2, 2), _Outer())
        self.head = nn.Linear(2, 2)


def test_patch_modules_raises_when_nothing_matches() -> None:
    model = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
    with pytest.raises(RuntimeError):
        patch_modules(model, [(_Outer, _OuterReplacement)])


def test_patch_modules_non_strict_returns_the_model_unchanged() -> None:
    model = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
    before = [type(m) for m in model.modules()]
    assert patch_modules(model, [(_Outer, _OuterReplacement)], strict=False) is model
    assert [type(m) for m in model.modules()] == before


def test_patch_modules_visits_the_children_of_a_replacement() -> None:
    """A class nested inside a replaced class is replaced too."""
    model = _Model()
    patch_modules(
        model, [(_Outer, _OuterReplacement), (_Leaf, lambda _: _LeafReplacement())]
    )

    kinds = {type(m) for m in model.modules()}
    assert _Outer not in kinds
    assert _Leaf not in kinds, (
        "the module nested inside the replacement was not visited"
    )
    assert isinstance(model.block[1], _OuterReplacement)
    assert isinstance(model.block[1].inner, _LeafReplacement)


def _transformers_integration():
    pytest.importorskip("transformers")
    # A torchvision built against another torch raises ImportError on the model.
    try:
        from transformers import DeformableDetrForObjectDetection  # noqa: F401, PLC0415
    except ImportError as exc:
        pytest.skip(f"transformers models unavailable: {exc}")
    from deformops.integrations import transformers  # noqa: PLC0415

    return transformers


def _mmcv_integration():
    pytest.importorskip("mmcv")
    from deformops.integrations import mmcv  # noqa: PLC0415

    return mmcv


@pytest.mark.parametrize(
    "load", [_transformers_integration, _mmcv_integration], ids=["transformers", "mmcv"]
)
def test_integration_patch_is_strict_by_default(load) -> None:
    """The public ``patch`` functions forward ``strict`` to the traversal."""
    module = load()
    model = nn.Sequential(nn.Linear(2, 2))
    with pytest.raises(RuntimeError):
        module.patch(model)
    assert module.patch(model, strict=False) is model
