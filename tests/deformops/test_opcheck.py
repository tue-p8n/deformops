r"""``torch.library.opcheck`` for every registered ``deformops`` operator.

``SAMPLE_INPUTS`` holds one builder per operator, and a completeness test fails by
name for any registered operator without one. Each opcheck subtest is its own test,
so one failure never hides another. Needs CUDA, and ``triton`` for the Triton ops.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA is not available", allow_module_level=True)

pytestmark = pytest.mark.gpu

# Imported for its side effect: it defines the public ``deformops::`` operators.
import deformops.registry  # noqa: E402, F401
from deformops.backends.cuda import msda_fp32acc  # noqa: E402
from deformops.backends.cuda.experimental import msda_half2atomic  # noqa: E402
from deformops.core.capabilities import has_triton  # noqa: E402
from deformops.core.tuning import (  # noqa: E402
    backward_thread_layout,
    forward_thread_layout,
)

#: Triton MSDA configurations; their operators share arguments, so builders are shared.
_TRITON_MSDA_CONFIGURATIONS = (
    "query_tiled",
    "query_tiled_fp32acc",
    "query_tiled_cached",
    "query_tiled_cached_fp32acc",
    "query_tiled_autotuned",
    "query_tiled_autotuned_fp32acc",
    "point_parallel",
    "atomic_wide",
    "block_reduce",
    "block_reduce_wide",
    "warp_reduce",
    "warp_reduce_wide",
    "sort_segscan",
    "sort_segscan_wide",
)

_HAS_TRITON = has_triton()

if _HAS_TRITON:
    # Imported to register their ``deformops_triton`` operators.
    import importlib

    for _module in (
        "attn_query_tiled",
        "attn_query_tiled_fp32acc",
        "conv_deform2d",
        *(
            f"msda_{name}"
            if name in ("query_tiled", "query_tiled_fp32acc")
            else f"experimental.msda_{name}"
            for name in _TRITON_MSDA_CONFIGURATIONS
        ),
    ):
        importlib.import_module(f"deformops.backends.triton.{_module}")

#: The ``(args, kwargs)`` a sample builder returns for one operator call.
OpArgs = tuple[tuple[Any, ...], dict[str, Any]]

# Minimal shapes: the machinery is under test, and several ops compile on first call.

_BATCH = 1
_HEIGHT = _WIDTH = 4
_SPATIAL = _HEIGHT * _WIDTH
_HEADS = 2
_DIM = 8
_LEVELS = 1
_POINTS = 4  # K; the production/variants CUDA kernels only template K in {4, 8}.
_QUERIES = 4
_IM2COL_STEP = 64


def _msda_tensors(
    *, shape_dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """A consistent ``(shapes, level_start_index, value, loc, attn)`` tuple.

    Shape tensors are ``int64`` for the CUDA operators and ``int32`` for Triton.
    """
    shapes = torch.tensor([[_HEIGHT, _WIDTH]], dtype=shape_dtype, device="cuda")
    level_start_index = torch.tensor([0], dtype=shape_dtype, device="cuda")
    value = torch.randn(
        _BATCH, _SPATIAL, _HEADS, _DIM, device="cuda", requires_grad=True
    )
    loc = (
        torch.rand(_BATCH, _QUERIES, _HEADS, _LEVELS, _POINTS, 2, device="cuda") * 0.5
        + 0.25
    ).requires_grad_(True)
    attn = torch.randn(
        _BATCH, _QUERIES, _HEADS, _LEVELS * _POINTS, device="cuda", requires_grad=True
    )
    return shapes, level_start_index, value, loc, attn


def _grad_output() -> torch.Tensor:
    return torch.randn(_BATCH, _QUERIES, _HEADS * _DIM, device="cuda")


def _cuda_forward_layout() -> tuple[int, int]:
    return forward_thread_layout(min(_BATCH, _IM2COL_STEP), _QUERIES, _HEADS, _DIM)


def _cuda_backward_layout() -> tuple[int, int]:
    return backward_thread_layout(min(_BATCH, _IM2COL_STEP), _QUERIES, _HEADS, _DIM)


# ``deformops_cuda::*``: the compiled CUDA operators.


def _sample_dm_forward() -> OpArgs:
    """Shared by the linear and softmax production forwards, which share a schema."""
    msda_fp32acc.load()
    shapes, start_index, value, loc, attn = _msda_tensors(shape_dtype=torch.long)
    channels_per_thread, threads_per_block = _cuda_forward_layout()
    return (
        (
            value,
            shapes,
            start_index,
            loc,
            attn,
            _IM2COL_STEP,
            _POINTS,
            channels_per_thread,
            threads_per_block,
        ),
        {},
    )


def _sample_dm_backward() -> OpArgs:
    msda_fp32acc.load()
    shapes, start_index, value, loc, attn = _msda_tensors(shape_dtype=torch.long)
    channels_per_thread, threads_per_block = _cuda_backward_layout()
    grad_output = _grad_output()
    return (
        (
            value,
            shapes,
            start_index,
            loc,
            attn,
            grad_output,
            _IM2COL_STEP,
            _POINTS,
            channels_per_thread,
            threads_per_block,
        ),
        {},
    )


def _sample_dm_half2atomic_forward() -> OpArgs:
    """One non-production configuration, covering the second registration path.

    All six share the schema and factory, so two cover them without building all.
    """
    msda_half2atomic.load()
    shapes, start_index, value, loc, attn = _msda_tensors(shape_dtype=torch.long)
    channels_per_thread, threads_per_block = _cuda_forward_layout()
    return (
        (
            value,
            shapes,
            start_index,
            loc,
            attn,
            _IM2COL_STEP,
            _POINTS,
            channels_per_thread,
            threads_per_block,
        ),
        {},
    )


def _sample_dm_half2atomic_backward() -> OpArgs:
    msda_half2atomic.load()
    shapes, start_index, value, loc, attn = _msda_tensors(shape_dtype=torch.long)
    channels_per_thread, threads_per_block = _cuda_backward_layout()
    grad_output = _grad_output()
    return (
        (
            value,
            shapes,
            start_index,
            loc,
            attn,
            grad_output,
            _IM2COL_STEP,
            _POINTS,
            channels_per_thread,
            threads_per_block,
        ),
        {},
    )


def _sample_public_msda() -> OpArgs:
    """``deformops::msda`` and ``deformops::msda_softmax``, on the selected backend.

    The attn and conv public operators reuse their Triton forward builders.
    """
    shapes, start_index, value, loc, attn = _msda_tensors(shape_dtype=torch.long)
    return (value, shapes, start_index, loc, attn), {}


# ``deformops_triton::msda_query_tiled*``: one argument shape per role.


def _sample_msda_triton_forward() -> OpArgs:
    shapes, start_index, value, loc, attn = _msda_tensors(shape_dtype=torch.int32)
    return (value, shapes, start_index, loc, attn), {}


def _sample_msda_triton_backward() -> OpArgs:
    shapes, start_index, value, loc, attn = _msda_tensors(shape_dtype=torch.int32)
    grad_output = _grad_output()
    return (grad_output, value, shapes, start_index, loc, attn), {}


def _sample_msda_triton_softmax_backward() -> OpArgs:
    """The caching softmax backward, which also takes the ``(B, Q, H)`` statistics."""
    shapes, start_index, value, loc, attn = _msda_tensors(shape_dtype=torch.int32)
    grad_output = _grad_output()
    statistics_shape = (value.shape[0], loc.shape[1], value.shape[2])
    running_max = torch.zeros(statistics_shape, device=value.device)
    running_sum = torch.ones(statistics_shape, device=value.device)
    return (
        grad_output,
        value,
        shapes,
        start_index,
        loc,
        attn,
        running_max,
        running_sum,
    ), {}


# ``deformops_triton::conv_deform2d_*``

_CONV_CHANNELS = 4
_CONV_HW = 5
_CONV_KERNEL = 3
_CONV_POINTS = _CONV_KERNEL * _CONV_KERNEL


def _sample_conv_deform2d_forward() -> OpArgs:
    value = torch.randn(
        1, _CONV_HW, _CONV_HW, _CONV_CHANNELS, device="cuda", requires_grad=True
    )
    offset = (
        torch.randn(1, _CONV_HW, _CONV_HW, 1, _CONV_POINTS, 2, device="cuda") * 0.5
    ).requires_grad_(True)
    mask = torch.rand(
        1, _CONV_HW, _CONV_HW, 1, _CONV_POINTS, device="cuda", requires_grad=True
    )
    weight = torch.randn(
        _CONV_CHANNELS,
        _CONV_CHANNELS,
        _CONV_KERNEL,
        _CONV_KERNEL,
        device="cuda",
        requires_grad=True,
    )
    bias = torch.randn(_CONV_CHANNELS, device="cuda", requires_grad=True)
    # stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups, offset_groups
    return (value, offset, mask, weight, bias, 1, 1, 1, 1, 1, 1, 1, 1), {}


def _sample_conv_deform2d_backward() -> OpArgs:
    value = torch.randn(
        1, _CONV_HW, _CONV_HW, _CONV_CHANNELS, device="cuda", requires_grad=True
    )
    offset = (
        torch.randn(1, _CONV_HW, _CONV_HW, 1, _CONV_POINTS, 2, device="cuda") * 0.5
    ).requires_grad_(True)
    mask = torch.rand(
        1, _CONV_HW, _CONV_HW, 1, _CONV_POINTS, device="cuda", requires_grad=True
    )
    weight = torch.randn(
        _CONV_CHANNELS,
        _CONV_CHANNELS,
        _CONV_KERNEL,
        _CONV_KERNEL,
        device="cuda",
        requires_grad=True,
    )
    grad_output = torch.randn(1, _CONV_HW, _CONV_HW, _CONV_CHANNELS, device="cuda")
    # stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups, offset_groups
    return (
        (grad_output, value, offset, mask, weight, 1, 1, 1, 1, 1, 1, 1, 1),
        {
            "needs_value_grad": True,
            "needs_offset_grad": True,
            "needs_mask_grad": True,
            "needs_weight_grad": True,
            "bias_dtype": torch.float32,
        },
    )


# ``deformops_triton::attn_query_tiled*``

_ATTN_GROUPS = 1
_ATTN_GROUP_DIMS = 8
_ATTN_KERNEL = 3
_ATTN_POINTS = _ATTN_KERNEL * _ATTN_KERNEL
_ATTN_HW = 5


def _attn_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    value = torch.randn(
        1,
        _ATTN_HW,
        _ATTN_HW,
        _ATTN_GROUPS * _ATTN_GROUP_DIMS,
        device="cuda",
        requires_grad=True,
    )
    offset_mask = torch.randn(
        1,
        _ATTN_HW,
        _ATTN_HW,
        _ATTN_GROUPS * _ATTN_POINTS * 3,
        device="cuda",
        requires_grad=True,
    )
    return value, offset_mask


def _sample_attn_forward() -> OpArgs:
    value, offset_mask = _attn_tensors()
    return (
        (
            value,
            offset_mask,
            _ATTN_KERNEL,
            _ATTN_KERNEL,
            1,
            1,
            1,
            1,
            1,
            1,
            _ATTN_GROUPS,
            _ATTN_GROUP_DIMS,
            1.0,
            False,
        ),
        {},
    )


def _sample_attn_backward() -> OpArgs:
    value, offset_mask = _attn_tensors()
    grad_output = torch.randn(
        1, _ATTN_HW, _ATTN_HW, _ATTN_GROUPS * _ATTN_GROUP_DIMS, device="cuda"
    )
    return (
        (
            grad_output,
            value,
            offset_mask,
            _ATTN_KERNEL,
            _ATTN_KERNEL,
            1,
            1,
            1,
            1,
            1,
            1,
            _ATTN_GROUPS,
            _ATTN_GROUP_DIMS,
            1.0,
            False,
        ),
        {},
    )


# The sample-input registry, enforced by ``test_sample_registry_is_complete``.

SAMPLE_INPUTS: dict[str, Callable[[], OpArgs]] = {
    "deformops::msda": _sample_public_msda,
    "deformops::msda_softmax": _sample_public_msda,
    "deformops::attn": _sample_attn_forward,
    "deformops::attn_softmax": _sample_attn_forward,
    "deformops::conv": _sample_conv_deform2d_forward,
    "deformops_cuda::msda_fp32acc_forward": _sample_dm_forward,
    "deformops_cuda::msda_fp32acc_backward": _sample_dm_backward,
    "deformops_cuda::msda_fp32acc_softmax_forward": _sample_dm_forward,
    "deformops_cuda::msda_fp32acc_softmax_backward": _sample_dm_backward,
    "deformops_cuda::msda_half2atomic_forward": _sample_dm_half2atomic_forward,
    "deformops_cuda::msda_half2atomic_backward": _sample_dm_half2atomic_backward,
    "deformops_cuda::msda_half2atomic_softmax_forward": (
        _sample_dm_half2atomic_forward
    ),
    "deformops_cuda::msda_half2atomic_softmax_backward": (
        _sample_dm_half2atomic_backward
    ),
}

#: Caching configurations, whose softmax backward takes two extra statistics.
_CACHING_CONFIGURATIONS = frozenset(
    {
        "query_tiled_cached",
        "query_tiled_cached_fp32acc",
        "query_tiled_autotuned",
        "query_tiled_autotuned_fp32acc",
    }
)

if _HAS_TRITON:
    for _name in _TRITON_MSDA_CONFIGURATIONS:
        _softmax_backward = (
            _sample_msda_triton_softmax_backward
            if _name in _CACHING_CONFIGURATIONS
            else _sample_msda_triton_backward
        )
        SAMPLE_INPUTS.update(
            {
                f"deformops_triton::msda_{_name}_forward": _sample_msda_triton_forward,
                f"deformops_triton::msda_{_name}_softmax_forward": (
                    _sample_msda_triton_forward
                ),
                f"deformops_triton::msda_{_name}_backward": (
                    _sample_msda_triton_backward
                ),
                f"deformops_triton::msda_{_name}_softmax_backward": _softmax_backward,
            }
        )
    SAMPLE_INPUTS.update(
        {
            "deformops_triton::conv_deform2d_forward": _sample_conv_deform2d_forward,
            "deformops_triton::conv_deform2d_backward": _sample_conv_deform2d_backward,
            "deformops_triton::attn_query_tiled_forward": _sample_attn_forward,
            "deformops_triton::attn_query_tiled_backward": _sample_attn_backward,
            "deformops_triton::attn_query_tiled_fp32acc_forward": _sample_attn_forward,
            "deformops_triton::attn_query_tiled_fp32acc_backward": _sample_attn_backward,
        }
    )


def _registered_ops() -> list[str]:
    """Every operator currently registered in the three namespaces."""
    names: list[str] = []
    for namespace in ("deformops", "deformops_cuda", "deformops_triton"):
        if hasattr(torch.ops, namespace):
            names += [
                f"{namespace}::{name}" for name in sorted(getattr(torch.ops, namespace))
            ]
    return names


def _op_handle(qualified_name: str):
    namespace, name = qualified_name.split("::", 1)
    return getattr(getattr(torch.ops, namespace), name)


#: The production and one other configuration, covering both registration paths.
_OPCHECKED_MSDA_PREFIXES = (
    "msda_fp32acc",
    "msda_half2atomic",
)

#: The other configurations share a factory and schema with one above; parity tests
#: cover their numerics.
_SCHEMA_EQUIVALENT_MSDA_PREFIXES = (
    "msda_serialreduce",
    "msda_half2atomic_serialreduce",
    "msda_scalaratomic",
    "msda_scalaratomic_serialreduce",
)


def _schema_equivalent_coverage() -> dict[str, str]:
    """Uncovered operator -> the opchecked operator whose schema it shares."""
    suffixes = tuple(
        f"{marker}_{direction}"
        for marker in ("", "_softmax")
        for direction in ("forward", "backward")
    )
    return {
        f"deformops_cuda::{prefix}{suffix}": (
            f"deformops_cuda::{_OPCHECKED_MSDA_PREFIXES[0]}{suffix}"
        )
        for prefix in _SCHEMA_EQUIVALENT_MSDA_PREFIXES
        for suffix in suffixes
    }


def _schema_body(qualified_name: str) -> str:
    """An operator's schema with its own name stripped off the front."""
    schema = str(_op_handle(qualified_name).default._schema)
    return schema[schema.index("(") :]


def test_sample_registry_is_complete():
    """Every registered operator has sample inputs or a schema-equivalent stand-in."""
    covered = _schema_equivalent_coverage()
    missing = [
        name
        for name in _registered_ops()
        if name not in SAMPLE_INPUTS and name not in covered
    ]
    assert not missing, (
        f"no sample inputs registered for: {missing!r}; add a builder to "
        "SAMPLE_INPUTS in tests/deformops/test_opcheck.py, or declare it "
        "schema-equivalent to one that has one"
    )


def test_unopchecked_operators_share_an_opchecked_schema():
    """A stand-in is valid only while the two schemas are identical."""
    registered = set(_registered_ops())
    for name, proxy in _schema_equivalent_coverage().items():
        if name not in registered:
            continue
        assert proxy in SAMPLE_INPUTS, (
            f"{proxy} stands in for {name} but is not sampled"
        )
        assert _schema_body(name) == _schema_body(proxy), (
            f"{name} is declared schema-equivalent to {proxy}, but their "
            f"schemas differ:\n  {_schema_body(name)}\n  {_schema_body(proxy)}"
        )


# One test per opcheck subtest: ``raise_exception=True`` stops at the first failure.

_SUBTESTS = (
    "test_schema",
    "test_autograd_registration",
    "test_faketensor",
    "test_aot_dispatch_static",
    "test_aot_dispatch_dynamic",
)

# The Triton backward ops have no autograd of their own, so the AOT subtests, which
# differentiate through them, hit custom_op's raising fallback; see docs/export.md.
_NO_AUTOGRAD_FORMULA_REASON = (
    "these are Triton-namespace raw *_backward helper custom_ops, never "
    "given their own register_autograd because they are only meant to be "
    "invoked from inside their forward op's autograd bridge (grad tracking "
    "off there). torch.library.custom_op installs a strict autograd "
    "fallback that raises when backward is attempted and none was "
    "registered. opcheck's test_autograd_registration requires a "
    "requires_grad=True input to run at all, and once present, AOTAutograd "
    "tries to build a backward graph through this op and hits the raise. "
    "Error: RuntimeError: Trying to backward through <op> but no autograd "
    "formula was registered. Please use register_autograd to add one."
)

#: Empty without Triton, where none of these operators is registered or sampled.
_BACKWARD_CUSTOM_OPS: tuple[str, ...] = (
    (
        "deformops_triton::attn_query_tiled_backward",
        "deformops_triton::attn_query_tiled_fp32acc_backward",
        "deformops_triton::conv_deform2d_backward",
        *(
            f"deformops_triton::msda_{name}_{suffix}"
            for name in _TRITON_MSDA_CONFIGURATIONS
            for suffix in ("backward", "softmax_backward")
        ),
    )
    if _HAS_TRITON
    else ()
)

# {qualified_name: {subtest: reason}}; anything absent is a hard assertion.
_XFAIL: dict[str, dict[str, str]] = {
    **{
        name: {
            "test_aot_dispatch_static": _NO_AUTOGRAD_FORMULA_REASON,
            "test_aot_dispatch_dynamic": _NO_AUTOGRAD_FORMULA_REASON,
        }
        for name in _BACKWARD_CUSTOM_OPS
    },
}


def test_xfail_registry_matches_known_operators_and_subtests() -> None:
    """``_XFAIL`` names real operators and subtests, so a rename cannot orphan one."""
    unknown_ops = [name for name in _BACKWARD_CUSTOM_OPS if name not in SAMPLE_INPUTS]
    assert not unknown_ops, (
        f"_BACKWARD_CUSTOM_OPS names operators absent from SAMPLE_INPUTS: "
        f"{unknown_ops!r}"
    )
    for name, subtests in _XFAIL.items():
        assert name in SAMPLE_INPUTS, (
            f"_XFAIL names an operator absent from SAMPLE_INPUTS: {name!r}"
        )
        unknown_subtests = [s for s in subtests if s not in _SUBTESTS]
        assert not unknown_subtests, (
            f"_XFAIL[{name!r}] names subtests absent from _SUBTESTS: "
            f"{unknown_subtests!r}"
        )


def _params_for(subtest: str) -> list[Any]:
    return [
        pytest.param(
            name,
            marks=(
                [
                    pytest.mark.xfail(
                        reason=_XFAIL[name][subtest].replace(
                            "<op>", name.split("::", 1)[1]
                        ),
                        strict=True,
                    )
                ]
                if subtest in _XFAIL.get(name, {})
                else []
            ),
            id=name.replace("::", "."),
        )
        for name in sorted(SAMPLE_INPUTS)
    ]


def _opcheck_one(qualified_name: str, subtest: str) -> None:
    op = _op_handle(qualified_name)
    args, kwargs = SAMPLE_INPUTS[qualified_name]()
    torch.library.opcheck(op, args, kwargs, test_utils=[subtest])


@pytest.mark.parametrize("qualified_name", _params_for("test_schema"))
def test_opcheck_schema(qualified_name: str) -> None:
    _opcheck_one(qualified_name, "test_schema")


@pytest.mark.parametrize("qualified_name", _params_for("test_autograd_registration"))
def test_opcheck_autograd_registration(qualified_name: str) -> None:
    _opcheck_one(qualified_name, "test_autograd_registration")


@pytest.mark.parametrize("qualified_name", _params_for("test_faketensor"))
def test_opcheck_faketensor(qualified_name: str) -> None:
    _opcheck_one(qualified_name, "test_faketensor")


@pytest.mark.parametrize("qualified_name", _params_for("test_aot_dispatch_static"))
def test_opcheck_aot_dispatch_static(qualified_name: str) -> None:
    _opcheck_one(qualified_name, "test_aot_dispatch_static")


@pytest.mark.parametrize("qualified_name", _params_for("test_aot_dispatch_dynamic"))
def test_opcheck_aot_dispatch_dynamic(qualified_name: str) -> None:
    _opcheck_one(qualified_name, "test_aot_dispatch_dynamic")


def test_opcheck_conv_backward_skipped_gradients() -> None:
    """Autograd asks for only some gradients.

    Each skipped gradient is its own zero-element placeholder, since returns may
    not alias; ``test_schema`` checks that. AOT subtests are excluded as above.
    """
    args, kwargs = _sample_conv_deform2d_backward()
    kwargs = {
        **kwargs,
        "needs_offset_grad": False,
        "needs_mask_grad": False,
        "bias_dtype": None,
    }
    op = torch.ops.deformops_triton.conv_deform2d_backward.default
    torch.library.opcheck(
        op,
        args,
        kwargs,
        test_utils=["test_schema", "test_faketensor", "test_autograd_registration"],
    )
