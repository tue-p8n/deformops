"""The public ``deformops::*`` operators, whose one kernel selects a backend.

One section per operator appends :class:`PublicOpCase` entries to :data:`CASES`;
the generic tests after them run every case. See ``docs/export.md``.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
import textwrap
from collections.abc import Callable
from typing import Any

import pytest
import torch
from deformops.core.capabilities import has_triton
from deformops.registry import Backend, attn, conv, msda
from torch._subclasses.fake_tensor import FakeTensorMode

#: Positional operands of one public operator call.
Operands = tuple[Any, ...]


@dataclasses.dataclass(frozen=True)
class PublicOpCase:
    """One public operator and how to call it.

    Attributes
    ----------
    name
        The operator name in the ``deformops`` namespace.
    schema
        The full schema string, namespace included.
    make_operands
        ``(batch, size, requires_grad)`` to the positional operands on CPU;
        ``size`` is the query count or spatial size.
    registry_call
        ``(operands, backend)`` to the output of the registry entry point.
    vmap_in_dims
        ``0`` for an operand that varies per sample, ``None`` for one that
        describes the batch.
    """

    name: str
    schema: str
    make_operands: Callable[[int, int, bool], Operands]
    registry_call: Callable[[Operands, Backend | None], torch.Tensor]
    vmap_in_dims: tuple[int | None, ...]

    @property
    def op(self) -> Any:
        """The operator packet ``torch.ops.deformops.<name>``."""
        return getattr(torch.ops.deformops, self.name)


#: Every public operator case the generic tests run.
CASES: list[PublicOpCase] = []


# Multi-scale deformable attention: deformops::msda, deformops::msda_softmax

_MSDA_SCHEMA = (
    "deformops::{name}(Tensor value, Tensor spatial_shapes, "
    "Tensor level_start_index, Tensor sampling_locations, "
    "Tensor attention_weights) -> Tensor"
)
_MSDA_LEVELS = ((4, 4), (2, 3))
_MSDA_HEADS, _MSDA_HEAD_DIM, _MSDA_POINTS = 2, 4, 3


def _msda_operands(batch: int, queries: int, requires_grad: bool) -> Operands:
    """``(value, spatial_shapes, level_start_index, locations, weights)``.

    Some locations fall outside ``[0, 1]``.
    """
    levels = len(_MSDA_LEVELS)
    sizes = [height * width for height, width in _MSDA_LEVELS]
    spatial_shapes = torch.tensor(_MSDA_LEVELS, dtype=torch.long)
    level_start_index = torch.tensor([0, *sizes[:-1]], dtype=torch.long).cumsum(0)
    value = torch.randn(batch, sum(sizes), _MSDA_HEADS, _MSDA_HEAD_DIM)
    locations = (
        torch.rand(batch, queries, _MSDA_HEADS, levels, _MSDA_POINTS, 2) * 1.2 - 0.1
    )
    weights = torch.randn(batch, queries, _MSDA_HEADS, levels * _MSDA_POINTS)
    for tensor in (value, locations, weights):
        tensor.requires_grad_(requires_grad)
    return value, spatial_shapes, level_start_index, locations, weights


def _msda_registry_call(method: str) -> Callable[[Operands, Backend | None], Any]:
    def call(operands: Operands, backend: Backend | None) -> Any:
        # Passed as the layer passes it; the public operator does not take it.
        return msda(*operands, method=method, chunk_step=128, backend=backend)

    return call


CASES += [
    PublicOpCase(
        name=name,
        schema=_MSDA_SCHEMA.format(name=name),
        make_operands=_msda_operands,
        registry_call=_msda_registry_call(method),
        vmap_in_dims=(0, None, None, 0, 0),
    )
    for name, method in (("msda", "linear"), ("msda_softmax", "softmax"))
]


def test_msda_vmap_over_the_level_table_raises() -> None:
    """The level table describes the batch, so batching it is an error."""
    value, spatial_shapes, level_start_index, locations, weights = _msda_operands(
        2, 3, requires_grad=False
    )
    batched_shapes = torch.stack([spatial_shapes, spatial_shapes])
    with pytest.raises(RuntimeError, match="spatial shapes and level start offsets"):
        torch.vmap(torch.ops.deformops.msda, in_dims=(None, 0, None, None, None))(
            value, batched_shapes, level_start_index, locations, weights
        )


def test_msda_backend_override_negotiates_in_python(monkeypatch) -> None:
    """A policy override keyword skips the public operator and still runs."""
    operands = _msda_operands(2, 3, requires_grad=False)
    calls: list[dict[str, Any]] = []
    run_selected = msda.run_selected

    def recording(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return run_selected(*args, **kwargs)

    monkeypatch.setattr(msda, "run_selected", recording)
    output = msda(*operands, fp32acc=True)
    # Negotiate once with the override, which the public kernel would not apply.
    assert len(calls) == 1
    assert calls[0]["fp32acc"] is True
    torch.testing.assert_close(output, msda(*operands, backend=Backend.EAGER))


def test_msda_first_call_compiles_before_any_backend_loads() -> None:
    """A fresh process can compile a ``backend=None`` call as its first call."""
    script = textwrap.dedent(
        """
        import torch
        from deformops.registry import Backend, msda

        value = torch.randn(1, 16, 2, 4)
        shapes = torch.tensor([[4, 4]])
        start = torch.tensor([0])
        loc = torch.rand(1, 3, 2, 1, 2, 2)
        weights = torch.randn(1, 3, 2, 2)
        compiled = torch.compile(
            lambda *args: msda(*args, method="softmax"),
            backend="eager",
            fullgraph=True,
        )
        output = compiled(value, shapes, start, loc, weights)
        expected = msda(
            value, shapes, start, loc, weights, method="softmax",
            backend=Backend.EAGER,
        )
        torch.testing.assert_close(output, expected)
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_msda_value_checks_skip_fake_and_traced_calls(monkeypatch) -> None:
    """``DEFORMOPS_VALIDATE`` reads the level table only when it has values."""
    monkeypatch.setenv("DEFORMOPS_VALIDATE", "1")
    with FakeTensorMode():
        output = torch.ops.deformops.msda(*_msda_operands(2, 3, requires_grad=True))
    assert output.shape == (2, 3, _MSDA_HEADS * _MSDA_HEAD_DIM)

    def call(*operands: Any) -> torch.Tensor:
        return msda(*operands)

    torch._dynamo.reset()
    compiled = torch.compile(call, backend="aot_eager", dynamic=True, fullgraph=True)
    operands = _msda_operands(2, 3, requires_grad=True)
    torch.testing.assert_close(
        compiled(*operands), msda(*operands, backend=Backend.EAGER)
    )

    value, spatial_shapes, level_start_index, locations, weights = operands
    with pytest.raises(ValueError, match="level_start_index"):
        msda(value, spatial_shapes, level_start_index + 1, locations, weights)


def test_msda_validation_errors_pass_through_the_operator() -> None:
    """A malformed ``backend=None`` call raises the same ``ValueError``."""
    value, spatial_shapes, level_start_index, locations, weights = _msda_operands(
        2, 3, requires_grad=False
    )
    with pytest.raises(ValueError, match="value must be 4-dimensional"):
        msda(value[0], spatial_shapes, level_start_index, locations, weights)
    with pytest.raises(TypeError, match="unexpected keyword"):
        msda(value, spatial_shapes, level_start_index, locations, weights, bogus=1)


#: 4 points per level and head channels divisible by 8, as the CUDA templates require.
_MSDA_GPU_HEADS, _MSDA_GPU_HEAD_DIM, _MSDA_GPU_POINTS = 2, 8, 4

_NEEDS_TRITON = pytest.mark.skipif(not has_triton(), reason="Triton is not installed")


def _msda_gpu_operands(
    device: torch.device, dtype: torch.dtype, requires_grad: bool
) -> Operands:
    """CUDA operands in the layout of :data:`_MSDA_GPU_HEADS` and its siblings."""
    levels = len(_MSDA_LEVELS)
    sizes = [height * width for height, width in _MSDA_LEVELS]
    spatial_shapes = torch.tensor(_MSDA_LEVELS, dtype=torch.long, device=device)
    level_start_index = torch.tensor(
        [0, *sizes[:-1]], dtype=torch.long, device=device
    ).cumsum(0)
    value = torch.randn(
        2, sum(sizes), _MSDA_GPU_HEADS, _MSDA_GPU_HEAD_DIM, device=device, dtype=dtype
    )
    shape = (2, 7, _MSDA_GPU_HEADS, levels, _MSDA_GPU_POINTS)
    locations = torch.rand(*shape, 2, device=device) * 1.2 - 0.1
    weights = torch.randn(*shape[:3], levels * _MSDA_GPU_POINTS, device=device)
    weights = weights.to(dtype)
    for tensor in (value, locations, weights):
        tensor.requires_grad_(requires_grad)
    return value, spatial_shapes, level_start_index, locations, weights


def _dispatched_op_names(call: Callable[[], Any]) -> set[str]:
    """Qualified names of every operator ``call`` dispatches below autograd."""
    from torch.utils._python_dispatch import TorchDispatchMode

    names: set[str] = set()

    class Recorder(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # noqa: ANN001
            names.add(func._schema.name)
            return func(*args, **(kwargs or {}))

    with Recorder():
        call()
    return names


def _aot_op_names(
    call: Callable[..., torch.Tensor], operands: Operands, *, backward: bool
) -> dict[str, set[str]]:
    """The operators in the AOTAutograd graphs of a compiled ``call``.

    Triton operators are kept whole, so the graphs name the selected kernel
    instead of an unnamed launch.
    """
    from functorch.compile import make_boxed_func
    from torch._dynamo.backends.common import aot_autograd

    names: dict[str, set[str]] = {"forward": set(), "backward": set()}

    def recorder(kind: str) -> Callable[..., Any]:
        def compiler(graph_module: torch.fx.GraphModule, _inputs: Any) -> Any:
            names[kind] |= _graph_op_names(graph_module)
            return make_boxed_func(graph_module.forward)

        return compiler

    torch._dynamo.reset()
    compiled = torch.compile(
        call,
        backend=aot_autograd(
            fw_compiler=recorder("forward"), bw_compiler=recorder("backward")
        ),
        fullgraph=True,
    )
    with torch._functorch.config.patch(decompose_custom_triton_ops=False):
        output = compiled(*operands)
        if backward:
            output.float().sum().backward()
    return names


@_NEEDS_TRITON
@pytest.mark.parametrize("method", ["linear", "softmax"])
@pytest.mark.parametrize(
    ("dtype", "training", "prefix"),
    [
        pytest.param(
            torch.bfloat16,
            True,
            "msda_query_tiled_fp32acc",
            marks=pytest.mark.skipif(
                torch.cuda.is_available()
                and torch.cuda.get_device_capability() >= (9, 0),
                reason="SM 9.0 and newer train on the native accumulator",
            ),
            id="bfloat16-training",
        ),
        pytest.param(
            torch.bfloat16, False, "msda_query_tiled", id="bfloat16-inference"
        ),
        pytest.param(torch.float32, True, "msda_query_tiled", id="float32-training"),
    ],
)
def test_msda_cuda_call_decomposes_to_the_policy_kernel(  # noqa: PLR0913
    device: torch.device,
    method: str,
    dtype: torch.dtype,
    training: bool,
    prefix: str,
) -> None:
    """Compiled and eager calls reach the Triton kernel the policy picks.

    Half-precision training before SM 9.0 takes the float32 accumulator.
    """
    operands = _msda_gpu_operands(device, dtype, requires_grad=training)
    public = "deformops::msda_softmax" if method == "softmax" else "deformops::msda"
    kernel = "_softmax" if method == "softmax" else ""
    forward = f"deformops_triton::{prefix}{kernel}_forward"

    def call(*args: Any) -> torch.Tensor:
        return msda(*args, method=method)

    names = _aot_op_names(call, operands, backward=training)
    assert forward in names["forward"], names["forward"]
    assert public not in names["forward"] | names["backward"]
    if training:
        backward = f"deformops_triton::{prefix}{kernel}_backward"
        assert backward in names["backward"], names["backward"]
    other = {"msda_query_tiled", "msda_query_tiled_fp32acc"} - {prefix}
    assert (
        not {f"deformops_triton::{p}{kernel}_forward" for p in other}
        & (names["forward"])
    )

    eager_names = _dispatched_op_names(lambda: call(*operands))
    assert forward in eager_names, eager_names


@pytest.mark.parametrize("method", ["linear", "softmax"])
def test_msda_explicit_cuda_backend_reaches_the_cuda_kernel(
    device: torch.device, method: str
) -> None:
    """``backend=Backend.CUDA`` runs the ``deformops_cuda`` operators."""
    operands = _msda_gpu_operands(device, torch.float32, requires_grad=True)
    kernel = "msda_fp32acc_softmax" if method == "softmax" else "msda_fp32acc"

    def call() -> None:
        msda(*operands, method=method, backend=Backend.CUDA).sum().backward()

    names = _dispatched_op_names(call)
    assert f"deformops_cuda::{kernel}_forward" in names, names
    assert f"deformops_cuda::{kernel}_backward" in names, names
    assert not any(name.startswith("deformops_triton::") for name in names), names


@_NEEDS_TRITON
@pytest.mark.parametrize("method", ["linear", "softmax"])
@pytest.mark.parametrize("backend", [Backend.CUDA, Backend.TRITON], ids=str)
def test_msda_public_op_on_cuda_matches_the_device_backends(
    device: torch.device, method: str, backend: Backend
) -> None:
    """Output and gradients match each device backend, at Triton's float32 tolerance."""
    operands = _msda_gpu_operands(device, torch.float32, requires_grad=True)
    through_op = _detached(operands, requires_grad=True)
    explicit = _detached(operands, requires_grad=True)

    output = msda(*through_op, method=method)
    expected = msda(*explicit, method=method, backend=backend)
    grad_output = torch.randn_like(expected)
    output.backward(grad_output)
    expected.backward(grad_output)

    torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        _gradients(through_op), _gradients(explicit), atol=1e-5, rtol=1e-5
    )


# Classic deformable convolution: deformops::conv

_CONV_SCHEMA = (
    "deformops::conv(Tensor value, Tensor offset, Tensor? mask, Tensor weight, "
    "Tensor? bias, int stride_h, int stride_w, int pad_h, int pad_w, "
    "int dilation_h, int dilation_w, int groups, int offset_groups) -> Tensor"
)
#: Per-axis padding and dilation that keep an S by S output, so an axis swap changes
#: values but no shape.
_CONV_GEOMETRY = (1, 1, 1, 2, 1, 2)
_CONV_IN, _CONV_OUT, _CONV_GROUPS, _CONV_OFFSET_GROUPS = 4, 6, 2, 2


def _conv_operands(batch: int, size: int, requires_grad: bool) -> Operands:
    """``(value, offset, mask, weight, bias, *geometry, groups, offset_groups)``.

    Some samples fall outside the input.
    """
    points = 9
    value = torch.randn(batch, size, size, _CONV_IN)
    offset = torch.randn(batch, size, size, _CONV_OFFSET_GROUPS, points, 2) * 2
    mask = torch.rand(batch, size, size, _CONV_OFFSET_GROUPS, points)
    weight = torch.randn(_CONV_OUT, _CONV_IN // _CONV_GROUPS, 3, 3)
    bias = torch.randn(_CONV_OUT)
    for tensor in (value, offset, mask, weight, bias):
        tensor.requires_grad_(requires_grad)
    return (
        value,
        offset,
        mask,
        weight,
        bias,
        *_CONV_GEOMETRY,
        _CONV_GROUPS,
        _CONV_OFFSET_GROUPS,
    )


def _conv_registry_call(operands: Operands, backend: Backend | None) -> Any:
    value, offset, mask, weight, bias, *geometry, groups, offset_groups = operands
    stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w = geometry
    return conv(
        value,
        offset,
        weight,
        bias=bias,
        mask=mask,
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        dilation=(dilation_h, dilation_w),
        groups=groups,
        offset_groups=offset_groups,
        backend=backend,
    )


CASES += [
    PublicOpCase(
        name="conv",
        schema=_CONV_SCHEMA,
        make_operands=_conv_operands,
        registry_call=_conv_registry_call,
        vmap_in_dims=(0, 0, 0, None, None, *([None] * 8)),
    )
]


@pytest.mark.parametrize(
    ("position", "match"), [(3, "a convolution weight"), (4, "a convolution bias")]
)
def test_conv_vmap_over_the_weight_or_bias_raises(position: int, match: str) -> None:
    """The weight and the bias describe the batch, so batching either raises."""
    operands = list(_conv_operands(2, 3, requires_grad=False))
    operands[position] = torch.stack([operands[position], operands[position]])
    in_dims = [0, 0, 0, None, None, *([None] * 8)]
    in_dims[position] = 0
    with pytest.raises(RuntimeError, match=match):
        torch.vmap(torch.ops.deformops.conv, in_dims=tuple(in_dims))(*operands)


def test_conv_without_mask_or_bias_matches_the_explicit_eager_path() -> None:
    """The unmodulated form passes ``None`` through both optional operands."""
    value, offset, _, weight, _, *rest = _conv_operands(2, 5, requires_grad=False)
    operands = (value, offset, None, weight, None, *rest)
    through_op = _detached(operands, requires_grad=True)
    explicit = _detached(operands, requires_grad=True)

    output = torch.ops.deformops.conv(*through_op)
    expected = _conv_registry_call(explicit, Backend.EAGER)
    grad_output = torch.randn_like(expected)
    output.backward(grad_output)
    expected.backward(grad_output)

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        _gradients(through_op), _gradients(explicit), rtol=0, atol=0
    )


def test_conv_validation_errors_pass_through_the_operator() -> None:
    """A malformed ``backend=None`` call raises what the explicit call raises."""
    value, offset, mask, weight, bias, *_ = _conv_operands(2, 5, requires_grad=False)
    with pytest.raises(ValueError, match="output grid"):
        conv(
            value,
            offset,
            weight,
            bias=bias,
            mask=mask,
            groups=_CONV_GROUPS,
            offset_groups=_CONV_OFFSET_GROUPS,
        )
    with pytest.raises(ValueError, match="padding must be an int or a pair"):
        conv(value, offset, weight, padding=(1, 1, 1))
    with pytest.raises(TypeError, match="unexpected keyword"):
        conv(value, offset, weight, padding=1, group=2)


def test_conv_first_call_compiles_before_any_backend_loads() -> None:
    """A fresh process can compile a ``backend=None`` call as its first call."""
    script = textwrap.dedent(
        """
        import torch
        from deformops.registry import Backend, conv

        value = torch.randn(1, 6, 6, 4)
        offset = torch.randn(1, 6, 6, 1, 9, 2)
        weight = torch.randn(4, 4, 3, 3)
        compiled = torch.compile(
            lambda *args: conv(*args, padding=1), backend="eager", fullgraph=True
        )
        output = compiled(value, offset, weight)
        expected = conv(value, offset, weight, padding=1, backend=Backend.EAGER)
        torch.testing.assert_close(output, expected)
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


# Single-scale deformable sampling: deformops::attn, deformops::attn_softmax

_ATTN_SCHEMA = (
    "deformops::{name}(Tensor value, Tensor offset_mask, int kernel_h, "
    "int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w, "
    "int dilation_h, int dilation_w, int groups, int group_dims, "
    "float spread, bool drop_center) -> Tensor"
)
#: An asymmetric geometry, so a height and width swap changes the output.
_ATTN_KERNEL, _ATTN_STRIDE = (3, 2), (1, 2)
_ATTN_PADDING, _ATTN_DILATION = (1, 0), (2, 1)
_ATTN_GROUPS, _ATTN_GROUP_DIMS, _ATTN_OFFSET_SCALE = 2, 3, 0.5


def _attn_output_extent(extent: int, axis: int) -> int:
    kernel, stride = _ATTN_KERNEL[axis], _ATTN_STRIDE[axis]
    padding, dilation = _ATTN_PADDING[axis], _ATTN_DILATION[axis]
    return (extent + 2 * padding - (dilation * (kernel - 1) + 1)) // stride + 1


def _attn_operands(drop_center: bool) -> Callable[[int, int, bool], Operands]:
    """Two tensors, then the geometry; the input is ``size`` high and one column wider.

    Some offsets leave the input, and ``offset_mask`` has one unread channel.
    """

    def make(batch: int, size: int, requires_grad: bool) -> Operands:
        height, width = size, size + 1
        points = _ATTN_KERNEL[0] * _ATTN_KERNEL[1] - (1 if drop_center else 0)
        value = torch.randn(batch, height, width, _ATTN_GROUPS * _ATTN_GROUP_DIMS)
        offset_mask = torch.randn(
            batch,
            _attn_output_extent(height, 0),
            _attn_output_extent(width, 1),
            _ATTN_GROUPS * points * 3 + 1,
        )
        offset_mask[..., : _ATTN_GROUPS * points * 3 : 3] *= 2
        for tensor in (value, offset_mask):
            tensor.requires_grad_(requires_grad)
        return (
            value,
            offset_mask,
            *_ATTN_KERNEL,
            *_ATTN_STRIDE,
            *_ATTN_PADDING,
            *_ATTN_DILATION,
            _ATTN_GROUPS,
            _ATTN_GROUP_DIMS,
            _ATTN_OFFSET_SCALE,
            drop_center,
        )

    return make


def _attn_keywords(operands: Operands) -> dict[str, Any]:
    """Registry keywords for the geometry, mixing the spellings it normalizes."""
    geometry = operands[2:]
    return {
        "kernel_size": (geometry[0], geometry[1]),
        "stride": (geometry[2], geometry[3]),
        "padding": (geometry[4], geometry[5]),
        "dilation_h": geometry[6],
        "dilation_w": geometry[7],
        "groups": geometry[8],
        "group_dims": geometry[9],
        "spread": geometry[10],
        "drop_center": geometry[11],
    }


def _attn_registry_call(method: str) -> Callable[[Operands, Backend | None], Any]:
    def call(operands: Operands, backend: Backend | None) -> Any:
        return attn(
            operands[0],
            operands[1],
            method=method,
            backend=backend,
            **_attn_keywords(operands),
        )

    return call


CASES += [
    PublicOpCase(
        name=name,
        schema=_ATTN_SCHEMA.format(name=name),
        make_operands=_attn_operands(drop_center),
        registry_call=_attn_registry_call(method),
        vmap_in_dims=(0, 0, *([None] * 12)),
    )
    for name, method, drop_center in (
        ("attn", "linear", False),
        ("attn_softmax", "softmax", True),
    )
]


def test_attn_backend_option_negotiates_in_python(monkeypatch) -> None:
    """``fp32acc`` skips the public operator and still runs."""
    operands = _attn_operands(False)(2, 5, False)
    value, offset_mask = operands[:2]
    keywords = _attn_keywords(operands)
    calls: list[dict[str, Any]] = []
    run_selected = attn.run_selected

    def recording(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return run_selected(*args, **kwargs)

    monkeypatch.setattr(attn, "run_selected", recording)
    output = attn(value, offset_mask, fp32acc=True, **keywords)
    # Negotiate once with the option, which the public kernel would not apply.
    assert len(calls) == 1
    assert calls[0]["fp32acc"] is True
    torch.testing.assert_close(
        output, attn(value, offset_mask, backend=Backend.EAGER, **keywords)
    )


def test_attn_first_call_compiles_before_any_backend_loads() -> None:
    """A fresh process can compile a ``backend=None`` call as its first call."""
    script = textwrap.dedent(
        """
        import torch
        from deformops.registry import Backend, attn

        value = torch.randn(1, 6, 6, 8)
        offset_mask = torch.randn(1, 6, 6, 2 * 9 * 3)
        compiled = torch.compile(
            lambda *args: attn(*args, groups=2, group_dims=4, method="softmax"),
            backend="eager",
            fullgraph=True,
        )
        output = compiled(value, offset_mask)
        expected = attn(
            value, offset_mask, groups=2, group_dims=4, method="softmax",
            backend=Backend.EAGER,
        )
        torch.testing.assert_close(output, expected)
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_attn_validation_errors_pass_through_the_operator() -> None:
    """A malformed ``backend=None`` call raises the same error as before."""
    operands = _attn_operands(False)(2, 5, False)
    value, offset_mask = operands[:2]
    keywords = _attn_keywords(operands)
    with pytest.raises(ValueError, match="value has size"):
        attn(value[..., :-1], offset_mask, **keywords)
    with pytest.raises(ValueError, match="output grid"):
        attn(value, offset_mask[:, :-1], **keywords)
    with pytest.raises(TypeError, match="unexpected keyword"):
        attn(value, offset_mask, bogus=1, **keywords)
    with pytest.raises(TypeError, match="pad_h and pad_w"):
        attn(value, offset_mask, pad_h=1, **keywords)


# Generic checks, run for every case above.


def _case_id(case: PublicOpCase) -> str:
    return case.name


def _graph_op_names(graph_module: torch.fx.GraphModule) -> set[str]:
    """Qualified names of every operator an FX graph calls."""
    names: set[str] = set()
    for node in graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        target = node.target
        if isinstance(target, torch._ops.OpOverload):
            names.add(target.name())
        elif isinstance(target, torch._ops.OpOverloadPacket):
            names.add(target._qualified_op_name)
    return names


def _qualified(case: PublicOpCase) -> str:
    return f"deformops::{case.name}"


def _detached(operands: Operands, requires_grad: bool) -> Operands:
    """Fresh leaves with the same values, for an independent backward."""
    return tuple(
        operand.detach().requires_grad_(requires_grad and operand.is_floating_point())
        if isinstance(operand, torch.Tensor)
        else operand
        for operand in operands
    )


def _gradients(operands: Operands) -> list[torch.Tensor]:
    return [
        operand.grad
        for operand in operands
        if isinstance(operand, torch.Tensor) and operand.requires_grad
    ]


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_schema(case: PublicOpCase) -> None:
    assert str(case.op.default._schema) == case.schema


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_forward_and_backward_match_the_explicit_eager_path(
    case: PublicOpCase,
) -> None:
    operands = case.make_operands(2, 5, True)
    through_op = _detached(operands, requires_grad=True)
    through_registry = _detached(operands, requires_grad=True)
    explicit = _detached(operands, requires_grad=True)

    outputs = [
        case.op(*through_op),
        case.registry_call(through_registry, None),
        case.registry_call(explicit, Backend.EAGER),
    ]
    grad_output = torch.randn_like(outputs[0])
    for output in outputs:
        output.backward(grad_output)

    torch.testing.assert_close(outputs[0], outputs[2], rtol=0, atol=0)
    torch.testing.assert_close(outputs[1], outputs[2], rtol=0, atol=0)
    for gradients in (_gradients(through_op), _gradients(through_registry)):
        torch.testing.assert_close(gradients, _gradients(explicit), rtol=0, atol=0)


@pytest.mark.parametrize("case", CASES, ids=_case_id)
@pytest.mark.parametrize("mode", [torch.no_grad, torch.inference_mode])
def test_runs_without_autograd(case: PublicOpCase, mode: Callable[[], Any]) -> None:
    operands = case.make_operands(2, 5, True)
    expected = case.registry_call(operands, Backend.EAGER).detach()
    with mode():
        output = case.op(*operands)
    assert not output.requires_grad
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_runs_under_fake_tensor_mode(case: PublicOpCase) -> None:
    expected = case.op(*case.make_operands(2, 5, False))
    with FakeTensorMode():
        output = case.op(*case.make_operands(2, 5, True))
    assert output.shape == expected.shape
    assert output.dtype == expected.dtype


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_compile_records_the_public_op_and_aot_the_backend(
    case: PublicOpCase,
) -> None:
    from functorch.compile import make_boxed_func
    from torch._dynamo.backends.common import aot_autograd

    dynamo_names: set[str] = set()
    aot_names: dict[str, set[str]] = {"forward": set(), "backward": set()}

    def recorder(kind: str) -> Callable[..., Any]:
        def compiler(graph_module: torch.fx.GraphModule, _inputs: Any) -> Any:
            aot_names[kind] |= _graph_op_names(graph_module)
            return make_boxed_func(graph_module.forward)

        return compiler

    aot_backend = aot_autograd(
        fw_compiler=recorder("forward"), bw_compiler=recorder("backward")
    )

    def backend(graph_module: torch.fx.GraphModule, inputs: Any) -> Any:
        dynamo_names.update(_graph_op_names(graph_module))
        return aot_backend(graph_module, inputs)

    torch._dynamo.reset()
    compiled = torch.compile(
        lambda *args: case.registry_call(args, None), backend=backend, fullgraph=True
    )
    operands = case.make_operands(2, 5, True)
    output = compiled(*operands)
    output.sum().backward()

    assert _qualified(case) in dynamo_names
    assert aot_names["forward"]
    assert aot_names["backward"]
    for kind, names in aot_names.items():
        assert _qualified(case) not in names, kind
    torch.testing.assert_close(
        output, case.registry_call(_detached(operands, False), Backend.EAGER)
    )


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_dynamic_compile_across_batch_and_size(case: PublicOpCase) -> None:
    torch._dynamo.reset()
    compiled = torch.compile(
        lambda *args: case.registry_call(args, None),
        backend="aot_eager",
        dynamic=True,
        fullgraph=True,
    )
    for batch, size in ((2, 5), (3, 7)):
        operands = case.make_operands(batch, size, True)
        output = compiled(*operands)
        output.sum().backward()
        torch.testing.assert_close(
            output, case.registry_call(_detached(operands, False), Backend.EAGER)
        )


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_export_keeps_the_public_op_until_decomposed(case: PublicOpCase) -> None:
    class Wrapper(torch.nn.Module):
        def forward(self, *operands: Any) -> torch.Tensor:
            return case.op(*operands)

    operands = case.make_operands(2, 5, False)
    exported = torch.export.export(Wrapper(), operands)
    assert _qualified(case) in _graph_op_names(exported.graph_module)

    decomposed = exported.run_decompositions()
    assert _qualified(case) not in _graph_op_names(decomposed.graph_module)
    torch.testing.assert_close(
        decomposed.module()(*operands), case.op(*operands), rtol=0, atol=0
    )


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_opcheck(case: PublicOpCase) -> None:
    torch.library.opcheck(case.op.default, case.make_operands(2, 5, True))


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_vmap_matches_a_loop_without_fallback(
    case: PublicOpCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = [case.make_operands(2, 5, False) for _ in range(3)]
    batched = tuple(
        operand
        if in_dim is None
        else torch.stack([sample[position] for sample in samples])
        for position, (operand, in_dim) in enumerate(
            zip(samples[0], case.vmap_in_dims, strict=True)
        )
    )
    shared = tuple(
        samples[0][position] if in_dim is None else None
        for position, in_dim in enumerate(case.vmap_in_dims)
    )
    looped = torch.stack(
        [
            case.op(
                *(
                    shared[position] if in_dim is None else sample[position]
                    for position, in_dim in enumerate(case.vmap_in_dims)
                )
            )
            for sample in samples
        ]
    )

    # The fallback warns from C++, so count kernel runs: a rule runs once, the
    # fallback once per sample.
    registry = {"msda": msda, "attn": attn, "conv": conv}[case.name.split("_")[0]]
    run_selected = registry.run_selected
    batches: list[int] = []

    def recording(*args: Any, **kwargs: Any) -> Any:
        batches.append(args[0].shape[0])
        return run_selected(*args, **kwargs)

    monkeypatch.setattr(registry, "run_selected", recording)
    vmapped = torch.vmap(case.op, in_dims=case.vmap_in_dims)(*batched)

    assert batches == [len(samples) * samples[0][0].shape[0]], batches
    torch.testing.assert_close(vmapped, looped)
