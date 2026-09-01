"""Parity with torchvision for classic deformable convolution, v1 and v2.

Covers the forward and every gradient across the arguments, and checks the reference
so tests can use it without torchvision. Tests assert torchvision's conventions;
see ``docs/conventions.md``.
"""

from __future__ import annotations

import pytest
import torch
from deformops.backends.reference.conv import reference as conv_reference
from deformops.integrations.torchvision import deform_conv2d

# A torchvision built for another torch imports but raises RuntimeError from
# ``torchvision.ops``, so catch everything and probe one op.
try:
    from torchvision import ops as torchvision_ops

    # A valid probe: a 3x3 weight needs 2 * 9 offset channels.
    torchvision_ops.deform_conv2d(
        torch.zeros(1, 1, 3, 3), torch.zeros(1, 18, 1, 1), torch.zeros(1, 1, 3, 3)
    )
except Exception as exc:  # noqa: BLE001
    torchvision_ops = None
    _UNAVAILABLE = f"torchvision's compiled ops are unusable here: {exc}"
else:
    _UNAVAILABLE = ""

pytestmark = pytest.mark.skipif(bool(_UNAVAILABLE), reason=_UNAVAILABLE or "available")


@pytest.fixture(scope="module")
def device(any_device):
    """The CPU always, CUDA when present; torchvision and the reference run on CPU."""
    return any_device


@pytest.fixture(autouse=True)
def _full_precision_matmul():
    """Compare like with like: disable the TF32 the suite conftest enables.

    torchvision's contraction runs through cuBLAS and this kernel runs in float32, so
    TF32 would add its own rounding. See ``docs/precision.md``.
    """
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(previous)


#: What a naive kernel gets wrong: groups, independent offset groups, stride,
#: dilation, asymmetric padding and non-square kernels.
CASES = [
    pytest.param(
        {
            "groups": 1,
            "offset_groups": 1,
            "kernel": (3, 3),
            "stride": (1, 1),
            "padding": (1, 1),
            "dilation": (1, 1),
            "bias": False,
            "mask": False,
        },
        id="v1-plain",
    ),
    pytest.param(
        {
            "groups": 1,
            "offset_groups": 1,
            "kernel": (3, 3),
            "stride": (1, 1),
            "padding": (1, 1),
            "dilation": (1, 1),
            "bias": True,
            "mask": True,
        },
        id="v2-plain",
    ),
    pytest.param(
        {
            "groups": 2,
            "offset_groups": 2,
            "kernel": (3, 3),
            "stride": (2, 2),
            "padding": (1, 1),
            "dilation": (1, 1),
            "bias": True,
            "mask": True,
        },
        id="grouped-strided",
    ),
    pytest.param(
        {
            "groups": 4,
            "offset_groups": 1,
            "kernel": (3, 3),
            "stride": (1, 1),
            "padding": (2, 2),
            "dilation": (2, 2),
            "bias": True,
            "mask": True,
        },
        id="dilated",
    ),
    pytest.param(
        {
            "groups": 1,
            "offset_groups": 4,
            "kernel": (1, 3),
            "stride": (1, 2),
            "padding": (0, 1),
            "dilation": (1, 1),
            "bias": False,
            "mask": True,
        },
        id="offset-groups-asymmetric",
    ),
    pytest.param(
        {
            "groups": 8,
            "offset_groups": 4,
            "kernel": (3, 3),
            "stride": (1, 1),
            "padding": (1, 1),
            "dilation": (1, 1),
            "bias": True,
            "mask": True,
        },
        id="groups-differ-from-offset-groups",
    ),
    pytest.param(
        {
            "groups": 1,
            "offset_groups": 1,
            "kernel": (5, 5),
            "stride": (1, 1),
            "padding": (2, 2),
            "dilation": (1, 1),
            "bias": True,
            "mask": True,
        },
        id="large-kernel",
    ),
]

_IN, _OUT, _HEIGHT, _WIDTH, _BATCH = 32, 32, 11, 13, 2


def _operands(spec: dict, device: str, dtype: torch.dtype = torch.float32) -> dict:
    kernel_h, kernel_w = spec["kernel"]
    points = kernel_h * kernel_w
    stride_h, stride_w = spec["stride"]
    pad_h, pad_w = spec["padding"]
    dil_h, dil_w = spec["dilation"]
    out_h = (_HEIGHT + 2 * pad_h - (dil_h * (kernel_h - 1) + 1)) // stride_h + 1
    out_w = (_WIDTH + 2 * pad_w - (dil_w * (kernel_w - 1) + 1)) // stride_w + 1
    og = spec["offset_groups"]
    opts = {"device": device, "dtype": dtype}
    return {
        "input": torch.randn(_BATCH, _IN, _HEIGHT, _WIDTH, **opts),
        "offset": torch.randn(_BATCH, 2 * og * points, out_h, out_w, **opts) * 0.5,
        "mask": (
            torch.rand(_BATCH, og * points, out_h, out_w, **opts)
            if spec["mask"]
            else None
        ),
        "weight": torch.randn(_OUT, _IN // spec["groups"], kernel_h, kernel_w, **opts),
        "bias": torch.randn(_OUT, **opts) if spec["bias"] else None,
        "grad_out": torch.randn(_BATCH, _OUT, out_h, out_w, **opts),
        "out_hw": (out_h, out_w),
    }


def _torchvision(spec: dict, ops: dict) -> torch.Tensor:
    return torchvision_ops.deform_conv2d(
        ops["input"],
        ops["offset"],
        ops["weight"],
        ops["bias"],
        spec["stride"],
        spec["padding"],
        spec["dilation"],
        ops["mask"],
    )


def _ours(spec: dict, ops: dict) -> torch.Tensor:
    return deform_conv2d(
        ops["input"],
        ops["offset"],
        ops["weight"],
        ops["bias"],
        spec["stride"],
        spec["padding"],
        spec["dilation"],
        ops["mask"],
    )


def _relative(ours: torch.Tensor | None, theirs: torch.Tensor | None) -> float:
    if ours is None or theirs is None:
        assert ours is theirs, (
            "one implementation produced a gradient and the other did not"
        )
        return 0.0
    scale = theirs.abs().max().clamp_min(1e-6)
    return ((ours - theirs).abs().max() / scale).item()


@pytest.mark.parametrize("spec", CASES)
def test_forward_matches_torchvision(spec: dict, device: torch.device) -> None:
    torch.manual_seed(0)
    ops = _operands(spec, str(device))
    assert _relative(_ours(spec, ops), _torchvision(spec, ops)) < 1e-4


@pytest.mark.parametrize("spec", CASES)
def test_gradients_match_torchvision(spec: dict, device: torch.device) -> None:
    """All five: input, offset, mask, weight and bias."""
    torch.manual_seed(0)
    ops = _operands(spec, str(device))

    grads = {}
    for label, run in (("theirs", _torchvision), ("ours", _ours)):
        leaves = {
            name: (
                ops[name].clone().requires_grad_(True)
                if ops[name] is not None
                else None
            )
            for name in ("input", "offset", "mask", "weight", "bias")
        }
        run(spec, {**ops, **leaves}).backward(ops["grad_out"])
        grads[label] = {
            n: (t.grad if t is not None else None) for n, t in leaves.items()
        }

    for name in ("input", "offset", "mask", "weight", "bias"):
        error = _relative(grads["ours"][name], grads["theirs"][name])
        assert error < 1e-4, f"gradient for {name} differs by {error:.2e}"


#: One geometry, exercised against every subset of gradients autograd can
#: ask for. Grouped and masked so no output is trivially absent.
_SUBSET_SPEC = {
    "groups": 2,
    "offset_groups": 2,
    "kernel": (3, 3),
    "stride": (1, 1),
    "padding": (1, 1),
    "dilation": (1, 1),
    "bias": True,
    "mask": True,
}

_ALL_LEAVES = ("input", "offset", "mask", "weight", "bias")


@pytest.mark.parametrize(
    "requested",
    [
        pytest.param(("input",), id="input-only"),
        pytest.param(("offset",), id="offset-only"),
        pytest.param(("mask",), id="mask-only"),
        pytest.param(("weight",), id="weight-only"),
        # The bias gradient is a host-side reduction, so no launch runs.
        pytest.param(("bias",), id="bias-only"),
        # A frozen weight, as in fine-tuning, takes the branch that skips its gradient.
        pytest.param(("input", "offset", "mask"), id="frozen-weight"),
        pytest.param(("input", "weight", "bias"), id="frozen-offset-and-mask"),
        pytest.param(_ALL_LEAVES, id="everything"),
    ],
)
def test_partial_gradients_match_torchvision(
    requested: tuple[str, ...], device: torch.device
) -> None:
    """Asking for a subset of gradients changes no answer and invents none.

    Outputs share intermediates in the specialized backward, so a condition on the
    wrong output gives wrong values without an error. Each subset is compared with
    torchvision's.
    """
    torch.manual_seed(0)
    ops = _operands(_SUBSET_SPEC, str(device))

    grads = {}
    for label, run in (("theirs", _torchvision), ("ours", _ours)):
        leaves = {
            name: (
                ops[name].clone().requires_grad_(name in requested)
                if ops[name] is not None
                else None
            )
            for name in _ALL_LEAVES
        }
        run(_SUBSET_SPEC, {**ops, **leaves}).backward(ops["grad_out"])
        grads[label] = {
            n: (t.grad if t is not None else None) for n, t in leaves.items()
        }

    for name in _ALL_LEAVES:
        if name in requested:
            assert grads["ours"][name] is not None, f"no gradient for requested {name}"
            error = _relative(grads["ours"][name], grads["theirs"][name])
            assert error < 1e-4, f"gradient for {name} differs by {error:.2e}"
        else:
            assert grads["ours"][name] is None, (
                f"produced a gradient for {name}, which was not asked for"
            )


def test_the_reference_matches_torchvision(device: torch.device) -> None:
    """The reference agrees, so it can replace an absent torchvision."""
    spec = CASES[2].values[0]
    torch.manual_seed(0)
    ops = _operands(spec, str(device))
    batch = ops["input"].shape[0]
    out_h, out_w = ops["out_hw"]
    kernel_h, kernel_w = spec["kernel"]
    points = kernel_h * kernel_w
    og = spec["offset_groups"]

    native_offset = (
        ops["offset"]
        .view(batch, og, points, 2, out_h, out_w)
        .permute(0, 4, 5, 1, 2, 3)
        .contiguous()
    )
    native_mask = (
        ops["mask"]
        .view(batch, og, points, out_h, out_w)
        .permute(0, 3, 4, 1, 2)
        .contiguous()
        if ops["mask"] is not None
        else None
    )
    got = conv_reference(
        ops["input"].permute(0, 2, 3, 1).contiguous(),
        native_offset,
        ops["weight"],
        ops["bias"],
        native_mask,
        *spec["stride"],
        *spec["padding"],
        *spec["dilation"],
        spec["groups"],
        og,
    ).permute(0, 3, 1, 2)
    assert _relative(got, _torchvision(spec, ops)) < 1e-4


@pytest.mark.parametrize("spec", CASES)
def test_our_precision_is_no_worse(spec: dict, device: torch.device) -> None:
    """Against float64, since torchvision and this kernel both run in float32."""
    if device.type != "cuda":
        pytest.skip("float64 oracle is only interesting against the device kernel")
    torch.manual_seed(0)
    ops = _operands(spec, str(device))
    exact = _torchvision(
        spec, {k: (v.double() if torch.is_tensor(v) else v) for k, v in ops.items()}
    ).double()
    scale = exact.abs().max().clamp_min(1e-12)

    theirs = ((_torchvision(spec, ops).double() - exact).abs().max() / scale).item()
    ours = ((_ours(spec, ops).double() - exact).abs().max() / scale).item()
    assert ours < 5e-6, f"absolute accuracy {ours:.2e} is worse than float32 should be"
    assert ours <= theirs * 4, (
        f"deformops error {ours:.2e} is disproportionately worse than torchvision's "
        f"{theirs:.2e} on the same operands"
    )


def test_v1_is_v2_with_unit_mask(device: torch.device) -> None:
    """No mask equals a mask of ones, which lets v1 and v2 share an implementation."""
    spec = dict(CASES[1].values[0])
    torch.manual_seed(0)
    ops = _operands(spec, str(device))
    without = _ours(spec, {**ops, "mask": None})
    with_ones = _ours(spec, {**ops, "mask": torch.ones_like(ops["mask"])})
    torch.testing.assert_close(without, with_ones, rtol=1e-5, atol=1e-5)


def test_zero_offset_is_a_plain_convolution(device: torch.device) -> None:
    """Zero offsets give a regular convolution, which checks the coordinate origin."""
    spec = dict(CASES[0].values[0])
    torch.manual_seed(0)
    ops = _operands(spec, str(device))
    out = _ours(spec, {**ops, "offset": torch.zeros_like(ops["offset"])})
    # cuDNN off: its convolution adds a second numerical difference, and its runtime
    # fails to load in some environments.
    previous = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    try:
        expected = torch.nn.functional.conv2d(
            ops["input"],
            ops["weight"],
            ops["bias"],
            spec["stride"],
            spec["padding"],
            spec["dilation"],
        )
    finally:
        torch.backends.cudnn.enabled = previous
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)


def test_channels_last_input_is_not_copied(device: torch.device) -> None:
    """The adapter is free for callers already using channels_last."""
    if device.type != "cuda":
        pytest.skip("layout conversion only matters on the device path")
    spec = dict(CASES[1].values[0])
    torch.manual_seed(0)
    ops = _operands(spec, str(device))
    contiguous = _ours(spec, ops)
    ops_cl = {**ops, "input": ops["input"].to(memory_format=torch.channels_last)}
    torch.testing.assert_close(_ours(spec, ops_cl), contiguous, rtol=1e-5, atol=1e-5)
