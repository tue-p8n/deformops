"""``deformops.integrations.torchvision`` is a drop-in for ``torchvision.ops``.

It also agrees with the native entry point; numerical parity with torchvision is
in ``tests/deformops/test_conv_torchvision_parity.py``.
"""

from __future__ import annotations

import pytest
import torch
from deformops.integrations.torchvision import DeformConv2d, deform_conv2d
from deformops.registry import conv as apply

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


_IN, _OUT, _HEIGHT, _WIDTH, _BATCH = 32, 32, 11, 13, 2

#: The "v2-plain" case from the parity suite's argument sweep.
_SPEC = {
    "groups": 1,
    "offset_groups": 1,
    "kernel": (3, 3),
    "stride": (1, 1),
    "padding": (1, 1),
    "dilation": (1, 1),
    "bias": True,
    "mask": True,
}


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
        "out_hw": (out_h, out_w),
    }


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


def _relative(ours: torch.Tensor, theirs: torch.Tensor) -> float:
    scale = theirs.abs().max().clamp_min(1e-6)
    return ((ours - theirs).abs().max() / scale).item()


def test_the_module_matches_the_torchvision_module(device: torch.device) -> None:
    """Same constructor, same forward, same result for shared weights."""
    torch.manual_seed(0)
    ours = DeformConv2d(_IN, _OUT, 3, padding=1, groups=2).to(device)
    theirs = torchvision_ops.DeformConv2d(_IN, _OUT, 3, padding=1, groups=2).to(device)
    theirs.load_state_dict(ours.state_dict())

    x = torch.randn(_BATCH, _IN, _HEIGHT, _WIDTH, device=device)
    offset = torch.randn(_BATCH, 2 * 9, _HEIGHT, _WIDTH, device=device) * 0.5
    mask = torch.rand(_BATCH, 9, _HEIGHT, _WIDTH, device=device)
    assert _relative(ours(x, offset, mask), theirs(x, offset, mask)) < 1e-4


def test_native_entry_point_agrees_with_the_compat_wrapper(
    device: torch.device,
) -> None:
    """The two public entry points must not diverge on the same computation."""
    spec = dict(_SPEC)
    torch.manual_seed(0)
    ops = _operands(spec, str(device))
    batch = ops["input"].shape[0]
    out_h, out_w = ops["out_hw"]
    points = 9
    native = apply(
        ops["input"].permute(0, 2, 3, 1).contiguous(),
        ops["offset"]
        .view(batch, 1, points, 2, out_h, out_w)
        .permute(0, 4, 5, 1, 2, 3)
        .contiguous(),
        ops["weight"],
        ops["bias"],
        ops["mask"]
        .view(batch, 1, points, out_h, out_w)
        .permute(0, 3, 4, 1, 2)
        .contiguous(),
        stride=spec["stride"],
        padding=spec["padding"],
        dilation=spec["dilation"],
    ).permute(0, 3, 1, 2)
    torch.testing.assert_close(native, _ours(spec, ops), rtol=1e-5, atol=1e-5)
