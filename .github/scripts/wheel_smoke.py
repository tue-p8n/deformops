"""Smoke-test an installed deformops wheel from outside the source tree.

Run against a clean virtual environment that has the built wheel and nothing
else from this repository on ``sys.path``. It catches a layout change that
builds cleanly but produces an unusable distribution, which the test suite
cannot detect because it does not run against an installed wheel.

Takes the expected version as its only argument, so a wheel built from a stale
tree fails here rather than passing under whatever version it happens to carry.

The script runs these checks in order, because the first one is only meaningful
before anything else has been imported.

1. Importing the top-level package pulls in no torch. That keeps
   ``import deformops`` cheap, and it depends on the package layout.
2. The version matches.
3. The package data ships: the CUDA kernel sources and shared headers the
   just-in-time build compiles, and the typing markers. A wheel missing them
   imports cleanly and fails only when a kernel first has to be compiled or a
   type checker reads the package.
4. Every public subpackage resolves, and the layer and dispatch surfaces
   expose what they advertise.
5. A real multi-scale deformable attention forward runs on the CPU through
   the registry. Importability alone would not catch a lazy import path that
   resolves to a module the wheel does not ship.
"""

import sys

_EXPECTED_LAYERS = (
    "DeformAttn2d",
    "DeformConv2d",
    "DeformSample2d",
    "MultiScaleDeformAttn2d",
)

_EXPECTED_REGISTRY = ("Backend", "attn", "conv", "msda")

#: Glob patterns, relative to the installed package, that must each match at
#: least one file.
_EXPECTED_PACKAGE_DATA = (
    "backends/cuda/*.cu",
    "backends/cuda/*.cpp",
    "backends/cuda/include/*.cuh",
    "backends/cuda/experimental/*.cu",
    "backends/cuda/experimental/*.cpp",
    "py.typed",
    "__init__.pyi",
)


def check_package_data() -> None:
    from importlib.resources import files
    from pathlib import Path

    root = Path(str(files("deformops")))
    missing = [
        pattern for pattern in _EXPECTED_PACKAGE_DATA if not any(root.glob(pattern))
    ]
    if missing:
        msg = f"installed package at {root} is missing {', '.join(missing)}"
        raise SystemExit(msg)
    print(f"package data: {', '.join(_EXPECTED_PACKAGE_DATA)}")


def main(expected_version: str) -> None:
    import deformops

    if "torch" in sys.modules:
        msg = (
            "importing deformops pulled in torch; the top-level package must "
            "stay import-cheap"
        )
        raise SystemExit(msg)

    if deformops.__version__ != expected_version:
        msg = (
            f"installed version {deformops.__version__!r} does not match the "
            f"expected {expected_version!r}"
        )
        raise SystemExit(msg)
    print(f"version: {deformops.__version__}")

    check_package_data()

    from deformops import nn, registry

    for name in _EXPECTED_LAYERS:
        getattr(nn, name)
    for name in _EXPECTED_REGISTRY:
        getattr(registry, name)
    print(f"layers: {', '.join(_EXPECTED_LAYERS)}")
    print(f"registry: {', '.join(_EXPECTED_REGISTRY)}")

    import torch

    torch.manual_seed(0)
    levels = [(8, 12), (4, 6)]
    spatial = sum(h * w for h, w in levels)
    heads, channels, points = 2, 8, 4
    layer = nn.MultiScaleDeformAttn2d(
        dim=heads * channels,
        num_heads=heads,
        num_levels=len(levels),
        num_points=points,
        backend=registry.Backend.EAGER,
    )
    query = torch.randn(1, 5, heads * channels)
    reference_points = torch.rand(1, 5, len(levels), 2)
    value = torch.randn(1, spatial, heads * channels)
    shapes = torch.tensor(levels, dtype=torch.long)
    level_index = torch.tensor([0, levels[0][0] * levels[0][1]], dtype=torch.long)
    out = layer(query, reference_points, value, shapes, level_index)

    if out.shape != query.shape:
        msg = f"forward returned {tuple(out.shape)}, expected {tuple(query.shape)}"
        raise SystemExit(msg)
    if not torch.isfinite(out).all():
        msg = "forward returned non-finite values"
        raise SystemExit(msg)
    print(f"eager forward: {tuple(out.shape)} finite")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        usage = "usage: wheel_smoke.py <expected-version>"
        raise SystemExit(usage)
    main(sys.argv[1])
