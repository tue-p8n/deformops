"""Centralized configuration for Deformops extensions."""

import os
import pathlib
import re
import shutil
from collections.abc import Callable, Mapping
from typing import Any

__all__ = [
    "EXTENSIONS_ENV_VAR",
    "EXTENSION_CONFIGS",
    "EXT_PACKAGE",
    "MINIMUM_TORCH_ABI",
    "MSDA_OP_PREFIX",
    "SHARED_CFLAGS",
    "SHARED_CUDA_CFLAGS",
    "SKIPPED_BUILD_FILE",
    "find_cuda_toolkit",
    "parse_nvcc_release",
    "select_extensions",
]

# Package holding prebuilt ``.so`` files. setup.py, build.py and the Nix derivations
# must agree on it: a mismatch fails no build but loads no kernel (test_ext_path.py).
EXT_PACKAGE = "deformops.core._ext"

# Extension name, op-schema prefix and file stem of the default MSDA kernel. The
# single-scale and convolution operators have no extension: their device kernels are
# Triton.
MSDA_OP_PREFIX = "msda_fp32acc"

#: Every MSDA kernel configuration, in ``_schema.OP_PREFIXES`` order. Repeated here
#: because setup.py loads this module by path; the tests compare the two.
_MSDA_PREFIXES: tuple[str, ...] = (
    MSDA_OP_PREFIX,
    "msda_serialreduce",
    "msda_half2atomic",
    "msda_half2atomic_serialreduce",
    "msda_scalaratomic",
    "msda_scalaratomic_serialreduce",
)

#: Stable-ABI target: the oldest supported torch, so newer builds still load on it.
#: It also makes torch reject any header outside its stable surface.
MINIMUM_TORCH_ABI = "0x020B000000000000"

SHARED_CFLAGS = ["-std=c++20", f"-DTORCH_TARGET_VERSION={MINIMUM_TORCH_ABI}"]
SHARED_CUDA_CFLAGS = [
    "-std=c++20",
    "--use_fast_math",
    "-DUSE_CUDA",
    f"-DTORCH_TARGET_VERSION={MINIMUM_TORCH_ABI}",
]

#: Why a source build skipped its selected kernels, reported when the CUDA backend
#: finds no bundled library.
SKIPPED_BUILD_FILE = "build-skipped.txt"

# Torch's extension build hides the CUDA half-precision operators, which the FP16
# and BF16 kernels need.
_HALF_BF16_FLAGS = [
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
]


def _cuda_flags(*extras: str) -> list[str]:
    return [*SHARED_CUDA_CFLAGS, *_HALF_BF16_FLAGS, *extras]


def _entry(package: str, stem: str, *, experimental: bool) -> dict[str, Any]:
    return {
        "package": package,
        "stem": stem,
        "experimental": experimental,
        "cflags": SHARED_CFLAGS,
        "cflags_cuda": _cuda_flags(),
    }


#: One extension per MSDA configuration, keyed by operator prefix (its file stem).
#: Experimental ones are left out of ``all`` and of the prebuilt libraries.
EXTENSION_CONFIGS: dict[str, dict[str, Any]] = {
    prefix: _entry(
        "deformops.backends.cuda"
        if prefix == MSDA_OP_PREFIX
        else "deformops.backends.cuda.experimental",
        prefix,
        experimental=prefix != MSDA_OP_PREFIX,
    )
    for prefix in _MSDA_PREFIXES
}


#: Build-time environment variable naming the extensions ``setup.py`` compiles.
EXTENSIONS_ENV_VAR = "DEFORMOPS_EXTENSIONS"

#: Values of :data:`EXTENSIONS_ENV_VAR` that compile nothing.
_NO_EXTENSIONS = frozenset({"", "none", "0", "false", "off", "no"})


def select_extensions(
    value: str | None, *, source_distribution: bool
) -> dict[str, dict[str, Any]]:
    """The extensions a build compiles, from :data:`EXTENSIONS_ENV_VAR`.

    Parameters
    ----------
    value
        The variable's value, or ``None`` when it is unset. ``"all"`` selects
        every non-experimental extension, ``"everything"`` adds the
        experimental ones, ``"none"`` selects nothing, and a comma-separated
        list names extensions directly.
    source_distribution
        Whether the build runs from an unpacked source distribution. Unset,
        such a build compiles ``"all"``, and a build from a repository
        checkout compiles nothing.

    Returns
    -------
    The selected entries of :data:`EXTENSION_CONFIGS`, by name.

    Raises
    ------
    ValueError
        If the value names an extension that does not exist.
    """
    if value is None:
        value = "all" if source_distribution else "none"
    choice = value.strip()
    if choice.lower() in _NO_EXTENSIONS:
        return {}
    if choice.lower() == "everything":
        return dict(EXTENSION_CONFIGS)
    if choice.lower() == "all":
        return {
            name: config
            for name, config in EXTENSION_CONFIGS.items()
            if not config["experimental"]
        }
    names = [name.strip() for name in choice.split(",") if name.strip()]
    unknown = sorted(set(names) - set(EXTENSION_CONFIGS))
    if unknown:
        msg = (
            f"Unknown extension(s) in {EXTENSIONS_ENV_VAR}: {', '.join(unknown)}. "
            f"Available: {', '.join(EXTENSION_CONFIGS)}, or all, everything, none."
        )
        raise ValueError(msg)
    return {name: EXTENSION_CONFIGS[name] for name in names}


def find_cuda_toolkit(
    environ: Mapping[str, str] | None = None,
    which: Callable[[str], str | None] = shutil.which,
    default_root: str = "/usr/local/cuda",
) -> str | None:
    """The root of a CUDA toolkit that can compile the kernels, or ``None``.

    Looks where :mod:`torch.utils.cpp_extension` looks, in the same order:
    ``CUDA_HOME``, ``CUDA_PATH``, the directory above ``nvcc`` on ``PATH``, and
    ``/usr/local/cuda``. A root counts only when it holds ``bin/nvcc``, so a
    runtime-only CUDA install is not a toolkit.

    Parameters
    ----------
    environ
        The environment to read. Defaults to :data:`os.environ`.
    which
        Resolves an executable on ``PATH``.
    default_root
        The conventional install location, checked last.
    """
    env = os.environ if environ is None else environ
    root = env.get("CUDA_HOME") or env.get("CUDA_PATH")
    if not root:
        nvcc = which("nvcc")
        root = str(pathlib.Path(nvcc).parent.parent) if nvcc else default_root
    return root if (pathlib.Path(root) / "bin" / "nvcc").is_file() else None


def parse_nvcc_release(output: str) -> tuple[int, int] | None:
    """The ``(major, minor)`` CUDA release in ``nvcc --version`` output."""
    match = re.search(r"release (\d+)\.(\d+)", output)
    return (int(match.group(1)), int(match.group(2))) if match else None
