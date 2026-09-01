"""Detection of optional capabilities in the environment.

Distribution metadata does not record selected extras, and a CUDA torch brings
Triton without ``deformops[triton]``, so each capability is detected, not inferred.
"""

from __future__ import annotations

import functools
import importlib.util

__all__ = ["has_jit_toolchain", "has_triton", "require_jit_toolchain", "require_triton"]


@functools.cache
def has_triton() -> bool:
    """Whether the Triton compiler is importable in this environment."""
    return importlib.util.find_spec("triton") is not None


def require_triton(backend_name: str) -> None:
    """Raise if a Triton-backed kernel was selected without Triton installed.

    Parameters
    ----------
    backend_name
        The kernel the caller asked for, named in the error.

    Raises
    ------
    ImportError
        If Triton is not importable.
    """
    if has_triton():
        return
    msg = (
        f"The {backend_name!r} backend needs Triton, which is not installed. "
        f"Install it with `pip install deformops[triton]`, or select a "
        f"different backend. A CUDA build of torch provides Triton already; "
        f"a CPU build does not."
    )
    raise ImportError(msg)


@functools.cache
def has_jit_toolchain() -> bool:
    """Whether torch can compile an extension just-in-time here.

    Checks ninja only. Torch reports a missing CUDA toolchain clearly, but a missing
    ninja raises a bare ``RuntimeError`` from inside torch.
    """
    import torch.utils.cpp_extension  # noqa: PLC0415

    try:
        torch.utils.cpp_extension.verify_ninja_availability()
    except RuntimeError:
        return False
    return True


def require_jit_toolchain(extension_name: str) -> None:
    """Raise if an extension has to be compiled and the build tool is missing.

    Parameters
    ----------
    extension_name
        The extension being compiled, named in the error.

    Raises
    ------
    KernelBuildError
        If ninja, the build tool torch compiles extensions with, is missing.
    """
    if has_jit_toolchain():
        return
    from .errors import KernelBuildError  # noqa: PLC0415

    msg = (
        f"Compiling the {extension_name!r} kernel library needs ninja, which is "
        f"not installed. Install it with `pip install deformops[jit]`, or set "
        f"DEFORMOPS_COMPILE=0 and use a prebuilt library."
    )
    raise KernelBuildError(msg)
