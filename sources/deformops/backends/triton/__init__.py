# pyright: reportUnsupportedDunderAll=false
r"""The Triton backend: kernels compiled on first use for float16, bfloat16 and float32.

The adapters :mod:`.msda`, :mod:`.attn` and :mod:`.conv` register only where
Triton imports and a CUDA device exists, and load lazily. Each ``msda_*`` or
``attn_*`` module is one kernel configuration exporting ``Backend``; the
``@triton.jit`` kernels live in :mod:`._kernels`.
"""

from typing import Any

from deformops.core.capabilities import require_triton

__all__ = ["attn", "conv", "msda"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    require_triton(name)
    import importlib  # noqa: PLC0415

    module = importlib.import_module(f".{name}", __name__)
    globals()[name] = module
    return module
