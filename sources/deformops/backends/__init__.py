# pyright: reportUnsupportedDunderAll=false
r"""Compute backends for the deformops operators.

``eager``, ``triton`` and ``cuda`` are the backends
:class:`~deformops.registry.Backend` names; ``reference`` holds the
implementations every backend is tested against and is not dispatchable.

``deformops.registry.msda.load(Backend.CUDA)`` and its ``attn`` and ``conv``
counterparts make a backend ready before its first call.
"""

from __future__ import annotations

from typing import Any

__all__ = ["cuda", "eager", "reference", "triton"]

LAZY_IMPORTS = frozenset(("cuda", "eager", "reference", "triton"))


def __getattr__(name: str) -> Any:
    if name in LAZY_IMPORTS:
        import importlib  # noqa: PLC0415

        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
