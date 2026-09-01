r"""Deformable sampling operators for PyTorch.

Single-scale deformable sampling, deformable convolution and multi-scale
deformable attention, as :mod:`deformops.nn` layers or :mod:`deformops.registry`
functions, on the backends in :mod:`deformops.backends`.
:mod:`deformops.integrations` and :mod:`deformops.models` adapt other packages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from . import backends, models, nn, registry, version

    __version__: str

__all__ = [
    "__version__",
    "backends",
    "models",
    "nn",
    "registry",
    "version",
]

LAZY_IMPORTS = frozenset(("version", "backends", "models", "nn", "registry"))


def __getattr__(name: str) -> Any:
    """Resolve attributes or submodules lazily."""
    if name == "__version__":
        from .version import package as value  # noqa: PLC0415

        globals()["__version__"] = value
        return value
    if name in LAZY_IMPORTS:
        import importlib  # noqa: PLC0415

        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
