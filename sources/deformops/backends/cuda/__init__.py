# pyright: reportUnsupportedDunderAll=false
r"""The CUDA backend: compiled multi-scale deformable attention kernels.

:mod:`.msda_fp32acc` is the kernel ``Backend.CUDA`` runs; :mod:`.experimental`
holds the configurations kept for comparison. Single-scale sampling and
deformable convolution have no compiled kernel, so :mod:`.attn` and :mod:`.conv`
run the Triton kernels under the CUDA backend name.
``deformops.registry.msda.load(Backend.CUDA)`` obtains the compiled library.
"""

from typing import Any

__all__ = ["attn", "conv", "experimental", "msda", "msda_fp32acc"]

_SUBMODULES: tuple[str, ...] = (
    "_msda_bridge",
    "_schema",
    "attn",
    "conv",
    "experimental",
    "msda",
    "msda_fp32acc",
)


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        import importlib  # noqa: PLC0415

        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
