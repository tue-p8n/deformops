r"""Deformable DETR and the deformops deformable-attention core.

Graph in ``model``, override in ``model_patches``, re-exported lazily. Needs
``deformops[deformable-detr]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .model import DEFAULT_CHECKPOINT, DeformableDETR
    from .model_patches import (
        PatchedMultiScaleDeformableAttention,
        patch_deformable_detr,
    )

__all__ = [
    "DEFAULT_CHECKPOINT",
    "DeformableDETR",
    "PatchedMultiScaleDeformableAttention",
    "patch_deformable_detr",
]

_LAZY_IMPORTS = {
    "DEFAULT_CHECKPOINT": ".model",
    "DeformableDETR": ".model",
    "PatchedMultiScaleDeformableAttention": ".model_patches",
    "patch_deformable_detr": ".model_patches",
}


def __getattr__(name: str) -> Any:
    """Resolve the graph and override exports without importing both."""
    if name in _LAZY_IMPORTS:
        import importlib  # noqa: PLC0415

        module = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
