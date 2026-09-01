r"""Mask2Former and the two Hugging Face modules deformops overrides.

Graph in ``model``, overrides in ``model_patches``, re-exported lazily. Needs
``deformops[mask2former]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .model import DEFAULT_CHECKPOINT, Mask2Former
    from .model_patches import (
        PatchedMask2FormerHungarianMatcher,
        PatchedMask2FormerLoss,
        PatchedMask2FormerMaskPredictor,
        PatchedMask2FormerPixelDecoderEncoderMultiscaleDeformableAttention,
        patch_mask2former,
    )

__all__ = [
    "DEFAULT_CHECKPOINT",
    "Mask2Former",
    "PatchedMask2FormerHungarianMatcher",
    "PatchedMask2FormerLoss",
    "PatchedMask2FormerMaskPredictor",
    "PatchedMask2FormerPixelDecoderEncoderMultiscaleDeformableAttention",
    "patch_mask2former",
]

_LAZY_IMPORTS = {
    "DEFAULT_CHECKPOINT": ".model",
    "Mask2Former": ".model",
    "PatchedMask2FormerHungarianMatcher": ".model_patches",
    "PatchedMask2FormerLoss": ".model_patches",
    "PatchedMask2FormerMaskPredictor": ".model_patches",
    "PatchedMask2FormerPixelDecoderEncoderMultiscaleDeformableAttention": (
        ".model_patches"
    ),
    "patch_mask2former": ".model_patches",
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
