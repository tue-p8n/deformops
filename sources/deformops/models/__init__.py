r"""Pre-built model graphs that exercise the deformops operators.

Each model package keeps its full graph and its deformops components in
separate modules, re-exported lazily, so importing a component does not import
the graph. A model imported without its extra raises a message naming it::

    from deformops.models import Mask2Former
    from deformops.models.mask2former import (
        PatchedMask2FormerPixelDecoderEncoderMultiscaleDeformableAttention,
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .deformable_detr import DeformableDETR
    from .mask2former import Mask2Former

__all__ = [
    "DeformableDETR",
    "Mask2Former",
]

_LAZY_IMPORTS = {
    "DeformableDETR": ".deformable_detr",
    "Mask2Former": ".mask2former",
}


def __getattr__(name: str) -> Any:
    """Resolve a full-graph export without importing its model package."""
    if name in _LAZY_IMPORTS:
        import importlib  # noqa: PLC0415

        module = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
