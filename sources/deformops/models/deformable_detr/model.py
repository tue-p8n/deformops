r"""Deformable DETR full graph, loaded from the Hugging Face hub.

:class:`~transformers.DeformableDetrForObjectDetection` with its encoder's
deformable attention routed through deformops; architecture, weights and heads
are untouched. Needs ``deformops[deformable-detr]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

from torch import nn

from .model_patches import patch_deformable_detr

if TYPE_CHECKING:
    from transformers import DeformableDetrForObjectDetection

    from deformops.registry import Backend, MSDABackend

try:
    from transformers import DeformableDetrForObjectDetection
except ModuleNotFoundError as exc:  # pragma: no cover - depends on install extras
    msg = (
        "deformops.models.deformable_detr requires the optional 'transformers' "
        "dependency. Install it with `pip install deformops[deformable-detr]`."
    )
    raise ModuleNotFoundError(msg) from exc

__all__ = ["DEFAULT_CHECKPOINT", "DeformableDETR"]

#: The published checkpoint used when none is named.
DEFAULT_CHECKPOINT = "SenseTime/deformable-detr"


class DeformableDETR(nn.Module):
    """Deformable DETR with a deformops deformable-attention encoder.

    Parameters
    ----------
    model
        A :class:`~transformers.DeformableDetrForObjectDetection`, patched in
        place.
    backend
        Backend for the attention layers; ``None`` selects one per call.
    """

    def __init__(
        self,
        model: DeformableDetrForObjectDetection,
        *,
        backend: Backend | MSDABackend | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.backend = backend
        patch_deformable_detr(self.model, backend=backend)

    @classmethod
    def from_pretrained(
        cls,
        ckpt: str = DEFAULT_CHECKPOINT,
        *,
        backend: Backend | MSDABackend | None = None,
        **kwargs: Any,
    ) -> DeformableDETR:
        """Load a checkpoint from the Hugging Face hub and patch it."""
        model = DeformableDetrForObjectDetection.from_pretrained(ckpt, **kwargs)
        return cls(model, backend=backend)

    @property
    def config(self) -> Any:
        """The wrapped Hugging Face config."""
        return self.model.config

    @override
    def forward(
        self,
        pixel_values: Any,
        pixel_mask: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Delegate to the wrapped graph, returning its output object."""
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, **kwargs)
