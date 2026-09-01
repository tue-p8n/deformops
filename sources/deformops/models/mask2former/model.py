r"""Mask2Former full graph, loaded from the Hugging Face hub.

:class:`~transformers.Mask2FormerForUniversalSegmentation` with the
:mod:`~deformops.models.mask2former.model_patches` modules swapped in;
architecture, weights and post-processing are untouched. The patches avoid what
breaks ``torch.export``, so the graph exports; see ``docs/export.md``. Needs
``deformops[mask2former]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

from torch import nn

from .model_patches import patch_mask2former

if TYPE_CHECKING:
    from transformers import Mask2FormerForUniversalSegmentation

    from deformops.registry import Backend, MSDABackend

try:
    from transformers import Mask2FormerForUniversalSegmentation
except ModuleNotFoundError as exc:  # pragma: no cover - depends on install extras
    msg = (
        "deformops.models.mask2former requires the optional 'transformers' "
        "dependency. Install it with `pip install deformops[mask2former]`."
    )
    raise ModuleNotFoundError(msg) from exc

__all__ = ["DEFAULT_CHECKPOINT", "Mask2Former"]

#: Default checkpoint; the other ``facebook/mask2former-swin-*`` sizes and
#: tasks load the same way.
DEFAULT_CHECKPOINT = "facebook/mask2former-swin-base-coco-instance"


class Mask2Former(nn.Module):
    """Mask2Former with the deformops pixel decoder.

    Parameters
    ----------
    model
        A :class:`~transformers.Mask2FormerForUniversalSegmentation`, patched
        in place.
    backend
        Backend for the attention layers; ``None`` selects one per call.
    """

    def __init__(
        self,
        model: Mask2FormerForUniversalSegmentation,
        *,
        backend: Backend | MSDABackend | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.backend = backend
        patch_mask2former(self.model, backend=backend)

    @classmethod
    def from_pretrained(
        cls,
        ckpt: str = DEFAULT_CHECKPOINT,
        *,
        backend: Backend | MSDABackend | None = None,
        **kwargs: Any,
    ) -> Mask2Former:
        """Load a checkpoint from the Hugging Face hub and patch it.

        Parameters
        ----------
        ckpt
            A Hugging Face model id, e.g.
            ``"facebook/mask2former-swin-base-coco-instance"``.
        backend
            Backend forwarded to the deformable attention layers.
        **kwargs
            Forwarded to ``from_pretrained`` (``revision``, ``cache_dir``,
            ``token``, ``dtype``, ...).

        Returns
        -------
        The patched model, ready to run or export.
        """
        model = Mask2FormerForUniversalSegmentation.from_pretrained(ckpt, **kwargs)
        return cls(model, backend=backend)

    @property
    def config(self) -> Any:
        """The wrapped Hugging Face config."""
        return self.model.config

    @property
    def pixel_decoder(self) -> nn.Module:
        """The pixel decoder whose attention runs on deformops."""
        return self.model.model.pixel_level_module.decoder

    @override
    def forward(self, pixel_values: Any, **kwargs: Any) -> Any:
        """Delegate to the wrapped graph, returning its output object."""
        return self.model(pixel_values, **kwargs)
