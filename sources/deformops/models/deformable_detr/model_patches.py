r"""Deformable DETR's deformops override.

Re-exports the parameter-free attention core of
:mod:`deformops.integrations.transformers` under the ``Patched`` name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from deformops.integrations import transformers as _transformers

if TYPE_CHECKING:
    from deformops.registry import Backend, MSDABackend

#: The integration's core. Needs ``deformops[deformable-detr]``.
PatchedMultiScaleDeformableAttention = _transformers.MultiScaleDeformableAttention

__all__ = [
    "PatchedMultiScaleDeformableAttention",
    "patch_deformable_detr",
]


def patch_deformable_detr(
    model: nn.Module,
    *,
    backend: Backend | MSDABackend | None = None,
    strict: bool = True,
) -> nn.Module:
    """Route ``model``'s multi-scale deformable attention through deformops.

    Parameters
    ----------
    model
        A Hugging Face Deformable DETR model, patched in place.
    backend
        Backend for every replacement; ``None`` selects one per call.
    strict
        Raise when no attention module was replaced.

    Returns
    -------
    ``model``, for chaining.
    """
    return _transformers.patch(model, backend=backend, strict=strict)
