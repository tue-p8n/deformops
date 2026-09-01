r"""Per-operator dispatch: contexts, backend protocols, policies and registries.

Import through this package, the public surface::

    from deformops.registry import Backend, msda, attn, conv

``Backend`` has no member for the reference implementations; see
``docs/dispatch.md``.
"""

from __future__ import annotations

from ..core.runtime import resolve_method
from ._dispatch import Backend, LoadedBackend, Registry, Resolution
from .attn import AttnBackend, AttnContext, AttnMethod, attn
from .conv import ConvBackend, ConvContext, conv
from .msda import (
    MSDABackend,
    MSDAContext,
    MSDAMethod,
    apply_msda_method,
    msda,
    msda_method_to_aggregate,
    resolve_msda_attn,
)

__all__ = [
    "Backend",
    "MSDABackend",
    "MSDAContext",
    "MSDAMethod",
    "AttnBackend",
    "AttnContext",
    "AttnMethod",
    "ConvBackend",
    "ConvContext",
    "LoadedBackend",
    "Registry",
    "Resolution",
    "apply_msda_method",
    "attn",
    "conv",
    "msda_method_to_aggregate",
    "msda",
    "resolve_msda_attn",
    "resolve_method",
]
