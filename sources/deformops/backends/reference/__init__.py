r"""Reference implementations.

Every backend is tested against these. The registry cannot select them; import
one directly::

    from deformops.backends.reference.attn import reference as attn_reference
    from deformops.backends.reference.conv import reference as conv_reference
    from deformops.backends.reference.msda import reference as msda_reference

A reference states the math; ``eager`` optimizes it.
"""

from __future__ import annotations

from . import attn, conv, msda

__all__ = ["attn", "conv", "msda"]
