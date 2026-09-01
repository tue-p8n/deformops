r"""The batch fold behind every Triton vmap rule, shared with the public operators."""

from __future__ import annotations

from ...registry._vmap import fold_into_batch, make_vmap_rule, unfold_from_batch

__all__ = ["fold_into_batch", "make_vmap_rule", "unfold_from_batch"]
