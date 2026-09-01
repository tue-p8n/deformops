r"""Deformable attention: output-pixel-tiled, input gradient in float32.

Avoids the 16-bit atomic add below SM 9.0; see ``docs/precision.md``. Registry
backend name: ``triton_query_tiled_fp32acc``.
"""

from ...core.capabilities import require_triton

require_triton(__name__.rsplit(".", 1)[-1])

from ._attn_bridge import install_attn_family

__all__ = ["Backend"]

_family = install_attn_family(
    op_prefix="attn_query_tiled_fp32acc",
    backend_name="triton_query_tiled_fp32acc",
    fp32acc=True,
    summary="output-pixel-tiled, input gradient in float32",
)

Backend = _family.backend
