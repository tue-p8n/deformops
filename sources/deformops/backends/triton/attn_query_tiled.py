r"""Deformable attention: output-pixel-tiled, input gradient in the input dtype.

Registry backend name: ``triton_query_tiled``.
"""

from ...core.capabilities import require_triton

require_triton(__name__.rsplit(".", 1)[-1])

from ._attn_bridge import install_attn_family

__all__ = ["Backend"]

_family = install_attn_family(
    op_prefix="attn_query_tiled",
    backend_name="triton_query_tiled",
    fp32acc=False,
    summary="output-pixel-tiled, input gradient in the input dtype",
)

Backend = _family.backend
