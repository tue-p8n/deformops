r"""MSDA: cached softmax statistics, value gradient in float32.

Registry backend name: ``triton_query_tiled_cached_fp32acc``.
"""

from ....core.capabilities import require_triton

require_triton(__name__.rsplit(".", 1)[-1])

from .._msda_bridge import install_msda_family
from ._kernels.msda_query_tiled_cached import float32_backward_launch, forward_launch

__all__ = ["Backend", "forward"]

_family = install_msda_family(
    op_prefix="msda_query_tiled_cached_fp32acc",
    backend_name="triton_query_tiled_cached_fp32acc",
    forward_launch=forward_launch,
    backward_launch=float32_backward_launch,
    summary="cached softmax statistics, value gradient in float32",
    caches_softmax_statistics=True,
)

forward = _family.forward
Backend = _family.backend
