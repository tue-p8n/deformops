r"""MSDA: cached softmax statistics, value gradient in the input dtype.

The softmax operators return and take the running max and sum-of-exp, two
``(B, Q, H)`` float32 tensors. Registry backend name:
``triton_query_tiled_cached``.
"""

from ....core.capabilities import require_triton

require_triton(__name__.rsplit(".", 1)[-1])

from .._msda_bridge import install_msda_family
from ._kernels.msda_query_tiled_cached import atomic_backward_launch, forward_launch

__all__ = ["Backend", "forward"]

_family = install_msda_family(
    op_prefix="msda_query_tiled_cached",
    backend_name="triton_query_tiled_cached",
    forward_launch=forward_launch,
    backward_launch=atomic_backward_launch,
    summary="cached softmax statistics, value gradient in the input dtype",
    caches_softmax_statistics=True,
)

forward = _family.forward
Backend = _family.backend
