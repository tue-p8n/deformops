r"""MSDA: point-parallel sampling, one program per sample.

The forward accumulates with atomics, so repeated calls can differ in the least
significant bits. Registry backend name: ``triton_point_parallel``.
"""

from ....core.capabilities import require_triton

require_triton(__name__.rsplit(".", 1)[-1])

from .._msda_bridge import install_msda_family
from ._kernels.msda_point_parallel import backward_launch, forward_launch

__all__ = ["Backend", "forward"]

_family = install_msda_family(
    op_prefix="msda_point_parallel",
    backend_name="triton_point_parallel",
    forward_launch=forward_launch,
    backward_launch=backward_launch,
    summary="one program per query, point, head and batch element",
)

forward = _family.forward
Backend = _family.backend
