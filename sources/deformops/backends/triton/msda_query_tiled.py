r"""MSDA: query-tiled sampling, value gradient accumulated in the input dtype.

The production Triton pick from SM 9.0 on; below it, reduced-precision training
uses :mod:`.msda_query_tiled_fp32acc`. Registry backend name:
``triton_query_tiled``.
"""

from ...core.capabilities import require_triton

require_triton(__name__.rsplit(".", 1)[-1])

from ._kernels.msda_query_tiled import atomic_backward_launch
from ._launch import query_tiled_forward_launch
from ._msda_bridge import install_msda_family

__all__ = ["Backend", "forward"]

_family = install_msda_family(
    op_prefix="msda_query_tiled",
    backend_name="triton_query_tiled",
    forward_launch=query_tiled_forward_launch,
    backward_launch=atomic_backward_launch,
    summary="query-tiled sampling, value gradient in the input dtype",
)

forward = _family.forward
Backend = _family.backend
