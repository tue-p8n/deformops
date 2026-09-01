r"""MSDA: query-tiled sampling, value gradient accumulated in float32.

The production pick for reduced-precision training below SM 9.0, at twice the
memory of the value-gradient buffer; see ``docs/precision.md``. Registry backend
name: ``triton_query_tiled_fp32acc``.
"""

from ...core.capabilities import require_triton

require_triton(__name__.rsplit(".", 1)[-1])

from ._kernels.msda_query_tiled import float32_backward_launch
from ._launch import query_tiled_forward_launch
from ._msda_bridge import install_msda_family

__all__ = ["Backend", "forward"]

_family = install_msda_family(
    op_prefix="msda_query_tiled_fp32acc",
    backend_name="triton_query_tiled_fp32acc",
    forward_launch=query_tiled_forward_launch,
    backward_launch=float32_backward_launch,
    summary="query-tiled sampling, value gradient in float32",
)

forward = _family.forward
Backend = _family.backend
