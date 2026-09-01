r"""MSDA with Morton query ordering.

Host-side Z-order reordering around the production kernel; see
``docs/kernels.md``.
"""

from .._msda_bridge import reference
from ..msda_fp32acc import forward as _kernel_forward
from ._query_order import install_reordering_family

__all__ = ["Backend", "forward", "reference"]

forward, Backend = install_reordering_family(
    curve="morton",
    backend_name="cuda_morton",
    base_forward=_kernel_forward,
    summary="Z-order (Morton) query reordering",
)
