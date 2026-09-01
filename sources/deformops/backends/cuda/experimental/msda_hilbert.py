r"""MSDA with Hilbert query ordering.

Host-side Hilbert-curve reordering around the production kernel; see
``docs/kernels.md``.
"""

from .._msda_bridge import reference
from ..msda_fp32acc import forward as _kernel_forward
from ._query_order import install_reordering_family

__all__ = ["Backend", "forward", "reference"]

forward, Backend = install_reordering_family(
    curve="hilbert",
    backend_name="cuda_hilbert",
    base_forward=_kernel_forward,
    summary="Hilbert-curve query reordering",
)
