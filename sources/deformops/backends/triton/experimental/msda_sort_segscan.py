r"""MSDA: sort and segment scan, default forward tiling.

Registry backend name: ``triton_sort_segscan``.
"""

from ....core.capabilities import require_triton

require_triton(__name__.rsplit(".", 1)[-1])

import functools

from .._launch import DEFAULT_TILES, query_tiled_forward_launch
from .._msda_bridge import install_msda_family
from ._kernels.msda_reduce import sort_segscan_backward_launch

__all__ = ["Backend", "forward"]

_family = install_msda_family(
    op_prefix="msda_sort_segscan",
    backend_name="triton_sort_segscan",
    forward_launch=functools.partial(query_tiled_forward_launch, tiles=DEFAULT_TILES),
    backward_launch=sort_segscan_backward_launch,
    summary="sort and segment scan, default forward tiling",
)

forward = _family.forward
Backend = _family.backend
