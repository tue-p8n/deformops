r"""MSDA: scattered atomics, wide forward tiling.

Registry backend name: ``triton_atomic_wide``.
"""

from ....core.capabilities import require_triton

require_triton(__name__.rsplit(".", 1)[-1])

import functools

from .._kernels.msda_query_tiled import atomic_backward_launch
from .._launch import WIDE_TILES, query_tiled_forward_launch
from .._msda_bridge import install_msda_family

__all__ = ["Backend", "forward"]

_family = install_msda_family(
    op_prefix="msda_atomic_wide",
    backend_name="triton_atomic_wide",
    forward_launch=functools.partial(query_tiled_forward_launch, tiles=WIDE_TILES),
    backward_launch=atomic_backward_launch,
    summary="scattered atomics, wide forward tiling",
)

forward = _family.forward
Backend = _family.backend
