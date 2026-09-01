r"""MSDA: block-scoped deduplication, default forward tiling.

Registry backend name: ``triton_block_reduce``.
"""

from ....core.capabilities import require_triton

require_triton(__name__.rsplit(".", 1)[-1])

import functools

from .._launch import DEFAULT_TILES, query_tiled_forward_launch
from .._msda_bridge import install_msda_family
from ._kernels.msda_reduce import BLOCK_SCOPE, reduce_backward_launch

__all__ = ["Backend", "forward"]

_family = install_msda_family(
    op_prefix="msda_block_reduce",
    backend_name="triton_block_reduce",
    forward_launch=functools.partial(query_tiled_forward_launch, tiles=DEFAULT_TILES),
    backward_launch=functools.partial(reduce_backward_launch, scope=BLOCK_SCOPE),
    summary="block-scoped deduplication, default forward tiling",
)

forward = _family.forward
Backend = _family.backend
