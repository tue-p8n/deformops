r"""DeformAttn under the CUDA backend name.

Single-scale deformable sampling has no C++/CUDA kernel. ``Backend.CUDA`` aliases the
Triton kernel where Triton and a CUDA device exist, and otherwise raises
:class:`~deformops.core.errors.NoBackendError` naming what is missing.
"""

from typing import Any

import torch

from ...core.capabilities import has_triton
from ...registry import Backend, attn


def attn_cuda(*args: Any, **kwargs: Any) -> Any:
    """Execute DeformAttn forward through the Triton kernel."""
    from ..triton.attn import attn_triton  # noqa: PLC0415

    return attn_triton(*args, **kwargs)


if has_triton() and torch.cuda.is_available():

    def _supports(context: Any) -> bool:
        """The Triton kernel's constraints: CUDA only, and no float64 path."""
        return context.device.type == "cuda" and context.dtype != torch.float64

    attn.register(Backend.CUDA, attn_cuda, supports=_supports)
else:
    attn.mark_unavailable(
        Backend.CUDA,
        "it runs the Triton kernel, which needs Triton and a CUDA device",
    )
