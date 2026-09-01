r"""DeformConv2d under the CUDA backend name.

Classic deformable convolution has no C++/CUDA kernel. ``Backend.CUDA`` aliases the
Triton kernel where Triton and a CUDA device exist, and otherwise raises
:class:`~deformops.core.errors.NoBackendError` naming what is missing.
"""

from typing import Any

import torch

from ...core.capabilities import has_triton
from ...registry import Backend, conv


def conv_cuda(*args: Any, **kwargs: Any) -> Any:
    """Execute DeformConv2d forward through the Triton kernel."""
    from ..triton.conv import conv_triton  # noqa: PLC0415

    return conv_triton(*args, **kwargs)


if has_triton() and torch.cuda.is_available():

    def _supports(context: Any) -> bool:
        """The Triton kernel's constraints: CUDA only, and no float64 path."""
        return context.device.type == "cuda" and context.dtype != torch.float64

    conv.register(Backend.CUDA, conv_cuda, supports=_supports)
else:
    conv.mark_unavailable(
        Backend.CUDA,
        "it runs the Triton kernel, which needs Triton and a CUDA device",
    )
