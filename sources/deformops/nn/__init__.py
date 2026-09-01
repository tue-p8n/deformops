r"""Neural network building blocks for PyTorch.

The ``torch.nn.Module`` layers:

- ``MultiScaleDeformAttn2d`` (multi-scale deformable attention),
- ``DeformConv2d`` (classic deformable convolution v1/v2),
- ``DeformAttn2d`` (single-scale deformable sampling layer),
- and ``DeformSample2d`` (single-scale depthwise offset sampler).
"""

from __future__ import annotations

from .attn import DeformAttn2d, DeformSample2d
from .conv import DeformConv2d
from .msda import MSDAMethod, MultiScaleDeformAttn2d

__all__ = [
    "DeformAttn2d",
    "DeformConv2d",
    "DeformSample2d",
    "MSDAMethod",
    "MultiScaleDeformAttn2d",
]
