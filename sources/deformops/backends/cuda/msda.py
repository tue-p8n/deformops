r"""MSDA production C++/CUDA kernel backend."""

from typing import Any

import torch

from ...registry import Backend, msda
from ...registry.msda import Method, resolve_method
from ._msda_bridge import supports_context
from .msda_fp32acc import forward as msda_cuda_forward
from .msda_fp32acc import load


def msda_cuda(
    value: Any,
    spatial_shapes: Any,
    level_start_index: Any,
    sampling_locations: Any,
    attention_weights: Any,
    *,
    chunk_step: int = 128,
    num_points: int | None = None,
    channels_per_thread: int = 0,
    threads_per_block: int = 0,
    method: Any = None,
    softmax: bool = False,
    **kwargs: Any,
) -> Any:
    """Run the MSDA forward on the CUDA kernel."""
    softmax = resolve_method(method, softmax=softmax) is Method.SOFTMAX
    # ``num_points`` selects the template; a mismatch silently reads 8 points as 4.
    if num_points is None:
        num_points = sampling_locations.shape[4]
    return msda_cuda_forward(
        value,
        spatial_shapes,
        level_start_index,
        sampling_locations,
        attention_weights,
        chunk_step=chunk_step,
        num_points=num_points,
        channels_per_thread=channels_per_thread,
        threads_per_block=threads_per_block,
        softmax=softmax,
    )


if torch.cuda.is_available():
    msda.register(Backend.CUDA, msda_cuda, supports=supports_context, load=load)
else:
    msda.mark_unavailable(Backend.CUDA, "no CUDA device is available")
