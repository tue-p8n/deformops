r"""MSDA Triton kernel backend."""

from typing import Any

import torch

from deformops.core.capabilities import has_triton
from deformops.registry import Backend, msda


def msda_triton(
    value: Any,
    spatial_shapes: Any,
    level_start_index: Any,
    sampling_locations: Any,
    attention_weights: Any,
    *,
    chunk_step: int = 128,
    fp32acc: bool = True,
    method: Any = None,
    softmax: bool = False,
    **kwargs: Any,
) -> Any:
    """Execute MSDA forward using the Triton query-tiled kernel.

    ``fp32acc`` picks the configuration module. The selection policy sets it
    per architecture; an explicit call to this backend defaults to the float32
    accumulator.
    """
    from deformops.registry.msda import resolve_method  # noqa: PLC0415

    if fp32acc:
        from .msda_query_tiled_fp32acc import (  # noqa: PLC0415
            Backend as TritonMSDABackend,
        )
    else:
        from .msda_query_tiled import Backend as TritonMSDABackend  # noqa: PLC0415

    method = resolve_method(method, softmax=softmax)
    return TritonMSDABackend().forward(
        value,
        spatial_shapes,
        level_start_index,
        sampling_locations,
        attention_weights,
        chunk_step,
        method=method,
    )


def _triton_unavailable_reason() -> str:
    if not has_triton():
        return (
            "Triton is not installed; install deformops[triton] or a CUDA build "
            "of torch, which includes it"
        )
    return "no CUDA device is available"


if has_triton() and torch.cuda.is_available():

    def _supports(context: Any) -> bool:
        """A CUDA tensor on SM 7.0 or newer, not float64."""
        return (
            context.device.type == "cuda"
            and context.dtype != torch.float64
            and context.sm_version >= (7, 0)
        )

    msda.register(Backend.TRITON, msda_triton, supports=_supports)
else:
    msda.mark_unavailable(Backend.TRITON, _triton_unavailable_reason())
