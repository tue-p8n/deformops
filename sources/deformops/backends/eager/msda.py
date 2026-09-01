r"""Registry adapter for the eager MSDA backend."""

from typing import Any

from ...registry import Backend, msda
from ...registry.msda import resolve_method
from .msda_pytorch import Backend as PyTorchMSDABackend

_backend_impl = PyTorchMSDABackend()


def msda_eager(
    value: Any,
    spatial_shapes: Any,
    level_start_index: Any,
    sampling_locations: Any,
    attention_weights: Any,
    *,
    chunk_step: int = 128,
    method: Any = None,
    softmax: bool = False,
    **kwargs: Any,
) -> Any:
    """Run the MSDA forward with PyTorch operators."""
    return _backend_impl.forward(
        value,
        spatial_shapes,
        level_start_index,
        sampling_locations,
        attention_weights,
        chunk_step,
        resolve_method(method, softmax=softmax),
    )


msda.register(Backend.EAGER, msda_eager, supports=lambda _context: True)
