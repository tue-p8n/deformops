r"""MSDA: fp32 accumulation, serial softmax reduction.

See ``docs/kernels.md`` for the traits.
"""

from .._msda_bridge import install_msda_family, reference

__all__ = ["Backend", "forward", "reference", "load"]

_family = install_msda_family(
    op_prefix="msda_serialreduce",
    backend_name="cuda_serialreduce",
    upcast_backward_inputs=True,
    summary="fp32 accumulation, serial softmax reduction",
)

forward = _family.forward
Backend = _family.backend
load = _family.load
