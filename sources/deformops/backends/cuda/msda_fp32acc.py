r"""MSDA: fp32 accumulation, warp-shuffle softmax reduction.

The configuration ``Backend.CUDA`` runs. See ``docs/kernels.md`` for the traits.
"""

from ._msda_bridge import install_msda_family, reference

__all__ = ["Backend", "forward", "reference", "load"]

_family = install_msda_family(
    op_prefix="msda_fp32acc",
    backend_name="cuda",
    upcast_backward_inputs=True,
    summary="fp32 accumulation, warp-shuffle softmax reduction",
)

forward = _family.forward
Backend = _family.backend
load = _family.load
