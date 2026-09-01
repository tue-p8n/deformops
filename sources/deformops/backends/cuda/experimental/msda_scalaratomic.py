r"""MSDA: native dtype with unpacked atomics, warp-shuffle softmax reduction.

See ``docs/kernels.md`` for the traits.
"""

from .._msda_bridge import install_msda_family, reference

__all__ = ["Backend", "forward", "reference", "load"]

_family = install_msda_family(
    op_prefix="msda_scalaratomic",
    backend_name="cuda_scalaratomic",
    upcast_backward_inputs=False,
    summary="native dtype with unpacked atomics, warp-shuffle softmax reduction",
)

forward = _family.forward
Backend = _family.backend
load = _family.load
