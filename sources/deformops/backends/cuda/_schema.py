r"""Operator schemas for the multi-scale deformable attention CUDA kernels.

Defined in Python so op handles and autograd exist before any ``.so`` loads;
each ``.cu`` registers its implementation against them, so
:func:`define_schemas` runs first. Not imported by ``deformops.backends.cuda``,
which keeps ``import deformops`` torch-free. See ``docs/export.md``.
"""

from __future__ import annotations

import torch.library

__all__ = ["OP_PREFIXES", "define_schemas"]

_NAMESPACE = "deformops_cuda"

#: One signature for every configuration: they differ in traits, not arguments.
_FORWARD_SCHEMA = (
    "(Tensor value, Tensor spatial_shapes, "
    "Tensor level_start_index, Tensor loc, Tensor attn, "
    "int chunk_step, int K, int channels_per_thread, "
    "int threads_per_block) -> Tensor"
)
_BACKWARD_SCHEMA = (
    "(Tensor value, Tensor spatial_shapes, "
    "Tensor level_start_index, Tensor loc, Tensor attn, Tensor grad_output, "
    "int chunk_step, int K, int channels_per_thread, int threads_per_block) -> "
    "(Tensor, Tensor, Tensor)"
)

#: One prefix per translation unit, and the ``EXTENSION_CONFIGS`` keys derived
#: from it. ``msda_fp32acc`` is production; the others name their differing traits.
OP_PREFIXES: tuple[str, ...] = (
    "msda_fp32acc",
    "msda_serialreduce",
    "msda_half2atomic",
    "msda_half2atomic_serialreduce",
    "msda_scalaratomic",
    "msda_scalaratomic_serialreduce",
)


def _schemas() -> tuple[tuple[str, str], ...]:
    """(operator name, schema) for every registered operator.

    Names exclude the namespace. Each prefix contributes a linear and a softmax
    forward/backward pair.
    """
    return tuple(
        (f"{prefix}{marker}_{direction}", schema)
        for prefix in OP_PREFIXES
        for marker in ("", "_softmax")
        for direction, schema in (
            ("forward", _FORWARD_SCHEMA),
            ("backward", _BACKWARD_SCHEMA),
        )
    )


def define_schemas() -> None:
    """Register every CUDA operator schema with ``torch.library``.

    Idempotent per operator, checked against ``torch.ops.deformops_cuda`` so it
    survives ``importlib.reload``; ``torch.library.define`` raises on a
    redefinition. Registers schemas only, never compiling or loading a ``.so``.
    """
    namespace = torch.ops.deformops_cuda  # type: ignore[attr-defined]
    for name, schema in _schemas():
        if hasattr(namespace, name):
            continue
        torch.library.define(f"{_NAMESPACE}::{name}", schema)
