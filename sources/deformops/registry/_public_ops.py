r"""Public ``deformops::`` operators whose one kernel selects a backend.

Each registers a ``CompositeImplicitAutograd`` kernel that runs the registry and
calls a backend-private operator. That operator has the autograd, fake and device
kernels. Imports only torch; see ``docs/export.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
import torch.library

from ._vmap import make_vmap_rule

__all__ = ["NAMESPACE", "define_public_op"]

#: The library namespace every public operator is defined in.
NAMESPACE = "deformops"

#: The fragment the public operators are defined through. Held for the life of
#: the process: a collected ``Library`` removes the operators it registered.
_LIBRARY = torch.library.Library(NAMESPACE, "FRAGMENT")


def define_public_op(
    name: str,
    schema: str,
    kernel: Callable[..., Any],
    *,
    shared: Mapping[int, str] | None = None,
) -> Any:
    """Define ``deformops::<name>`` with a backend-selecting kernel and vmap rule.

    Idempotent: an existing operator is returned as registered.

    Parameters
    ----------
    name
        The operator name, without the namespace.
    schema
        The argument and return list, without the operator name, for example
        ``"(Tensor value, int groups) -> Tensor"``.
    kernel
        The ``CompositeImplicitAutograd`` kernel, taking the schema's arguments
        positionally. It must build its result from differentiable operators,
        since the public operator has no autograd formula.
    shared
        Argument positions that describe the batch, for
        :func:`deformops.registry._vmap.make_vmap_rule`.

    Returns
    -------
    The operator packet ``torch.ops.deformops.<name>``.
    """
    namespace = getattr(torch.ops, NAMESPACE)
    if hasattr(namespace, name):
        return getattr(namespace, name)
    _LIBRARY.define(f"{name}{schema}")
    _LIBRARY.impl(name, kernel, "CompositeImplicitAutograd")
    op = getattr(namespace, name)
    torch.library.register_vmap(
        f"{NAMESPACE}::{name}", make_vmap_rule(op, shared=shared), lib=_LIBRARY
    )
    return op
