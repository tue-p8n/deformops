r"""Operator-agnostic runtime helpers: the method enum and call-context helpers."""

from __future__ import annotations

import enum
import functools
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["Method", "is_training_from", "resolve_method", "sm_version_for"]


class Method(enum.StrEnum):
    """How an operator normalizes its attention or modulation weights.

    Shared by the multi-scale and single-scale operators. The string values are
    ``"linear"`` and ``"softmax"``, and every ``method`` argument accepts either
    the member or its value.
    """

    #: Use the weights as given, with no normalization.
    LINEAR = enum.auto()
    #: Softmax jointly over the weighted points: the ``L * K`` samples of each head
    #: (multi-scale) or the ``K`` points of each group (single-scale).
    SOFTMAX = enum.auto()


def resolve_method(method: str | Method | None, *, softmax: bool = False) -> Method:
    """Resolve the aggregation method for one operator call.

    An explicit ``method`` wins; otherwise ``softmax`` chooses. Every adapter routes
    through here, so the default, :attr:`Method.LINEAR`, lives in one place; see
    ``docs/conventions.md``.

    Parameters
    ----------
    method
        An explicit method, as a :class:`Method` member or its string value.
        ``None`` defers to ``softmax``.
    softmax
        Selects :attr:`Method.SOFTMAX` when no explicit ``method`` is given.

    Returns
    -------
    The resolved method.
    """
    if method is not None:
        return Method(method)
    return Method.SOFTMAX if softmax else Method.LINEAR


@functools.cache
def sm_version_for(device: torch.device) -> tuple[int, int]:
    """Return the CUDA compute capability (SM version) of ``device``.

    ``(0, 0)`` off CUDA, so a context built at runtime is defined on every host.

    Parameters
    ----------
    device
        The device to query.

    Returns
    -------
    The ``(major, minor)`` compute capability, or ``(0, 0)`` off CUDA.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return (0, 0)
    return torch.cuda.get_device_capability(device)


def is_training_from(*tensors: Tensor) -> bool:
    """Whether a backward pass could run for an operator call.

    Needs grad mode *and* an input that requires grad, so an inference call is not
    misclassified as training because grad mode is on by default.

    Parameters
    ----------
    *tensors
        The call's differentiable inputs; only ``requires_grad`` is read.

    Returns
    -------
    Whether a backward pass could run for this call.
    """
    return torch.is_grad_enabled() and any(t.requires_grad for t in tensors)
