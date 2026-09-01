r"""Module replacement shared by the integrations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from torch import nn

__all__ = ["ModuleFactory", "patch_modules", "replace_module_recursively"]

#: Builds a replacement for one upstream module instance.
ModuleFactory = Callable[[Any], nn.Module]


def replace_module_recursively(
    module: nn.Module,
    old_class: type[Any],
    factory: ModuleFactory,
    *,
    strict: bool = True,
) -> nn.Module:
    """Replace every ``old_class`` module with ``factory(child)``, in place.

    Single-class shorthand for :func:`patch_modules`.

    Parameters
    ----------
    module
        The model (or subtree) to patch.
    old_class
        The upstream module class to replace.
    factory
        Called with each matched child; returns its replacement.
    strict
        Raise when no module matched. See :func:`patch_modules`.

    Returns
    -------
    ``module``, for chaining.
    """
    return patch_modules(module, [(old_class, factory)], strict=strict)


def patch_modules(
    module: nn.Module,
    replacements: Sequence[tuple[type[Any], ModuleFactory]],
    *,
    strict: bool = True,
) -> nn.Module:
    """Replace several upstream module classes in one traversal, in place.

    The children of a replacement are visited too, so nested classes (a
    Mask2Former attention inside its encoder layer) are found. A patch that
    matches nothing usually means upstream renamed the class; ``strict`` makes
    that raise instead of leaving the upstream implementation running.

    Parameters
    ----------
    module
        The model (or subtree) to patch.
    replacements
        ``(old_class, factory)`` pairs. The first class matching a child wins.
    strict
        Raise when no module was replaced; ``False`` patches opportunistically.

    Returns
    -------
    ``module``, for chaining.

    Raises
    ------
    RuntimeError
        If ``strict`` is set and no module matched any of the classes.
    """
    replaced = 0

    def _visit(container: nn.Module) -> None:
        nonlocal replaced
        for name, child in list(container.named_children()):
            for old_class, factory in replacements:
                if isinstance(child, old_class):
                    replacement = factory(child)
                    setattr(container, name, replacement)
                    replaced += 1
                    _visit(replacement)
                    break
            else:
                _visit(child)

    _visit(module)
    if strict and replaced == 0:
        names = ", ".join(
            f"{cls.__module__}.{cls.__qualname__}" for cls, _ in replacements
        )
        msg = (
            f"No module of {type(module).__name__} matched any class to replace "
            f"({names}). The installed upstream library may define these under a "
            f"different name, in which case the model still runs its own "
            f"implementation. Pass strict=False to allow a model without them."
        )
        raise RuntimeError(msg)
    return module
