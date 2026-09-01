r"""Operator-agnostic backend dispatch.

One :class:`Registry` per operator maps :class:`Backend` members to the
functions that run the operator on that backend. The operator supplies a
``context_of`` hook that summarizes a call, and a ``select`` policy that turns
the summary into an ordered list of candidate backends. This module imports no
torch.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import logging
import os
import threading
import warnings
from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from typing import Any, TypeGuard, cast

from ..core.errors import BackendFallbackWarning, KernelNotProvidedError, NoBackendError

__all__ = ["Backend", "LoadedBackend", "Registry", "Resolution"]

logger = logging.getLogger(__name__)

#: Environment variable that silences :class:`BackendFallbackWarning`.
FALLBACK_WARNINGS_ENV_VAR = "DEFORMOPS_FALLBACK_WARNINGS"

_FALSY = frozenset({"0", "false", "no", "off"})


class Backend(StrEnum):
    """Supported compute backends."""

    CUDA = "cuda"
    TRITON = "triton"
    EAGER = "eager"


#: A policy's candidate: a backend, or a backend and the options it runs with.
Candidate = Backend | tuple[Backend, dict[str, Any]]


@dataclasses.dataclass(frozen=True)
class Resolution:
    """The backend a call runs on, and why earlier candidates were passed over.

    Attributes
    ----------
    backend
        The backend that runs the call.
    options
        Keyword options the policy passes to that backend.
    rejected
        ``(backend, reason)`` for every candidate the policy preferred over
        ``backend`` and that could not run the call, in the policy's order.
    """

    backend: Backend
    options: dict[str, Any]
    rejected: tuple[tuple[Backend, str], ...] = ()


@dataclasses.dataclass(frozen=True)
class LoadedBackend:
    """An operator backend made ready by :meth:`Registry.load`.

    Attributes
    ----------
    operator
        The operator name: ``"msda"``, ``"attn"`` or ``"conv"``.
    backend
        The backend that was loaded.
    library
        Path of the loaded kernel library, or ``None`` for a backend without one.
    """

    operator: str
    backend: Backend
    library: str | None


def _fallback_warnings_enabled() -> bool:
    value = os.getenv(FALLBACK_WARNINGS_ENV_VAR, "").strip().lower()
    return value not in _FALSY


def _is_pair(item: object) -> TypeGuard[tuple[Backend, Mapping[str, Any]]]:
    """Whether ``item`` is one ``(backend, options)`` candidate."""
    if not isinstance(item, tuple):
        return False
    parts = cast("tuple[object, ...]", item)
    return (
        len(parts) == 2  # noqa: PLR2004
        and isinstance(parts[0], Backend)
        and isinstance(parts[1], Mapping)
    )


def _candidates(
    selected: Candidate | Sequence[Candidate],
) -> list[tuple[Backend, dict[str, Any]]]:
    """Normalize a policy's answer to an ordered list of ``(backend, options)``."""
    if isinstance(selected, Backend):
        return [(selected, {})]
    if _is_pair(selected):
        backend, options = selected
        return [(backend, dict(options))]
    candidates: list[tuple[Backend, dict[str, Any]]] = []
    for item in cast("Sequence[Candidate]", selected):
        if isinstance(item, Backend):
            candidates.append((item, {}))
        else:
            backend, options = item
            candidates.append((backend, dict(options)))
    return candidates


class Registry[OpCallable: Callable[..., Any]]:
    """Maps backends to operator implementations for one operator.

    An explicit ``backend=`` runs that backend or raises. ``backend=None`` runs
    the first policy candidate that is registered, supports the call and has
    its kernel, memoized per context; see ``docs/dispatch.md``.
    """

    def __init__(
        self,
        name: str,
        *,
        context_of: Callable[..., Any],
        select: Callable[[Any], Candidate | Sequence[Candidate]],
        describe: Callable[..., Any] | None = None,
    ) -> None:
        """Create the registry for one operator.

        Parameters
        ----------
        name
            The operator name, which is also the adapter module name each
            backend package provides.
        context_of
            Builds the call context from the call's arguments.
        select
            The selection policy: from a context to the candidate backends in
            order of preference, each optionally paired with options.
        describe
            A function whose signature and docstring the registry copies onto
            the instance, for ``help()`` and :func:`inspect.signature`.
        """
        self.name = name
        if describe is not None:
            self.__doc__ = describe.__doc__
            self.__signature__ = inspect.signature(describe)
        self._backends: dict[Backend, OpCallable] = {}
        self._supports: dict[Backend, Callable[[Any], bool]] = {}
        self._loaders: dict[Backend, Callable[..., Any]] = {}
        self._unavailable: dict[Backend, str] = {}
        self._context_of = context_of
        self._select = select
        self._resolved: dict[Any, Resolution] = {}
        self._backends_loaded = False
        self._load_lock = threading.RLock()
        self._selected_route: Callable[..., Any] | None = None

    def register(
        self,
        backend: Backend,
        op_handle: OpCallable,
        *,
        supports: Callable[[Any], bool] | None = None,
        load: Callable[..., Any] | None = None,
    ) -> None:
        """Register an operator handle for a backend.

        Parameters
        ----------
        backend
            The backend the handle runs on.
        op_handle
            The function that runs the operator.
        supports
            Whether the backend can run a given call context. Omitted means
            every context.
        load
            Makes the backend's kernel available, taking ``download`` and
            ``compile`` keywords and returning the loaded library. Raises
            :class:`~deformops.core.errors.KernelNotProvidedError` when the
            environment disallows the download or compile it needs.
        """
        if not isinstance(backend, Backend):  # type: ignore[unreachable] # pyright: ignore[reportUnnecessaryIsInstance]
            msg = (
                f"backend must be a Backend StrEnum instance, got "
                f"{type(backend).__name__!r}. Raw strings are not permitted."
            )
            raise TypeError(msg)
        self._backends[backend] = op_handle
        self._unavailable.pop(backend, None)
        if supports is not None:
            self._supports[backend] = supports
        if load is not None:
            self._loaders[backend] = load

    def route_selected_calls(self, route: Callable[..., Any]) -> None:
        """Send every ``backend=None`` call through ``route``, a public operator.

        Tracing then records the operator, not the backend selection. The
        operator's kernel must call :meth:`run_selected`, not the registry,
        which would route back to the operator.
        """
        self._selected_route = route

    def mark_unavailable(self, backend: Backend, reason: str) -> None:
        """Record why a backend did not register, for error messages."""
        if backend not in self._backends:
            self._unavailable[backend] = reason

    def _ensure_backends_loaded(self) -> None:
        """Import this operator's backend adapters once, which registers them.

        A failed import warns and marks the backend unavailable. The loaded flag
        is set under the lock after every import, so a concurrent first call
        never sees a partial registry.
        """
        if self._backends_loaded:
            return
        with self._load_lock:
            if self._backends_loaded:
                return
            for backend in Backend:
                module = f"deformops.backends.{backend.value}.{self.name}"
                try:
                    importlib.import_module(module)
                except Exception as error:  # noqa: BLE001
                    logger.debug("Failed to import %s", module, exc_info=True)
                    reason = (
                        f"its adapter failed to import "
                        f"({type(error).__name__}: {error})"
                    )
                    self.mark_unavailable(backend, reason)
                    warnings.warn(
                        f"The {backend.value!r} backend for {self.name!r} is "
                        f"unavailable: {reason}",
                        RuntimeWarning,
                        stacklevel=3,
                    )
            self._backends_loaded = True

    def load(
        self,
        backend: Backend,
        *,
        download: bool | None = None,
        compile: bool | str | None = None,  # noqa: A002
    ) -> LoadedBackend:
        """Make a backend ready before this operator's first call.

        For a backend with a compiled kernel library, such as the CUDA
        multi-scale backend, this obtains the library as a first call would.
        For the others it registers the operators and checks that the backend
        can run on this host. Load before running an exported or compiled
        program that calls the backend's operators. ``download`` and
        ``compile`` apply to the first load only; see
        ``docs/kernel-loading.md``.

        Parameters
        ----------
        backend
            A :class:`Backend` member.
        download
            Allow the prebuilt download; ``None`` reads ``DEFORMOPS_DOWNLOAD``.
        compile
            Allow a local compile: ``True``, ``False``, or a
            :class:`~deformops.core.build.CompileMode` name. ``None`` reads
            ``DEFORMOPS_COMPILE``.

        Returns
        -------
        What was loaded.

        Raises
        ------
        TypeError
            If ``backend`` is not a :class:`Backend` member.
        NoBackendError
            If the backend cannot run on this host.
        KernelUnavailableError
            If the kernel library cannot be provided; the subclass says why.
        """
        if not isinstance(backend, Backend):  # pyright: ignore[reportUnnecessaryIsInstance]
            msg = f"backend must be a Backend member, got {type(backend).__name__!r}"
            raise TypeError(msg)
        if not self.is_registered(backend):
            msg = (
                f"The {backend.value!r} backend for {self.name} is unavailable: "
                f"{self.unavailable_reason(backend)}."
            )
            raise NoBackendError(msg)
        loader = self._loaders.get(backend)
        library = None
        if loader is not None:
            library = str(loader(download=download, compile=compile))
        return LoadedBackend(self.name, backend, library)

    def is_registered(self, backend: Backend) -> bool:
        """Whether a backend is registered for this operator."""
        self._ensure_backends_loaded()
        return backend in self._backends

    def get(self, backend: Backend) -> OpCallable | None:
        """The operator handle for a backend, or ``None`` if not registered."""
        self._ensure_backends_loaded()
        return self._backends.get(backend)

    def available(self) -> list[str]:
        """The names of every registered backend, sorted."""
        self._ensure_backends_loaded()
        return sorted(backend.value for backend in self._backends)

    def _supports_backend(self, backend: Backend, context: Any) -> bool:
        predicate = self._supports.get(backend)
        return predicate is None or predicate(context)

    def unavailable_reason(self, backend: Backend) -> str:
        """Why a backend is not registered for this operator on this host."""
        return self._unavailable.get(backend, "it is not registered on this host")

    def _negotiate(self, context: Any) -> Resolution:
        rejected: list[tuple[Backend, str]] = []
        for backend, options in _candidates(self._select(context)):
            if backend not in self._backends:
                rejected.append((backend, self.unavailable_reason(backend)))
                continue
            if not self._supports_backend(backend, context):
                rejected.append((backend, "it does not support this call"))
                continue
            loader = self._loaders.get(backend)
            if loader is not None:
                try:
                    loader()
                except KernelNotProvidedError as error:
                    rejected.append((backend, str(error)))
                    continue
            resolution = Resolution(backend, options, tuple(rejected))
            if rejected and _fallback_warnings_enabled():
                warnings.warn(
                    f"{self.name}: running the {backend.value!r} backend because "
                    + "; ".join(
                        f"{skipped.value!r} was passed over: {reason}"
                        for skipped, reason in rejected
                    )
                    + f". Set {FALLBACK_WARNINGS_ENV_VAR}=0 to silence this warning.",
                    BackendFallbackWarning,
                    stacklevel=4,
                )
            return resolution
        details = "; ".join(f"{b.value!r}: {reason}" for b, reason in rejected)
        msg = (
            f"No backend can run this {self.name} call ({context!r}). "
            f"Candidates in order of preference: {details or 'none'}."
        )
        raise NoBackendError(msg)

    def _resolve_context(self, context: Any) -> Resolution:
        cached = self._resolved.get(context)
        if cached is not None:
            return cached
        resolution = self._negotiate(context)
        self._resolved[context] = resolution
        return resolution

    @staticmethod
    def _check_backend_argument(backend: object) -> None:
        if backend is not None and not isinstance(backend, Backend):
            msg = (
                f"backend argument must be a Backend StrEnum instance or None, "
                f"got {type(backend).__name__!r}. Raw strings are not permitted."
            )
            raise TypeError(msg)

    def _explicit(self, backend: Backend, context: Any) -> OpCallable:
        handler = self._backends.get(backend)
        if handler is None:
            msg = (
                f"The {backend.value!r} backend for {self.name} is unavailable: "
                f"{self.unavailable_reason(backend)}. "
                f"Registered backends: {self.available()}."
            )
            raise NoBackendError(msg)
        if not self._supports_backend(backend, context):
            msg = (
                f"The {backend.value!r} backend does not support this {self.name} "
                f"call: {context!r}"
            )
            raise NoBackendError(msg)
        return handler

    def resolve(
        self, *args: Any, backend: Backend | None = None, **kwargs: Any
    ) -> Resolution:
        """Report which backend a call with these arguments would run on.

        Loads the kernel as the call would, so the report matches running it.

        Raises
        ------
        NoBackendError
            If no backend, or not the requested one, can run the call.
        """
        self._check_backend_argument(backend)
        self._ensure_backends_loaded()
        context = self._context_of(*args, **kwargs)
        if backend is not None:
            self._explicit(backend, context)
            loader = self._loaders.get(backend)
            if loader is not None:
                loader()
            return Resolution(backend, {})
        return self._resolve_context(context)

    def run_selected(self, *args: Any, **kwargs: Any) -> Any:
        """Run a call on the selected backend, bypassing any route.

        Explicit keyword arguments override the policy's options. A public
        operator's kernel calls this.
        """
        self._ensure_backends_loaded()
        context = self._context_of(*args, **kwargs)
        resolution = self._resolve_context(context)
        return self._backends[resolution.backend](
            *args, **{**resolution.options, **kwargs}
        )

    def __call__(
        self, *args: Any, backend: Backend | None = None, **kwargs: Any
    ) -> Any:
        """Run the operator on the requested or selected backend.

        A routed ``backend=None`` call leaves loading to the operator's kernel,
        so Dynamo traces a first call without entering the load lock.
        """
        self._check_backend_argument(backend)
        if backend is not None:
            self._ensure_backends_loaded()
            context = self._context_of(*args, **kwargs)
            return self._explicit(backend, context)(*args, **kwargs)
        if self._selected_route is not None:
            return self._selected_route(*args, **kwargs)
        return self.run_selected(*args, **kwargs)

    apply = __call__
