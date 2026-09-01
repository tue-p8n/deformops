"""Exceptions and warnings raised when an operator backend cannot run."""

from __future__ import annotations

__all__ = [
    "BackendFallbackWarning",
    "KernelBuildError",
    "KernelDownloadError",
    "KernelNotProvidedError",
    "KernelUnavailableError",
    "NoBackendError",
]


class KernelUnavailableError(RuntimeError):
    """A compiled kernel library could not be made available."""


class KernelNotProvidedError(KernelUnavailableError):
    """No library is bundled, cached or downloadable, and compiling is disabled.

    Selection with ``backend=None`` treats this as "backend unavailable" and
    moves on to the next backend; an explicitly requested backend raises it.
    """


class KernelDownloadError(KernelUnavailableError):
    """Downloading the prebuilt library was enabled and failed.

    Always raised, including under ``backend=None``: set
    ``DEFORMOPS_DOWNLOAD=0`` to skip the download.
    """


class KernelBuildError(KernelUnavailableError):
    """Compiling the library was enabled and could not be done."""


class NoBackendError(RuntimeError):
    """No backend can run an operator call."""


class BackendFallbackWarning(UserWarning):
    """``backend=None`` ran a backend other than its first choice.

    Set ``DEFORMOPS_FALLBACK_WARNINGS=0`` to silence it.
    """
