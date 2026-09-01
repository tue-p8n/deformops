r"""Resolve, download and compile the CUDA kernel libraries.

The loader tries four sources in order: a library bundled with the install, one
already compiled on this machine for the same sources and environment, a download
from the prebuilt index, and a local compile. ``DEFORMOPS_DOWNLOAD`` and
``DEFORMOPS_COMPILE`` (both off by default) enable the last two; an enabled download
that fails raises.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.resources
import json
import logging
import os
import pathlib
import re
import shutil
import stat
import tempfile
import threading
import urllib.request
import warnings
from collections.abc import Callable, Generator, Iterable
from enum import StrEnum
from typing import Any, Literal

try:
    import fcntl
except ImportError:  # Windows: serialized_build becomes a no-op.
    fcntl = None

import torch.utils.cpp_extension

import deformops.version

from .configs import EXT_PACKAGE, EXTENSION_CONFIGS, SKIPPED_BUILD_FILE
from .errors import KernelBuildError, KernelDownloadError, KernelNotProvidedError

# Resolved from this file, so source lookups use this tree even when a prebuilt or
# Nix store package shadows it on sys.path.
_DEFORMOPS_SRC_ROOT: pathlib.Path = pathlib.Path(__file__).parent.parent

__all__: list[str] = [
    "COMPILE_ENV_VAR",
    "DOWNLOAD_ENV_VAR",
    "CompileMode",
    "compile_mode",
    "define_library",
    "download_enabled",
    "lazy_extension",
    "load_op",
    "locate_build",
    "op_library",
    "register_op_pair",
    "used_prebuilt",
]

logger = logging.getLogger(__name__)


def op_library(op_prefix: str) -> str:
    """Return the ``deformops_cuda::<op_prefix>`` string ``torch.library`` takes.

    Parameters
    ----------
    op_prefix
        The op-name prefix an extension registers under.
    """
    return f"deformops_cuda::{op_prefix}"


def lazy_extension(
    name: str,
    build_dir: str | pathlib.Path | None = None,
    *,
    with_cuda: bool = True,
    opt_level: Literal[0, 1, 2, 3] = 2,
    cflags: Iterable[str] = (),
    cflags_cuda: Iterable[str] = (),
) -> Callable[[], Any]:
    r"""Defer ``define_library(name, ...)`` to first call, then memoize it.

    Importing a configuration module that calls this loads nothing. Concurrent
    first calls load once; a failed load is not memoized, so a later call retries.

    Parameters
    ----------
    name, build_dir, with_cuda, opt_level, cflags, cflags_cuda
        Forwarded to :func:`define_library` on the first successful load.

    Returns
    -------
    ``load(*, download=None, compile=None)``, which forwards ``download`` and
    ``compile`` to :func:`define_library` and returns the loaded library,
    memoized after the first success.
    """
    locks = [threading.Lock()]
    loaded: list[Any] = []

    def _reset_lock_in_child() -> None:
        # A fork while another thread holds the lock (a DataLoader worker, a
        # background warm-up) would otherwise leave the child waiting forever.
        locks[0] = threading.Lock()

    os.register_at_fork(after_in_child=_reset_lock_in_child)

    def bind(*, download: bool | None = None, compile: bool | str | None = None) -> Any:
        if loaded:
            if download is not None or compile is not None:
                logger.debug(
                    "The %s kernel library is already loaded; this call ignores "
                    "download=%s and compile=%s.",
                    name,
                    download,
                    compile,
                )
            return loaded[0]
        with locks[0]:
            if not loaded:
                loaded.append(
                    define_library(
                        name,
                        build_dir,
                        with_cuda=with_cuda,
                        opt_level=opt_level,
                        cflags=cflags,
                        cflags_cuda=cflags_cuda,
                        download=download,
                        compile=compile,
                    )
                )
        return loaded[0]

    return bind


def load_op(
    op_prefix: str,
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Return the ``<op_prefix>_forward`` and ``<op_prefix>_backward`` ops.

    Load the hosting library with :func:`define_library` first. One extension may
    register several pairs, so call this once per prefix.

    Parameters
    ----------
    op_prefix
        The op-name prefix an extension registers under.
    """
    namespace = torch.ops.deformops_cuda  # type: ignore[attr-defined]
    forward_op = getattr(namespace, f"{op_prefix}_forward")
    backward_op = getattr(namespace, f"{op_prefix}_backward").default
    return forward_op, backward_op


def register_op_pair(
    *,
    linear_op_library: str,
    softmax_op_library: str,
    setup_context: Callable[..., Any],
    backward_linear: Callable[..., Any],
    backward_softmax: Callable[..., Any],
    forward_meta: Callable[..., Any],
    backward_meta: Callable[..., Any],
) -> None:
    """Register meta and autograd for a (linear, softmax) op pair.

    The variants share schemas and meta functions; each gets its own autograd
    registration because its backward op differs. The hooks must follow these rules,
    which nothing enforces:

    - ``setup_context(ctx, inputs, output)`` saves the differentiable inputs with
      ``ctx.save_for_backward``. Non-tensor state, such as the launch geometry,
      goes in attributes on ``ctx``.
    - The backwards use ``@once_differentiable`` and disable autocast. Each calls
      the backward op at its configuration's dtype, copying to float32 only where
      the configuration upcasts. It casts each gradient back with ``.type_as``
      and returns a ``None``-padded tuple, one entry per forward input.
    - The meta functions project shapes on fake tensors and compute nothing.

    Parameters
    ----------
    linear_op_library, softmax_op_library
        The :func:`op_library` strings of the linear and softmax variants.
    setup_context, backward_linear, backward_softmax
        See the rules above; ``setup_context`` serves both variants.
    forward_meta, backward_meta
        Fake-tensor shape functions shared by both variants.
    """
    import torch.library  # local import to keep build-time imports minimal

    for lib in (linear_op_library, softmax_op_library):
        torch.library.impl(f"{lib}_forward", "meta")(forward_meta)
        torch.library.impl(f"{lib}_backward", "meta")(backward_meta)
    torch.library.register_autograd(
        f"{linear_op_library}_forward",
        backward_linear,
        setup_context=setup_context,
    )
    torch.library.register_autograd(
        f"{softmax_op_library}_forward",
        backward_softmax,
        setup_context=setup_context,
    )


#: Environment variable enabling the prebuilt-library download (default off).
DOWNLOAD_ENV_VAR = "DEFORMOPS_DOWNLOAD"

#: Environment variable selecting when a library is compiled locally.
COMPILE_ENV_VAR = "DEFORMOPS_COMPILE"

_FALSY = frozenset({"", "0", "false", "no", "off"})
_TRUTHY = frozenset({"1", "true", "yes", "on"})


class CompileMode(StrEnum):
    """When a kernel library is compiled locally.

    - ``NEVER``: never compile; a library must be bundled, already compiled
      on this machine, or downloaded.
    - ``FALLBACK``: compile when no bundled, compiled or downloaded library
      is available.
    - ``ALWAYS``: skip bundled and downloadable libraries; use or compile a
      library on this machine.
    - ``REBUILD``: like ``ALWAYS``, after clearing this build's directory.
    """

    NEVER = "never"
    FALLBACK = "fallback"
    ALWAYS = "always"
    REBUILD = "rebuild"

    @classmethod
    def coerce(cls, value: bool | str | CompileMode) -> CompileMode:
        """Interpret a boolean, a mode name, or an environment value."""
        if isinstance(value, CompileMode):
            return value
        if isinstance(value, bool):
            return cls.FALLBACK if value else cls.NEVER
        normalized = value.strip().lower()
        if normalized in _FALSY:
            return cls.NEVER
        if normalized in _TRUTHY:
            return cls.FALLBACK
        try:
            return cls(normalized)
        except ValueError:
            msg = (
                f"Invalid {COMPILE_ENV_VAR} value {value!r}: expected 0, 1, "
                f"{', '.join(mode.value for mode in cls)}."
            )
            raise ValueError(msg) from None


def compile_mode() -> CompileMode:
    """The compile mode set by ``DEFORMOPS_COMPILE`` (default ``never``)."""
    return CompileMode.coerce(os.getenv(COMPILE_ENV_VAR, ""))


def download_enabled() -> bool:
    """Whether ``DEFORMOPS_DOWNLOAD`` allows the prebuilt download (default off)."""
    value = os.getenv(DOWNLOAD_ENV_VAR, "").strip().lower()
    return value not in _FALSY


#: Sidecar the flock is held on, distinct from the builder's ``lock`` file below.
_JIT_LOCK_FILENAME = ".deformops-jit.flock"

#: The lock file the extension builder creates and may abandon.
_STALE_LOCK_FILENAME = "lock"


def _clear_build_dir(build_dir: pathlib.Path, name: str) -> None:
    """Remove every file under ``build_dir`` to force a fresh ninja build.

    Keeps ``_JIT_LOCK_FILENAME``, whose flock is held: unlinking it would let a
    concurrent builder lock a fresh inode at the same path and build alongside.
    """
    logger.info(
        "Clearing build directory %s for %s before rebuilding",
        str(build_dir),
        name,
    )
    for child in build_dir.iterdir():
        if child.name == _JIT_LOCK_FILENAME:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


@contextlib.contextmanager
def serialized_build(build_dir: pathlib.Path | None) -> Generator[None]:
    r"""Serialize JIT builds of ``build_dir``, clearing a stale torch lock.

    Torch's build lock has no liveness check, so a killed build leaves every later
    build waiting forever. An ``flock`` is released when its holder exits, even on
    SIGKILL, so while it is held a leftover torch lock is safe to remove. Without a
    lock to take (``None``, no ``fcntl``, an unwritable directory) this does
    nothing. See ``docs/kernel-loading.md``.
    """
    if build_dir is None or fcntl is None:
        yield
        return

    try:
        build_dir.mkdir(parents=True, exist_ok=True)
        handle = (build_dir / _JIT_LOCK_FILENAME).open("w")
    except OSError as err:
        logger.warning(
            "Could not open a build lock in %s (%s); concurrent builds of this "
            "extension will not be serialized.",
            str(build_dir),
            err,
        )
        yield
        return

    with handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX)
        except OSError as err:
            logger.warning(
                "Could not lock the build directory %s (%s); concurrent builds "
                "of this extension will not be serialized.",
                str(build_dir),
                err,
            )
            yield
            return
        try:
            stale_lock = build_dir / _STALE_LOCK_FILENAME
            if stale_lock.exists():
                logger.warning(
                    "Removing stale torch cpp_extension lock file %s; a "
                    "previous build of this extension was likely killed "
                    "before it finished (that lock has no liveness check "
                    "and would otherwise spin forever).",
                    str(stale_lock),
                )
                stale_lock.unlink(missing_ok=True)
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _extensions_root() -> pathlib.Path:
    """Return the root of every extension's build directory.

    Shared with :func:`_find_cached_library_for_any_arch`, which globs under it.
    """
    try:
        ext_root = pathlib.Path(os.environ["TORCH_EXTENSIONS_DIR"])
        logger.debug("Found TORCH_EXTENSIONS_DIR in environment: %s", str(ext_root))
    except KeyError:
        ext_root = (
            pathlib.Path(torch.utils.cpp_extension.get_default_build_root())
            / deformops.version.package
            / deformops.version.environment
        )

        # The default root is already namespaced by package and environment,
        # so builds from different installations cannot collide there.
        logger.debug(
            "TORCH_EXTENSIONS_DIR not set. Using default build root: %s", str(ext_root)
        )
    return ext_root


def locate_build(name: str, *, tag: str = "") -> pathlib.Path:
    """Get the build directory for an extension.

    Parameters
    ----------
    name
        The name of the extension.
    tag
        Optional fingerprint, as ``<name>-<tag>``, so changed inputs get a new
        directory and never reuse a stale ``.so``.

    Returns
    -------
    The build directory, created if missing.
    """
    ext_root = _extensions_root()

    dir_name = f"{name}-{tag}" if tag else name
    build_directory = ext_root / dir_name
    logger.debug("Using %s as build directory for %s", str(build_directory), name)

    if not build_directory.is_dir():
        logger.info("Creating build directory: %s", str(build_directory))
        build_directory.mkdir(exist_ok=True, parents=True)

    return build_directory


def _try_load_prebuilt(name: str) -> str | None:
    """Load a bundled stable-ABI ``.so`` with ``torch.ops.load_library``.

    It has no ``PyInit_*`` symbol, so it is dlopened instead of imported. Loading
    runs the static initializers that register ``torch.ops.deformops_cuda``.
    """
    import torch

    ext_pkg = EXT_PACKAGE
    so_name = f"{name}.so"
    try:
        so_path = importlib.resources.files(ext_pkg) / so_name
        so_str = str(so_path)
    except (ModuleNotFoundError, FileNotFoundError, TypeError):
        return None
    if not pathlib.Path(so_str).is_file():
        return None
    try:
        torch.ops.load_library(so_str)
    except (OSError, RuntimeError) as e:
        logger.warning(
            "Failed to load the bundled %s library (%s); trying the next source.",
            name,
            e,
        )
        return None
    logger.info("Successfully loaded pre-built extension: %s", name)
    return so_str


# Default host for the stable-ABI operator index (GitHub Pages).
# ``DEFORMOPS_OPS_INDEX_URL`` points at a private copy instead.
_DEFAULT_OPS_INDEX_URL = "https://tue-p8n.github.io/deformops/ops"


def _cuda_tag() -> str | None:
    """Return the ``cuXXX`` tag for the running torch build, or ``None`` on CPU.

    E.g. ``torch.version.cuda == "12.6"`` -> ``"cu126"``.
    """
    import torch  # noqa: PLC0415

    cuda_version = getattr(torch.version, "cuda", None)
    if not cuda_version:
        return None
    major, _, rest = cuda_version.partition(".")
    minor = rest.partition(".")[0]
    return f"cu{major}{minor}"


def _no_cuda_arch_error() -> RuntimeError:
    """Build the error for no visible device and no ``TORCH_CUDA_ARCH_LIST``.

    Raised only once a compile is about to happen; see :func:`_resolve_cuda_archs`.
    """
    msg = (
        "Cannot determine which CUDA architecture to build for: no "
        "CUDA device is visible and TORCH_CUDA_ARCH_LIST is not set. "
        "Set TORCH_CUDA_ARCH_LIST (e.g. TORCH_CUDA_ARCH_LIST='8.6') "
        "to the target GPU's compute capability, or run this build "
        "where a CUDA device is visible."
    )
    return RuntimeError(msg)


def _resolve_cuda_archs() -> list[str] | None:
    r"""Resolve the CUDA arch identifiers to build for, without raising.

    ``TORCH_CUDA_ARCH_LIST`` wins unless unset or ``"native"``. Otherwise there is one
    entry per distinct visible compute capability, since a mixed host needs every
    device's gencode. Entries sort numerically (a string sort puts ``"10.0"``
    before ``"8.6"``), with ``+PTX`` on the highest. Returns ``None`` with neither,
    because only a compile needs an architecture.
    """
    import torch  # noqa: PLC0415

    raw = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
    if raw and raw.lower() != "native":
        return [arch for arch in re.split(r"[\s;]+", raw) if arch]

    device_count = torch.cuda.device_count()
    if device_count == 0:
        return None

    seen: set[tuple[int, int]] = set()
    capabilities: list[tuple[int, int]] = []
    for index in range(device_count):
        capability = torch.cuda.get_device_capability(index)
        if capability not in seen:
            seen.add(capability)
            capabilities.append(capability)
    capabilities.sort()  # numeric: (major, minor) tuples, not "major.minor" strings
    archs = [f"{major}.{minor}" for major, minor in capabilities]
    archs[-1] += "+PTX"
    return archs


def _arch_flags_for(archs: Iterable[str]) -> list[str]:
    r"""Turn arch identifiers (``"8.6"``, ``"8.6+PTX"``) into ``-gencode`` flags.

    Explicit flags bypass torch's autodetection, which raises an unguarded
    ``IndexError`` with no visible device and the variable unset.

    Raises
    ------
    ValueError
        If an entry is not ``major.minor``, optionally with ``+PTX``. Named aliases
        such as ``"Ampere"`` are rejected: their expansion depends on torch.

    Notes
    -----
    Sorted numerically, ``compute_*`` before ``sm_*`` within an arch. This matches
    torch for one arch and differs where arch numbers differ in digit count.
    """
    seen: set[str] = set()
    entries: list[tuple[int, str]] = []
    for arch in archs:
        has_ptx = arch.upper().endswith("+PTX")
        version = arch[: -len("+PTX")] if has_ptx else arch
        if version.count(".") != 1:
            msg = (
                f"Cannot parse TORCH_CUDA_ARCH_LIST entry {arch!r}: expected "
                "a 'major.minor' compute capability, optionally suffixed "
                "'+PTX' (e.g. '8.6' or '8.6+PTX'). Named aliases (e.g. "
                "'Ampere') are not supported; spell out the compute "
                "capability instead."
            )
            raise ValueError(msg)
        arch_major, arch_minor = version.split(".")
        num = f"{arch_major}{arch_minor}"
        # Leading digits only, so a suffixed arch such as "9.0a" still sorts.
        num_key = int(re.match(r"\d+", num).group())  # type: ignore[union-attr]
        sm_flag = f"-gencode=arch=compute_{num},code=sm_{num}"
        if sm_flag not in seen:
            seen.add(sm_flag)
            entries.append((num_key, sm_flag))
        if has_ptx:
            ptx_flag = f"-gencode=arch=compute_{num},code=compute_{num}"
            if ptx_flag not in seen:
                seen.add(ptx_flag)
                entries.append((num_key, ptx_flag))
    entries.sort(key=lambda entry: (entry[0], entry[1]))
    return [flag for _, flag in entries]


_OPT_LEVELS: dict[str, Literal[0, 1, 2, 3]] = {"0": 0, "1": 1, "2": 2, "3": 3}


def _resolve_opt_level(opt_level: Literal[0, 1, 2, 3]) -> Literal[0, 1, 2, 3]:
    r"""Resolve the JIT optimization level, honoring ``DEFORMOPS_OPT_LEVEL``.

    Lets a development shell request a debuggable build without editing a call
    site.

    Parameters
    ----------
    opt_level
        The caller's level, used when the variable is unset or empty.

    Returns
    -------
    The optimization level the build uses.

    Raises
    ------
    ValueError
        If ``DEFORMOPS_OPT_LEVEL`` is not one of ``0`` to ``3``.
    """
    env_opt = os.getenv("DEFORMOPS_OPT_LEVEL", "").strip()
    if not env_opt:
        return opt_level
    if env_opt not in _OPT_LEVELS:
        msg = f"Invalid DEFORMOPS_OPT_LEVEL: {env_opt!r}. Expected one of: 0, 1, 2, 3."
        raise ValueError(msg)
    return _OPT_LEVELS[env_opt]


# Called only from tests, which pyright does not scan.
def _cuda_arch_flags() -> list[str]:  # pyright: ignore[reportUnusedFunction]
    r"""Resolve explicit CUDA ``-gencode`` flags for the JIT compile.

    Raises at once when no architecture resolves; :func:`define_library` calls the
    two halves separately to defer that past a cache hit.

    Raises
    ------
    RuntimeError
        If ``TORCH_CUDA_ARCH_LIST`` is unset or ``"native"`` and no device is visible.
    ValueError
        If ``TORCH_CUDA_ARCH_LIST`` holds an entry that is not ``major.minor``.
    """
    archs = _resolve_cuda_archs()
    if archs is None:
        raise _no_cuda_arch_error()
    return _arch_flags_for(archs)


def _ops_cache_root() -> pathlib.Path:
    """Root directory under which downloaded operator libraries are cached.

    Falls back to the temp directory where ``Path.home()`` raises.
    """
    override = os.getenv("DEFORMOPS_OPS_CACHE_DIR")
    if override:
        return pathlib.Path(override)
    try:
        return pathlib.Path.home() / ".cache" / "deformops" / "ops"
    except (RuntimeError, KeyError):
        # Scoped by UID so users of a shared host do not collide.
        uid = f"-{os.getuid()}" if hasattr(os, "getuid") else ""
        return pathlib.Path(tempfile.gettempdir()) / f"deformops{uid}" / "ops"


def _ops_cache_dir(pkgver: str, cuda_tag: str) -> pathlib.Path:
    """Local cache directory for downloaded operator libraries.

    Keyed on package version and CUDA tag, not architecture, which holds only
    because ``nix/ops.nix`` builds every published ``.so`` for all compute
    capabilities of its toolkit.
    """
    return _ops_cache_root() / pkgver / cuda_tag


def _cache_owner_ok(cached: pathlib.Path) -> bool:
    """Whether the cache path is safe to write and load on a shared system.

    A co-tenant could pre-create the predictable ``/tmp`` fallback and swap the
    ``.so`` between verification and ``dlopen``. Every directory from the file up to
    the trust boundary must be owned by this user and deny group and other write.
    Always true without ``os.getuid``. See ``docs/kernel-loading.md``.
    """
    getuid = getattr(os, "getuid", None)
    if getuid is None:
        return True
    uid = getuid()
    root = _ops_cache_root()
    # Trust boundary: the cache root, extended to a ``deformops*`` parent in the
    # /tmp fallback (that predictable parent is the hijack target).
    boundary = root.parent if root.parent.name.startswith("deformops") else root
    path = cached
    while True:
        try:
            st = path.stat()
        except FileNotFoundError:
            pass  # not created yet; nothing to hijack at this level
        except OSError:
            return False
        else:
            if st.st_uid != uid:
                return False
            if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
                return False
        if path in (boundary, path.parent):
            return True
        path = path.parent


#: Largest manifest that will be read. The real one is a few hundred bytes
#: per extension; anything approaching this is a misconfigured or hostile host.
_MAX_MANIFEST_BYTES = 1 << 20

#: Largest operator library that will be read. The real ones are tens of
#: megabytes.
_MAX_LIBRARY_BYTES = 512 << 20


def _plaintext_http_allowed() -> bool:
    """Whether ``DEFORMOPS_OPS_INDEX_URL`` may use plaintext ``http://``.

    Off by default: the manifest holds the digests, so plaintext lets an interposer
    replace both library and digest. Use it only for a copy on a trusted network.
    """
    val = os.getenv("DEFORMOPS_OPS_ALLOW_INSECURE", "0").strip().lower()
    return val not in {"0", "false", "no", "off", ""}


def _fetch_url(url: str, *, max_bytes: int, timeout: float = 10.0) -> bytes:
    """Read at most ``max_bytes`` of ``url`` (honors ``HTTPS_PROXY`` via urllib).

    The cap applies to the bytes read, not ``Content-Length``, which a host can
    under-report or omit. A named User-Agent avoids CDN and firewall rejections.
    """
    # Only http(s): a misconfigured index URL must never open file:// or the like.
    if not url.startswith(("http://", "https://")):
        msg = f"URL scheme not allowed: {url}"
        raise ValueError(msg)
    if url.startswith("http://") and not _plaintext_http_allowed():
        msg = (
            f"Refusing to fetch the ops index over plaintext HTTP: {url}. The "
            f"manifest served there carries the sha256 each library is checked "
            f"against, so plaintext defeats the check. Use https, or set "
            f"DEFORMOPS_OPS_ALLOW_INSECURE=1 for a mirror on a trusted network."
        )
        raise ValueError(msg)
    req = urllib.request.Request(
        url, headers={"User-Agent": f"deformops/{deformops.version.package}"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        # One byte past the cap detects an over-long body instead of truncating it.
        payload = resp.read(max_bytes + 1)
    if len(payload) > max_bytes:
        msg = f"Response from {url} exceeds the {max_bytes} byte limit"
        raise ValueError(msg)
    return payload


def _is_plain_filename(name: str) -> bool:
    """Whether ``name`` is a bare filename, safe to append to a directory URL.

    Defense in depth: the name shares a manifest with the digest, but refusing a
    traversal also keeps a malformed manifest from producing a confusing URL.
    """
    return (
        bool(name)
        and "/" not in name
        and "\\" not in name
        and name
        not in {
            ".",
            "..",
        }
    )


#: Manifests by index directory URL. A failure is kept as ``None``, so an
#: unreachable index costs one timeout per process, not one per extension.
_MANIFESTS: dict[str, dict[str, Any] | None] = {}

#: Why a memoized manifest fetch failed, by index directory URL.
_MANIFEST_ERRORS: dict[str, str] = {}
_MANIFESTS_LOCK = threading.Lock()


def _fetch_manifest(dir_url: str) -> dict[str, Any] | None:
    """Fetch and parse ``manifest.json`` under ``dir_url``, once per process.

    Returns ``None`` when the index is unreachable or the manifest is not a
    JSON object.
    """
    with _MANIFESTS_LOCK:
        if dir_url in _MANIFESTS:
            return _MANIFESTS[dir_url]
        manifest: dict[str, Any] | None
        # Any failure is recorded for the caller to report, not raised.
        try:
            raw = _fetch_url(f"{dir_url}/manifest.json", max_bytes=_MAX_MANIFEST_BYTES)
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                manifest = parsed
            else:
                manifest = None
                _MANIFEST_ERRORS[dir_url] = "the manifest is not a JSON object"
        except Exception as e:  # noqa: BLE001
            logger.debug("Ops index unreachable at %s: %s", dir_url, e)
            manifest = None
            _MANIFEST_ERRORS[dir_url] = f"{type(e).__name__}: {e}"
        _MANIFESTS[dir_url] = manifest
        return manifest


def _ops_manifest_entry(name: str, dir_url: str) -> tuple[str, str] | None:
    """Look up ``name`` in the manifest under ``dir_url``.

    Returns ``(sha256, file)``, or ``None`` if the manifest is unreachable,
    malformed, or lacks an entry for ``name``.
    """
    manifest = _fetch_manifest(dir_url)
    if manifest is None:
        return None
    try:
        entry = manifest[name]
        so_file = entry.get("file", f"{name}.so")
        if not _is_plain_filename(so_file):
            msg = f"manifest entry for {name} names a non-file path: {so_file!r}"
            raise ValueError(msg)
        # Normalize to lowercase so an upper/mixed-case digest still verifies.
        return entry["sha256"].lower(), so_file
    except Exception as e:  # noqa: BLE001
        logger.debug("No ops-index entry for %s at %s: %s", name, dir_url, e)
        return None


def _ensure_cached(
    cached: pathlib.Path, dir_url: str, so_file: str, sha256: str
) -> bool:
    """Download the ``.so`` into ``cached`` unless a verified copy is there.

    Returns ``True`` when ``cached`` is present and matches ``sha256`` (either
    already cached or freshly downloaded), ``False`` on download / checksum /
    write failure.
    """
    # Every read, download or write failure is reported to the caller as
    # ``False``, which raises KernelDownloadError with the reason.
    try:
        if cached.is_file() and _sha256_file(cached) == sha256:
            return True
    except OSError as e:
        logger.debug("Failed to read cached %s: %s", cached, e)
    try:
        payload = _fetch_url(f"{dir_url}/{so_file}", max_bytes=_MAX_LIBRARY_BYTES)
    except Exception as e:  # noqa: BLE001
        logger.debug("Failed to download %s: %s", so_file, e)
        return False
    if hashlib.sha256(payload).hexdigest() != sha256:
        # Corruption, a truncating proxy, or tampering: always warn.
        logger.warning(
            "Checksum mismatch for downloaded %s from %s; not loading it.",
            so_file,
            dir_url,
        )
        return False
    # A unique temp file and an atomic replace prevent collisions between ranks or
    # threads. ``tmp`` is bound before the write so ``finally`` always cleans up.
    tmp = None
    try:
        cached.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=cached.parent, prefix=f"{cached.name}.", suffix=".tmp", delete=False
        ) as tmp_file:
            tmp = pathlib.Path(tmp_file.name)
            tmp_file.write(payload)
        tmp.replace(cached)
    except OSError as e:
        # A concurrent process (e.g. another DDP rank) may have already cached a
        # valid copy; treat that as success.
        try:
            if cached.is_file() and _sha256_file(cached) == sha256:
                return True
        except OSError:
            pass
        logger.debug("Failed to cache downloaded %s: %s", so_file, e)
        return False
    finally:
        # On success replace() consumed tmp; otherwise drop the leftover.
        if tmp is not None:
            tmp.unlink(missing_ok=True)
    return True


def _skipped_build_note() -> str:
    """Why this install has no compiled kernels, when its build recorded it."""
    try:
        marker = importlib.resources.files(EXT_PACKAGE) / SKIPPED_BUILD_FILE
        reason = marker.read_text(encoding="utf-8").strip()
    except (ModuleNotFoundError, FileNotFoundError, OSError, TypeError):
        return ""
    return (
        f" This install was built without its CUDA kernels: {reason}" if reason else ""
    )


def _download_failure(name: str, where: str | None, reason: str) -> str:
    """The message for a download that was enabled and did not succeed."""
    source = f" from {where}" if where else ""
    return (
        f"Could not obtain the prebuilt {name!r} kernel library for deformops "
        f"{deformops.version.package}{source}: {reason}. Set {DOWNLOAD_ENV_VAR}=0 "
        f"to skip the download. The CUDA backend then needs a library compiled on "
        f"this machine ({COMPILE_ENV_VAR}=1, which needs a CUDA toolkit and "
        f"deformops[jit]), and backend=None moves on to its next backend."
        f"{_skipped_build_note()}"
    )


def _download_prebuilt(name: str) -> str:
    """Download the prebuilt library for ``name`` from the index and load it.

    The library must match the package version and torch's CUDA version; it is
    checked against the manifest sha256 and cached under ``~/.cache/deformops/ops``
    (or ``DEFORMOPS_OPS_CACHE_DIR``). An unreachable index falls back to a verified
    cached copy.

    Returns
    -------
    The path of the loaded library.

    Raises
    ------
    KernelDownloadError
        If the library cannot be obtained or loaded, naming the reason.
    """
    import torch  # noqa: PLC0415

    cuda_tag = _cuda_tag()
    if cuda_tag is None:
        raise KernelDownloadError(
            _download_failure(name, None, "the installed torch has no CUDA runtime")
        )

    pkgver = deformops.version.package
    base_url = os.getenv("DEFORMOPS_OPS_INDEX_URL", _DEFAULT_OPS_INDEX_URL).rstrip("/")
    dir_url = f"{base_url}/{pkgver}/{cuda_tag}"

    cached = _ops_cache_dir(pkgver, cuda_tag) / f"{name}.so"
    if not _cache_owner_ok(cached):
        reason = f"the cache directory {cached.parent} is not owned by the current user"
        raise KernelDownloadError(_download_failure(name, dir_url, reason))

    if _fetch_manifest(dir_url) is None:
        offline = _load_cached_offline(name, cached)
        if offline is not None:
            return offline
        error = _MANIFEST_ERRORS.get(dir_url, "unknown error")
        reason = (
            f"its manifest could not be fetched ({error}) and no verified "
            f"copy is cached"
        )
        raise KernelDownloadError(_download_failure(name, dir_url, reason))
    entry = _ops_manifest_entry(name, dir_url)
    if entry is None:
        # A cached copy is not used: a library leaves the index when it must
        # no longer be loaded.
        raise KernelDownloadError(
            _download_failure(name, dir_url, "the index does not list this library")
        )
    sha256, so_file = entry

    if not _ensure_cached(cached, dir_url, so_file, sha256):
        reason = "the download failed or did not match its checksum"
        raise KernelDownloadError(_download_failure(name, dir_url, reason))

    try:
        torch.ops.load_library(str(cached))
    except (OSError, RuntimeError) as error:
        reason = f"the downloaded library failed to load ({error})"
        raise KernelDownloadError(_download_failure(name, dir_url, reason)) from error
    logger.info("Loaded prebuilt extension %s from the index.", name)
    return str(cached)


def _load_cached_offline(name: str, cached: pathlib.Path) -> str | None:
    """Load an already-cached ``.so`` when the index is unreachable.

    The file was verified when written, so reuse needs no manifest. Returns the
    path, or ``None`` when it is missing or fails to load.
    """
    import torch  # noqa: PLC0415

    if not cached.is_file():
        return None
    try:
        torch.ops.load_library(str(cached))
    except (OSError, RuntimeError) as e:
        logger.debug("Failed to load cached extension %s offline: %s", name, e)
        return None
    logger.info("Loaded cached pre-built extension %s (offline fallback).", name)
    return str(cached)


def _sha256_file(path: pathlib.Path) -> str:
    """Return the hex sha256 digest of a file's contents (read in chunks)."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _content_fingerprint(
    sources: Iterable[str],
    include_root: pathlib.Path,
    extra_cflags: Iterable[str],
    extra_cuda_cflags: Iterable[str],
    with_cuda: bool,
) -> str:
    """Hash the exact inputs that determine the compiled ``.so``.

    Covers every source, the flags, ``with_cuda`` and the torch/CUDA environment.
    It also covers every header under ``include_root`` or beside a source,
    recursively, since a kernel may keep its header next to its ``.cu``. An edit
    never reuses a stale binary.
    """
    source_paths = sorted(pathlib.Path(src) for src in sources)
    digest = hashlib.sha256()
    digest.update(deformops.version.environment.encode())
    digest.update(b"\0with_cuda=" + str(with_cuda).encode())
    for flag in (*extra_cflags, *extra_cuda_cflags):
        digest.update(b"\0flag=" + flag.encode())
    for src in source_paths:
        digest.update(b"\0src=" + str(src).encode())
        digest.update(src.read_bytes())
    header_roots = {include_root, *(src.parent for src in source_paths)}
    headers = sorted(
        {
            found
            for root in header_roots
            for found in root.rglob("*")
            if found.suffix in {".cuh", ".h", ".hpp"}
        }
    )
    for hdr in headers:
        # Keyed on the path: a same-named header in both roots would otherwise collide.
        digest.update(b"\0hdr=" + str(hdr.relative_to(_DEFORMOPS_SRC_ROOT)).encode())
        digest.update(hdr.read_bytes())
    return digest.hexdigest()[:16]


def _arch_tag(archs: Iterable[str]) -> str:
    """Short, deterministic tag for a resolved set of CUDA arch identifiers.

    A separate directory component, since :func:`_content_fingerprint` says nothing
    about architecture and a build must not be reused for another one.
    """
    digest = hashlib.sha256()
    for arch in sorted(archs):
        digest.update(b"\0arch=" + arch.encode())
    return digest.hexdigest()[:16]


def _find_cached_library_for_any_arch(
    name: str, environment: str, fingerprint: str
) -> pathlib.Path | None:
    r"""Find a build directory for ``name`` matching all but the architecture.

    Used only when no architecture resolves and no ``build_dir`` was given: with no
    visible device no kernel runs here, so any architecture's build serves. Returns
    the directory, or ``None`` when a compile is required.
    """
    root = _extensions_root()
    pattern = f"{name}-{environment}-*-{fingerprint}"
    matches = sorted(
        candidate
        for candidate in root.glob(pattern)
        if (candidate / f"{name}.so").is_file()
    )
    if not matches:
        return None
    return matches[0]


def _locate_jit_build_dir(
    name: str,
    build_dir: str | pathlib.Path | None,
    archs: list[str] | None,
    sources: Iterable[str],
    include_root: pathlib.Path,
    extra_cflags: Iterable[str],
    extra_cuda_cflags: Iterable[str],
    with_cuda: bool,
    rebuild: bool,
) -> pathlib.Path:
    r"""Resolve the build directory, ``<name>-<environment>-<archtag>-<hash>``.

    The environment names the torch, CUDA and cuDNN versions, and the hash covers
    sources, headers and flags, so any change yields a new directory. An explicit
    ``build_dir`` is returned unchanged. The architecture is left out of the hash, so
    a masked device still finds its build. A rebuild skips that search, because
    clearing the directory must not clear another architecture's build.
    """
    if build_dir is not None:
        # serialized_build() in define_library creates an explicit build_dir
        # before _clear_build_dir needs it to exist.
        return pathlib.Path(build_dir)

    fingerprint = _content_fingerprint(
        sources, include_root, extra_cflags, extra_cuda_cflags, with_cuda
    )
    environment = deformops.version.environment
    if archs is not None:
        return locate_build(name, tag=f"{environment}-{_arch_tag(archs)}-{fingerprint}")
    if not rebuild:
        found = _find_cached_library_for_any_arch(name, environment, fingerprint)
        if found is not None:
            return found
    return locate_build(name, tag=f"{environment}-unresolved-{fingerprint}")


#: Names whose latest :func:`define_library` call loaded a bundled or downloaded
#: library; see :func:`used_prebuilt`.
_prebuilt_resolved: set[str] = set()


def used_prebuilt(name: str) -> bool:
    r"""Whether ``name``'s latest :func:`define_library` call used a prebuilt library.

    Bundled or downloaded counts. A name never loaded, or whose prebuilt path was
    skipped or failed, reports ``False``.

    Parameters
    ----------
    name
        The extension name passed to :func:`define_library`.
    """
    return name in _prebuilt_resolved


def _load_cached_library(name: str, lib_file: pathlib.Path) -> bool:
    """Load a previously compiled library, discarding it if it is unloadable.

    A build killed mid-link leaves a truncated library under an unchanged
    fingerprint, so an unloadable file is removed and compiled again.

    Returns
    -------
    Whether the library was found and loaded.
    """
    if not lib_file.is_file():
        return False
    logger.info("Found existing library file for %s: %s", name, str(lib_file))
    try:
        torch.ops.load_library(str(lib_file))
    except OSError as error:
        warnings.warn(
            f"Discarding the cached {name!r} library at {lib_file}, which failed "
            f"to load ({error}); rebuilding it.",
            RuntimeWarning,
            stacklevel=3,
        )
        lib_file.unlink(missing_ok=True)
        return False
    return True


def define_library(  # noqa: PLR0913, PLR0915, C901
    name: str,
    build_dir: str | pathlib.Path | None = None,
    *,
    with_cuda: bool = True,
    opt_level: Literal[0, 1, 2, 3] = 2,
    cflags: Iterable[str] = (),
    cflags_cuda: Iterable[str] = (),
    download: bool | None = None,
    compile: bool | str | CompileMode | None = None,
) -> Any:
    r"""Make the kernel library ``name`` available and load it.

    Sources, in order: a library bundled with the install; one compiled on this
    machine for the same sources, flags and environment; a download from the
    prebuilt index; a local compile. ``compile`` set to ``always`` or
    ``rebuild`` skips the bundled and downloadable libraries.

    Parameters
    ----------
    name
        The ``EXTENSION_CONFIGS`` key; also the library file name
        (``f"{name}.so"``).
    build_dir
        Explicit build directory. When ``None``, one is derived from a content
        fingerprint of the sources, headers, flags and environment.
    with_cuda
        Whether to compile the CUDA source alongside the C++ source.
    opt_level
        Optimization level passed to both compilers.
    cflags, cflags_cuda
        Extra compiler flags, appended after the configured ones.
    download
        Allow the prebuilt download. ``None`` reads ``DEFORMOPS_DOWNLOAD``.
    compile
        A :class:`CompileMode`, its name, or a boolean (``True`` is
        ``fallback``). ``None`` reads ``DEFORMOPS_COMPILE``.

    Returns
    -------
    The path of the loaded library, or the compiled module.

    Raises
    ------
    KernelDownloadError
        If the download is enabled, needed, and fails.
    KernelNotProvidedError
        If no library is available and compiling is disabled.
    KernelBuildError
        If a compile is needed and the CUDA toolkit, the build tool (ninja) or a
        target architecture is missing.
    ValueError
        If ``DEFORMOPS_COMPILE``, ``DEFORMOPS_OPT_LEVEL`` or
        ``TORCH_CUDA_ARCH_LIST`` holds an invalid value.
    """
    mode = compile_mode() if compile is None else CompileMode.coerce(compile)
    allow_download = download_enabled() if download is None else download
    opt_level = _resolve_opt_level(opt_level)
    use_prebuilt = mode in (CompileMode.NEVER, CompileMode.FALLBACK)
    rebuild = mode is CompileMode.REBUILD

    if use_prebuilt:
        bundled = _try_load_prebuilt(name)
        if bundled is not None:
            _prebuilt_resolved.add(name)
            return bundled
    _prebuilt_resolved.discard(name)

    config = EXTENSION_CONFIGS[name]
    pkg_root = _DEFORMOPS_SRC_ROOT / config["package"].removeprefix(
        "deformops."
    ).replace(".", "/")
    include_root = _DEFORMOPS_SRC_ROOT / "backends" / "cuda" / "include"
    sources: list[str] = [str(pkg_root / f"{config['stem']}.cpp")]
    if with_cuda:
        sources.append(str(pkg_root / f"{config['stem']}.cu"))
    extra_include_paths: list[str] = [str(include_root)]
    extra_cflags: list[str] = [
        "-fdiagnostics-color=always",
        f"-O{opt_level}",
        *config["cflags"],
        *cflags,
    ]
    extra_cuda_cflags: list[str] = [
        f"-O{opt_level}",
        *config["cflags_cuda"],
        *cflags_cuda,
    ]
    archs = _resolve_cuda_archs() if with_cuda else None

    # The architecture is kept out of the content fingerprint that names this
    # directory; see _locate_jit_build_dir.
    build_dir = _locate_jit_build_dir(
        name,
        build_dir,
        archs,
        sources,
        include_root,
        extra_cflags,
        extra_cuda_cflags,
        with_cuda,
        rebuild,
    )
    lib_file = build_dir / f"{name}.so"

    if not rebuild:
        with serialized_build(build_dir):
            if _load_cached_library(name, lib_file):
                return str(lib_file)

    if use_prebuilt and allow_download:
        path = _download_prebuilt(name)
        _prebuilt_resolved.add(name)
        return path

    if mode is CompileMode.NEVER:
        why = (
            "downloading is disabled"
            if download is not None
            else f"downloading is off ({DOWNLOAD_ENV_VAR})"
        )
        msg = (
            f"The {name!r} kernel library is not bundled with this install, has "
            f"not been compiled on this machine, and {why}. Set "
            f"{COMPILE_ENV_VAR}=1 to compile it (needs a CUDA toolkit and "
            f"deformops[jit]), or set {DOWNLOAD_ENV_VAR}=1 to download it from a "
            f"prebuilt index.{_skipped_build_note()}"
        )
        raise KernelNotProvidedError(msg)

    if with_cuda:
        cuda_home = (
            os.getenv("CUDA_HOME")
            or os.getenv("CUDA_PATH")
            or torch.utils.cpp_extension.CUDA_HOME
        )
        if cuda_home is None or not pathlib.Path(cuda_home).is_dir():
            msg = (
                f"Compiling the {name!r} kernel library needs a CUDA toolkit, and "
                f"none was found. Install one and set CUDA_HOME to its root, or "
                f"set {COMPILE_ENV_VAR}=0 and use a prebuilt library."
            )
            raise KernelBuildError(msg)
        logger.debug("Using CUDA_HOME: %s", cuda_home)

    from .capabilities import require_jit_toolchain  # noqa: PLC0415

    require_jit_toolchain(name)

    with serialized_build(build_dir):
        if rebuild:
            _clear_build_dir(build_dir, name)
        elif _load_cached_library(name, lib_file):
            return str(lib_file)

        if with_cuda:
            if archs is None:
                raise KernelBuildError(str(_no_cuda_arch_error()))
            extra_cuda_cflags = [*extra_cuda_cflags, *_arch_flags_for(archs)]

        logger.info("Compiling the %s kernel library; this may take a while.", name)
        logger.debug("Sources: %s", sources)
        logger.debug("Extra include paths: %s", extra_include_paths)
        logger.debug("Extra CFLAGS: %s", extra_cflags)
        logger.debug("Extra CUDA CFLAGS: %s", extra_cuda_cflags)

        module = torch.utils.cpp_extension.load(
            name,
            build_directory=str(build_dir),
            with_cuda=with_cuda,
            sources=sources,
            extra_include_paths=extra_include_paths,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            keep_intermediates=False,
            is_python_module=False,
            is_standalone=False,
            verbose=True,
        )
        logger.info("Compiled the %s kernel library.", name)
    return module
