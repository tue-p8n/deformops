"""Version and environment description utilities for deformops."""

from typing import TYPE_CHECKING

__all__ = ["environment", "local", "package"]


def _get_package() -> str:
    """Return the installed distribution version, or the checkout's VERSION file.

    Returns
    -------
    The version string.

    Raises
    ------
    importlib.metadata.PackageNotFoundError
        If neither an installed distribution nor a checkout VERSION file
        can be found.
    """
    from importlib.metadata import PackageNotFoundError  # noqa: PLC0415
    from importlib.metadata import version as _meta_version  # noqa: PLC0415

    try:
        return _meta_version("deformops")
    except PackageNotFoundError:
        pass

    # A checkout without dist-info: VERSION is the single source of truth.
    from pathlib import Path  # noqa: PLC0415

    version_file = Path(__file__).parent.parent.parent / "VERSION"
    if not version_file.is_file():
        msg = "deformops"
        raise PackageNotFoundError(msg)
    return version_file.read_text(encoding="utf-8").strip()


def _get_environment() -> str:
    r"""Describe the current build environment in a one-line string.

    Notes
    -----
    This is agnostic of the Python version currently installed.
    """
    import platform  # noqa: PLC0415
    import sys  # noqa: PLC0415

    import torch  # noqa: PLC0415

    abi_str = f"{sys.platform}_{platform.machine()}"

    v_torch = torch.__version__.split("+", 1)[0]
    torch_str = f"torch{v_torch}"

    if getattr(torch.version, "cuda", None) is None:
        backend_str = "cpu"
    else:
        v_cuda = f"cu{torch.version.cuda}"

        if torch.backends.cudnn.is_available() and torch.backends.cudnn.enabled:
            backend_str = f"{v_cuda}-cudnn{torch.backends.cudnn.version()}"
        else:
            backend_str = f"{v_cuda}-nocudnn"

    return "-".join((abi_str, torch_str, backend_str)).replace(".", "_")


def _get_install() -> str:
    """Return the package version joined with the build environment string.

    Returns
    -------
    ``"<package version>+<environment>"``.
    """
    return f"{_get_package()}+{_get_environment()}"


if TYPE_CHECKING:
    package: str
    environment: str
    local: str

__cache__: dict[str, str] = {}


def __getattr__(name: str) -> str:
    """Resolve and cache one of this module's lazy version attributes.

    Parameters
    ----------
    name
        One of ``"package"``, ``"environment"``, ``"local"``.

    Returns
    -------
    The resolved, cached string.

    Raises
    ------
    AttributeError
        If ``name`` is not one of the supported attributes.
    """
    match name:
        case "package":
            return __cache__.setdefault(name, _get_package())
        case "environment":
            return __cache__.setdefault(name, _get_environment())
        case "local":
            return __cache__.setdefault(name, _get_install())
        case _:
            pass
    msg = f"Unsupported version attribute: {name}"
    raise AttributeError(msg)
