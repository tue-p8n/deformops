import os
import pathlib
import subprocess
import sys
from importlib.machinery import SourceFileLoader

from setuptools import find_packages, setup

# VERSION holds the base version; it is ``dynamic`` so CUDA wheels can add +cuXXX.
_BASE_VERSION: str = (
    (pathlib.Path(__file__).parent / "VERSION").read_text(encoding="utf-8").strip()
)

project_root = pathlib.Path(__file__).parent
sources_root = project_root / "sources"
configs_mod = SourceFileLoader(
    "deformops.core.configs",
    str(sources_root / "deformops" / "core" / "configs.py"),
).load_module()

EXTENSION_CONFIGS = configs_mod.EXTENSION_CONFIGS
EXTENSIONS_ENV_VAR = configs_mod.EXTENSIONS_ENV_VAR

# PKG-INFO marks an unpacked sdist, which compiles the non-experimental kernels
# by default; a checkout compiles nothing. See docs/kernel-loading.md.
SELECTED_EXTENSIONS = configs_mod.select_extensions(
    os.getenv(EXTENSIONS_ENV_VAR),
    source_distribution=(project_root / "PKG-INFO").is_file(),
)

# Explicitly requested kernels must compile. The default compiles only against
# an importable CUDA torch, so an isolated build or a missing toolkit skips.
EXTENSIONS_REQUESTED = (
    os.getenv(EXTENSIONS_ENV_VAR) is not None or os.getenv("FORCE_CUDA") == "1"
)
CUDA_TOOLKIT = configs_mod.find_cuda_toolkit()
if not EXTENSIONS_REQUESTED and CUDA_TOOLKIT is None:
    SELECTED_EXTENSIONS = {}

# Only these commands compile; metadata and sdist need neither torch nor CUDA.
_COMPILE_COMMANDS = frozenset(
    {"build", "build_ext", "bdist_wheel", "install", "develop", "editable_wheel"}
)

# Usage: OPT_LEVEL=-O1 DEFORMOPS_EXTENSIONS=all uv pip install -e .
OPT_LEVEL = os.getenv("OPT_LEVEL", "-O3")

# The extension package comes from configs, shared with the loader and Nix.
# The include path is absolute because ninja compiles from build/temp.*.
_EXT_PARENT_PACKAGE = configs_mod.EXT_PACKAGE
_INCLUDE_DIR = str(sources_root / "deformops" / "backends" / "cuda" / "include")


def _package_to_path(package: str) -> str:
    """Translate a dotted package name to its source directory."""
    return os.path.join("sources", *package.split("."))


_SKIP_MARKER = sources_root.joinpath(
    *configs_mod.EXT_PACKAGE.split("."), configs_mod.SKIPPED_BUILD_FILE
)


def _record_skipped_build(reason: str, *directories: pathlib.Path) -> None:
    """Write why the kernels were not compiled to stderr and the runtime's marker."""
    sys.stderr.write(f"deformops: installing without compiled CUDA kernels: {reason}\n")
    for directory in (_SKIP_MARKER.parent, *directories):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / configs_mod.SKIPPED_BUILD_FILE).write_text(
            reason + "\n", encoding="utf-8"
        )


def _toolkit_release() -> tuple[int, int] | None:
    if CUDA_TOOLKIT is None:
        return None
    try:
        output = subprocess.run(
            [os.path.join(CUDA_TOOLKIT, "bin", "nvcc"), "--version"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return configs_mod.parse_nvcc_release(output)


class _SkipBuild(Exception):  # noqa: N818
    """The selected kernels cannot be compiled in this build environment."""

    def __init__(self, reason: str, *, report: bool = True) -> None:
        super().__init__(reason)
        self.reason = reason
        self.report = report


_REINSTALL = "reinstall deformops with --no-build-isolation"


def _check_build_environment():  # type: ignore[no-untyped-def]
    """The CUDA extension class, once the environment can compile the kernels.

    Raises
    ------
    _SkipBuild
        With the reason the kernels cannot be compiled here.
    """
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME, CUDAExtension
    except ImportError as exc:
        reason = (
            "torch is not importable in the build environment, which is isolated "
            "from the one deformops runs in. Install torch with CUDA support "
            f"first, then {_REINSTALL} to compile against it."
        )
        raise _SkipBuild(reason) from exc
    if torch.version.cuda is None:
        # Nothing to report: a CPU torch cannot run the kernels.
        reason = "the installed torch is a CPU build."
        raise _SkipBuild(reason, report=False)
    if CUDA_HOME is None:
        reason = "no CUDA toolkit was found. Set CUDA_HOME to its location."
        raise _SkipBuild(reason)
    release = _toolkit_release()
    torch_major = int(torch.version.cuda.split(".")[0])
    if release is not None and release[0] != torch_major:
        reason = (
            f"the CUDA toolkit at {CUDA_TOOLKIT} is CUDA {release[0]}.{release[1]}, "
            f"and torch {torch.__version__} is built for CUDA {torch.version.cuda}. "
            f"Point CUDA_HOME at a CUDA {torch_major} toolkit and {_REINSTALL}."
        )
        raise _SkipBuild(reason)
    if not (
        torch.cuda.is_available()
        or os.getenv("FORCE_CUDA", "0") == "1"
        or os.getenv("TORCH_CUDA_ARCH_LIST")
    ):
        reason = (
            "no CUDA device is visible to pick the target architecture. Set "
            f"TORCH_CUDA_ARCH_LIST and {_REINSTALL}."
        )
        raise _SkipBuild(reason)
    return CUDAExtension


def get_extensions() -> list:  # type: ignore[type-arg]
    """The CUDA extensions this build compiles, from ``SELECTED_EXTENSIONS``."""
    if not SELECTED_EXTENSIONS or not _COMPILE_COMMANDS.intersection(sys.argv[1:]):
        return []
    _SKIP_MARKER.unlink(missing_ok=True)
    selected = SELECTED_EXTENSIONS
    try:
        cuda_extension = _check_build_environment()
    except _SkipBuild as skip:
        if EXTENSIONS_REQUESTED:
            msg = (
                f"Cannot compile the deformops CUDA kernels ({', '.join(selected)}): "
                f"{skip.reason} Set {EXTENSIONS_ENV_VAR}=none to install without "
                f"compiled kernels."
            )
            raise RuntimeError(msg) from skip
        if skip.report:
            _record_skipped_build(skip.reason)
        return []

    extensions = []
    # An editable install needs the destination package to exist.
    ext_parent = _package_to_path(_EXT_PARENT_PACKAGE)
    os.makedirs(ext_parent, exist_ok=True)

    for name, config in selected.items():
        # Same package/stem mapping as the JIT path in core/build.py.
        src_dir = _package_to_path(config["package"])
        stem = config["stem"]
        cpp_source = os.path.join(src_dir, f"{stem}.cpp")
        cu_source = os.path.join(src_dir, f"{stem}.cu")

        if not os.path.isfile(cpp_source):
            msg = (
                f"Extension {name!r} declares package={config['package']!r} "
                f"and stem={stem!r}, but {cpp_source} is missing."
            )
            raise FileNotFoundError(msg)

        sources = [cpp_source]
        # Ahead-of-time builds drop float64, which only gradient checks use; the
        # JIT path keeps it. See docs/kernel-loading.md.
        cflags = [OPT_LEVEL, "-DDEFORMOPS_NO_FP64"] + list(config["cflags"])
        cflags_cuda = [OPT_LEVEL, "-DDEFORMOPS_NO_FP64"] + list(config["cflags_cuda"])

        if not os.path.isfile(cu_source):
            msg = (
                f"Extension {name!r} requested CUDA build but {cu_source} "
                f"does not exist."
            )
            raise FileNotFoundError(msg)
        sources.append(cu_source)
        extension = cuda_extension(
            name=f"{_EXT_PARENT_PACKAGE}.{name}",
            sources=sources,
            extra_compile_args={
                "cxx": cflags,
                "nvcc": cflags_cuda,
            },
            include_dirs=[_INCLUDE_DIR],
        )
        extensions.append(extension)

    return extensions


# A +cuXXX local version lets CUDA variants coexist on one package index.
version = _BASE_VERSION
cuda_version_env = os.getenv("CUDA_VERSION")
if cuda_version_env:
    version += f"+cu{cuda_version_env.replace('.', '')}"
elif os.getenv("FORCE_CUDA") == "1":
    import torch

    torch_cuda = torch.version.cuda
    if torch.cuda.is_available() and torch_cuda is not None:
        major, minor = torch_cuda.split(".")
        version += f"+cu{major}{minor}"


def _platlib_bdist_wheel():
    """A ``bdist_wheel`` tagging ``py3-none-<platform>``.

    Kernels ship as package data, which setuptools would tag ``any``; a
    stable-ABI ``.so`` needs a platform tag but no Python or ABI tag.
    """
    try:
        from setuptools.command.bdist_wheel import bdist_wheel as _bw
    except ImportError:  # pragma: no cover - older setuptools
        from wheel.bdist_wheel import bdist_wheel as _bw

    class _PlatBdistWheel(_bw):  # type: ignore[valid-type, misc]
        def finalize_options(self) -> None:
            super().finalize_options()
            self.root_is_pure = False

        def get_tag(self):  # type: ignore[no-untyped-def]
            _python, _abi, plat = super().get_tag()
            return "py3", "none", plat

    return _PlatBdistWheel


def _build_ext_with_cuda_objects():
    """A ``build_ext`` writing bare ``<name>.so`` libraries.

    ``.cu`` sources compile to ``<stem>.cuda.o``, as torch's JIT names them, so
    the ``<stem>.cpp``/``<stem>.cu`` pair does not collide on ``<stem>.o``.
    """
    from torch.utils.cpp_extension import BuildExtension

    class _BuildExt(BuildExtension):  # type: ignore[valid-type, misc]
        def get_ext_filename(self, fullname: str) -> str:
            # Loaded with torch.ops.load_library, not imported.
            return os.path.join(*fullname.split(".")) + ".so"

        def build_extensions(self) -> None:
            original = self.compiler.object_filenames

            def object_filenames(source_filenames, strip_dir=False, output_dir=""):
                sources = list(source_filenames)
                objects = original(sources, strip_dir, output_dir)
                return [
                    f"{os.path.splitext(obj)[0]}.cuda.o"
                    if os.path.splitext(src)[1] == ".cu"
                    else obj
                    for src, obj in zip(sources, objects, strict=True)
                ]

            self.compiler.object_filenames = object_filenames
            try:
                super().build_extensions()
            except Exception as exc:
                if EXTENSIONS_REQUESTED:
                    raise
                # The default selection never fails the install.
                build_lib = pathlib.Path(self.build_lib).joinpath(
                    *_EXT_PARENT_PACKAGE.split(".")
                )
                _record_skipped_build(
                    f"compiling failed ({type(exc).__name__}: {exc}). Fix the "
                    f"build environment and reinstall deformops with "
                    f"--no-build-isolation.",
                    build_lib,
                )

    return _BuildExt


ext_modules = get_extensions()
cmdclass = {}
if ext_modules:
    cmdclass["build_ext"] = _build_ext_with_cuda_objects()

# CUDA wheels bundle prebuilt kernels and need a platform tag.
if os.getenv("DEFORMOPS_WHEEL_PLATLIB") == "1":
    cmdclass["bdist_wheel"] = _platlib_bdist_wheel()

setup(
    name="deformops",
    version=version,
    packages=find_packages(where="sources"),
    package_dir={"": "sources"},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    # Wheels need this list; ``include-package-data`` covers the sdist.
    package_data={
        "deformops.core": [
            "_ext/*.so",
            "_ext/*.pyd",
            "_ext/build-skipped.txt",
        ],
        # Kernel sources, so an install without a prebuilt library can compile.
        "deformops.backends.cuda": [
            "*.cpp",
            "*.cu",
            "include/*.cuh",
        ],
        "deformops.backends.cuda.experimental": [
            "*.cpp",
            "*.cu",
        ],
    },
)
