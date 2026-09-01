"""Explicit CUDA architecture flags in the JIT build path.

Without flags, ``cpp_extension.load()`` raises an unguarded ``IndexError`` when no
device is visible and ``TORCH_CUDA_ARCH_LIST`` is unset; see
``docs/kernel-loading.md``.
"""

import hashlib
from pathlib import Path

import pytest
import torch
from deformops.core import build as _build
from deformops.core import capabilities
from deformops.core.configs import MSDA_OP_PREFIX
from deformops.core.errors import KernelBuildError


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
    # A compile reached here runs a stubbed compiler; the build tool check
    # must not depend on what this host has installed.
    monkeypatch.setattr(capabilities, "has_jit_toolchain", lambda: True)


def test_raises_actionable_error_without_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

    with pytest.raises(RuntimeError, match="TORCH_CUDA_ARCH_LIST"):
        _build._cuda_arch_flags()


def test_derives_flags_from_current_device_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A visible device gets a ``+PTX`` fallback too, as torch's autodetection does."""
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _index: (8, 6))

    assert _build._cuda_arch_flags() == [
        "-gencode=arch=compute_86,code=compute_86",
        "-gencode=arch=compute_86,code=sm_86",
    ]


def test_derives_flags_for_every_visible_device_not_just_device_0(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gencode covers every visible device, including those after device 0.

    A library built for device 0 alone fails on the other cards of a mixed node.
    """
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    capabilities = {0: (8, 0), 1: (8, 6)}
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda index: capabilities[index]
    )

    assert _build._cuda_arch_flags() == [
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_86,code=compute_86",
        "-gencode=arch=compute_86,code=sm_86",
    ]


def test_multi_device_arch_flags_are_deterministic_across_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The flag order is stable, since it feeds the build-directory fingerprint."""
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    capabilities = {0: (8, 6), 1: (7, 5), 2: (9, 0)}
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda index: capabilities[index]
    )

    first = _build._cuda_arch_flags()
    second = _build._cuda_arch_flags()

    assert first == second
    assert first  # sanity: not vacuously equal empty lists


def test_multi_device_sort_is_numeric_not_lexical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``10.0`` sorts after ``8.6``, so ``+PTX`` applies to the newest card.

    A string sort gets this backwards: ``"10.0" < "8.6"``.
    """
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    capabilities = {0: (10, 0), 1: (8, 6)}
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda index: capabilities[index]
    )

    flags = _build._cuda_arch_flags()
    assert "-gencode=arch=compute_100,code=compute_100" in flags, (
        "the newest card (10.0) must get the +PTX fallback, not 8.6"
    )
    assert "-gencode=arch=compute_86,code=sm_86" in flags
    assert "-gencode=arch=compute_100,code=sm_100" in flags
    assert "-gencode=arch=compute_86,code=compute_86" not in flags, (
        "+PTX must not land on the older card"
    )


def test_multi_device_arch_flags_deduplicate_identical_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Several devices sharing the same compute capability (the common case
    on a homogeneous multi-GPU node) must not duplicate gencode flags."""
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _index: (8, 6))

    assert _build._cuda_arch_flags() == [
        "-gencode=arch=compute_86,code=compute_86",
        "-gencode=arch=compute_86,code=sm_86",
    ]


@pytest.mark.parametrize("value", ["native", "NATIVE", " native "])
def test_native_arch_list_falls_back_to_device_capability(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """``TORCH_CUDA_ARCH_LIST=native`` means autodetect, the same as unset."""
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", value)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _index: (8, 6))

    assert _build._cuda_arch_flags() == [
        "-gencode=arch=compute_86,code=compute_86",
        "-gencode=arch=compute_86,code=sm_86",
    ]


def test_native_arch_list_without_device_still_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "native")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

    with pytest.raises(RuntimeError, match="TORCH_CUDA_ARCH_LIST"):
        _build._cuda_arch_flags()


def test_malformed_arch_list_entry_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "Ampere")

    with pytest.raises(ValueError, match="TORCH_CUDA_ARCH_LIST"):
        _build._cuda_arch_flags()


def test_honors_explicit_arch_list_without_touching_the_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "7.5")

    def _boom() -> int:
        raise AssertionError(
            "device_count() should not be queried when TORCH_CUDA_ARCH_LIST "
            "is set explicitly"
        )

    monkeypatch.setattr(torch.cuda, "device_count", _boom)

    assert _build._cuda_arch_flags() == [
        "-gencode=arch=compute_75,code=sm_75",
    ]


def test_honors_explicit_arch_list_with_ptx_and_multiple_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "6.1;8.6+PTX")

    assert _build._cuda_arch_flags() == [
        "-gencode=arch=compute_61,code=sm_61",
        "-gencode=arch=compute_86,code=compute_86",
        "-gencode=arch=compute_86,code=sm_86",
    ]


def test_ptx_suffix_is_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.6+ptx")

    assert _build._cuda_arch_flags() == [
        "-gencode=arch=compute_86,code=compute_86",
        "-gencode=arch=compute_86,code=sm_86",
    ]


def test_define_library_passes_all_device_archs_to_the_compiler(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The multi-device flags reach ``load()``'s ``extra_cuda_cflags``."""
    monkeypatch.setenv("DEFORMOPS_COMPILE", "always")
    monkeypatch.setenv("CUDA_HOME", str(tmp_path))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    capabilities = {0: (8, 0), 1: (8, 6)}
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda index: capabilities[index]
    )

    captured: dict[str, object] = {}

    def _fake_load(name: str, **kwargs: object) -> str:
        captured.update(kwargs)
        return "sentinel-module"

    monkeypatch.setattr(torch.utils.cpp_extension, "load", _fake_load)

    result = _build.define_library(
        MSDA_OP_PREFIX,
        build_dir=str(tmp_path / "build"),
        with_cuda=True,
    )

    assert result == "sentinel-module"
    extra_cuda_cflags = captured["extra_cuda_cflags"]
    assert "-gencode=arch=compute_80,code=sm_80" in extra_cuda_cflags
    assert "-gencode=arch=compute_86,code=sm_86" in extra_cuda_cflags
    assert "-gencode=arch=compute_86,code=compute_86" in extra_cuda_cflags


def test_define_library_uses_different_directories_for_different_archs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two calls that differ only in architecture get separate directories.

    The architecture is outside the content fingerprint, so the directory name
    includes it. Uses the production ``build_dir=None`` path.
    """
    monkeypatch.setenv("DEFORMOPS_COMPILE", "always")
    cuda_home = tmp_path / "cuda_home"
    cuda_home.mkdir()
    monkeypatch.setenv("CUDA_HOME", str(cuda_home))
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "extensions"))
    monkeypatch.setattr(_build, "_content_fingerprint", _fake_content_fingerprint)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    captured: list[str] = []

    def _fake_load(name: str, *, build_directory: str, **_kwargs: object) -> str:
        captured.append(build_directory)
        (Path(build_directory) / f"{name}.so").write_bytes(b"fake")
        return "sentinel-module"

    monkeypatch.setattr(torch.utils.cpp_extension, "load", _fake_load)
    # A shared directory would load the fake library from the cache, and the
    # assertion on `captured` would fail before dlopen.
    monkeypatch.setattr(
        torch.ops, "load_library", lambda *_a, **_k: None, raising=False
    )

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _index: (8, 6))
    _build.define_library(MSDA_OP_PREFIX, with_cuda=True)

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _index: (8, 0))
    _build.define_library(MSDA_OP_PREFIX, with_cuda=True)

    assert len(captured) == 2, (
        "the second call must not have hit the first call's cached .so; "
        "that would mean the two architectures collapsed onto one directory"
    )
    assert captured[0] != captured[1]


def test_define_library_raises_actionable_error_without_device(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With no device and no cached library, a CUDA compile raises ``KernelBuildError``.

    An explicit ``build_dir`` skips content fingerprinting, which is not under test.
    """
    monkeypatch.setenv("DEFORMOPS_COMPILE", "always")
    monkeypatch.setenv("CUDA_HOME", str(tmp_path))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

    with pytest.raises(KernelBuildError, match="TORCH_CUDA_ARCH_LIST"):
        _build.define_library(
            MSDA_OP_PREFIX,
            build_dir=str(tmp_path / "build"),
            with_cuda=True,
        )


def test_define_library_returns_cached_library_without_device(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Locating an already-built library needs no CUDA architecture; compiling does.

    The explicit ``build_dir`` skips the fingerprint; the path without it is covered by
    ``test_define_library_reuses_cached_library_after_device_becomes_invisible``.
    """
    monkeypatch.setenv("DEFORMOPS_COMPILE", "always")
    monkeypatch.setenv("CUDA_HOME", str(tmp_path))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

    build_dir = tmp_path / "build"
    build_dir.mkdir()
    lib_file = build_dir / f"{MSDA_OP_PREFIX}.so"
    lib_file.write_bytes(b"not actually a shared object")

    loaded: list[str] = []
    monkeypatch.setattr(torch.ops, "load_library", loaded.append, raising=False)

    result = _build.define_library(
        MSDA_OP_PREFIX, build_dir=str(build_dir), with_cuda=True
    )

    assert result == str(lib_file)
    assert loaded == [str(lib_file)]


def _fake_content_fingerprint(
    sources: object,
    include_root: object,
    extra_cflags: object,
    extra_cuda_cflags: object,
    with_cuda: object,
) -> str:
    """Hash the arguments of ``_content_fingerprint`` instead of the files on disk.

    It still depends on ``extra_cuda_cflags``: whether architecture flags join that
    list before fingerprinting is the coupling under test.
    """
    payload = repr(
        (
            list(sources),
            str(include_root),
            list(extra_cflags),
            list(extra_cuda_cflags),
            with_cuda,
        )
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def test_define_library_reuses_cached_library_after_device_becomes_invisible(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A library built with a device visible is found once the device is gone.

    Covers the path without ``build_dir``, where architecture flags are not part of
    the fingerprint that names the build directory. Builds once with a device, then
    calls again with none and ``TORCH_CUDA_ARCH_LIST`` unset, and expects the cache.
    """
    monkeypatch.setenv("DEFORMOPS_COMPILE", "always")
    cuda_home = tmp_path / "cuda_home"
    cuda_home.mkdir()
    monkeypatch.setenv("CUDA_HOME", str(cuda_home))
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "extensions"))
    monkeypatch.setattr(_build, "_content_fingerprint", _fake_content_fingerprint)

    built_at: dict[str, str] = {}

    def _fake_load(name: str, *, build_directory: str, **_kwargs: object) -> str:
        so_path = Path(build_directory) / f"{name}.so"
        so_path.write_bytes(b"not actually a shared object")
        built_at["build_directory"] = build_directory
        return "sentinel-module"

    monkeypatch.setattr(torch.utils.cpp_extension, "load", _fake_load)

    # Build once with a device visible.
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _index: (8, 6))

    first = _build.define_library(MSDA_OP_PREFIX, with_cuda=True)
    assert first == "sentinel-module"
    assert "build_directory" in built_at

    # Re-enter with the device gone and no explicit arch list.
    loaded: list[str] = []
    monkeypatch.setattr(torch.ops, "load_library", loaded.append, raising=False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

    expected = str(Path(built_at["build_directory"]) / f"{MSDA_OP_PREFIX}.so")
    second = _build.define_library(MSDA_OP_PREFIX, with_cuda=True)

    assert second == expected
    assert loaded == [expected]


def test_rebuild_strategy_never_clears_another_architectures_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``CompileMode.REBUILD`` skips the any-architecture cache lookup.

    ``REBUILD`` wipes the directory it resolves, and that lookup finds a directory
    built for a different architecture.
    """
    monkeypatch.setenv("DEFORMOPS_COMPILE", "rebuild")
    cuda_home = tmp_path / "cuda_home"
    cuda_home.mkdir()
    monkeypatch.setenv("CUDA_HOME", str(cuda_home))
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "extensions"))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

    # Fix this call's fingerprint so a planted directory matches the lookup glob.
    monkeypatch.setattr(
        _build, "_content_fingerprint", lambda *args, **kwargs: "somefingerprint"
    )
    environment = _build.deformops.version.environment

    # Plant a library as if a *different*, previously-resolved architecture
    # had already built one matching this same fingerprint.
    other_arch_dir = (
        tmp_path
        / "extensions"
        / f"{MSDA_OP_PREFIX}-{environment}-somearchtag-somefingerprint"
    )
    other_arch_dir.mkdir(parents=True)
    other_so = other_arch_dir / f"{MSDA_OP_PREFIX}.so"
    other_so.write_bytes(b"belongs to a different architecture's build")

    with pytest.raises(KernelBuildError, match="TORCH_CUDA_ARCH_LIST"):
        _build.define_library(MSDA_OP_PREFIX, with_cuda=True)

    assert other_so.is_file(), (
        "REBUILD must never delete a different architecture's cached library"
    )
