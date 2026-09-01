"""Where a CUDA kernel library comes from, and the public call that loads one.

Resolution order and the download and compile switches are described in
``docs/kernel-loading.md``. Every source is stubbed; nothing touches the network,
a compiler or the CUDA toolkit.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import torch
from deformops.core import build as _build
from deformops.core import capabilities
from deformops.core.build import CompileMode, compile_mode, download_enabled
from deformops.core.configs import MSDA_OP_PREFIX
from deformops.core.errors import (
    KernelBuildError,
    KernelDownloadError,
    KernelNotProvidedError,
    KernelUnavailableError,
    NoBackendError,
)
from deformops.registry import Backend, LoadedBackend, attn, conv, msda

NAME = MSDA_OP_PREFIX

requires_cpu_host = pytest.mark.skipif(
    torch.cuda.is_available(), reason="checks a host without a CUDA device"
)


@dataclasses.dataclass
class Sources:
    """Records which library sources ``define_library`` consulted."""

    build_dir: Path
    bundled: str | None = None
    downloaded: str | Exception | None = None
    calls: list[str] = dataclasses.field(default_factory=list)

    @property
    def library(self) -> Path:
        return self.build_dir / f"{NAME}.so"

    def plant_compiled_library(self) -> Path:
        self.build_dir.mkdir(parents=True, exist_ok=True)
        self.library.write_bytes(b"compiled earlier")
        return self.library


@pytest.fixture
def sources(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Sources:
    state = Sources(build_dir=tmp_path / "build")
    monkeypatch.delenv(_build.COMPILE_ENV_VAR, raising=False)
    monkeypatch.delenv(_build.DOWNLOAD_ENV_VAR, raising=False)
    saved = set(_build._prebuilt_resolved)
    _build._prebuilt_resolved.discard(NAME)

    def bundled(_name: str) -> str | None:
        state.calls.append("bundled")
        return state.bundled

    def downloaded(_name: str) -> str:
        state.calls.append("download")
        if isinstance(state.downloaded, Exception):
            raise state.downloaded
        if state.downloaded is None:
            msg = "the download was not expected"
            raise AssertionError(msg)
        return state.downloaded

    def cached(path: str) -> None:
        state.calls.append("cached")
        assert path == str(state.library)

    def compiled(name: str, *, build_directory: str, **_kwargs: object) -> str:
        state.calls.append("compile")
        assert name == NAME
        assert build_directory == str(state.build_dir)
        return "compiled-module"

    monkeypatch.setattr(_build, "_try_load_prebuilt", bundled)
    monkeypatch.setattr(_build, "_download_prebuilt", downloaded)
    monkeypatch.setattr(torch.ops, "load_library", cached, raising=False)
    monkeypatch.setattr(torch.utils.cpp_extension, "load", compiled)
    monkeypatch.setattr(capabilities, "has_jit_toolchain", lambda: True)
    yield state
    _build._prebuilt_resolved.clear()
    _build._prebuilt_resolved.update(saved)


def _define(state: Sources, **kwargs: object):
    return _build.define_library(
        NAME, build_dir=state.build_dir, with_cuda=False, **kwargs
    )


# --------------------------------------------------------------------------- #
# Resolution order
# --------------------------------------------------------------------------- #


def test_bundled_library_wins(sources):
    sources.bundled = "/bundled/lib.so"
    sources.downloaded = "/downloaded/lib.so"
    sources.plant_compiled_library()

    assert _define(sources, download=True, compile=True) == "/bundled/lib.so"
    assert sources.calls == ["bundled"]


def test_compiled_library_is_used_before_the_download(sources):
    library = sources.plant_compiled_library()

    assert _define(sources, download=True, compile=True) == str(library)
    assert sources.calls == ["bundled", "cached"]


def test_download_is_used_before_a_compile(sources):
    sources.downloaded = "/downloaded/lib.so"

    assert _define(sources, download=True, compile=True) == "/downloaded/lib.so"
    assert sources.calls == ["bundled", "download"]
    assert _build.used_prebuilt(NAME) is True


def test_a_failed_download_raises_even_when_compiling_is_enabled(sources):
    sources.downloaded = KernelDownloadError("index unreachable")

    with pytest.raises(KernelDownloadError, match="index unreachable"):
        _define(sources, download=True, compile=True)
    assert "compile" not in sources.calls


def test_compile_is_the_last_resort(sources):
    assert _define(sources, download=False, compile=True) == "compiled-module"
    assert sources.calls == ["bundled", "compile"]
    assert _build.used_prebuilt(NAME) is False


def test_download_and_compile_disabled_raises_not_provided(sources):
    with pytest.raises(KernelNotProvidedError, match="DEFORMOPS_COMPILE") as caught:
        _define(sources, download=False, compile=False)
    assert isinstance(caught.value, KernelUnavailableError)
    assert sources.calls == ["bundled"]


def test_not_provided_quotes_why_the_install_built_no_kernels(
    sources, monkeypatch, tmp_path
):
    (tmp_path / "build-skipped.txt").write_text("torch was not importable.\n")
    monkeypatch.setattr(_build.importlib.resources, "files", lambda _pkg: tmp_path)

    with pytest.raises(KernelNotProvidedError, match="torch was not importable"):
        _define(sources, download=False, compile=False)


def test_download_failure_quotes_why_the_install_built_no_kernels(
    monkeypatch, tmp_path
):
    (tmp_path / "build-skipped.txt").write_text("no CUDA device is visible.\n")
    monkeypatch.setattr(_build.importlib.resources, "files", lambda _pkg: tmp_path)

    message = _build._download_failure(NAME, None, "HTTP 404")  # noqa: SLF001
    assert message.endswith(
        "This install was built without its CUDA kernels: no CUDA device is visible."
    )


def test_not_provided_names_the_download_variable_when_the_env_disabled_it(
    sources, monkeypatch
):
    monkeypatch.setenv("DEFORMOPS_DOWNLOAD", "0")
    monkeypatch.setenv("DEFORMOPS_COMPILE", "0")

    with pytest.raises(KernelNotProvidedError, match="DEFORMOPS_DOWNLOAD"):
        _define(sources)


def test_compile_always_skips_the_bundled_library_and_the_download(sources):
    sources.bundled = "/bundled/lib.so"
    sources.downloaded = "/downloaded/lib.so"

    assert _define(sources, download=True, compile="always") == "compiled-module"
    assert sources.calls == ["compile"]


def test_compile_always_reuses_the_compiled_library(sources):
    library = sources.plant_compiled_library()

    assert _define(sources, download=True, compile="always") == str(library)
    assert sources.calls == ["cached"]


def test_compile_rebuild_clears_the_build_directory(sources):
    sources.bundled = "/bundled/lib.so"
    library = sources.plant_compiled_library()
    stale_object = sources.build_dir / "kernel.o"
    stale_object.write_bytes(b"stale")

    assert _define(sources, download=True, compile="rebuild") == "compiled-module"
    assert sources.calls == ["compile"]
    assert not library.exists()
    assert not stale_object.exists()


def test_a_missing_build_tool_raises_build_error(sources, monkeypatch):
    monkeypatch.setattr(capabilities, "has_jit_toolchain", lambda: False)

    with pytest.raises(KernelBuildError, match="ninja"):
        _define(sources, download=False, compile=True)
    assert "compile" not in sources.calls


def test_a_missing_cuda_toolkit_raises_build_error(sources, monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_HOME", str(tmp_path / "missing"))
    monkeypatch.delenv("CUDA_PATH", raising=False)

    with pytest.raises(KernelBuildError, match="CUDA toolkit"):
        _build.define_library(
            NAME, build_dir=sources.build_dir, download=False, compile=True
        )
    assert "compile" not in sources.calls


# --------------------------------------------------------------------------- #
# Environment defaults and keyword overrides
# --------------------------------------------------------------------------- #


def test_defaults_neither_download_nor_compile(sources):
    sources.downloaded = "/downloaded/lib.so"

    with pytest.raises(KernelNotProvidedError, match="DEFORMOPS_DOWNLOAD=1"):
        _define(sources)
    assert "download" not in sources.calls


def test_environment_enables_the_download(sources, monkeypatch):
    monkeypatch.setenv("DEFORMOPS_DOWNLOAD", "1")
    sources.downloaded = "/downloaded/lib.so"

    assert _define(sources) == "/downloaded/lib.so"
    assert sources.calls == ["bundled", "download"]


def test_environment_enables_the_compile(sources, monkeypatch):
    monkeypatch.setenv("DEFORMOPS_DOWNLOAD", "0")
    monkeypatch.setenv("DEFORMOPS_COMPILE", "1")

    assert _define(sources) == "compiled-module"


def test_keywords_override_the_environment(sources, monkeypatch):
    monkeypatch.setenv("DEFORMOPS_DOWNLOAD", "1")
    monkeypatch.setenv("DEFORMOPS_COMPILE", "0")

    assert _define(sources, download=False, compile=True) == "compiled-module"
    assert "download" not in sources.calls


def test_an_invalid_compile_variable_raises(sources, monkeypatch):
    monkeypatch.setenv("DEFORMOPS_COMPILE", "sometimes")

    with pytest.raises(ValueError, match="DEFORMOPS_COMPILE"):
        _define(sources)
    assert sources.calls == []


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", CompileMode.NEVER),
        ("0", CompileMode.NEVER),
        ("false", CompileMode.NEVER),
        ("no", CompileMode.NEVER),
        ("off", CompileMode.NEVER),
        ("never", CompileMode.NEVER),
        (False, CompileMode.NEVER),
        ("1", CompileMode.FALLBACK),
        ("true", CompileMode.FALLBACK),
        ("yes", CompileMode.FALLBACK),
        ("on", CompileMode.FALLBACK),
        ("fallback", CompileMode.FALLBACK),
        (True, CompileMode.FALLBACK),
        ("always", CompileMode.ALWAYS),
        (" Always ", CompileMode.ALWAYS),
        ("rebuild", CompileMode.REBUILD),
        ("REBUILD", CompileMode.REBUILD),
        (CompileMode.ALWAYS, CompileMode.ALWAYS),
    ],
)
def test_compile_mode_coerce(value, expected):
    assert CompileMode.coerce(value) is expected


@pytest.mark.parametrize("value", ["2", "sometimes", "jit", "prebuilt"])
def test_compile_mode_coerce_rejects_other_values(value):
    with pytest.raises(ValueError, match="DEFORMOPS_COMPILE"):
        CompileMode.coerce(value)


def test_compile_mode_defaults_to_never(monkeypatch):
    monkeypatch.delenv("DEFORMOPS_COMPILE", raising=False)
    assert compile_mode() is CompileMode.NEVER


@pytest.mark.parametrize(("value", "expected"), [("rebuild", CompileMode.REBUILD)])
def test_compile_mode_reads_the_environment(monkeypatch, value, expected):
    monkeypatch.setenv("DEFORMOPS_COMPILE", value)
    assert compile_mode() is expected


def test_download_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("DEFORMOPS_DOWNLOAD", raising=False)
    assert download_enabled() is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", False),
        ("1", True),
        ("true", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("off", False),
        (" OFF ", False),
    ],
)
def test_download_enabled_values(monkeypatch, value, expected):
    monkeypatch.setenv("DEFORMOPS_DOWNLOAD", value)
    assert download_enabled() is expected


# --------------------------------------------------------------------------- #
# Registry.load
# --------------------------------------------------------------------------- #


def test_load_rejects_a_backend_name_string():
    with pytest.raises(TypeError, match="Backend"):
        msda.load("cuda")  # type: ignore[arg-type]


@pytest.mark.parametrize("registry", [msda, attn, conv], ids=lambda r: r.name)
def test_load_eager_has_no_library(registry):
    expected = LoadedBackend(registry.name, Backend.EAGER, None)
    assert registry.load(Backend.EAGER) == expected


@requires_cpu_host
@pytest.mark.parametrize("backend", [Backend.TRITON, Backend.CUDA])
@pytest.mark.parametrize("registry", [msda, attn, conv], ids=lambda r: r.name)
def test_load_raises_no_backend_for_a_device_backend_on_a_cpu_host(registry, backend):
    with pytest.raises(NoBackendError, match="unavailable") as caught:
        registry.load(backend)
    assert not isinstance(caught.value, KernelUnavailableError)


def test_load_cuda_msda_loads_the_production_library(monkeypatch):
    """``load`` forwards to the production configuration and reports the path."""

    msda.is_registered(Backend.CUDA)  # import the adapters first
    monkeypatch.setitem(msda._backends, Backend.CUDA, lambda *_a, **_k: None)
    calls: list[dict[str, object]] = []

    def fake_load(**kwargs: object) -> str:
        calls.append(kwargs)
        return "/compiled/msda_fp32acc.so"

    monkeypatch.setitem(msda._loaders, Backend.CUDA, fake_load)

    loaded = msda.load(Backend.CUDA, download=False, compile="always")

    assert loaded == LoadedBackend("msda", Backend.CUDA, "/compiled/msda_fp32acc.so")
    assert calls == [{"download": False, "compile": "always"}]


def test_load_cuda_msda_propagates_a_kernel_error(monkeypatch):

    msda.is_registered(Backend.CUDA)
    monkeypatch.setitem(msda._backends, Backend.CUDA, lambda *_a, **_k: None)

    def fake_load(**_kwargs: object) -> str:
        msg = "no library"
        raise KernelNotProvidedError(msg)

    monkeypatch.setitem(msda._loaders, Backend.CUDA, fake_load)

    with pytest.raises(KernelNotProvidedError, match="no library"):
        msda.load(Backend.CUDA)
