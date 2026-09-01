"""``DEFORMOPS_OPT_LEVEL`` overrides ``define_library``'s ``opt_level`` (default 2)."""

from pathlib import Path

import pytest
import torch
from deformops.core import build as _build
from deformops.core import capabilities
from deformops.core.configs import MSDA_OP_PREFIX


@pytest.fixture(autouse=True)
def _compile_locally(monkeypatch: pytest.MonkeyPatch) -> None:
    """Take the compile path with a stubbed compiler on any host."""
    monkeypatch.setenv("DEFORMOPS_COMPILE", "always")
    monkeypatch.setattr(capabilities, "has_jit_toolchain", lambda: True)


def test_env_override_changes_emitted_optimization_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DEFORMOPS_OPT_LEVEL", "0")

    captured: dict[str, object] = {}

    def _fake_load(name: str, **kwargs: object) -> str:
        captured.update(kwargs)
        return "sentinel-module"

    monkeypatch.setattr(torch.utils.cpp_extension, "load", _fake_load)

    result = _build.define_library(
        MSDA_OP_PREFIX,
        build_dir=str(tmp_path / "build"),
        with_cuda=False,
    )

    assert result == "sentinel-module"
    assert "-O0" in captured["extra_cflags"]
    assert "-O2" not in captured["extra_cflags"]


def test_env_override_absent_keeps_the_parameter_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("DEFORMOPS_OPT_LEVEL", raising=False)

    captured: dict[str, object] = {}

    def _fake_load(name: str, **kwargs: object) -> str:
        captured.update(kwargs)
        return "sentinel-module"

    monkeypatch.setattr(torch.utils.cpp_extension, "load", _fake_load)

    _build.define_library(
        MSDA_OP_PREFIX,
        build_dir=str(tmp_path / "build"),
        with_cuda=False,
    )

    assert "-O2" in captured["extra_cflags"]


def test_invalid_env_override_raises_rather_than_being_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEFORMOPS_OPT_LEVEL", "5")

    with pytest.raises(ValueError, match="DEFORMOPS_OPT_LEVEL"):
        _build.define_library(MSDA_OP_PREFIX)


def test_different_opt_levels_use_different_build_directories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two calls that differ only in optimization level get separate directories.

    The level is part of the directory name, so an ``-O0`` library is never served to
    an ``-O2`` caller. Uses the production ``build_dir=None`` path and the real
    fingerprint; ``with_cuda=False`` avoids the CUDA toolchain.
    """
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "extensions"))

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

    monkeypatch.setenv("DEFORMOPS_OPT_LEVEL", "0")
    _build.define_library(MSDA_OP_PREFIX, with_cuda=False)

    monkeypatch.setenv("DEFORMOPS_OPT_LEVEL", "2")
    _build.define_library(MSDA_OP_PREFIX, with_cuda=False)

    assert len(captured) == 2, (
        "the second call must not have hit the first call's cached .so; "
        "that would mean the two optimization levels collapsed onto one "
        "directory"
    )
    assert captured[0] != captured[1]
