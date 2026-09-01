"""``core.build.used_prebuilt``: whether the last resolution loaded a non-local library.

Every test runs ``define_library`` offline, with the bundled lookup, the download
and the compiler stubbed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from deformops.core import build as _build
from deformops.core import capabilities
from deformops.core.build import used_prebuilt
from deformops.core.configs import MSDA_OP_PREFIX
from deformops.core.errors import KernelNotProvidedError

NAME = MSDA_OP_PREFIX
OTHER = "another_extension"


@pytest.fixture(autouse=True)
def _clean_slate(monkeypatch: pytest.MonkeyPatch):
    """Every test starts with no name recorded, and a stubbed compiler."""
    saved = set(_build._prebuilt_resolved)
    _build._prebuilt_resolved.difference_update({NAME, OTHER})
    monkeypatch.setattr(capabilities, "has_jit_toolchain", lambda: True)
    monkeypatch.setattr(
        torch.utils.cpp_extension, "load", lambda *_a, **_k: "compiled-module"
    )
    yield
    _build._prebuilt_resolved.clear()
    _build._prebuilt_resolved.update(saved)


def _sources(
    monkeypatch: pytest.MonkeyPatch,
    *,
    bundled: str | None = None,
    downloaded: str | None = None,
) -> None:
    monkeypatch.setattr(_build, "_try_load_prebuilt", lambda _name: bundled)

    def _download(name: str) -> str:
        if downloaded is None:
            msg = f"no download stubbed for {name}"
            raise AssertionError(msg)
        return downloaded

    monkeypatch.setattr(_build, "_download_prebuilt", _download)


def _resolve(tmp_path: Path, **kwargs):
    return _build.define_library(
        NAME, build_dir=tmp_path / "build", with_cuda=False, **kwargs
    )


def test_unresolved_name_reports_false():
    """A name never resolved is not a working prebuilt, the safe default."""
    assert used_prebuilt(NAME) is False


def test_bundled_hit_reports_true(monkeypatch, tmp_path):
    _sources(monkeypatch, bundled=f"/fake/{NAME}.so")

    assert _resolve(tmp_path, download=False, compile="never") == f"/fake/{NAME}.so"
    assert used_prebuilt(NAME) is True


def test_downloaded_hit_reports_true(monkeypatch, tmp_path):
    _sources(monkeypatch, downloaded=f"/fake/cache/{NAME}.so")

    assert _resolve(tmp_path, download=True, compile="never") == (
        f"/fake/cache/{NAME}.so"
    )
    assert used_prebuilt(NAME) is True


def test_compiled_library_reports_false(monkeypatch, tmp_path):
    _sources(monkeypatch)

    assert _resolve(tmp_path, download=False, compile=True) == "compiled-module"
    assert used_prebuilt(NAME) is False


def test_cached_compiled_library_reports_false(monkeypatch, tmp_path):
    _sources(monkeypatch)
    library = tmp_path / "build" / f"{NAME}.so"
    library.parent.mkdir()
    library.write_bytes(b"compiled earlier")
    monkeypatch.setattr(torch.ops, "load_library", lambda _path: None, raising=False)

    assert _resolve(tmp_path, download=True, compile="never") == str(library)
    assert used_prebuilt(NAME) is False


def test_miss_raises_and_reports_false(monkeypatch, tmp_path):
    _sources(monkeypatch)

    with pytest.raises(KernelNotProvidedError):
        _resolve(tmp_path, download=False, compile="never")
    assert used_prebuilt(NAME) is False


def test_compile_always_clears_an_earlier_prebuilt_hit(monkeypatch, tmp_path):
    """``always`` skips the bundled library, so an earlier ``True`` does not persist."""
    _sources(monkeypatch, bundled=f"/fake/{NAME}.so")
    _resolve(tmp_path, download=False, compile="never")
    assert used_prebuilt(NAME) is True

    assert _resolve(tmp_path, download=False, compile="always") == "compiled-module"
    assert used_prebuilt(NAME) is False


def test_names_are_tracked_independently(monkeypatch, tmp_path):
    _sources(monkeypatch, bundled=f"/fake/{NAME}.so")

    _resolve(tmp_path, download=False, compile="never")

    assert used_prebuilt(NAME) is True
    assert used_prebuilt(OTHER) is False
