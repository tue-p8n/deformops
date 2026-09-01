"""Keep every build system pointed at the same pre-built extension directory.

``setup.py``, the Nix ops and wheel derivations and ``build.py`` each name the
directory; only ``build.py`` runs in the suite. Each must agree with
:data:`deformops.core.configs.EXT_PACKAGE`.
"""

from __future__ import annotations

import pathlib
import re

import pytest
from deformops.core.configs import EXT_PACKAGE

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

#: Where ``setup.py build_ext --inplace`` writes, relative to the repository root,
#: derived from the package name's single declaration.
EXT_SOURCE_DIR = pathlib.PurePosixPath("sources", *EXT_PACKAGE.split("."))

#: Files that hardcode the directory as a shell path; Nix cannot import Python.
_NIX_FILES = ("nix/ops.nix", "nix/wheel.nix")

#: Any ``sources/deformops/**/_ext`` path appearing in a build script.
_EXT_PATH_RE = re.compile(r"sources/deformops/[A-Za-z0-9_/]*_ext")


def _skip_without_checkout(path: pathlib.Path) -> str:
    if not path.is_file():
        pytest.skip(f"running outside a source checkout; {path.name} is not shipped")
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("relative", _NIX_FILES)
def test_nix_derivation_uses_the_current_ext_dir(relative: str) -> None:
    """Every ``_ext`` path in a Nix derivation points at the current package."""
    text = _skip_without_checkout(_REPO_ROOT / relative)
    found = set(_EXT_PATH_RE.findall(text))
    assert found, (
        f"{relative} mentions no _ext path. Either the derivation does not copy "
        f"pre-built libraries, and this test does not apply to it, or the path "
        f"has a spelling this test cannot see."
    )
    assert found == {str(EXT_SOURCE_DIR)}, (
        f"{relative} refers to {sorted(found)}, but setup.py builds into "
        f"{EXT_SOURCE_DIR} (from configs.EXT_PACKAGE = {EXT_PACKAGE!r}). "
        f"The per-CUDA build would produce no libraries, or libraries at a "
        f"path the runtime loader does not look in."
    )


def test_setup_py_derives_the_ext_dir_rather_than_repeating_it() -> None:
    """``setup.py`` takes the package name from configs instead of literals."""
    text = _skip_without_checkout(_REPO_ROOT / "setup.py")
    assert "configs_mod.EXT_PACKAGE" in text, (
        "setup.py should derive the pre-built extension package from "
        "configs.EXT_PACKAGE so it cannot drift from the runtime loader."
    )
    assert not _EXT_PATH_RE.search(text), (
        "setup.py hardcodes an _ext source path; derive it from "
        "configs.EXT_PACKAGE via _package_to_path instead."
    )


def test_runtime_loader_derives_the_ext_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``build.py`` resolves ``EXT_PACKAGE`` at call time instead of using a literal.

    Drives the real function, since a text match passes on a dead branch or comment.
    """
    from deformops.core import build

    sentinel = "sentinel.package.that.does.not.exist"
    monkeypatch.setattr(build, "EXT_PACKAGE", sentinel)

    requested: list[str] = []
    original_files = build.importlib.resources.files

    def _spy(package: str):
        requested.append(package)
        return original_files(package)

    monkeypatch.setattr(build.importlib.resources, "files", _spy)

    result = build._try_load_prebuilt("does-not-matter")

    assert requested == [sentinel], (
        "_try_load_prebuilt should look up the patched build.EXT_PACKAGE at "
        "call time, not a package name baked in elsewhere."
    )
    assert result is None
