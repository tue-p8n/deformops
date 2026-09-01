"""Keep the three declarations of the supported CUDA versions in step.

Parses and compares the ``pyproject.toml`` extras, the ``flake.nix`` variants and
the Pages workflow matrix. The manual workflow finds a mismatch only in a release
build.
"""

from __future__ import annotations

import pathlib
import re
import tomllib

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

_EXTRA_RE = re.compile(r"^cu\d{3}$")
_FLAKE_EXTRA_RE = re.compile(r'extra\s*=\s*"(cu\d{3})"')
_MATRIX_RE = re.compile(r"^\s*cuda:\s*\[([^\]]*)\]", re.MULTILINE)


def _read(relative: str) -> str:
    path = _REPO_ROOT / relative
    if not path.is_file():
        pytest.skip(f"running outside a source checkout; {relative} is not shipped")
    return path.read_text(encoding="utf-8")


def _pyproject_extras() -> set[str]:
    data = tomllib.loads(_read("pyproject.toml"))
    extras = data["project"]["optional-dependencies"]
    return {name for name in extras if _EXTRA_RE.match(name)}


def _flake_variants() -> set[str]:
    return set(_FLAKE_EXTRA_RE.findall(_read("flake.nix")))


def _pages_matrices() -> list[set[str]]:
    text = _read(".github/workflows/pages.yml")
    return [
        {entry.strip() for entry in match.group(1).split(",") if entry.strip()}
        for match in _MATRIX_RE.finditer(text)
    ]


def test_each_source_declares_something() -> None:
    """Each parse matches at least one entry."""
    assert _pyproject_extras(), "no cuXXX extras found in pyproject.toml"
    assert _flake_variants(), "no cuXXX variants found in flake.nix"
    matrices = _pages_matrices()
    assert matrices, "no cuda matrix found in pages.yml"
    # The pairwise comparison below passes vacuously if the regex matches only
    # one of the two jobs' matrices, so require both here.
    assert len(matrices) >= 2, (
        f"expected at least 2 cuda matrices (the ops job and the wheel job) "
        f"in pages.yml, found {len(matrices)}; either a job's matrix stopped "
        f"matching the parser or one was removed"
    )


def test_flake_builds_every_version_the_release_extras_offer() -> None:
    """A published extra needs kernels, so the flake must define its build."""
    missing = _pyproject_extras() - _flake_variants()
    assert not missing, (
        f"pyproject.toml offers extras {sorted(missing)} with no matching variant "
        f"in flake.nix, so `nix build .#ops-<v>` cannot produce their kernels"
    )


@pytest.mark.parametrize("index", [0, 1])
def test_pages_matrix_matches_the_flake(index: int) -> None:
    """Every matrix entry must name a derivation the flake defines."""
    matrices = _pages_matrices()
    assert index < len(matrices), (
        f"pages.yml declares only {len(matrices)} cuda matrices, expected at "
        f"least {index + 1}; a matrix that stopped matching the parser goes "
        f"unchecked against the flake rather than failing this test"
    )
    matrix = matrices[index]
    variants = _flake_variants()
    assert matrix <= variants, (
        f"pages.yml matrix {sorted(matrix - variants)} has no flake variant; that "
        f"job fails on `nix build .#ops-<v>` the next time the workflow is "
        f"dispatched"
    )


def test_pages_matrices_agree_with_each_other() -> None:
    """The ops job and the wheel job must build the same set."""
    matrices = _pages_matrices()
    # A loop over `matrices[1:]` passes vacuously with fewer than two matrices.
    assert len(matrices) >= 2, (
        f"expected at least 2 cuda matrices in pages.yml to compare, found "
        f"{len(matrices)}"
    )
    first = matrices[0]
    for other in matrices[1:]:
        assert other == first, (
            f"pages.yml declares differing cuda matrices: {sorted(first)} and "
            f"{sorted(other)}; the wheel job would bundle kernels the ops job "
            f"did not build, or vice versa"
        )
