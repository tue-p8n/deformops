"""Every operator package must be importable and match its declared path."""

import importlib
import pathlib

import pytest
from deformops.core.configs import EXTENSION_CONFIGS

OPERATORS = ("backends", "models", "nn", "integrations", "core")
REPO_ROOT = pathlib.Path(__file__).parents[2]


def _sources_or_skip() -> pathlib.Path:
    sources = REPO_ROOT / "sources"
    if not sources.is_dir():
        pytest.skip("running outside a source checkout; sources/ is not shipped")
    return sources


@pytest.mark.parametrize("operator", OPERATORS)
def test_operator_package_imports(operator):
    module = importlib.import_module(f"deformops.{operator}")
    assert module.__name__ == f"deformops.{operator}"


def test_integrations_package_imports():
    module = importlib.import_module("deformops.integrations")
    assert module.__name__ == "deformops.integrations"


def test_no_hyphenated_package_directories_remain():
    stale = sorted(p.name for p in _sources_or_skip().glob("deformops-*"))
    assert not stale, f"hyphenated, unimportable directories remain: {stale}"


def test_every_extension_config_resolves_to_real_sources():
    sources = _sources_or_skip()
    for name, config in EXTENSION_CONFIGS.items():
        src_dir = sources / pathlib.Path(*config["package"].split("."))
        stem = config["stem"]
        assert (src_dir / f"{stem}.cpp").is_file(), (
            f"{name}: missing {stem}.cpp in {src_dir}"
        )
        assert (src_dir / f"{stem}.cu").is_file(), (
            f"{name}: missing {stem}.cu in {src_dir}"
        )
