import re

import pytest
from deformops import version


def test_version_attributes():
    """Verify that expected version attributes are accessible."""
    assert isinstance(version.package, str)
    assert isinstance(version.environment, str)
    assert isinstance(version.local, str)


def test_version_string_format():
    """Verify that the version strings follow expected formats."""
    # Package version should be x.y.z
    package_pattern = r"^\d+\.\d+\.\d+(?:[a-zA-Z0-9\.]+)?$"
    assert re.match(package_pattern, version.package)

    # Local version includes environment info: package+environment
    assert version.local == f"{version.package}+{version.environment}"

    # Environment string should contain platform info
    assert (
        "linux" in version.environment
        or "darwin" in version.environment
        or "win" in version.environment
    )
    assert "torch" in version.environment


def test_version_package_fallback_on_metadata_error(monkeypatch: pytest.MonkeyPatch):
    """When importlib.metadata cannot find the dist, _get_package falls back
    to reading the VERSION file instead of raising."""
    import importlib.metadata as md
    from pathlib import Path

    def _raise(_name: str) -> str:
        raise md.PackageNotFoundError

    monkeypatch.setattr(md, "version", _raise)

    version_file = Path(__file__).parent.parent.parent / "VERSION"
    expected = version_file.read_text(encoding="utf-8").strip()

    assert version._get_package() == expected  # pyright: ignore[reportPrivateUsage]


def test_version_getattr_unknown_attribute_raises():
    """version.__getattr__ on an unknown name raises AttributeError."""
    with pytest.raises(AttributeError, match="Unsupported version attribute"):
        _ = version.unknown_attribute  # type: ignore[attr-defined]


def test_version_components_parsing():
    """The package version parses as a semantic version."""
    v_parts = version.package.split(".")
    assert len(v_parts) >= 3

    major = int(v_parts[0])
    minor = int(v_parts[1])
    # Patch might contain suffix like 0rc1
    patch_str = v_parts[2]
    # Simple regex to extract leading digits
    patch_match = re.match(r"^\d+", patch_str)
    assert patch_match
    patch = int(patch_match.group(0))

    assert major >= 0
    assert minor >= 0
    assert patch >= 0
