import deformops


def test_package_exports():
    """The package exposes the expected top-level modules and version."""
    assert hasattr(deformops, "__version__")
    assert hasattr(deformops, "backends")
    assert hasattr(deformops, "nn")
    assert hasattr(deformops, "registry")
    assert hasattr(deformops, "version")


def test_all_definitions():
    """``__all__`` lists every top-level subpackage and the version."""
    assert set(deformops.__all__) == {
        "__version__",
        "backends",
        "models",
        "nn",
        "registry",
        "version",
    }
