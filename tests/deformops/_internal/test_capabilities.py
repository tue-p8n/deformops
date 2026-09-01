"""Optional capabilities are detected, never assumed."""

import importlib
import importlib.metadata
import sys

import pytest
from deformops.core import capabilities


def test_has_triton_matches_actual_importability():
    import importlib.util

    assert capabilities.has_triton() == (importlib.util.find_spec("triton") is not None)


def test_triton_is_not_a_base_dependency():
    requires = importlib.metadata.requires("deformops") or []
    base = [r for r in requires if "extra ==" not in r]
    assert not [r for r in base if r.startswith("triton")], (
        "triton must be an extra: a CUDA torch already provides it, and a CPU "
        "install cannot use it"
    )


def test_triton_extra_is_declared():
    metadata = importlib.metadata.metadata("deformops")
    assert "triton" in set(metadata.get_all("Provides-Extra") or [])


def test_require_triton_names_the_extra_when_missing(monkeypatch):
    monkeypatch.setattr(capabilities, "has_triton", lambda: False)
    with pytest.raises(ImportError, match=r"deformops\[triton\]"):
        capabilities.require_triton("TritonQueryTiled")


def test_require_triton_is_silent_when_present(monkeypatch):
    monkeypatch.setattr(capabilities, "has_triton", lambda: True)
    capabilities.require_triton("TritonQueryTiled")


def test_registry_dispatch_names_the_extra_when_triton_missing(monkeypatch):
    """Accessing triton backends without triton raises an extra-naming error."""
    import deformops.backends.triton as triton_mod

    monkeypatch.setattr(capabilities, "has_triton", lambda: False)
    triton_mod.__dict__.pop("msda", None)
    with pytest.raises(ImportError, match=r"deformops\[triton\]"):
        _ = triton_mod.msda


@pytest.mark.parametrize(
    "dotted_module",
    [
        "deformops.backends.triton.msda_query_tiled",
        "deformops.backends.triton.attn_query_tiled",
        "deformops.backends.triton.conv_deform2d",
    ],
)
def test_direct_leaf_import_names_the_extra_when_triton_missing(
    monkeypatch, dotted_module
):
    """A direct import of a Triton leaf module names the extra.

    A dotted import bypasses the package ``__getattr__`` and the registry, so the
    leaf module checks for Triton before its own ``import triton``.
    """
    monkeypatch.setattr(capabilities, "has_triton", lambda: False)
    sys.modules.pop(dotted_module, None)
    try:
        with pytest.raises(ImportError, match=r"deformops\[triton\]"):
            importlib.import_module(dotted_module)
    finally:
        # Drop the failed import so a later real import gets no stale entry.
        sys.modules.pop(dotted_module, None)


def test_jit_extra_is_declared():
    metadata = importlib.metadata.metadata("deformops")
    assert "jit" in set(metadata.get_all("Provides-Extra") or [])


def test_has_jit_toolchain_matches_ninja_availability():
    from torch.utils import cpp_extension

    try:
        cpp_extension.verify_ninja_availability()
    except RuntimeError:
        expected = False
    else:
        expected = True
    assert capabilities.has_jit_toolchain() == expected


def test_require_jit_toolchain_names_the_extra_when_missing(monkeypatch):
    from deformops.core.errors import KernelBuildError

    monkeypatch.setattr(capabilities, "has_jit_toolchain", lambda: False)
    with pytest.raises(KernelBuildError, match=r"deformops\[jit\]"):
        capabilities.require_jit_toolchain("msda_fp32acc")


def test_no_prebuilt_download_is_attempted_without_a_cuda_torch(monkeypatch):
    """A CPU torch fetches nothing and raises a download error naming the opt-out."""
    from deformops.core import build
    from deformops.core.errors import KernelDownloadError

    monkeypatch.setattr(build, "_cuda_tag", lambda: None)

    def _fail(*args, **kwargs):
        msg = "the operator index must not be contacted without a CUDA torch"
        raise AssertionError(msg)

    monkeypatch.setattr(build, "_fetch_url", _fail)
    with pytest.raises(KernelDownloadError, match="DEFORMOPS_DOWNLOAD=0"):
        build._download_prebuilt("msda_fp32acc")


def test_readme_documents_every_declared_extra():
    import pathlib

    metadata = importlib.metadata.metadata("deformops")
    extras = set(metadata.get_all("Provides-Extra") or [])
    readme = (pathlib.Path(__file__).parents[3] / "README.md").read_text(
        encoding="utf-8"
    )
    undocumented = sorted(e for e in extras if f"deformops[{e}]" not in readme)
    assert not undocumented, f"extras missing from README: {undocumented}"
