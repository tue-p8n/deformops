r"""Loading a CUDA multi-scale kernel library before the first call.

``load`` and ``forward`` share one memoized bind, so loading early never loads twice.
See ``docs/kernel-loading.md``.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import textwrap
from typing import Any

import pytest
import torch
from deformops.backends.cuda import msda_fp32acc as production
from deformops.backends.cuda._msda_bridge import MSDAFamily
from deformops.backends.cuda.experimental import msda_half2atomic
from deformops.core import build
from deformops.registry import Backend, LoadedBackend, msda

#: Every CUDA multi-scale configuration that owns a compiled library.
KERNEL_MODULES = (
    "msda_fp32acc",
    "experimental.msda_serialreduce",
    "experimental.msda_half2atomic",
    "experimental.msda_half2atomic_serialreduce",
    "experimental.msda_scalaratomic",
    "experimental.msda_scalaratomic_serialreduce",
)

#: Query reorderings around the production kernel, with no library of their own.
REORDERING_MODULES = ("experimental.msda_morton", "experimental.msda_hilbert")


def _module(name: str):
    return importlib.import_module(f"deformops.backends.cuda.{name}")


# Mechanism: ``define_library`` stubbed, no CUDA required.


def _counting_stub(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, ...]]:
    """Stub ``define_library`` and give two configuration modules a fresh bind.

    The real bind memoizes for the process, so an earlier load would hide the stub.
    """
    calls: list[tuple[Any, ...]] = []

    def fake_define_library(name: str, build_dir: Any = None, **kwargs: Any) -> str:
        calls.append((name, build_dir, kwargs))
        return f"/fake/{name}.so"

    monkeypatch.setattr(build, "define_library", fake_define_library)
    monkeypatch.setattr(production, "load", build.lazy_extension("msda_fp32acc"))
    msda.is_registered(Backend.CUDA)  # import the adapters first
    monkeypatch.setitem(msda._loaders, Backend.CUDA, production.load)
    monkeypatch.setattr(
        msda_half2atomic,
        "load",
        build.lazy_extension("msda_half2atomic"),
    )
    return calls


@pytest.fixture
def cuda_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register a placeholder CUDA handle, as a host with a CUDA device does."""
    msda.is_registered(Backend.CUDA)  # import the adapters first
    monkeypatch.setitem(msda._backends, Backend.CUDA, lambda *_a, **_k: None)


def test_load_loads_the_production_library(monkeypatch, cuda_registered) -> None:
    calls = _counting_stub(monkeypatch)

    loaded = msda.load(Backend.CUDA)

    expected = LoadedBackend("msda", Backend.CUDA, "/fake/msda_fp32acc.so")
    assert loaded == expected
    assert [name for name, _build_dir, _kwargs in calls] == ["msda_fp32acc"]


def test_load_forwards_download_and_compile(monkeypatch, cuda_registered) -> None:
    calls = _counting_stub(monkeypatch)

    msda.load(Backend.CUDA, download=False, compile="rebuild")

    _name, _build_dir, kwargs = calls[0]
    assert kwargs["download"] is False
    assert kwargs["compile"] == "rebuild"


def test_load_is_idempotent(monkeypatch, cuda_registered) -> None:
    """A second load loads no library, even with different keywords."""
    calls = _counting_stub(monkeypatch)

    first = msda.load(Backend.CUDA)
    second = msda.load(Backend.CUDA, compile="rebuild")

    assert len(calls) == 1
    assert first == second


def test_load_and_the_module_load_share_one_memoized_load(
    monkeypatch, cuda_registered
) -> None:
    calls = _counting_stub(monkeypatch)

    msda.load(Backend.CUDA)
    production.load()

    assert len(calls) == 1, "the module's own load must not repeat the library load"


def test_an_experimental_module_loads_its_own_library(monkeypatch) -> None:
    calls = _counting_stub(monkeypatch)

    msda_half2atomic.load()

    assert [name for name, _build_dir, _kwargs in calls] == ["msda_half2atomic"]


def test_configurations_load_independently(monkeypatch) -> None:
    """Loading one configuration must not appear to satisfy another."""
    calls = _counting_stub(monkeypatch)

    production.load()
    msda_half2atomic.load()

    assert {name for name, _build_dir, _kwargs in calls} == {
        "msda_fp32acc",
        "msda_half2atomic",
    }


@pytest.mark.parametrize("name", KERNEL_MODULES)
def test_every_kernel_configuration_exports_load(name: str) -> None:
    module = _module(name)
    assert callable(module.load)
    assert "load" in module.__all__
    assert module.load is module._family.load
    assert isinstance(module._family, MSDAFamily)


@pytest.mark.parametrize("name", REORDERING_MODULES)
def test_reordering_configurations_export_no_load(name: str) -> None:
    assert not hasattr(_module(name), "load")


@pytest.mark.parametrize("removed", ["warm_up", "so_path", "MSDA_KERNEL_MODULES"])
def test_the_cuda_package_exposes_no_other_loading_entry_point(removed: str) -> None:
    from deformops.backends import cuda

    assert not hasattr(cuda, removed)


# Shared memoization: the production module's own bind.


@pytest.mark.parametrize("name", KERNEL_MODULES)
def test_forward_calls_the_exported_load(name: str) -> None:
    """``forward`` binds through the closure ``load`` exports."""
    module = _module(name)
    cells = [cell.cell_contents for cell in module.forward.__closure__ or ()]
    assert any(content is module.load for content in cells)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_module_load_then_forward_loads_once_gpu() -> None:
    """In a fresh interpreter, ``load`` then ``forward`` loads the library once."""
    script = textwrap.dedent(
        """
        import torch

        import deformops.core.build as build

        calls = []
        real_define_library = build.define_library

        def counting(*args, **kwargs):
            calls.append(args[0])
            return real_define_library(*args, **kwargs)

        build.define_library = counting

        from deformops.backends.cuda import msda_fp32acc

        msda_fp32acc.load()
        assert calls == ["msda_fp32acc"], calls

        device = torch.device("cuda")
        values = torch.randn(1, 16, 2, 8, device=device)
        shapes = torch.tensor([[4, 4]], device=device)
        start_index = torch.tensor([0], device=device)
        locs = torch.rand(1, 8, 2, 1, 4, 2, device=device)
        attn = torch.rand(1, 8, 2, 4, device=device)
        msda_fp32acc.forward(values, shapes, start_index, locs, attn, num_points=4)
        assert calls == ["msda_fp32acc"], calls
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
