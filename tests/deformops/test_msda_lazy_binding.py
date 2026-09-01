r"""Lazy binding: a configuration's library loads on its first call, not on import.

Mechanism tests stub ``define_library`` with a counter; integration tests load the
real library in a fresh subprocess. See ``docs/kernel-loading.md``.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import threading
import time
from typing import Any

import pytest
import torch
from deformops.core import build

_GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# Mechanism: ``lazy_extension`` with ``define_library`` stubbed.


def test_lazy_extension_constructor_calls_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Building the callable must not itself invoke ``define_library``."""
    calls: list[tuple[Any, ...]] = []

    def fake_define_library(name: str, build_dir: Any = None, **kwargs: Any) -> str:
        calls.append((name, build_dir, kwargs))
        return f"/fake/{name}.so"

    monkeypatch.setattr(build, "define_library", fake_define_library)

    build.lazy_extension("some_extension")

    assert calls == []


def test_first_call_binds_once_second_call_binds_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first call loads once; later calls reuse the memoized result."""
    calls: list[tuple[Any, ...]] = []

    def fake_define_library(name: str, build_dir: Any = None, **kwargs: Any) -> str:
        calls.append((name, build_dir, kwargs))
        return f"/fake/{name}.so"

    monkeypatch.setattr(build, "define_library", fake_define_library)

    bind = build.lazy_extension("some_extension")
    assert calls == []

    first = bind()
    assert len(calls) == 1
    assert first == "/fake/some_extension.so"

    second = bind()
    assert len(calls) == 1, "a second call must not repeat the load"
    third = bind()
    assert len(calls) == 1, "a third call must not repeat the load either"
    assert second == first
    assert third == first


def test_lazy_extension_delegates_without_bypassing_define_library(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The deferred call is ``define_library``, with ``lazy_extension``'s arguments."""
    calls: list[tuple[Any, ...]] = []

    def fake_define_library(name: str, build_dir: Any = None, **kwargs: Any) -> str:
        calls.append((name, build_dir, kwargs))
        return f"/fake/{name}.so"

    monkeypatch.setattr(build, "define_library", fake_define_library)

    bind = build.lazy_extension(
        "some_extension",
        with_cuda=False,
        opt_level=0,
        cflags=("-DFOO",),
        cflags_cuda=("-DBAR",),
    )
    bind()

    assert calls == [
        (
            "some_extension",
            None,
            {
                "with_cuda": False,
                "opt_level": 0,
                "cflags": ("-DFOO",),
                "cflags_cuda": ("-DBAR",),
                "download": None,
                "compile": None,
            },
        )
    ]


def test_lazy_extension_forwards_download_and_compile_on_the_first_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first call's keywords reach ``define_library``; later ones are ignored."""
    calls: list[dict[str, Any]] = []

    def fake_define_library(name: str, build_dir: Any = None, **kwargs: Any) -> str:
        calls.append(kwargs)
        return f"/fake/{name}.so"

    monkeypatch.setattr(build, "define_library", fake_define_library)

    bind = build.lazy_extension("some_extension")
    assert bind(download=False, compile="always") == "/fake/some_extension.so"
    assert bind(download=True, compile="rebuild") == "/fake/some_extension.so"

    assert len(calls) == 1
    assert calls[0]["download"] is False
    assert calls[0]["compile"] == "always"


def test_lazy_extension_retries_after_a_failed_first_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed load is not memoized, so a later call can still succeed."""
    from deformops.core.errors import KernelNotProvidedError

    calls: list[dict[str, Any]] = []

    def fake_define_library(name: str, build_dir: Any = None, **kwargs: Any) -> str:
        calls.append(kwargs)
        if not kwargs["compile"]:
            msg = "compiling is disabled"
            raise KernelNotProvidedError(msg)
        return f"/fake/{name}.so"

    monkeypatch.setattr(build, "define_library", fake_define_library)

    bind = build.lazy_extension("some_extension")
    with pytest.raises(KernelNotProvidedError):
        bind(compile=False)
    assert bind(compile=True) == "/fake/some_extension.so"
    assert len(calls) == 2


def test_lazy_extension_is_safe_under_concurrent_first_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two threads racing into the first call must still only load once."""
    calls: list[int] = []
    entered = threading.Event()
    release = threading.Event()

    def slow_define_library(name: str, build_dir: Any = None, **kwargs: Any) -> str:
        calls.append(1)
        entered.set()
        release.wait(timeout=5)
        return "/fake.so"

    monkeypatch.setattr(build, "define_library", slow_define_library)
    bind = build.lazy_extension("ext")

    results: list[Any] = []

    def call() -> None:
        results.append(bind())

    t1 = threading.Thread(target=call)
    t2 = threading.Thread(target=call)
    t1.start()
    assert entered.wait(timeout=5), "first thread never entered define_library"
    t2.start()
    time.sleep(0.05)  # give the second thread a chance to race into the lock
    release.set()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert len(calls) == 1
    assert results == ["/fake.so", "/fake.so"]


# Integration: the real ``msda_fp32acc`` load, in a fresh subprocess.


def _run(script: str, *, timeout: float = 600) -> subprocess.CompletedProcess[str]:
    """Run ``script`` in a fresh interpreter and return the completed process."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def test_importing_cuda_kernel_module_triggers_no_load() -> None:
    """Importing ``msda_fp32acc.py`` does not call ``define_library``."""
    result = _run(
        """
        import deformops.core.build as build

        calls = []
        real_define_library = build.define_library

        def counting(*args, **kwargs):
            calls.append((args, kwargs))
            return real_define_library(*args, **kwargs)

        build.define_library = counting

        import deformops.backends.cuda.msda_fp32acc  # noqa: F401

        assert calls == [], f"import triggered {len(calls)} load(s): {calls}"
        print("OK")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


@pytest.mark.gpu
@_GPU
def test_first_forward_call_binds_once_second_call_binds_none_gpu() -> None:
    """Import loads nothing, the first ``forward`` loads once, and the second does not.

    Runs in a fresh subprocess, since the suite may already have loaded the library.
    """
    result = _run(
        """
        import torch

        import deformops.core.build as build

        calls = []
        real_define_library = build.define_library

        def counting(*args, **kwargs):
            calls.append((args, kwargs))
            return real_define_library(*args, **kwargs)

        build.define_library = counting

        from deformops.backends.cuda import msda_fp32acc

        assert calls == [], f"import triggered {len(calls)} load(s): {calls}"

        # Shapes from test_msda_fake_tensors.py, where D divides channels_per_thread.
        device = torch.device("cuda")
        values = torch.randn(1, 16, 2, 8, device=device)
        shapes = torch.tensor([[4, 4]], dtype=torch.long, device=device)
        start_index = torch.tensor([0], dtype=torch.long, device=device)
        locs = torch.rand(1, 8, 2, 1, 4, 2, device=device)
        attn = torch.rand(1, 8, 2, 4, device=device)

        out1 = msda_fp32acc.forward(values, shapes, start_index, locs, attn, num_points=4)
        assert len(calls) == 1, f"first call triggered {len(calls)} load(s), want 1"

        out2 = msda_fp32acc.forward(values, shapes, start_index, locs, attn, num_points=4)
        assert len(calls) == 1, (
            f"second call triggered {len(calls) - 1} additional load(s), want 0"
        )
        assert out1.shape == out2.shape == (1, 8, 16)
        print("OK")
        """,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
