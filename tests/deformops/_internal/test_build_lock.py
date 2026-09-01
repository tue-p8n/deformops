"""The JIT build lock: a kernel ``flock`` instead of torch's stale lock file.

See ``docs/kernel-loading.md``.
"""

import threading
import time
from pathlib import Path

import pytest
from deformops.core import build as build_module
from deformops.core.build import (
    _JIT_LOCK_FILENAME,
    _clear_build_dir,
    serialized_build,
)


def test_removes_stale_torch_lock_file(tmp_path: Path) -> None:
    stale = tmp_path / "lock"
    stale.touch()

    with serialized_build(tmp_path):
        assert not stale.exists()


def test_noop_when_no_stale_lock_present(tmp_path: Path) -> None:
    with serialized_build(tmp_path):
        pass

    assert not (tmp_path / "lock").exists()


def test_does_not_remove_unrelated_files(tmp_path: Path) -> None:
    keep = tmp_path / "msda.so"
    keep.write_bytes(b"not actually a shared object")

    with serialized_build(tmp_path):
        pass

    assert keep.is_file()


def test_clear_build_dir_keeps_jit_lock_file(tmp_path: Path) -> None:
    lock_file = tmp_path / _JIT_LOCK_FILENAME
    lock_file.touch()
    stale_object = tmp_path / "msda.o"
    stale_object.touch()
    stale_subdir = tmp_path / "some_intermediate_dir"
    stale_subdir.mkdir()

    _clear_build_dir(tmp_path, "dummy")

    assert lock_file.exists()
    assert not stale_object.exists()
    assert not stale_subdir.exists()


def test_rebuild_strategy_preserves_mutual_exclusion(tmp_path: Path) -> None:
    """``CompileMode.REBUILD`` clears the build directory without deleting the lock.

    A recreated lock file would let a second builder flock a fresh inode and build
    concurrently.
    """
    order: list[str] = []

    def rebuilding_worker() -> None:
        with serialized_build(tmp_path):
            order.append("first-start")
            _clear_build_dir(tmp_path, "dummy")
            time.sleep(0.2)
            order.append("first-end")

    def second_worker() -> None:
        with serialized_build(tmp_path):
            order.append("second-start")
            order.append("second-end")

    first = threading.Thread(target=rebuilding_worker)
    second = threading.Thread(target=second_worker)

    first.start()
    time.sleep(0.05)  # let `first` acquire the lock before `second` starts
    second.start()
    first.join()
    second.join()

    assert order == ["first-start", "first-end", "second-start", "second-end"]


def test_degrades_to_noop_without_fcntl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ``fcntl`` (Windows) the lock is a no-op and does not crash."""
    monkeypatch.setattr(build_module, "fcntl", None)

    with serialized_build(tmp_path):
        pass  # must not raise


def test_concurrent_builds_are_mutually_exclusive(tmp_path: Path) -> None:
    order: list[str] = []

    def worker(label: str, hold_seconds: float) -> None:
        with serialized_build(tmp_path):
            order.append(f"{label}-start")
            time.sleep(hold_seconds)
            order.append(f"{label}-end")

    first = threading.Thread(target=worker, args=("first", 0.2))
    second = threading.Thread(target=worker, args=("second", 0.0))

    first.start()
    time.sleep(0.05)  # let `first` acquire the lock before `second` starts
    second.start()
    first.join()
    second.join()

    assert order == ["first-start", "first-end", "second-start", "second-end"]
