"""The prebuilt-library download in ``core/build.py``, fully offline.

``_fetch_url`` and ``torch.ops.load_library`` are monkeypatched. Every failure of
an enabled download raises ``KernelDownloadError``.
"""

import hashlib
import json
import urllib.error

import pytest
import torch
from deformops.core import build as _build
from deformops.core.errors import KernelDownloadError

PAYLOAD = b"\x7fELF fake stable-abi shared object"
SHA = hashlib.sha256(PAYLOAD).hexdigest()
NAME = "attn"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Set a known CUDA tag and the default index for every test here."""
    monkeypatch.delenv("DEFORMOPS_OPS_INDEX_URL", raising=False)
    monkeypatch.setattr(torch.version, "cuda", "12.6", raising=False)
    # The manifest is memoized per process; each test starts from a clean memo.
    monkeypatch.setattr(_build, "_MANIFESTS", {})


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DEFORMOPS_OPS_CACHE_DIR", str(tmp_path))
    return tmp_path


def _manifest(sha=SHA, *, with_file=True):
    entry = {"sha256": sha, "size": len(PAYLOAD)}
    if with_file:
        entry["file"] = f"{NAME}.so"
    return {NAME: entry}


def _make_fetch(manifest=None, payload=PAYLOAD, *, fail_on=None):
    """Return a fake ``_fetch_url`` branching on the requested URL."""
    manifest = _manifest() if manifest is None else manifest

    def _fetch(url, *, max_bytes=None, timeout=30.0):
        if fail_on is not None and fail_on in url:
            raise urllib.error.URLError("boom")
        if url.endswith("manifest.json"):
            return json.dumps(manifest).encode()
        return payload

    return _fetch


@pytest.fixture
def record_load(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.ops, "load_library", calls.append, raising=False)
    return calls


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("cuda", "expected"),
    [("12.6", "cu126"), ("13.0", "cu130"), ("12.6.1", "cu126"), (None, None)],
)
def test_cuda_tag(monkeypatch, cuda, expected):
    monkeypatch.setattr(torch.version, "cuda", cuda, raising=False)
    assert _build._cuda_tag() == expected


def test_ops_cache_dir_override(monkeypatch, tmp_path):
    monkeypatch.setenv("DEFORMOPS_OPS_CACHE_DIR", str(tmp_path))
    assert _build._ops_cache_dir("2.0.0", "cu126") == tmp_path / "2.0.0" / "cu126"


def test_ops_cache_dir_default(monkeypatch):
    monkeypatch.delenv("DEFORMOPS_OPS_CACHE_DIR", raising=False)
    got = _build._ops_cache_dir("2.0.0", "cu126")
    assert got.parts[-3:] == ("ops", "2.0.0", "cu126")
    assert any("deformops" in p for p in got.parts)


def test_ops_cache_dir_home_failure(monkeypatch):
    # Path.home() raises RuntimeError in restricted environments -> temp fallback.
    monkeypatch.delenv("DEFORMOPS_OPS_CACHE_DIR", raising=False)

    def _no_home():
        raise RuntimeError("no home")

    monkeypatch.setattr(_build.pathlib.Path, "home", staticmethod(_no_home))
    got = _build._ops_cache_dir("2.0.0", "cu126")
    assert got.parts[-2:] == ("2.0.0", "cu126")
    # temp fallback dir is UID-scoped ("deformops-<uid>") on shared hosts
    assert any(p.startswith("deformops") for p in got.parts)


def _fake_urlopen(monkeypatch, body: bytes):
    """Patch urlopen with a response that honors ``read(n)``."""

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, amount=None):
            return body if amount is None else body[:amount]

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp())


def test_fetch_url(monkeypatch):
    _fake_urlopen(monkeypatch, b"hello")
    assert _build._fetch_url("https://example/x", max_bytes=1024) == b"hello"


def test_fetch_url_rejects_non_http_scheme():
    with pytest.raises(ValueError, match="scheme not allowed"):
        _build._fetch_url("file:///etc/passwd", max_bytes=1024)


def test_fetch_url_rejects_plaintext_http_by_default(monkeypatch):
    """The manifest holds the digest, so plaintext HTTP would defeat the checksum."""
    monkeypatch.delenv("DEFORMOPS_OPS_ALLOW_INSECURE", raising=False)
    _fake_urlopen(monkeypatch, b"hello")
    with pytest.raises(ValueError, match="plaintext HTTP"):
        _build._fetch_url("http://example/x", max_bytes=1024)


@pytest.mark.parametrize("setting", ["1", "true", "on"])
def test_fetch_url_allows_plaintext_http_on_opt_in(monkeypatch, setting):
    monkeypatch.setenv("DEFORMOPS_OPS_ALLOW_INSECURE", setting)
    _fake_urlopen(monkeypatch, b"hello")
    assert _build._fetch_url("http://example/x", max_bytes=1024) == b"hello"


def test_fetch_url_caps_the_response_size(monkeypatch):
    """An over-long body raises before it is read into memory."""
    _fake_urlopen(monkeypatch, b"x" * 4096)
    with pytest.raises(ValueError, match="exceeds the 16 byte limit"):
        _build._fetch_url("https://example/x", max_bytes=16)


def test_fetch_url_accepts_a_body_at_the_limit(monkeypatch):
    _fake_urlopen(monkeypatch, b"x" * 16)
    assert _build._fetch_url("https://example/x", max_bytes=16) == b"x" * 16


@pytest.mark.parametrize(
    "name", ["../evil.so", "/abs/evil.so", "sub/dir.so", "", ".", ".."]
)
def test_manifest_file_field_must_be_a_bare_filename(monkeypatch, cache_dir, name):
    """A traversal in the manifest is refused before it reaches a URL."""
    manifest = {NAME: {"sha256": SHA, "file": name}}
    monkeypatch.setattr(_build, "_fetch_url", _make_fetch(manifest))
    assert _build._ops_manifest_entry(NAME, "https://host/ops/2.0.0/cu126") is None


def test_sha256_file(tmp_path):
    p = tmp_path / "blob"
    p.write_bytes(PAYLOAD)
    assert _build._sha256_file(p) == SHA


# --------------------------------------------------------------------------- #
# _try_load_prebuilt (bundled _ext/<name>.so)
# --------------------------------------------------------------------------- #
def test_try_load_prebuilt_success(monkeypatch, tmp_path, record_load):
    so = tmp_path / f"{NAME}.so"
    so.write_bytes(PAYLOAD)
    monkeypatch.setattr(_build.importlib.resources, "files", lambda _pkg: tmp_path)
    assert _build._try_load_prebuilt(NAME) == str(so)
    assert record_load == [str(so)]


def test_try_load_prebuilt_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(_build.importlib.resources, "files", lambda _pkg: tmp_path)
    assert _build._try_load_prebuilt(NAME) is None


def test_try_load_prebuilt_load_error(monkeypatch, tmp_path):
    so = tmp_path / f"{NAME}.so"
    so.write_bytes(PAYLOAD)
    monkeypatch.setattr(_build.importlib.resources, "files", lambda _pkg: tmp_path)

    def _boom(_):
        raise OSError("bad so")

    monkeypatch.setattr(torch.ops, "load_library", _boom, raising=False)
    assert _build._try_load_prebuilt(NAME) is None


# --------------------------------------------------------------------------- #
# _download_prebuilt
# --------------------------------------------------------------------------- #
def _cached_path(cache_dir):
    return cache_dir / _build.deformops.version.package / "cu126" / f"{NAME}.so"


def _raises_download_error(match: str | None = None):
    return pytest.raises(KernelDownloadError, match=match)


def test_failure_names_the_opt_out(monkeypatch, cache_dir):
    monkeypatch.setattr(_build, "_fetch_url", _make_fetch(manifest={}))
    with _raises_download_error("DEFORMOPS_DOWNLOAD=0"):
        _build._download_prebuilt(NAME)


def test_cpu_only_torch_raises(monkeypatch):
    monkeypatch.setattr(torch.version, "cuda", None, raising=False)
    with _raises_download_error("no CUDA runtime"):
        _build._download_prebuilt(NAME)


def test_manifest_unreachable_without_a_cached_copy_raises(monkeypatch, cache_dir):
    monkeypatch.setattr(_build, "_fetch_url", _make_fetch(fail_on="manifest.json"))
    with _raises_download_error("no verified copy is cached"):
        _build._download_prebuilt(NAME)


def test_missing_entry_raises(monkeypatch, cache_dir):
    monkeypatch.setattr(_build, "_fetch_url", _make_fetch(manifest={}))
    with _raises_download_error("does not list"):
        _build._download_prebuilt(NAME)


def test_offline_fallback_loads_cache(monkeypatch, cache_dir, record_load):
    # Manifest unreachable but a verified copy is cached: reuse it.
    cached = _cached_path(cache_dir)
    cached.parent.mkdir(parents=True)
    cached.write_bytes(PAYLOAD)
    monkeypatch.setattr(_build, "_fetch_url", _make_fetch(fail_on="manifest.json"))
    assert _build._download_prebuilt(NAME) == str(cached)
    assert record_load == [str(cached)]


def test_offline_fallback_load_error_raises(monkeypatch, cache_dir):
    cached = _cached_path(cache_dir)
    cached.parent.mkdir(parents=True)
    cached.write_bytes(PAYLOAD)
    monkeypatch.setattr(_build, "_fetch_url", _make_fetch(fail_on="manifest.json"))

    def _boom(_):
        raise RuntimeError("bad abi")

    monkeypatch.setattr(torch.ops, "load_library", _boom, raising=False)
    with _raises_download_error("no verified copy is cached"):
        _build._download_prebuilt(NAME)


def test_happy_path(monkeypatch, cache_dir, record_load):
    monkeypatch.setattr(_build, "_fetch_url", _make_fetch())
    result = _build._download_prebuilt(NAME)
    cached = _cached_path(cache_dir)
    assert result == str(cached)
    assert cached.read_bytes() == PAYLOAD
    assert record_load == [str(cached)]


def test_happy_path_default_filename(monkeypatch, cache_dir, record_load):
    # The manifest entry omits "file", so the loader derives f"{name}.so".
    monkeypatch.setattr(
        _build, "_fetch_url", _make_fetch(manifest=_manifest(with_file=False))
    )
    assert _build._download_prebuilt(NAME) == str(_cached_path(cache_dir))
    assert len(record_load) == 1


def test_uses_valid_cache_without_download(monkeypatch, cache_dir, record_load):
    cached = _cached_path(cache_dir)
    cached.parent.mkdir(parents=True)
    cached.write_bytes(PAYLOAD)
    # The library fetch would raise; a valid cache must skip it.
    monkeypatch.setattr(_build, "_fetch_url", _make_fetch(fail_on=f"{NAME}.so"))
    assert _build._download_prebuilt(NAME) == str(cached)
    assert record_load == [str(cached)]


def test_stale_cache_is_redownloaded(monkeypatch, cache_dir, record_load):
    cached = _cached_path(cache_dir)
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"stale")
    monkeypatch.setattr(_build, "_fetch_url", _make_fetch())
    assert _build._download_prebuilt(NAME) == str(cached)
    assert cached.read_bytes() == PAYLOAD


def test_checksum_mismatch_raises(monkeypatch, cache_dir, record_load):
    monkeypatch.setattr(
        _build, "_fetch_url", _make_fetch(manifest=_manifest(sha="deadbeef"))
    )
    with _raises_download_error("checksum"):
        _build._download_prebuilt(NAME)
    assert record_load == []


def test_download_fetch_error_raises(monkeypatch, cache_dir):
    monkeypatch.setattr(_build, "_fetch_url", _make_fetch(fail_on=f"{NAME}.so"))
    with _raises_download_error("download failed"):
        _build._download_prebuilt(NAME)


def test_cache_write_error_raises(monkeypatch, tmp_path):
    # A cache root that is a file makes mkdir(parents=True) raise OSError.
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    monkeypatch.setenv("DEFORMOPS_OPS_CACHE_DIR", str(blocker))
    monkeypatch.setattr(_build, "_fetch_url", _make_fetch())
    with _raises_download_error():
        _build._download_prebuilt(NAME)


def test_cache_read_error_falls_through(monkeypatch, cache_dir, record_load):
    # A cached file whose sha256 read raises must not crash; the loader
    # downloads and verifies the fresh payload instead.
    cached = _cached_path(cache_dir)
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"unreadable")

    def _boom(_):
        raise OSError("perm")

    monkeypatch.setattr(_build, "_sha256_file", _boom)
    monkeypatch.setattr(_build, "_fetch_url", _make_fetch())
    assert _build._download_prebuilt(NAME) == str(cached)
    assert cached.read_bytes() == PAYLOAD


def test_replace_failure_cleans_tmp(monkeypatch, cache_dir):
    # A directory at the cached path makes tmp.replace(cached) fail; the
    # leftover temporary file is removed and the download raises.
    cached = _cached_path(cache_dir)
    cached.mkdir(parents=True)
    monkeypatch.setattr(_build, "_fetch_url", _make_fetch())
    with _raises_download_error():
        _build._download_prebuilt(NAME)
    assert not list(cached.parent.glob("*.tmp"))


def test_ensure_cached_concurrent_winner(monkeypatch, cache_dir, record_load):
    # replace() fails, but a concurrent writer already left a valid cached copy,
    # which counts as success.
    cached = _cached_path(cache_dir)
    cached.parent.mkdir(parents=True)
    cached.write_bytes(PAYLOAD)

    calls = {"n": 0}

    def _sha(_path):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("transient")
        return SHA

    def _replace_fail(_self, _target):
        raise OSError("locked")

    monkeypatch.setattr(_build, "_sha256_file", _sha)
    monkeypatch.setattr(_build.pathlib.Path, "replace", _replace_fail)
    monkeypatch.setattr(_build, "_fetch_url", _make_fetch())
    assert _build._download_prebuilt(NAME) == str(cached)
    assert record_load == [str(cached)]


def test_owner_mismatch_raises_without_loading(monkeypatch, cache_dir, record_load):
    # A cache directory owned by another uid (simulated) is refused, so a
    # co-tenant on a shared /tmp cannot get a planted library loaded.
    cached = _cached_path(cache_dir)
    cached.parent.mkdir(parents=True)
    cached.write_bytes(PAYLOAD)
    monkeypatch.setattr(_build, "_fetch_url", _make_fetch())
    real = _build.os.getuid()
    monkeypatch.setattr(_build.os, "getuid", lambda: real + 1)
    with _raises_download_error("not owned"):
        _build._download_prebuilt(NAME)
    assert record_load == []


def test_cache_owner_ok_no_getuid(monkeypatch, tmp_path):
    monkeypatch.delattr(_build.os, "getuid", raising=False)
    assert _build._cache_owner_ok(tmp_path / "x.so") is True


def test_cache_owner_ok_stat_error(monkeypatch, cache_dir):
    cached = _cached_path(cache_dir)
    cached.parent.mkdir(parents=True)

    def _boom(_self, *_a, **_k):
        raise PermissionError("nope")

    monkeypatch.setattr(_build.pathlib.Path, "stat", _boom)
    assert _build._cache_owner_ok(cached) is False


def test_load_library_error_raises(monkeypatch, cache_dir):
    monkeypatch.setattr(_build, "_fetch_url", _make_fetch())

    def _boom(_):
        raise RuntimeError("bad abi")

    monkeypatch.setattr(torch.ops, "load_library", _boom, raising=False)
    with _raises_download_error("failed to load") as caught:
        _build._download_prebuilt(NAME)
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_removed_entry_does_not_load_a_cached_copy(monkeypatch, cache_dir, record_load):
    # The index no longer lists the library, so a cached copy must not load.
    cached = _cached_path(cache_dir)
    cached.parent.mkdir(parents=True)
    cached.write_bytes(PAYLOAD)
    monkeypatch.setattr(_build, "_fetch_url", _make_fetch(manifest={}))
    with _raises_download_error("does not list"):
        _build._download_prebuilt(NAME)
    assert record_load == []


def test_manifest_is_fetched_once_per_index(monkeypatch, cache_dir, record_load):
    requests = []
    fetch = _make_fetch(fail_on="manifest.json")

    def counting_fetch(url, **kwargs):
        requests.append(url)
        return fetch(url, **kwargs)

    monkeypatch.setattr(_build, "_fetch_url", counting_fetch)
    with _raises_download_error():
        _build._download_prebuilt(NAME)
    with _raises_download_error():
        _build._download_prebuilt("another_extension")
    assert sum(url.endswith("manifest.json") for url in requests) == 1
