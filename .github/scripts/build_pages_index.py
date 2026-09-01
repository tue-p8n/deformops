#!/usr/bin/env python3
"""Assemble the deformops GitHub Pages site.

The site hosts two things behind a single Pages deployment:

* ``ops/<pkgver>/<cuXXX>/``: stable-ABI ``.so`` + per-dir ``manifest.json``,
  fetched at runtime by ``core/build.py::_try_download_prebuilt`` for the
  generic (PyPI) install.
* ``whl/<cuXXX>/``: a PEP 503 "simple" index of per-CUDA wheels that bundle
  the prebuilt kernels, consumable via
  ``pip install deformops --extra-index-url https://<owner>.github.io/deformops/whl/<cuXXX>``
  (the ``+cuXXX`` local version outranks the plain PyPI release, PyTorch-style).

Usage:
    build_pages_index.py <public_dir> <wheel_staging_dir>

``<public_dir>`` already contains the merged ``ops/`` tree (downloaded from the
``ops-*`` artifacts). ``<wheel_staging_dir>`` contains one subdirectory per
wheel artifact named ``whl-<cuXXX>`` holding that variant's ``*.whl``.
"""

from __future__ import annotations

import hashlib
import html
import json
import pathlib
import sys


def _aggregate_ops(public: pathlib.Path) -> dict:
    """Write ``ops/manifest.json`` aggregating every per-dir manifest."""
    ops = public / "ops"
    agg: dict[str, dict] = {}
    if ops.is_dir():
        for manifest in sorted(ops.rglob("manifest.json")):
            rel = manifest.parent.relative_to(ops).parts
            if len(rel) != 2:
                continue
            ver, cu = rel
            agg.setdefault(ver, {})[cu] = json.loads(
                manifest.read_text(encoding="utf-8")
            )
        (ops / "manifest.json").write_text(
            json.dumps(agg, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return agg


def _lay_out_wheels(public: pathlib.Path, staging: pathlib.Path) -> dict:
    """Copy wheels into a PEP 503 tree and return ``{cuXXX: [(file, sha)]}``."""
    by_cuda: dict[str, list[tuple[str, str]]] = {}
    if not staging.is_dir():
        return by_cuda
    for wheel in sorted(staging.glob("*/*.whl")):
        cuda = wheel.parent.name.removeprefix("whl-")
        dest_dir = public / "whl" / cuda / "deformops"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / wheel.name
        data = wheel.read_bytes()
        dest.write_bytes(data)
        sha = hashlib.sha256(data).hexdigest()
        by_cuda.setdefault(cuda, []).append((wheel.name, sha))
    return by_cuda


def _write_pep503(public: pathlib.Path, by_cuda: dict) -> None:
    """Write PEP 503 simple-index pages for each CUDA variant."""
    for cuda, files in sorted(by_cuda.items()):
        proj = public / "whl" / cuda / "deformops"
        links = "\n".join(
            f'    <a href="{html.escape(name)}#sha256={sha}">'
            f"{html.escape(name)}</a><br>"
            for name, sha in sorted(files)
        )
        # Project page: pip fetches "<extra-index>/deformops/". PEP 503/629:
        # UTF-8 pages MUST carry <meta charset="utf-8">.
        proj.joinpath("index.html").write_text(
            "<!DOCTYPE html>\n<html><head>"
            '<meta charset="utf-8">'
            '<meta name="pypi:repository-version" content="1.0">'
            "<title>deformops</title></head><body>\n"
            f"{links}\n</body></html>\n",
            encoding="utf-8",
        )
        # Root of the simple index (links to the single project).
        (public / "whl" / cuda).joinpath("index.html").write_text(
            "<!DOCTYPE html>\n<html><head>"
            '<meta charset="utf-8">'
            '<meta name="pypi:repository-version" content="1.0">'
            "<title>Simple index</title></head><body>\n"
            '    <a href="deformops/">deformops</a><br>\n'
            "</body></html>\n",
            encoding="utf-8",
        )


def _write_landing(public: pathlib.Path, ops: dict, by_cuda: dict) -> None:
    ops_items = "\n".join(
        f"<li><code>{html.escape(ver)}</code>: "
        + ", ".join(f"<code>{html.escape(cu)}</code>" for cu in sorted(cudas))
        + "</li>"
        for ver, cudas in sorted(ops.items())
    )
    whl_items = "\n".join(
        f'<li><a href="whl/{html.escape(cu)}/deformops/">whl/{html.escape(cu)}</a> '
        f"({len(files)} wheel(s))</li>"
        for cu, files in sorted(by_cuda.items())
    )
    (public / "index.html").write_text(
        "<!DOCTYPE html>\n<html><head><meta charset='utf-8'>"
        "<title>deformops package index</title></head><body>\n"
        "<h1>deformops package index</h1>\n"
        "<h2>Prebuilt wheels (per CUDA)</h2>\n"
        "<p>Install a CUDA build (the <code>+cuXXX</code> local version "
        "outranks the plain PyPI release):</p>\n"
        "<pre>pip install deformops --extra-index-url "
        "https://tue-p8n.github.io/deformops/whl/cu128</pre>\n"
        f"<ul>{whl_items or '<li><em>none</em></li>'}</ul>\n"
        "<h2>Stable-ABI operator libraries</h2>\n"
        "<p>Fetched at runtime by the generic (PyPI) install; see "
        "<code>ops/manifest.json</code>.</p>\n"
        f"<ul>{ops_items or '<li><em>none</em></li>'}</ul>\n"
        "</body></html>\n",
        encoding="utf-8",
    )


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        sys.stderr.write(f"usage: {argv[0]} <public_dir> <wheel_staging_dir>\n")
        return 2
    public = pathlib.Path(argv[1])
    staging = pathlib.Path(argv[2])
    public.mkdir(parents=True, exist_ok=True)
    ops = _aggregate_ops(public)
    by_cuda = _lay_out_wheels(public, staging)
    _write_pep503(public, by_cuda)
    _write_landing(public, ops, by_cuda)
    sys.stdout.write(
        f"ops versions: {sorted(ops)}; wheel variants: {sorted(by_cuda)}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
