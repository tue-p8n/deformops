"""A bundled prebuilt extension computes what a JIT build computes.

Each library runs in its own subprocess; full outputs and gradients are compared.
"""

import csv
import importlib.resources
import importlib.util
import json
import os
import pathlib
import subprocess
import sys

import pytest
import torch
from deformops.core.configs import EXT_PACKAGE, MSDA_OP_PREFIX

_SNIPPETS = pathlib.Path(__file__).parent / "snippets"
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]

#: The environment that makes a run take its library from one source: the
#: bundled library only, or a library compiled on this machine.
_SOURCE_ENV = {
    "bundled": {"DEFORMOPS_COMPILE": "0", "DEFORMOPS_DOWNLOAD": "0"},
    "compiled": {"DEFORMOPS_COMPILE": "always", "DEFORMOPS_DOWNLOAD": "0"},
}

#: The compiled library each operator's CUDA backend loads.
_LIBRARIES = {"msda": MSDA_OP_PREFIX}


def _bundled_library(name: str) -> pathlib.Path | None:
    """The pre-built ``.so`` for ``name`` in the extension package, if any.

    ``core.build`` loads from here before trying a download or a compile.
    """
    if importlib.util.find_spec(EXT_PACKAGE) is None:
        return None
    candidate = pathlib.Path(str(importlib.resources.files(EXT_PACKAGE) / f"{name}.so"))
    return candidate if candidate.is_file() else None


def run_snippet(
    snippet_name: str, *, source: str, op: str, dtype: str, out_path: pathlib.Path
) -> dict:
    """Run a snippet in a subprocess and return its saved tensors and timing."""
    snippet_path = _SNIPPETS / f"{snippet_name}.py"
    if not snippet_path.exists():
        msg = f"Snippet {snippet_name} not found at {snippet_path}"
        raise FileNotFoundError(msg)

    env = os.environ.copy()
    env.update(_SOURCE_ENV[source])
    env["DEFORMOPS_SEED"] = "1958"
    env["DEFORMOPS_OP"] = op
    env["DEFORMOPS_DTYPE"] = dtype
    env["DEFORMOPS_PARITY_OUT"] = str(out_path)

    sources_path = str(_PROJECT_ROOT / "sources")
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (sources_path, env.get("PYTHONPATH")) if p
    )

    result = subprocess.run(
        [sys.executable, str(snippet_path)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,  # a JIT run may compile the extension first
    )
    assert result.returncode == 0, (
        f"{source} run of {snippet_name} failed with exit code "
        f"{result.returncode}\n--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )

    # Build logs may precede the JSON, which is the last line.
    timing = json.loads(result.stdout.strip().split("\n")[-1])
    return {**timing, **torch.load(out_path)}


@pytest.mark.gpu
@pytest.mark.parametrize("op", ["msda"])
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_binary_jit_parity(op: str, dtype: str, tmp_path: pathlib.Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if _bundled_library(_LIBRARIES[op]) is None:
        pytest.skip(f"no pre-built {_LIBRARIES[op]}.so is bundled in {EXT_PACKAGE}")

    jit = run_snippet(
        "bench_parity",
        source="compiled",
        op=op,
        dtype=dtype,
        out_path=tmp_path / "jit.pt",
    )
    # A bundled library that fails to load or run is a failure, not a skip.
    binary = run_snippet(
        "bench_parity",
        source="bundled",
        op=op,
        dtype=dtype,
        out_path=tmp_path / "prebuilt.pt",
    )

    with (tmp_path / "binary_jit_bench.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["op", "dtype", "mode", "avg_time"])
        writer.writeheader()
        writer.writerow(
            {"op": op, "dtype": dtype, "mode": "JIT", "avg_time": jit["avg_time"]}
        )
        writer.writerow(
            {"op": op, "dtype": dtype, "mode": "Binary", "avg_time": binary["avg_time"]}
        )

    # Both libraries compile the same sources with the same flags, so the
    # tolerance only absorbs nondeterministic accumulation order.
    tol = 1e-2 if dtype in ("float16", "bfloat16") else 1e-4
    torch.testing.assert_close(binary["output"], jit["output"], rtol=tol, atol=tol)
    assert binary["grads"].keys() == jit["grads"].keys()
    for name, expected in jit["grads"].items():
        torch.testing.assert_close(
            binary["grads"][name],
            expected,
            rtol=tol,
            atol=tol,
            msg=lambda m, name=name: f"{name} gradient differs:\n{m}",
        )
