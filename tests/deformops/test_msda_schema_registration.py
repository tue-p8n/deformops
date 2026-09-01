r"""Operator handles exist from the Python schemas before any library loads.

Checks run in a fresh subprocess, since the suite may already have loaded one.
See ``docs/kernel-loading.md``.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

# Four operators per configuration, written out so a change to ``OP_PREFIXES`` fails.
_PREFIXES = (
    "msda_fp32acc",
    "msda_serialreduce",
    "msda_half2atomic",
    "msda_half2atomic_serialreduce",
    "msda_scalaratomic",
    "msda_scalaratomic_serialreduce",
)

_EXPECTED_OPS = tuple(
    f"{prefix}{marker}_{direction}"
    for prefix in _PREFIXES
    for marker in ("", "_softmax")
    for direction in ("forward", "backward")
)

#: The configuration modules under ``deformops.backends.cuda``; each stem is its prefix.
_KERNEL_MODULES = tuple(
    (
        "deformops.backends.cuda"
        if prefix == "msda_fp32acc"
        else "deformops.backends.cuda.experimental"
    )
    + f".{prefix}"
    for prefix in _PREFIXES
)


def _run(script: str) -> subprocess.CompletedProcess[str]:
    """Run ``script`` in a fresh interpreter and return the completed process."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_bare_kernels_import_touches_no_heavy_dependency() -> None:
    """Importing the top-level package alone does not import torch."""
    result = _run(
        """
        import sys

        import deformops  # noqa: F401

        assert "torch" not in sys.modules, sorted(sys.modules)
        print("OK")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_define_schemas_registers_every_operator_without_loading_a_library() -> None:
    """Every operator resolves from the schemas, with no configuration imported."""
    result = _run(
        f"""
        import sys

        import torch

        from deformops.backends.cuda import _schema

        _schema.define_schemas()

        for name in {_EXPECTED_OPS!r}:
            assert hasattr(torch.ops.deformops_cuda, name), name

        for qualname in {_KERNEL_MODULES!r}:
            assert qualname not in sys.modules, qualname
        print("OK")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_define_schemas_is_idempotent() -> None:
    """A second and third call do nothing and raise no "redefined" error."""
    result = _run(
        """
        from deformops.backends.cuda import _schema

        _schema.define_schemas()
        _schema.define_schemas()
        _schema.define_schemas()
        print("OK")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_reimport_of_schema_module_is_idempotent() -> None:
    """Reloading ``_schema`` and calling ``define_schemas`` again is a no-op."""
    result = _run(
        """
        import importlib

        from deformops.backends.cuda import _schema

        _schema.define_schemas()
        importlib.reload(_schema)
        _schema.define_schemas()
        print("OK")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
