"""Importing the models package is lazy and does not import torch."""

from __future__ import annotations

import subprocess
import sys


def test_full_graph_exports_are_declared() -> None:
    from deformops import models  # noqa: PLC0415

    assert set(models.__all__) == {
        "DeformableDETR",
        "Mask2Former",
    }


def test_importing_deformops_pulls_in_no_model_or_torch() -> None:
    """``deformops`` and ``deformops.models`` import neither torch nor a model."""
    code = (
        "import sys; import deformops; "
        "assert 'torch' not in sys.modules, 'torch imported'; "
        "import deformops.models; "
        "assert 'deformops.models' in sys.modules; "
        "extra = [m for m in sys.modules if m.startswith('deformops.models.')]; "
        "assert not extra, extra"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
