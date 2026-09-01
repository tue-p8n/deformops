"""Collection settings for the build/JIT and registry-mechanics tests."""

from __future__ import annotations

# `snippets/` holds subprocess scripts that run CUDA work at import; doctest
# collection of them aborts the session on a CPU-only host.
collect_ignore = ["snippets"]
