"""Double precision is compiled in by default; only ahead-of-time builds opt out.

``DEFORMOPS_NO_FP64`` removes the ``Double`` branch, so the JIT build, which never
passes it, keeps float64. Each flag list is checked on its own; see
``docs/kernel-loading.md``.
"""

import pathlib
import re

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

_FLAG = "-DDEFORMOPS_NO_FP64"


def _get_extensions_source() -> str:
    text = (_REPO_ROOT / "setup.py").read_text(encoding="utf-8")
    match = re.search(r"def get_extensions\(\).*?(?=\ndef |\Z)", text, re.DOTALL)
    assert match, "setup.py does not define get_extensions()"
    return match.group(0)


def _define_library_source() -> str:
    text = (_REPO_ROOT / "sources/deformops/core/build.py").read_text(encoding="utf-8")
    match = re.search(r"def define_library\(.*?(?=\ndef |\Z)", text, re.DOTALL)
    assert match, "build.py does not define define_library()"
    return match.group(0)


def _assignment_line(source: str, name: str) -> str:
    """The line assigning ``name`` itself; ``cflags`` is a prefix of ``cflags_cuda``."""
    match = re.search(rf"^\s*{re.escape(name)}\s*=.*$", source, re.MULTILINE)
    assert match, f"could not find a `{name} = ...` assignment in get_extensions()"
    return match.group(0)


def test_setup_py_cxx_flags_opt_out_of_double_precision() -> None:
    """The ahead-of-time build's C++ flags (`cflags`) include the macro."""
    line = _assignment_line(_get_extensions_source(), "cflags")
    assert _FLAG in line, (
        f"setup.py's get_extensions() should add {_FLAG} to `cflags`; found: {line!r}"
    )


def test_setup_py_cuda_flags_opt_out_of_double_precision() -> None:
    """The ahead-of-time ``cflags_cuda`` include the macro on their own."""
    line = _assignment_line(_get_extensions_source(), "cflags_cuda")
    assert _FLAG in line, (
        f"setup.py's get_extensions() should add {_FLAG} to `cflags_cuda`; "
        f"found: {line!r}"
    )


def test_define_library_keeps_double_precision() -> None:
    """The just-in-time build never adds the opt-out macro."""
    source = _define_library_source()
    assert _FLAG not in source, (
        f"define_library() must not add {_FLAG}; the JIT path is supposed "
        f"to keep full dtype coverage, including double precision, for "
        f"builds that set DEFORMOPS_COMPILE."
    )
