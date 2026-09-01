"""Run the README's Python examples on CPU, so the front page cannot drift from the API.

``.cuda()`` is dropped and registry ``load`` calls are bound to its signature.
Snippets that download weights or need a missing extra are compiled and skipped.
"""

from __future__ import annotations

import importlib
import pathlib
import re

import pytest

_README = pathlib.Path(__file__).resolve().parents[2] / "README.md"

#: A fenced block tagged ``python`` or ``py``.
_BLOCK = re.compile(r"^```(?:python|py)\n(.*?)^```", re.MULTILINE | re.DOTALL)

#: Marks a snippet that downloads pretrained weights, which the offline runner skips.
_NEEDS_NETWORK = "from_pretrained"

#: Rewrites that run a GPU-facing snippet on CPU without changing what it shows.
_SUBSTITUTIONS = (
    # Without the device call, the snippet runs on the eager backend.
    (".cuda()", ""),
)

#: A registry ``load`` call, bound to its signature instead of loading, with
#: its arguments still evaluated so a rename fails.
_LOAD_CALL = re.compile(
    r"^(?P<indent>[ \t]*)(?P<callee>(?:deformops\.)?(?:registry\.)?(?:msda|attn|conv)\.load)"
    r"\((?P<args>.*)\)(?P<comment>[ \t]*#.*)?$",
    re.MULTILINE,
)


def _bind_load_call(match: re.Match[str]) -> str:
    callee, args = match["callee"], match["args"]
    return (
        f"{match['indent']}__import__('inspect').signature({callee}).bind({args})"
        f"{match['comment'] or ''}"
    )


def _snippets() -> list[str]:
    if not _README.is_file():
        return []
    return [m.group(1) for m in _BLOCK.finditer(_README.read_text(encoding="utf-8"))]


def _runnable(source: str) -> str:
    for old, new in _SUBSTITUTIONS:
        source = source.replace(old, new)
    return _LOAD_CALL.sub(_bind_load_call, source)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "attn.load(Backend.TRITON)  # registers the operators",
            "__import__('inspect').signature(attn.load).bind(Backend.TRITON)"
            "  # registers the operators",
        ),
        (
            "    registry.msda.load(Backend.CUDA, compile=True)",
            "    __import__('inspect').signature(registry.msda.load)"
            ".bind(Backend.CUDA, compile=True)",
        ),
        ("state = torch.load(path)", "state = torch.load(path)"),
        ("model.load_state_dict(state)", "model.load_state_dict(state)"),
    ],
)
def test_load_calls_are_bound_rather_than_run(source: str, expected: str) -> None:
    assert _runnable(source) == expected


#: Extras a snippet can need, keyed by the signaling deformops module, each mapped to an
#: import that fails when unusable. A model class, since transformers imports lazily.
_EXTRA_PROBES = {
    "transformers": ("transformers", "DeformableDetrForObjectDetection"),
    "torchmatch": ("torchmatch", None),
    "mmcv-lite": ("mmcv", None),
}
_EXTRAS_BY_MODULE = {
    "deformops.integrations.transformers": ("transformers",),
    "deformops.integrations.mmcv": ("mmcv-lite",),
    "deformops.models.deformable_detr": ("transformers",),
    "deformops.models.mask2former": ("transformers", "torchmatch"),
}


def _extra_is_usable(extra: str) -> bool:
    module_name, attribute = _EXTRA_PROBES[extra]
    try:
        module = importlib.import_module(module_name)
        if attribute is not None:
            getattr(module, attribute)
    except ImportError:
        return False
    return True


def _missing_extras(source: str) -> list[str]:
    needed = {
        extra
        for module, extras in _EXTRAS_BY_MODULE.items()
        if module in source
        for extra in extras
    }
    return sorted(extra for extra in needed if not _extra_is_usable(extra))


_ALL_SNIPPETS = _snippets()
_NETWORK_SNIPPETS = [s for s in _ALL_SNIPPETS if _NEEDS_NETWORK in s]
_EXTRA_SNIPPETS = [
    (s, missing)
    for s in _ALL_SNIPPETS
    if _NEEDS_NETWORK not in s and (missing := _missing_extras(s))
]
_SNIPPETS = [
    s
    for s in _ALL_SNIPPETS
    if _NEEDS_NETWORK not in s and not any(s is e for e, _ in _EXTRA_SNIPPETS)
]


def test_the_readme_has_examples_to_check() -> None:
    """The extraction matches at least one block."""
    if not _README.is_file():
        pytest.skip("running outside a source checkout; README.md is not shipped")
    assert len(_ALL_SNIPPETS) >= 3, (
        f"only {len(_ALL_SNIPPETS)} python blocks found in the README; the extraction "
        f"or the document structure changed"
    )


def test_readme_examples_run() -> None:
    """Every README example runs, in document order against one namespace."""
    if not _README.is_file():
        pytest.skip("running outside a source checkout; README.md is not shipped")
    namespace: dict[str, object] = {"__name__": "readme_example"}
    exec(compile("import torch", "<readme>", "exec"), namespace)  # noqa: S102
    for index, source in enumerate(_SNIPPETS):
        try:
            exec(
                compile(_runnable(source), f"<readme block {index}>", "exec"), namespace
            )  # noqa: S102
        except Exception as exc:  # noqa: BLE001
            pytest.fail(
                f"README python block {index} failed: {type(exc).__name__}: {exc}\n"
                f"--- block ---\n{source}"
            )


@pytest.mark.parametrize(
    "source",
    [pytest.param(s, id=f"network-block-{i}") for i, s in enumerate(_NETWORK_SNIPPETS)]
    or [pytest.param(None, id="no-network-blocks")],
)
def test_readme_network_examples_compile(source: str | None) -> None:
    """Snippets that download weights are compiled, then skipped offline."""
    if source is None:
        pytest.skip("the README has no example that downloads pretrained weights")
    compile(_runnable(source), "<readme network block>", "exec")
    pytest.skip(
        f"example calls {_NEEDS_NETWORK}, which downloads pretrained weights; "
        f"the test session is offline"
    )


@pytest.mark.parametrize(
    ("source", "missing"),
    [
        pytest.param(source, missing, id=f"extras-block-{index}")
        for index, (source, missing) in enumerate(_EXTRA_SNIPPETS)
    ]
    or [pytest.param(None, None, id="no-blocks-missing-extras")],
)
def test_readme_examples_needing_missing_extras_compile(
    source: str | None, missing: list[str] | None
) -> None:
    """Snippets whose extras cannot be imported are compiled; otherwise they run."""
    if source is None:
        pytest.skip("every README example's optional extras are importable here")
    compile(_runnable(source), "<readme extras block>", "exec")
    pytest.skip(f"example needs optional extras that do not import here: {missing}")
