"""The compiled dispatch packs match what the tuning heuristic can emit.

An extra value is dead code; a missing one turns a valid shape into a TORCH_CHECK
failure. Each site is checked against its own reachable set in both directions,
since a union check misses a one-sided narrowing.
"""

import itertools
import pathlib
import re
from collections.abc import Callable

import pytest
from deformops.core.tuning import backward_thread_layout, forward_thread_layout

INCLUDE_DIR = (
    pathlib.Path(__file__).parents[3] / "sources/deformops/backends/cuda/include"
)


@pytest.fixture(autouse=True)
def _skip_without_checkout() -> None:
    """The headers are read from the source tree, which only a checkout has."""
    if not INCLUDE_DIR.is_dir():
        pytest.skip(
            f"running outside a source checkout; {INCLUDE_DIR.name}/ is not present"
        )


# Shapes wide enough to exercise both branches of each heuristic.
BATCH = (1, 2)
WORK = (16, 64, 256)
GROUPS = (1, 4, 8)
CHANNELS = (1, 2, 4, 8, 16, 32, 64, 128, 256)

# Sites by enclosing function: independent of order, and a moved site fails the
# lookup instead of skipping the check.
_FORWARD_SITE = "forward_launch_inner"
_BACKWARD_SITE = "backward_launch_inner"

_SITE_HEADERS = {
    _FORWARD_SITE: INCLUDE_DIR / "msda_forward.cuh",
    _BACKWARD_SITE: INCLUDE_DIR / "msda_backward.cuh",
}


def _reachable(layout: Callable[[int, int, int, int], tuple[int, int]]) -> set[int]:
    return {
        layout(b, w, g, c)[0]
        for b, w, g, c in itertools.product(BATCH, WORK, GROUPS, CHANNELS)
    }


def _packs() -> list[list[int]]:
    """Every ``channels_per_thread`` pack under ``include/``, catching a new site."""
    return [
        [int(v) for v in m.split(",")]
        for header in sorted(INCLUDE_DIR.glob("*.cuh"))
        for m in re.findall(
            r"dispatch_constant<([\d,\s]+)>\(channels_per_thread",
            header.read_text(encoding="utf-8"),
        )
    ]


def _pack_for_site(site: str) -> set[int]:
    """The first ``channels_per_thread`` pack after ``site``'s own definition."""
    header = _SITE_HEADERS[site]
    text = header.read_text(encoding="utf-8")
    definition = re.search(rf"\bvoid\s+{re.escape(site)}\s*\(", text)
    assert definition, f"could not find the definition of {site} in {header}"
    dispatch = re.search(
        r"dispatch_constant<([\d,\s]+)>\(channels_per_thread", text[definition.end() :]
    )
    assert dispatch, (
        f"could not find a channels_per_thread dispatch_constant call after "
        f"{site}'s definition"
    )
    return {int(v) for v in dispatch.group(1).split(",")}


def test_two_channels_per_thread_dispatch_sites_exist():
    assert len(_packs()) == 2, (
        f"expected one forward and one backward site, got {_packs()}"
    )


def test_forward_pack_covers_every_reachable_value():
    reachable = _reachable(forward_thread_layout)
    compiled = _pack_for_site(_FORWARD_SITE)
    missing = sorted(reachable - compiled)
    assert not missing, (
        f"{_FORWARD_SITE} can be asked for {missing} but its pack does not compile it"
    )


def test_forward_pack_compiles_nothing_unreachable():
    reachable = _reachable(forward_thread_layout)
    compiled = _pack_for_site(_FORWARD_SITE)
    dead = sorted(compiled - reachable)
    assert not dead, (
        f"{_FORWARD_SITE}'s pack compiles {dead}, which the heuristic cannot ask for"
    )


def test_backward_pack_covers_every_reachable_value():
    reachable = _reachable(backward_thread_layout)
    compiled = _pack_for_site(_BACKWARD_SITE)
    missing = sorted(reachable - compiled)
    assert not missing, (
        f"{_BACKWARD_SITE} can be asked for {missing} but its pack does not compile it"
    )


def test_backward_pack_compiles_nothing_unreachable():
    reachable = _reachable(backward_thread_layout)
    compiled = _pack_for_site(_BACKWARD_SITE)
    dead = sorted(compiled - reachable)
    assert not dead, (
        f"{_BACKWARD_SITE}'s pack compiles {dead}, which the heuristic cannot ask for"
    )
