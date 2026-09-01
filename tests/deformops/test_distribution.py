"""What the published distribution may and may not contain.

Only ``sources/`` is published. Package discovery and ``setup.py``'s package data
set the boundary, so both the declarations and what they match on disk are checked.
"""

from __future__ import annotations

import ast
import pathlib
import re
import tomllib

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: Suffixes of the files the CUDA backend compiles from at install or first use.
_CUDA_SOURCE_SUFFIXES = (".cu", ".cpp", ".cuh")


def _pyproject() -> dict:
    path = _REPO_ROOT / "pyproject.toml"
    if not path.is_file():
        pytest.skip("running outside a source checkout; pyproject.toml is not shipped")
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_package_discovery_is_scoped_to_the_library_tree() -> None:
    """Discovery must not reach past the library's own sources."""
    find = _pyproject()["tool"]["setuptools"]["packages"]["find"]
    assert find["where"] == ["sources"], (
        f"package discovery searches {find['where']}; widening it past 'sources' "
        f"would sweep whatever else sits at the repository root into the wheel"
    )


def test_every_operator_package_is_discoverable() -> None:
    """Every subpackage on disk is discovered, so none is left out of the wheel."""
    from setuptools import find_packages

    find = _pyproject()["tool"]["setuptools"]["packages"]["find"]
    where = _REPO_ROOT / find["where"][0]
    if not where.is_dir():
        pytest.skip("running outside a source checkout; sources/ is not shipped")
    discovered = set(find_packages(where=str(where), exclude=find.get("exclude", [])))
    expected = {
        "deformops",
        "deformops.backends",
        "deformops.backends.cuda",
        "deformops.backends.cuda.experimental",
        "deformops.backends.triton",
        "deformops.backends.triton.experimental",
        "deformops.backends.eager",
        "deformops.backends.reference",
        "deformops.nn",
        "deformops.models",
        "deformops.models.deformable_detr",
        "deformops.models.mask2former",
        "deformops.integrations",
        "deformops.core",
        "deformops.registry",
    }
    missing = expected - discovered
    assert not missing, f"not discovered by setuptools: {sorted(missing)}"


def _declared_package_data() -> dict[str, list[str]]:
    """``setup.py``'s ``package_data`` literal, parsed without importing torch."""
    path = _REPO_ROOT / "setup.py"
    if not path.is_file():
        pytest.skip("running outside a source checkout; setup.py is not shipped")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "setup":
            for keyword in node.keywords:
                if keyword.arg == "package_data":
                    return ast.literal_eval(keyword.value)
    pytest.fail("setup.py has no setup(package_data=...) literal to check")


def test_cuda_backend_package_data_covers_every_kernel_source() -> None:
    """Every file the CUDA kernels compile from ships in the wheel.

    A first-use compile reads them from the installed package; the patterns expand
    as globs relative to the package, as setuptools does.
    """
    sources_root = _REPO_ROOT / "sources"
    package = "deformops.backends.cuda"
    package_dir = sources_root / pathlib.Path(*package.split("."))
    sources = sorted(
        path.relative_to(sources_root).as_posix()
        for path in package_dir.rglob("*")
        if path.is_file() and path.suffix in _CUDA_SOURCE_SUFFIXES
    )
    assert sources, f"no kernel sources found under {package_dir}"

    declared = _declared_package_data()
    assert package in declared, f"setup.py declares no package_data for {package}"
    shipped = {
        path.relative_to(sources_root).as_posix()
        for name, patterns in declared.items()
        for pattern in patterns
        for path in (sources_root / pathlib.Path(*name.split("."))).glob(pattern)
        if path.is_file()
    }
    uncovered = [source for source in sources if source not in shipped]
    assert not uncovered, (
        f"package_data misses kernel sources the wheel needs to compile from: "
        f"{uncovered}"
    )


# ---------------------------------------------------------------------------
# Kernels compiled at install time
# ---------------------------------------------------------------------------


def _non_experimental() -> set[str]:
    from deformops.core.configs import EXTENSION_CONFIGS

    return {n for n, c in EXTENSION_CONFIGS.items() if not c["experimental"]}


def test_a_source_distribution_compiles_the_non_experimental_kernels() -> None:
    from deformops.core.configs import select_extensions

    selected = select_extensions(None, source_distribution=True)
    assert set(selected) == _non_experimental()
    assert set(selected), "no non-experimental extension is declared"


def test_a_checkout_compiles_nothing_by_default() -> None:
    from deformops.core.configs import select_extensions

    assert select_extensions(None, source_distribution=False) == {}


@pytest.mark.parametrize("value", ["none", "NONE", "", " 0 ", "off"])
@pytest.mark.parametrize("source_distribution", [True, False])
def test_the_variable_opts_out(value: str, source_distribution: bool) -> None:
    from deformops.core.configs import select_extensions

    assert select_extensions(value, source_distribution=source_distribution) == {}


def test_the_variable_selects_explicitly() -> None:
    from deformops.core.configs import EXTENSION_CONFIGS, select_extensions

    everything = select_extensions("everything", source_distribution=False)
    assert set(everything) == set(EXTENSION_CONFIGS)
    assert set(select_extensions("all", source_distribution=False)) == (
        _non_experimental()
    )
    name = next(iter(EXTENSION_CONFIGS))
    assert set(select_extensions(f" {name} ,", source_distribution=True)) == {name}


def test_the_variable_rejects_an_unknown_extension() -> None:
    from deformops.core.configs import select_extensions

    with pytest.raises(ValueError, match="not_a_kernel"):
        select_extensions("not_a_kernel", source_distribution=True)


@pytest.mark.parametrize(
    "workflow", [".github/workflows/release.yml", ".github/workflows/ci.yml"]
)
def test_the_generic_wheel_builds_opt_out_of_compiling(workflow: str) -> None:
    """Release and CI build the generic wheel from the sdist without compiling."""
    path = _REPO_ROOT / workflow
    if not path.is_file():
        pytest.skip("running outside a source checkout")
    text = path.read_text(encoding="utf-8")
    for match in re.finditer(r"run:[^\n]*uv build[^\n]*", text):
        preceding = text[max(0, match.start() - 200) : match.start()]
        assert "DEFORMOPS_EXTENSIONS: none" in preceding, (
            f"{workflow} builds a wheel without DEFORMOPS_EXTENSIONS: none"
        )


def _toolkit(tmp_path: pathlib.Path, name: str) -> pathlib.Path:
    root = tmp_path / name
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "nvcc").write_text("")
    return root


def test_the_toolkit_is_found_where_torch_looks(tmp_path: pathlib.Path) -> None:
    from deformops.core.configs import find_cuda_toolkit

    home = _toolkit(tmp_path, "home")
    on_path = _toolkit(tmp_path, "path")
    default = _toolkit(tmp_path, "default")
    nvcc = str(on_path / "bin" / "nvcc")

    def found(environ: dict[str, str], which_result: str | None) -> str | None:
        return find_cuda_toolkit(
            environ, lambda _name: which_result, default_root=str(default)
        )

    assert found({"CUDA_HOME": str(home)}, nvcc) == str(home)
    assert found({"CUDA_PATH": str(home)}, nvcc) == str(home)
    assert found({}, nvcc) == str(on_path)
    assert found({}, None) == str(default)


def test_a_machine_without_nvcc_has_no_toolkit(tmp_path: pathlib.Path) -> None:
    from deformops.core.configs import find_cuda_toolkit

    runtime_only = tmp_path / "runtime"
    (runtime_only / "lib64").mkdir(parents=True)
    missing = str(tmp_path / "missing")
    assert find_cuda_toolkit({}, lambda _name: None, default_root=missing) is None
    assert (
        find_cuda_toolkit(
            {"CUDA_HOME": str(runtime_only)}, lambda _name: None, default_root=missing
        )
        is None
    )


@pytest.mark.parametrize(
    ("output", "release"),
    [
        ("Cuda compilation tools, release 12.8, V12.8.93", (12, 8)),
        ("Cuda compilation tools, release 13.0, V13.0.88", (13, 0)),
        ("nvcc: command output without a release", None),
    ],
)
def test_the_nvcc_release_is_parsed(output: str, release) -> None:
    from deformops.core.configs import parse_nvcc_release

    assert parse_nvcc_release(output) == release


def test_the_skipped_build_marker_ships_in_the_wheel() -> None:
    from deformops.core.configs import SKIPPED_BUILD_FILE

    assert f"_ext/{SKIPPED_BUILD_FILE}" in _declared_package_data()["deformops.core"]


def test_every_kernel_targets_the_minimum_torch_abi() -> None:
    """The target version also makes torch reject non-stable headers."""
    from deformops.core.configs import EXTENSION_CONFIGS, MINIMUM_TORCH_ABI

    flag = f"-DTORCH_TARGET_VERSION={MINIMUM_TORCH_ABI}"
    for name, config in EXTENSION_CONFIGS.items():
        assert flag in config["cflags"], name
        assert flag in config["cflags_cuda"], name
