from __future__ import annotations

import logging
import os
import random
import warnings

import pytest
import torch

logger = logging.getLogger(__name__)

#: Variables ``pytest_configure`` sets: no prebuilt index serves a source checkout,
#: so downloads are off and kernels compile on demand.
_ENV_OVERRIDES = {"DEFORMOPS_DOWNLOAD": "0"}

#: Environment variables ``pytest_configure`` sets only when the caller left
#: them unset, so a session can still ask for ``always`` or ``rebuild``.
_ENV_DEFAULTS = {"DEFORMOPS_COMPILE": "1"}

_saved_env: dict[str, str | None] = {}
_saved_cudnn_enabled: bool | None = None


def _prebuild_extensions(config) -> None:
    """Load the production CUDA multi-scale kernel once, before tests start.

    Other configurations compile on first use, since each would add minutes. A
    failure warns once instead of aborting. ``DEFORMOPS_TEST_PREBUILD=0`` skips it.
    """
    if os.environ.get("DEFORMOPS_TEST_PREBUILD") == "0":
        logger.info("Extension pre-build disabled by DEFORMOPS_TEST_PREBUILD=0")
        return
    if not torch.cuda.is_available():
        # No CUDA kernel can run here, so there is nothing to load.
        return

    from deformops.registry import Backend, msda

    # Workers re-run this so their coverage tracer records the module-level
    # code the main process already executed; only the main process reports.
    is_worker = hasattr(config, "workerinput")

    try:
        msda.load(Backend.CUDA)
    except Exception as exc:  # noqa: BLE001
        # Expected: KernelUnavailableError, NoBackendError, or a compiler RuntimeError.
        if is_worker:
            return
        logger.warning("Pre-build of the CUDA msda kernel failed", exc_info=True)
        config.issue_config_time_warning(
            pytest.PytestWarning(
                f"loading the CUDA msda kernel library failed "
                f"({type(exc).__name__}: {exc}); tests that need it load it on "
                f"first use and fail or skip individually"
            ),
            stacklevel=2,
        )


def pytest_configure(config):
    global _saved_cudnn_enabled  # noqa: PLW0603

    # Keep the test session offline and let it compile what it needs; see
    # ``_ENV_OVERRIDES`` and ``_ENV_DEFAULTS``.
    for key, value in _ENV_OVERRIDES.items():
        _saved_env[key] = os.environ.get(key)
        os.environ[key] = value
    for key, value in _ENV_DEFAULTS.items():
        _saved_env[key] = os.environ.get(key)
        if not os.environ.get(key, "").strip():
            os.environ[key] = value

    # The canonical API warns on PyTorch 2.9+, and setting ``fp32_precision`` directly
    # breaks Inductor's ``allow_tf32`` probe, so only the deprecation is silenced.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*new API settings to control TF32 behavior.*",
            category=UserWarning,
        )
        torch.set_float32_matmul_precision("high")

    if torch.cuda.is_available():
        # Warm the CUDA context and cuBLAS handles on the autograd worker thread too, or
        # ``gradcheck``'s first backward warns about a missing CUDA context.
        torch.cuda.init()
        _warmup = torch.eye(2, device="cuda", requires_grad=True)
        (_warmup @ _warmup).sum().backward()
        del _warmup
    elif torch.version.cuda is not None:
        # A CUDA torch on a CPU-only host aborts in ``cudnn.version()``, which the
        # environment tag reads; disabling cudnn gives the ``-nocudnn`` tag instead.
        _saved_cudnn_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False

    _prebuild_extensions(config)

    seed = 1958
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def pytest_unconfigure(config):  # noqa: ARG001
    """Undo the process-wide changes ``pytest_configure`` made."""
    global _saved_cudnn_enabled  # noqa: PLW0603

    for key, previous in _saved_env.items():
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous
    _saved_env.clear()

    if _saved_cudnn_enabled is not None:
        torch.backends.cudnn.enabled = _saved_cudnn_enabled
        _saved_cudnn_enabled = None


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    """Mark every test that takes the CUDA-only ``device`` fixture as ``gpu``.

    Runs before ``-m`` deselection. Tests whose ``device`` comes from
    :func:`any_device` or ``parametrize`` mark their CUDA cases themselves.
    """
    for item in items:
        info = getattr(item, "_fixtureinfo", None)
        definitions = info.name2fixturedefs.get("device") if info else None
        if (
            definitions
            and getattr(definitions[-1].func, "__module__", None) == __name__
        ):
            item.add_marker(pytest.mark.gpu)


@pytest.fixture(scope="session")
def device():
    """The CUDA device, skipping the test where none is available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return torch.device("cuda")


@pytest.fixture(
    scope="session",
    params=[
        pytest.param("cpu", id="cpu"),
        pytest.param(
            "cuda",
            id="cuda",
            marks=[
                pytest.mark.gpu,
                pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA is not available"
                ),
            ],
        ),
    ],
)
def any_device(request):
    """Every device a CPU-capable test can run on: the CPU always, CUDA when present.

    Modules override ``device`` with this so the CPU job runs the GPU comparison;
    the CUDA case has the ``gpu`` marker.
    """
    return torch.device(request.param)
