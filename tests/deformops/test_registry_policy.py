"""Backend selection: the candidate policies, the candidate walk, memoization.

Policies are functions of the context, so synthetic contexts test them on CPU; a
throwaway operator exercises the generic dispatch machinery.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
import textwrap
import warnings

import pytest
import torch
from deformops.core.errors import (
    BackendFallbackWarning,
    KernelDownloadError,
    KernelNotProvidedError,
    NoBackendError,
)
from deformops.registry import Backend, Registry, Resolution, attn, conv, msda
from deformops.registry.attn import AttnContext
from deformops.registry.attn import select_backend as select_attn_backend
from deformops.registry.conv import ConvContext
from deformops.registry.conv import select_backend as select_conv_backend
from deformops.registry.msda import MSDAContext, select_backend

# The production policies


def _context(
    *,
    device: str = "cuda",
    sm: tuple[int, int] = (8, 6),
    dtype: torch.dtype = torch.bfloat16,
    training: bool = True,
    points: int = 4,
) -> MSDAContext:
    return MSDAContext(
        device=torch.device(device),
        num_levels=3,
        num_heads=8,
        num_points=points,
        dtype=dtype,
        sm_version=sm,
        is_training=training,
        num_queries=100,
    )


def _triton_then_cuda(fp32acc: bool) -> list:
    return [(Backend.TRITON, {"fp32acc": fp32acc}), Backend.CUDA]


@pytest.mark.parametrize(
    ("context", "expected"),
    [
        pytest.param(_context(device="cpu"), [Backend.EAGER], id="cpu"),
        pytest.param(
            _context(device="cpu", dtype=torch.float32, training=False),
            [Backend.EAGER],
            id="cpu-fp32-inference",
        ),
        pytest.param(
            _context(device="cuda", dtype=torch.float64),
            [Backend.EAGER],
            id="float64",
        ),
        pytest.param(
            _context(device="cuda", sm=(7, 5)),
            _triton_then_cuda(fp32acc=True),
            id="turing-bf16-training",
        ),
        pytest.param(
            _context(device="cuda", sm=(8, 6), dtype=torch.float16, training=True),
            _triton_then_cuda(fp32acc=True),
            id="ampere-half-training",
        ),
        pytest.param(
            _context(device="cuda", sm=(8, 0), dtype=torch.bfloat16, training=True),
            _triton_then_cuda(fp32acc=True),
            id="ampere-bf16-training",
        ),
        pytest.param(
            _context(device="cuda", sm=(8, 6), dtype=torch.bfloat16, training=False),
            _triton_then_cuda(fp32acc=False),
            id="ampere-bf16-inference",
        ),
        pytest.param(
            _context(device="cuda", sm=(8, 6), dtype=torch.float32, training=True),
            _triton_then_cuda(fp32acc=False),
            id="ampere-fp32-training",
        ),
        pytest.param(
            _context(device="cuda", sm=(9, 0), dtype=torch.bfloat16, training=True),
            _triton_then_cuda(fp32acc=False),
            id="hopper-bf16",
        ),
    ],
)
def test_msda_policy(context, expected) -> None:
    assert select_backend(context) == expected


@pytest.mark.parametrize("sm", [(7, 5), (8, 6), (9, 0)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("training", [True, False])
def test_msda_policy_never_offers_eager_for_a_cuda_tensor(sm, dtype, training) -> None:
    candidates = select_backend(
        _context(device="cuda", sm=sm, dtype=dtype, training=training)
    )
    backends = [c if isinstance(c, Backend) else c[0] for c in candidates]
    assert Backend.EAGER not in backends


@pytest.mark.parametrize(
    ("context", "expected"),
    [
        pytest.param(
            AttnContext(device=torch.device("cpu"), dtype=torch.float32),
            [Backend.EAGER],
            id="cpu",
        ),
        pytest.param(
            AttnContext(device=torch.device("cuda"), dtype=torch.float64),
            [Backend.EAGER],
            id="float64",
        ),
        pytest.param(
            AttnContext(
                device=torch.device("cuda"),
                dtype=torch.float16,
                sm_version=(8, 6),
                is_training=True,
            ),
            [(Backend.TRITON, {"fp32acc": True})],
            id="ampere-half-training",
        ),
    ],
)
def test_attn_policy(context, expected) -> None:
    assert select_attn_backend(context) == expected


@pytest.mark.parametrize(
    ("device", "dtype", "expected"),
    [
        ("cpu", torch.float32, [Backend.EAGER]),
        ("cuda", torch.float64, [Backend.EAGER]),
        ("cuda", torch.float16, [Backend.TRITON]),
        ("cuda", torch.float32, [Backend.TRITON]),
    ],
)
def test_conv_policy(device, dtype, expected) -> None:
    context = ConvContext(device=torch.device(device), dtype=dtype)
    assert select_conv_backend(context) == expected


def test_context_is_hashable() -> None:
    """The context is also the registry's memoization key."""
    assert hash(_context()) == hash(_context())
    assert _context() == _context()


# Generic dispatch machinery, with a throwaway operator


@dataclasses.dataclass(frozen=True)
class _Context:
    device: str
    dtype: str


def _machinery(  # noqa: PLR0913
    *,
    policy,
    eager_supports=lambda _context: True,
    triton_supports=lambda _context: True,
    triton_load=None,
) -> tuple[Registry, list[_Context]]:
    calls: list[_Context] = []

    def _policy(context: _Context):
        calls.append(context)
        return policy(context)

    registry: Registry = Registry(
        "nonexistent", context_of=lambda context, **_options: context, select=_policy
    )
    # A synthetic operator has no adapter modules; its backends are registered below.
    registry._backends_loaded = True
    registry.register(
        Backend.EAGER,
        lambda _context, **options: ("eager", options),
        supports=eager_supports,
    )
    registry.register(
        Backend.TRITON,
        lambda _context, **options: ("triton", options),
        supports=triton_supports,
        load=triton_load,
    )
    return registry, calls


def _ran(result) -> str:
    return result[0]


@pytest.fixture(autouse=True)
def _fallback_warnings_enabled(monkeypatch) -> None:
    monkeypatch.delenv("DEFORMOPS_FALLBACK_WARNINGS", raising=False)


def test_auto_runs_the_first_candidate_without_a_warning() -> None:
    registry, _ = _machinery(policy=lambda _context: [Backend.TRITON, Backend.EAGER])
    with warnings.catch_warnings():
        warnings.simplefilter("error", BackendFallbackWarning)
        assert _ran(registry(_Context("cuda", "float32"))) == "triton"


def test_auto_accepts_a_bare_backend_and_a_single_pair() -> None:
    registry, _ = _machinery(policy=lambda _context: Backend.TRITON)
    assert _ran(registry(_Context("cuda", "float32"))) == "triton"

    registry, _ = _machinery(
        policy=lambda _context: (Backend.TRITON, {"fp32acc": True})
    )
    assert registry(_Context("cuda", "float32")) == ("triton", {"fp32acc": True})


def test_auto_follows_the_candidate_order() -> None:
    registry, _ = _machinery(policy=lambda _context: [Backend.EAGER, Backend.TRITON])
    assert _ran(registry(_Context("cuda", "float32"))) == "eager"


def test_auto_passes_the_candidate_options_to_the_backend() -> None:
    registry, _ = _machinery(
        policy=lambda _context: [(Backend.TRITON, {"fp32acc": False})]
    )
    assert registry(_Context("cuda", "float32")) == ("triton", {"fp32acc": False})


def test_auto_falls_back_past_an_unregistered_candidate_with_a_warning() -> None:
    registry, _ = _machinery(policy=lambda _context: [Backend.CUDA, Backend.TRITON])
    registry.mark_unavailable(Backend.CUDA, "no CUDA device is available")
    with pytest.warns(BackendFallbackWarning, match="no CUDA device is available"):
        assert _ran(registry(_Context("cuda", "float32"))) == "triton"


def test_auto_falls_back_past_an_unsupported_candidate_with_a_warning() -> None:
    registry, _ = _machinery(
        policy=lambda _context: [Backend.TRITON, Backend.EAGER],
        triton_supports=lambda _context: False,
    )
    with pytest.warns(BackendFallbackWarning, match="does not support this call"):
        assert _ran(registry(_Context("cuda", "float32"))) == "eager"


def test_auto_does_not_fall_back_beyond_the_policy() -> None:
    """A registered backend the policy did not offer is never run."""
    registry, _ = _machinery(
        policy=lambda _context: [Backend.TRITON],
        triton_supports=lambda _context: False,
    )
    with pytest.raises(NoBackendError):
        registry(_Context("cuda", "float32"))


def test_fallback_warning_names_every_skipped_candidate() -> None:
    registry, _ = _machinery(
        policy=lambda _context: [Backend.CUDA, Backend.TRITON, Backend.EAGER],
        triton_supports=lambda _context: False,
    )
    with pytest.warns(BackendFallbackWarning) as record:
        assert _ran(registry(_Context("cuda", "float32"))) == "eager"
    assert len(record) == 1
    message = str(record[0].message)
    assert "'cuda'" in message
    assert "it is not registered on this host" in message
    assert "'triton'" in message
    assert "does not support this call" in message
    assert "DEFORMOPS_FALLBACK_WARNINGS=0" in message


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF"])
def test_fallback_warning_is_silenced_by_the_environment(monkeypatch, value) -> None:
    monkeypatch.setenv("DEFORMOPS_FALLBACK_WARNINGS", value)
    registry, _ = _machinery(policy=lambda _context: [Backend.CUDA, Backend.TRITON])
    with warnings.catch_warnings():
        warnings.simplefilter("error", BackendFallbackWarning)
        assert _ran(registry(_Context("cuda", "float32"))) == "triton"


def test_fallback_warning_is_emitted_once_per_context() -> None:
    registry, calls = _machinery(policy=lambda _context: [Backend.CUDA, Backend.TRITON])
    context = _Context("cuda", "float32")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        results = [_ran(registry(context)) for _ in range(5)]
    assert results == ["triton"] * 5
    assert [type(w.message) for w in record] == [BackendFallbackWarning]
    assert calls == [context]


def test_no_candidate_raises_no_backend_with_every_reason() -> None:
    registry, _ = _machinery(
        policy=lambda _context: [Backend.CUDA, Backend.TRITON],
        triton_supports=lambda _context: False,
    )
    registry.mark_unavailable(Backend.CUDA, "no CUDA device is available")
    with pytest.raises(NoBackendError) as caught:
        registry(_Context("cuda", "float32"))
    message = str(caught.value)
    assert "'cuda': no CUDA device is available" in message
    assert "'triton': it does not support this call" in message


def test_a_loader_without_a_kernel_falls_back_to_the_next_candidate() -> None:
    def triton_load():
        msg = "the kernel library is not bundled and compiling is disabled"
        raise KernelNotProvidedError(msg)

    registry, _ = _machinery(
        policy=lambda _context: [Backend.TRITON, Backend.EAGER],
        triton_load=triton_load,
    )
    with pytest.warns(BackendFallbackWarning, match="compiling is disabled"):
        assert _ran(registry(_Context("cuda", "float32"))) == "eager"


def test_a_failed_download_propagates_instead_of_falling_back() -> None:
    def triton_load():
        msg = "the index is unreachable"
        raise KernelDownloadError(msg)

    registry, _ = _machinery(
        policy=lambda _context: [Backend.TRITON, Backend.EAGER],
        triton_load=triton_load,
    )
    with pytest.raises(KernelDownloadError, match="unreachable"):
        registry(_Context("cuda", "float32"))


def test_a_loader_runs_before_its_backend_is_chosen() -> None:
    loads: list[int] = []
    registry, _ = _machinery(
        policy=lambda _context: [Backend.TRITON],
        triton_load=lambda: loads.append(1),
    )
    context = _Context("cuda", "float32")
    assert _ran(registry(context)) == "triton"
    assert _ran(registry(context)) == "triton"
    assert loads == [1]


def test_auto_policy_runs_once_per_context() -> None:
    registry, calls = _machinery(policy=lambda _context: [Backend.EAGER])
    context = _Context("cpu", "float32")
    for _ in range(5):
        assert _ran(registry(context)) == "eager"
    assert calls == [context]


def test_resolve_reports_the_backend_options_and_rejections() -> None:
    registry, _ = _machinery(
        policy=lambda _context: [
            Backend.CUDA,
            (Backend.TRITON, {"fp32acc": True}),
        ]
    )
    with pytest.warns(BackendFallbackWarning):
        resolution = registry.resolve(_Context("cuda", "float32"))
    assert resolution == Resolution(
        Backend.TRITON,
        {"fp32acc": True},
        ((Backend.CUDA, "it is not registered on this host"),),
    )


def test_resolve_matches_the_call() -> None:
    registry, _ = _machinery(policy=lambda _context: [Backend.EAGER])
    context = _Context("cpu", "float32")
    resolution = registry.resolve(context)
    assert resolution == Resolution(Backend.EAGER, {}, ())
    assert _ran(registry(context)) == resolution.backend.value


def test_resolve_with_an_explicit_backend_checks_it() -> None:
    registry, _ = _machinery(
        policy=lambda _context: [Backend.EAGER],
        triton_supports=lambda context: context.dtype != "float64",
    )
    assert registry.resolve(
        _Context("cuda", "float32"), backend=Backend.TRITON
    ) == Resolution(Backend.TRITON, {}, ())
    with pytest.raises(NoBackendError, match="does not support"):
        registry.resolve(_Context("cuda", "float64"), backend=Backend.TRITON)
    with pytest.raises(TypeError, match="Raw strings are not permitted"):
        registry.resolve(_Context("cuda", "float32"), backend="triton")


def test_resolve_with_an_explicit_backend_loads_its_kernel() -> None:
    """``resolve(backend=...)`` makes the kernel available, as the call would."""
    loads: list[int] = []
    registry, _ = _machinery(
        policy=lambda _context: [Backend.EAGER],
        triton_load=lambda: loads.append(1),
    )
    context = _Context("cuda", "float32")
    assert registry.resolve(context, backend=Backend.TRITON) == Resolution(
        Backend.TRITON, {}, ()
    )
    assert loads == [1]


def test_resolve_with_an_explicit_backend_propagates_a_loader_error() -> None:
    def triton_load():
        msg = "the kernel library is not bundled and compiling is disabled"
        raise KernelNotProvidedError(msg)

    registry, _ = _machinery(
        policy=lambda _context: [Backend.EAGER],
        triton_load=triton_load,
    )
    with pytest.raises(KernelNotProvidedError, match="compiling is disabled"):
        registry.resolve(_Context("cuda", "float32"), backend=Backend.TRITON)


def test_explicit_pin_runs_the_named_backend() -> None:
    registry, _ = _machinery(policy=lambda _context: [Backend.CUDA])
    assert (
        _ran(registry(_Context("cuda", "float32"), backend=Backend.TRITON)) == "triton"
    )


def test_explicit_pin_rejects_unsupported_context() -> None:
    registry, _ = _machinery(
        policy=lambda _context: [Backend.EAGER],
        triton_supports=lambda context: context.dtype != "float64",
    )
    with pytest.raises(NoBackendError, match="does not support this nonexistent call"):
        registry(_Context("cuda", "float64"), backend=Backend.TRITON)


def test_explicit_pin_rejects_unregistered_backend() -> None:
    registry, _ = _machinery(policy=lambda _context: [Backend.EAGER])
    registry.mark_unavailable(Backend.CUDA, "no CUDA device is available")
    with pytest.raises(NoBackendError, match="unavailable: no CUDA device"):
        registry(_Context("cpu", "float32"), backend=Backend.CUDA)
    assert registry.unavailable_reason(Backend.CUDA) == "no CUDA device is available"


def test_registering_a_backend_clears_its_unavailable_reason() -> None:
    registry, _ = _machinery(policy=lambda _context: [Backend.EAGER])
    registry.mark_unavailable(Backend.CUDA, "no CUDA device is available")
    registry.register(Backend.CUDA, lambda _context, **_options: ("cuda", {}))
    assert registry.unavailable_reason(Backend.CUDA) == (
        "it is not registered on this host"
    )
    registry.mark_unavailable(Backend.CUDA, "ignored once registered")
    assert registry.is_registered(Backend.CUDA)


def test_raw_string_backend_is_rejected() -> None:
    registry, _ = _machinery(policy=lambda _context: [Backend.EAGER])
    with pytest.raises(TypeError, match="Raw strings are not permitted"):
        registry(_Context("cpu", "float32"), backend="eager")


# The production registries on a host without a CUDA device

requires_cpu_host = pytest.mark.skipif(
    torch.cuda.is_available(), reason="checks a host without a CUDA device"
)


def _msda_operands():
    shapes = torch.tensor([[8, 8], [4, 4]])
    starts = torch.tensor([0, 64])
    value = torch.randn(1, 80, 4, 8)
    loc = torch.rand(1, 5, 4, 2, 4, 2)
    weights = torch.rand(1, 5, 4, 2 * 4)
    return value, shapes, starts, loc, weights


def test_msda_on_cpu_tensors_runs_eager_without_warnings() -> None:
    operands = _msda_operands()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        output = msda(*operands)
        resolution = msda.resolve(*operands)
    assert output.shape == (1, 5, 32)
    assert resolution == Resolution(Backend.EAGER, {}, ())
    torch.testing.assert_close(output, msda(*operands, backend=Backend.EAGER))


def test_msda_on_cpu_tensors_warns_nothing_in_a_fresh_interpreter() -> None:
    """The first call, which imports every adapter, warns nothing."""
    script = textwrap.dedent(
        """
        import warnings

        warnings.simplefilter("error")

        import torch
        from deformops.registry import Backend, msda

        shapes = torch.tensor([[8, 8], [4, 4]])
        starts = torch.tensor([0, 64])
        value = torch.randn(1, 80, 4, 8)
        loc = torch.rand(1, 5, 4, 2, 4, 2)
        weights = torch.rand(1, 5, 4, 2 * 4)
        assert msda.resolve(value, shapes, starts, loc, weights).backend is (
            Backend.EAGER
        )
        assert msda(value, shapes, starts, loc, weights).shape == (1, 5, 32)
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_attn_and_conv_on_cpu_tensors_resolve_to_eager() -> None:
    value = torch.randn(1, 6, 6, 32)
    offset_mask = torch.zeros(1, 6, 6, 2 * 9 * 3)
    assert attn.resolve(value, offset_mask, groups=2, group_dims=16).backend is (
        Backend.EAGER
    )
    conv_value = torch.randn(1, 6, 6, 4)
    offset = torch.zeros(1, 6, 6, 1, 9, 2)
    weight = torch.randn(4, 4, 3, 3)
    assert conv.resolve(conv_value, offset, weight, padding=1).backend is (
        Backend.EAGER
    )


@requires_cpu_host
@pytest.mark.parametrize("backend", [Backend.TRITON, Backend.CUDA])
def test_msda_device_backends_are_unavailable_with_a_reason(backend) -> None:
    assert not msda.is_registered(backend)
    reason = msda.unavailable_reason(backend)
    assert reason != "it is not registered on this host"
    with pytest.raises(NoBackendError, match="unavailable"):
        msda(*_msda_operands(), backend=backend)
