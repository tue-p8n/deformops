"""The provided models compile and export on every backend and with ``backend=None``.

Each recipe is compared with the uncompiled model; see ``docs/export.md``.
"""

from __future__ import annotations

import copy
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import deformops
import pytest
import torch
from deformops.registry import Backend

from ._tiny import MODELS, flat_model

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]

#: ``None`` runs the public ``deformops::`` operators, which select a backend per
#: call; each member selects one explicitly.
_BACKENDS: list[Backend | None] = [None, Backend.EAGER, Backend.TRITON, Backend.CUDA]
_CLOSE = {"rtol": 1e-3, "atol": 1e-3}


def _backend_id(backend: Backend | None) -> str:
    return "selected" if backend is None else backend.value


def _setup(name: str, backend: Backend | None, batch: int = 2):
    torch.manual_seed(0)
    model = flat_model(name, backend, "cuda").eval()
    size = MODELS[name].image_size
    generator = torch.Generator().manual_seed(1)
    image = torch.randn(batch, 3, size, size, generator=generator).cuda()
    return model, image


def _assert_outputs_close(actual, expected) -> None:
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got.float(), want.float(), **_CLOSE)


@pytest.fixture(autouse=True)
def _fresh_dynamo():
    torch._dynamo.reset()  # noqa: SLF001
    yield
    torch._dynamo.reset()  # noqa: SLF001


@pytest.mark.parametrize("backend", _BACKENDS, ids=_backend_id)
@pytest.mark.parametrize("name", sorted(MODELS))
def test_compile_fullgraph_inference(name: str, backend: Backend | None) -> None:
    model, image = _setup(name, backend)
    with torch.no_grad():
        expected = model(image)
        actual = torch.compile(model, fullgraph=True)(image)
    _assert_outputs_close(actual, expected)


@pytest.mark.parametrize("backend", _BACKENDS, ids=_backend_id)
@pytest.mark.parametrize("name", sorted(MODELS))
def test_compile_fullgraph_backward(name: str, backend: Backend | None) -> None:
    """A compiled forward and backward give the uncompiled gradients.

    Eval mode disables dropout and the models' ``isfinite`` check, which break graphs.
    """
    model, image = _setup(name, backend)
    compiled_model = copy.deepcopy(model)
    compiled = torch.compile(compiled_model, fullgraph=True)

    losses = []
    for runner in (model, compiled):
        outputs = runner(image)
        loss = sum(output.float().square().mean() for output in outputs)
        loss.backward()
        losses.append(loss.detach())

    torch.testing.assert_close(losses[1], losses[0], **_CLOSE)
    for (key, eager), (_, traced) in zip(
        model.named_parameters(), compiled_model.named_parameters(), strict=True
    ):
        if eager.grad is None:
            assert traced.grad is None, key
            continue
        torch.testing.assert_close(
            traced.grad, eager.grad, rtol=1e-2, atol=1e-4, msg=key
        )


@pytest.mark.parametrize("backend", _BACKENDS, ids=_backend_id)
@pytest.mark.parametrize("name", sorted(MODELS))
def test_compile_training_breaks_only_outside_deformops(
    name: str, backend: Backend | None
) -> None:
    """In training mode, no graph break happens inside deformops.

    The upstream models break the graph on ``isfinite(...).all()`` regardless.
    """
    model, image = _setup(name, backend)
    model.train()
    explanation = torch._dynamo.explain(model)(image)  # noqa: SLF001
    package = Path(deformops.__file__).parent
    for reason in explanation.break_reasons:
        frame = Path(reason.user_stack[-1].filename)
        assert package not in frame.parents, reason


@pytest.mark.parametrize("backend", _BACKENDS, ids=_backend_id)
@pytest.mark.parametrize("name", sorted(MODELS))
def test_export_static(name: str, backend: Backend | None) -> None:
    model, image = _setup(name, backend)
    with torch.no_grad():
        expected = model(image)
        program = torch.export.export(model, (image,))
        actual = program.module()(image)
    _assert_outputs_close(actual, expected)


def _public_operators(program: torch.export.ExportedProgram) -> set[str]:
    """The public ``deformops::`` operators an exported graph calls."""
    return {
        node.target.name()
        for node in program.graph_module.graph.nodes
        if node.op == "call_function"
        and isinstance(node.target, torch._ops.OpOverload)  # noqa: SLF001
        and node.target.namespace == "deformops"
    }


@pytest.mark.parametrize("name", sorted(MODELS))
def test_export_keeps_the_public_operator_until_decomposed(name: str) -> None:
    """With ``backend=None``, the export holds the public operator.

    ``run_decompositions`` swaps in the private operator this host selects.
    """
    model, image = _setup(name, None)
    with torch.no_grad():
        expected = model(image)
        program = torch.export.export(model, (image,))
        assert _public_operators(program) & {
            "deformops::msda",
            "deformops::msda_softmax",
        }
        decomposed = program.run_decompositions()
        assert not _public_operators(decomposed)
        actual = decomposed.module()(image)
    _assert_outputs_close(actual, expected)


@pytest.mark.parametrize("backend", _BACKENDS, ids=_backend_id)
def test_export_dynamic_batch(backend: Backend | None) -> None:
    """Mask2Former exports with a dynamic batch size; Deformable DETR cannot.

    Its mask resize puts the batch in the channel position, which CUDA specializes on.
    """
    model, image = _setup("mask2former", backend)
    batch = torch.export.Dim("batch", min=1, max=16)
    with torch.no_grad():
        program = torch.export.export(
            model, (image,), dynamic_shapes={"pixel_values": {0: batch}}
        )
        other = torch.cat([image, image.flip(0), image])
        _assert_outputs_close(program.module()(other), model(other))


_LOADER = textwrap.dedent(
    """
    import sys

    import torch

    if sys.argv[3] == "cuda":
        from deformops.registry import Backend, msda

        msda.load(Backend.CUDA)
    package = torch._inductor.aoti_load_package(sys.argv[1])
    if sys.argv[3] != "cuda":
        assert not any(name.startswith("deformops") for name in sys.modules)
    saved = torch.load(sys.argv[2])
    outputs = package(saved["image"].cuda())
    for got, want in zip(outputs, saved["expected"], strict=True):
        torch.testing.assert_close(got.float().cpu(), want.float(), rtol=1e-3, atol=1e-3)
    print("loaded")
    """
)


@pytest.mark.parametrize("backend", _BACKENDS, ids=_backend_id)
@pytest.mark.parametrize("name", sorted(MODELS))
def test_aoti_package_loads_in_a_fresh_process(
    name: str, backend: Backend | None, tmp_path: Path
) -> None:
    """The package runs in a fresh process.

    Only a program calling the CUDA operators needs their library loaded first.
    """
    model, image = _setup(name, backend)
    with torch.no_grad():
        expected = model(image)
        program = torch.export.export(model, (image,))
    package = torch._inductor.aoti_compile_and_package(  # noqa: SLF001
        program, package_path=str(tmp_path / "model.pt2")
    )
    saved = tmp_path / "io.pt"
    torch.save({"image": image.cpu(), "expected": [t.cpu() for t in expected]}, saved)

    script = tmp_path / "load.py"
    script.write_text(_LOADER)
    environment = dict(os.environ)
    source_root = str(Path(deformops.__file__).parents[1])
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [source_root, environment.get("PYTHONPATH")])
    )
    result = subprocess.run(
        [sys.executable, str(script), package, str(saved), _backend_id(backend)],
        capture_output=True,
        text=True,
        env=environment,
        cwd=tmp_path,
        check=False,
        timeout=900,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "loaded" in result.stdout


@pytest.mark.parametrize("backend", _BACKENDS, ids=_backend_id)
@pytest.mark.parametrize("name", sorted(MODELS))
def test_torch_tensorrt(name: str, backend: Backend | None) -> None:
    torch_tensorrt = pytest.importorskip("torch_tensorrt")
    model, image = _setup(name, backend)
    with torch.no_grad():
        expected = model(image)
        program = torch.export.export(model, (image,))
        compiled = torch_tensorrt.dynamo.compile(
            program, inputs=[image], min_block_size=1
        )
        actual = compiled(image)
    _assert_outputs_close(actual, expected)
