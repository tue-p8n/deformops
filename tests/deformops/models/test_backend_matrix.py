"""Every provided model runs on every backend and matches eager.

Compares outputs and the gradients of every deformable attention parameter.
"""

from __future__ import annotations

import copy

import pytest
import torch
from deformops.registry import Backend

from ._tiny import MODELS

_GPU = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]


_BACKENDS = [
    pytest.param("cpu", Backend.EAGER, id="eager-cpu"),
    pytest.param("cuda", Backend.EAGER, id="eager-cuda", marks=_GPU),
    pytest.param("cuda", Backend.TRITON, id="triton", marks=_GPU),
    pytest.param("cuda", Backend.CUDA, id="cuda", marks=_GPU),
    pytest.param("cuda", None, id="auto-cuda", marks=_GPU),
]


def _attention_parameters(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if any(key in name for key in ("sampling_offsets", "attention_weights"))
    }


def _run(name: str, device: str, backend: Backend | None, net: torch.nn.Module):
    spec = MODELS[name]
    model = spec.wrapper(copy.deepcopy(net).to(device), backend=backend).eval()
    size, outputs_of = spec.image_size, spec.outputs_of
    generator = torch.Generator().manual_seed(1)
    image = torch.randn(2, 3, size, size, generator=generator).to(device)
    outputs = outputs_of(model(image))
    loss = sum(
        (
            output.float()
            * torch.linspace(-1, 1, output.numel(), device=device).view_as(output)
        ).sum()
        for output in outputs
    )
    loss.backward()
    grads = {
        key: parameter.grad.detach().float().cpu()
        for key, parameter in _attention_parameters(model).items()
    }
    return [output.detach().float().cpu() for output in outputs], grads


@pytest.mark.parametrize(("device", "backend"), _BACKENDS)
@pytest.mark.parametrize("name", sorted(MODELS))
def test_model_matches_eager_on_every_backend(
    name: str, device: str, backend: Backend | None
) -> None:
    torch.manual_seed(0)
    net = MODELS[name].network()
    expected_outputs, expected_grads = _run(name, "cpu", Backend.EAGER, net)
    outputs, grads = _run(name, device, backend, net)

    assert expected_grads, "the model has no deformable attention parameters"
    for got, want in zip(outputs, expected_outputs, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)
    assert grads.keys() == expected_grads.keys()
    for key, want in expected_grads.items():
        torch.testing.assert_close(grads[key], want, rtol=1e-3, atol=1e-4, msg=key)
