import torch
from deformops.registry import msda


def test_msda_cpu_fallback():
    N, M, D, L, P, Q = 1, 2, 4, 1, 4, 8
    shapes = torch.tensor([[4, 4]], dtype=torch.long)
    start_index = torch.tensor([0], dtype=torch.long)
    values = torch.randn((N, 16, M, D))
    locs = torch.rand((N, Q, M, L, P, 2))
    attn = torch.randn((N, Q, M, L, P)).softmax(-1)

    # This should trigger reference() fallback in forward()
    output = msda(values, shapes, start_index, locs, attn)
    assert output.device.type == "cpu"
    assert output.shape == (N, Q, M * D)


def test_msda_cpu_fallback_aggregate_softmax():
    """Softmax aggregation on CPU runs on the eager backend."""
    N, M, D, L, P, Q = 1, 2, 4, 1, 4, 8
    shapes = torch.tensor([[4, 4]], dtype=torch.long)
    start_index = torch.tensor([0], dtype=torch.long)
    values = torch.randn((N, 16, M, D))
    locs = torch.rand((N, Q, M, L, P, 2))
    attn_logits = torch.randn((N, Q, M, L, P))

    # The linear method with weights normalized beforehand.
    out_pre = msda(
        values, shapes, start_index, locs, attn_logits.softmax(-1), method="linear"
    )
    # The softmax method with raw logits; the eager backend normalizes them.
    out_agg = msda(values, shapes, start_index, locs, attn_logits, method="softmax")

    assert out_pre.shape == out_agg.shape == (N, Q, M * D)
    assert out_pre.device.type == "cpu"
    torch.testing.assert_close(out_pre, out_agg)
