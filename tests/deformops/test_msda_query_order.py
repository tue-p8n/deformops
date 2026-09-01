"""Space-filling-curve query reordering; only the kernel calls need CUDA."""

import pytest
import torch
from deformops.backends.cuda.experimental._query_order import (
    _gather_along_dim1,
    _hilbert_index,
    _morton_index,
    _query_sort_indices,
    _scatter_along_dim1,
    install_reordering_family,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_NEEDS_GPU = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available"
)


def _shapes(N: int, Q: int, G: int, L: int, P: int):
    locs = torch.rand(N, Q, G, L, P, 2, device=DEVICE)
    attn = torch.randn(N, Q, G, L * P, device=DEVICE)
    return locs, attn


def test_gather_matches_naive_per_batch_loop():
    torch.manual_seed(0)
    locs, attn = _shapes(N=3, Q=17, G=2, L=3, P=4)
    sort_indices = _query_sort_indices(locs)

    locs_ref = torch.stack([locs[i, sort_indices[i]] for i in range(locs.shape[0])])
    attn_ref = torch.stack([attn[i, sort_indices[i]] for i in range(attn.shape[0])])

    assert torch.equal(_gather_along_dim1(locs, sort_indices), locs_ref)
    assert torch.equal(_gather_along_dim1(attn, sort_indices), attn_ref)


def test_scatter_matches_naive_per_batch_loop():
    torch.manual_seed(0)
    locs, _ = _shapes(N=3, Q=17, G=2, L=3, P=4)
    sort_indices = _query_sort_indices(locs)
    result = torch.randn(3, 17, 8, device=DEVICE)

    final_ref = torch.empty_like(result)
    for i in range(result.shape[0]):
        final_ref[i, sort_indices[i]] = result[i]

    assert torch.equal(_scatter_along_dim1(result, sort_indices), final_ref)


def test_gather_then_scatter_round_trips():
    torch.manual_seed(0)
    locs, attn = _shapes(N=4, Q=23, G=1, L=4, P=4)
    sort_indices = _query_sort_indices(locs)

    gathered = _gather_along_dim1(attn, sort_indices)
    recovered = _scatter_along_dim1(gathered, sort_indices)

    assert torch.equal(recovered, attn)


def test_sort_indices_are_a_permutation_per_batch_row():
    torch.manual_seed(1)
    locs, _ = _shapes(N=5, Q=31, G=2, L=2, P=8)
    sort_indices = _query_sort_indices(locs)

    expected_row = torch.arange(31, device=DEVICE)
    for row in sort_indices:
        assert torch.equal(row.sort()[0], expected_row)


# Hilbert curve


def test_morton_and_hilbert_index_clamp_out_of_bounds_coordinates():
    # Locations routinely leave [0, 1]; unclamped, a negative one is a garbage index.
    x = torch.tensor([-0.5, -0.1, 0.0, 0.5, 1.0, 1.1, 1.5], device=DEVICE)
    y = torch.tensor([-0.3, 1.2, 0.5, 0.5, 0.5, -0.2, 2.0], device=DEVICE)
    x_clamped = x.clamp(0.0, 1.0)
    y_clamped = y.clamp(0.0, 1.0)

    morton = _morton_index(x, y)
    hilbert = _hilbert_index(x, y)

    assert (morton >= 0).all()
    assert (hilbert >= 0).all()
    # The function itself clamps: it matches pre-clamped input.
    assert torch.equal(morton, _morton_index(x_clamped, y_clamped))
    assert torch.equal(hilbert, _hilbert_index(x_clamped, y_clamped))


def test_hilbert_index_matches_known_order2_traversal():
    # Order-2 (4x4) Hilbert curve, per https://en.wikipedia.org/wiki/Hilbert_curve
    expected = {
        (0, 0): 0,
        (1, 0): 1,
        (1, 1): 2,
        (0, 1): 3,
        (0, 2): 4,
        (0, 3): 5,
        (1, 3): 6,
        (1, 2): 7,
        (2, 2): 8,
        (2, 3): 9,
        (3, 3): 10,
        (3, 2): 11,
        (3, 1): 12,
        (2, 1): 13,
        (2, 0): 14,
        (3, 0): 15,
    }
    bits = 2
    n = 1 << bits
    coords = [(ix, iy) for ix in range(n) for iy in range(n)]
    x = torch.tensor([c[0] for c in coords], dtype=torch.float32, device=DEVICE) / (
        n - 1
    )
    y = torch.tensor([c[1] for c in coords], dtype=torch.float32, device=DEVICE) / (
        n - 1
    )

    d = _hilbert_index(x, y, bits=bits)

    for coord, value in zip(coords, d.tolist(), strict=True):
        assert value == expected[coord]


def test_hilbert_index_is_a_bijection_over_the_full_grid():
    bits = 4
    n = 1 << bits
    coords = torch.cartesian_prod(
        torch.arange(n, device=DEVICE), torch.arange(n, device=DEVICE)
    ).float()
    x = coords[:, 0] / (n - 1)
    y = coords[:, 1] / (n - 1)

    d = _hilbert_index(x, y, bits=bits)

    assert torch.equal(d.sort()[0], torch.arange(n * n, device=DEVICE))


def test_hilbert_index_never_jumps_more_than_one_cell():
    # Unlike Morton order, consecutive indices are always adjacent cells.
    bits = 4
    n = 1 << bits
    coords = torch.cartesian_prod(
        torch.arange(n, device=DEVICE), torch.arange(n, device=DEVICE)
    ).float()
    x = coords[:, 0] / (n - 1)
    y = coords[:, 1] / (n - 1)

    d = _hilbert_index(x, y, bits=bits)
    order = d.argsort()
    path = coords[order]

    steps = (path[1:] - path[:-1]).abs()
    chebyshev = steps.max(dim=-1).values
    assert torch.equal(chebyshev, torch.ones_like(chebyshev))


def test_query_sort_indices_hilbert_curve_is_a_permutation():
    torch.manual_seed(2)
    locs, _ = _shapes(N=4, Q=29, G=1, L=3, P=4)
    sort_indices = _query_sort_indices(locs, curve="hilbert")

    expected_row = torch.arange(29, device=DEVICE)
    for row in sort_indices:
        assert torch.equal(row.sort()[0], expected_row)


def test_installing_a_family_rejects_an_unknown_curve():
    with pytest.raises(ValueError, match="curve"):
        install_reordering_family(
            curve="not_a_curve",  # type: ignore[arg-type]
            backend_name="cuda_nonsense",
            base_forward=lambda *a, **k: torch.empty(0),
            summary="",
        )


@_NEEDS_GPU
@pytest.mark.gpu
@pytest.mark.fp64
def test_hilbert_ordering_matches_the_unordered_output():
    # Reordering and scattering back leaves every query's result unchanged.
    from deformops.backends.cuda import msda_fp32acc
    from deformops.backends.cuda.experimental import msda_hilbert

    torch.manual_seed(3)
    N, Q, G, D, L, P = 2, 25, 2, 16, 3, 4
    shapes = torch.tensor([[4, 4], [3, 4], [2, 4]], device=DEVICE, dtype=torch.long)
    start_index = torch.tensor([0, 16, 28], device=DEVICE, dtype=torch.long)
    total_spatial = int((shapes[:, 0] * shapes[:, 1]).sum())
    values = torch.randn(N, total_spatial, G, D, device=DEVICE, dtype=torch.float64)
    locs = torch.rand(N, Q, G, L, P, 2, device=DEVICE, dtype=torch.float64)
    attn = torch.randn(N, Q, G, L * P, device=DEVICE, dtype=torch.float64)

    baseline = msda_fp32acc.forward(
        values, shapes, start_index, locs, attn, num_points=P
    )
    reordered = msda_hilbert.forward(
        values, shapes, start_index, locs, attn, num_points=P
    )

    torch.testing.assert_close(baseline, reordered, atol=1e-6, rtol=1e-6)
