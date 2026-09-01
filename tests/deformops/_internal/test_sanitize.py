"""Direct unit tests for ``core.sanitize``, runnable without CUDA."""

import pytest
from deformops.core.sanitize import (
    check_2power,
    check_divisible,
    check_msda_params,
    check_positive,
)


def test_check_2power_accepts_powers_of_two():
    for n in (1, 2, 4, 8, 16, 32, 64, 128):
        check_2power(n)


def test_check_2power_rejects_non_power_of_two():
    with pytest.raises(ValueError, match="power of 2"):
        check_2power(12)


def test_check_2power_rejects_zero():
    with pytest.raises(ValueError, match="positive integer"):
        check_2power(0)


def test_check_2power_rejects_negative():
    with pytest.raises(ValueError, match="positive integer"):
        check_2power(-4)


def test_check_divisible_passes():
    check_divisible(12, 4)
    check_divisible(0, 1)


def test_check_divisible_raises():
    with pytest.raises(ValueError, match="divisible"):
        check_divisible(7, 2)


def test_check_divisible_rejects_zero_divisor():
    with pytest.raises(ValueError, match="non-zero divisor"):
        check_divisible(5, 0)


def test_check_positive_passes():
    check_positive(1, "x")


def test_check_positive_rejects_zero():
    with pytest.raises(ValueError, match="positive integer"):
        check_positive(0, "x")


def test_check_positive_rejects_negative():
    with pytest.raises(ValueError, match="positive integer"):
        check_positive(-1, "x")


def test_check_msda_params_passes():
    check_msda_params(dim=16, num_heads=2, num_levels=1, num_points=1)


def test_check_msda_params_rejects_zero_dim():
    with pytest.raises(ValueError, match="positive integer"):
        check_msda_params(dim=0, num_heads=1, num_levels=1, num_points=1)


def test_check_msda_params_rejects_non_divisible_dim():
    with pytest.raises(ValueError, match="divisible"):
        check_msda_params(dim=15, num_heads=4, num_levels=1, num_points=1)
