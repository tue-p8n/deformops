import pytest
from deformops.core.tuning import find_factors
from hypothesis import given
from hypothesis import strategies as st


def test_find_factors_simple():
    """Verify factors for simple known numbers."""
    assert find_factors(1) == [1]
    assert find_factors(2) == [1, 2]
    assert find_factors(3) == [1, 3]
    assert find_factors(4) == [1, 2, 4]
    assert find_factors(12) == [1, 2, 3, 4, 6, 12]
    assert find_factors(16) == [1, 2, 4, 8, 16]


def test_find_factors_error():
    """Verify error handling for invalid inputs."""
    with pytest.raises(ValueError, match="Expected a positive integer"):
        find_factors(-1)


@given(n=st.integers(min_value=1, max_value=10000))
def test_find_factors_properties(n):
    """Property-based tests for finding factors."""
    factors = find_factors(n)

    # 1. 1 and n should always be factors
    assert 1 in factors
    assert n in factors

    # 2. All factors must divide n without remainder
    for f in factors:
        assert n % f == 0

    # 3. Product of any factor pair (f, n/f) must be n
    for f in factors:
        assert (n // f) in factors


@given(n=st.integers(min_value=1, max_value=100))
def test_find_factors_completeness(n):
    """Verify that every factor is found."""
    factors = set(find_factors(n))

    # Brute force check for small numbers
    expected = set()
    for i in range(1, n + 1):
        if n % i == 0:
            expected.add(i)

    assert factors == expected
