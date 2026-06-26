import pytest

from snowcat_demo.model.workload import GemmWorkload, divisors


def test_divisors_returns_sorted_positive_divisors() -> None:
    assert divisors(12) == [1, 2, 3, 4, 6, 12]


def test_divisors_rejects_non_positive_input() -> None:
    with pytest.raises(ValueError):
        divisors(0)


def test_algorithmic_minimum_and_operations() -> None:
    workload = GemmWorkload(m=4, k=8, n=2, bytes_per_element=2)

    assert workload.operations == 128
    assert workload.algorithmic_minimum_bytes == (4 * 8 + 8 * 2 + 4 * 2) * 2


def test_tile_counts_rejects_non_divisor_tile() -> None:
    workload = GemmWorkload(m=8, k=8, n=8, bytes_per_element=1)

    with pytest.raises(ValueError):
        workload.tile_counts(3, 2, 2)

