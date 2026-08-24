"""Tests for :mod:`cds.probability.extra_distributions`."""

from __future__ import annotations

import pytest

from cds.probability.extra_distributions import (
    geometric_cdf,
    geometric_pmf,
    hypergeometric_cdf,
    hypergeometric_pmf,
    negative_binomial_cdf,
    negative_binomial_pmf,
)

# --------------------------------------------------------------------- #
# Geometric                                                              #
# --------------------------------------------------------------------- #


def test_geometric_pmf_fair_coin_first_heads() -> None:
    assert geometric_pmf(1, 0.5) == pytest.approx(0.5)
    assert geometric_pmf(2, 0.5) == pytest.approx(0.25)
    assert geometric_pmf(3, 0.5) == pytest.approx(0.125)


def test_geometric_normalization() -> None:
    total = sum(geometric_pmf(k, 0.3) for k in range(1, 400))
    assert total == pytest.approx(1.0, abs=1e-12)


def test_geometric_edge_p_one() -> None:
    assert geometric_pmf(1, 1.0) == 1.0
    assert geometric_pmf(2, 1.0) == 0.0
    assert geometric_pmf(9, 1.0) == 0.0
    assert geometric_cdf(7, 1.0) == 1.0
    assert geometric_cdf(1, 1.0) == 1.0


def test_geometric_edge_p_zero() -> None:
    assert geometric_pmf(1, 0.0) == 0.0
    assert geometric_pmf(50, 0.0) == 0.0
    assert geometric_cdf(25, 0.0) == 0.0


def test_geometric_cdf_matches_closed_form() -> None:
    for k in (1, 2, 5, 17):
        assert geometric_cdf(k, 0.4) == pytest.approx(1.0 - 0.6**k)


def test_geometric_cdf_below_support_is_zero() -> None:
    assert geometric_cdf(0, 0.5) == 0.0
    assert geometric_cdf(-3, 0.5) == 0.0


def test_geometric_cdf_monotone() -> None:
    values = [geometric_cdf(k, 0.25) for k in range(1, 30)]
    assert all(a <= b for a, b in zip(values, values[1:]))


def test_geometric_validation() -> None:
    with pytest.raises(ValueError, match=r"p must be in \[0, 1\]"):
        geometric_pmf(1, -0.1)
    with pytest.raises(ValueError, match=r"p must be in \[0, 1\]"):
        geometric_pmf(1, 1.5)
    with pytest.raises(ValueError, match="k must be at least 1"):
        geometric_pmf(0, 0.5)
    with pytest.raises(ValueError, match="k must be at least 1"):
        geometric_pmf(-4, 0.5)
    with pytest.raises(ValueError, match=r"p must be in \[0, 1\]"):
        geometric_cdf(3, 2.0)


# --------------------------------------------------------------------- #
# Hypergeometric                                                         #
# --------------------------------------------------------------------- #


def test_hypergeometric_card_example() -> None:
    p_two_red = hypergeometric_pmf(2, 26, 26, 5)
    assert p_two_red == pytest.approx(845000 / 2598960)
    assert p_two_red == pytest.approx(0.3251301, abs=1e-6)
    assert hypergeometric_pmf(0, 26, 26, 5) == pytest.approx(65780 / 2598960)


def test_hypergeometric_normalization() -> None:
    lo, hi = max(0, 4 - 5), min(4, 7)
    total = sum(hypergeometric_pmf(k, 7, 5, 4) for k in range(lo, hi + 1))
    assert total == pytest.approx(1.0, abs=1e-12)


def test_hypergeometric_exhaustive_draw_is_degenerate() -> None:
    assert hypergeometric_pmf(3, 3, 2, 5) == pytest.approx(1.0)
    assert hypergeometric_cdf(3, 3, 2, 5) == pytest.approx(1.0)
    assert hypergeometric_cdf(10, 3, 2, 5) == pytest.approx(1.0)


def test_hypergeometric_forced_minimum_support_point() -> None:
    assert hypergeometric_pmf(2, 2, 1, 3) == pytest.approx(1.0)
    with pytest.raises(ValueError, match=r"k must be in \[2, 2\]"):
        hypergeometric_pmf(1, 2, 1, 3)


def test_hypergeometric_empty_population_zero_draws() -> None:
    assert hypergeometric_pmf(0, 0, 0, 0) == pytest.approx(1.0)


def test_hypergeometric_cdf_partial_sums() -> None:
    assert hypergeometric_cdf(0, 26, 26, 5) == pytest.approx(65780 / 2598960)
    partial = hypergeometric_pmf(0, 26, 26, 5) + hypergeometric_pmf(1, 26, 26, 5)
    assert hypergeometric_cdf(1, 26, 26, 5) == pytest.approx(partial)
    expected_2 = sum(hypergeometric_pmf(j, 7, 5, 4) for j in range(0, 3))
    assert hypergeometric_cdf(2, 7, 5, 4) == pytest.approx(expected_2)


def test_hypergeometric_cdf_clamps_outside_support() -> None:
    assert hypergeometric_cdf(1, 5, 2, 4) == 0.0
    assert hypergeometric_cdf(4, 5, 2, 4) == pytest.approx(1.0)
    assert hypergeometric_cdf(99, 5, 2, 4) == pytest.approx(1.0)


def test_hypergeometric_cdf_monotone() -> None:
    values = [hypergeometric_cdf(k, 8, 3, 6) for k in range(0, 9)]
    assert all(a <= b for a, b in zip(values, values[1:]))


def test_hypergeometric_validation() -> None:
    with pytest.raises(ValueError, match="population_successes must be non-negative"):
        hypergeometric_pmf(0, -1, 2, 1)
    with pytest.raises(ValueError, match="population_failures must be non-negative"):
        hypergeometric_pmf(0, 2, -1, 1)
    with pytest.raises(ValueError, match="draws must be non-negative"):
        hypergeometric_pmf(0, 2, 2, -1)
    with pytest.raises(ValueError, match="draws must not exceed the population size"):
        hypergeometric_pmf(0, 2, 2, 5)
    with pytest.raises(ValueError, match=r"k must be in \[0, 1\]"):
        hypergeometric_pmf(3, 2, 2, 1)
    with pytest.raises(ValueError, match=r"k must be in \[0, 1\]"):
        hypergeometric_pmf(-1, 2, 2, 1)
    with pytest.raises(ValueError, match="draws must not exceed the population size"):
        hypergeometric_cdf(0, 1, 1, 3)
    with pytest.raises(ValueError, match="population_successes must be non-negative"):
        hypergeometric_cdf(0, -5, 2, 1)


# --------------------------------------------------------------------- #
# Negative binomial                                                      #
# --------------------------------------------------------------------- #


def test_negative_binomial_hand_values() -> None:
    assert negative_binomial_pmf(0, 2, 0.5) == pytest.approx(0.25)
    assert negative_binomial_pmf(1, 2, 0.5) == pytest.approx(0.25)
    assert negative_binomial_pmf(2, 2, 0.5) == pytest.approx(0.1875)


def test_negative_binomial_r_one_reduces_to_first_success_law() -> None:
    for k in range(6):
        assert negative_binomial_pmf(k, 1, 0.3) == pytest.approx((0.7**k) * 0.3)


def test_negative_binomial_normalization() -> None:
    total = sum(negative_binomial_pmf(k, 3, 0.25) for k in range(400))
    assert total == pytest.approx(1.0, abs=1e-12)


def test_negative_binomial_edge_p_one() -> None:
    assert negative_binomial_pmf(0, 4, 1.0) == 1.0
    assert negative_binomial_pmf(1, 4, 1.0) == 0.0
    assert negative_binomial_cdf(0, 4, 1.0) == 1.0
    assert negative_binomial_cdf(5, 4, 1.0) == 1.0


def test_negative_binomial_edge_p_zero() -> None:
    assert negative_binomial_pmf(0, 2, 0.0) == 0.0
    assert negative_binomial_pmf(30, 2, 0.0) == 0.0
    assert negative_binomial_cdf(30, 2, 0.0) == 0.0


def test_negative_binomial_cdf_partial_sums() -> None:
    assert negative_binomial_cdf(-1, 2, 0.5) == 0.0
    assert negative_binomial_cdf(1, 2, 0.5) == pytest.approx(0.5)
    expected = sum(negative_binomial_pmf(j, 2, 0.5) for j in range(0, 5))
    assert negative_binomial_cdf(4, 2, 0.5) == pytest.approx(expected)


def test_negative_binomial_cdf_monotone() -> None:
    values = [negative_binomial_cdf(k, 3, 0.4) for k in range(0, 25)]
    assert all(a <= b for a, b in zip(values, values[1:]))


def test_negative_binomial_validation() -> None:
    with pytest.raises(ValueError, match=r"p must be in \[0, 1\]"):
        negative_binomial_pmf(0, 2, -0.5)
    with pytest.raises(ValueError, match=r"p must be in \[0, 1\]"):
        negative_binomial_pmf(0, 2, 1.5)
    with pytest.raises(ValueError, match="r must be a positive integer"):
        negative_binomial_pmf(0, 0, 0.5)
    with pytest.raises(ValueError, match="r must be a positive integer"):
        negative_binomial_pmf(0, -2, 0.5)
    with pytest.raises(ValueError, match="k must be non-negative"):
        negative_binomial_pmf(-1, 2, 0.5)
    with pytest.raises(ValueError, match=r"p must be in \[0, 1\]"):
        negative_binomial_cdf(2, 2, 1.5)
    with pytest.raises(ValueError, match="r must be a positive integer"):
        negative_binomial_cdf(2, -3, 0.5)
