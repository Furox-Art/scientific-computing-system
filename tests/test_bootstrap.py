"""Tests for :mod:`cds.stats.bootstrap` (percentile bootstrap CIs)."""

from __future__ import annotations

import pytest

from cds.stats.bootstrap import BootstrapResult, bootstrap_ci, bootstrap_diff_ci
from cds.stats.descriptive import median


def test_bootstrap_ci_fixed_seed_reproduces_exact_bounds() -> None:
    data = [float(v) for v in range(1, 21)]
    first = bootstrap_ci(data, n_resamples=200, seed=12345)
    second = bootstrap_ci(data, n_resamples=200, seed=12345)
    assert isinstance(first, BootstrapResult)
    assert first == second


def test_bootstrap_ci_different_seeds_give_different_bounds() -> None:
    data = [float(v) for v in range(1, 21)]
    one = bootstrap_ci(data, n_resamples=200, seed=1)
    two = bootstrap_ci(data, n_resamples=200, seed=2)
    assert (one.lower, one.upper) != (two.lower, two.upper)


def test_bootstrap_ci_balanced_binary_data_is_sane() -> None:
    data = [0.0] * 50 + [1.0] * 50
    res = bootstrap_ci(data, n_resamples=500, seed=7)
    assert res.estimate == pytest.approx(0.5)
    assert 0.3 < res.lower < res.upper < 0.7
    assert res.lower < 0.5 < res.upper
    assert res.se > 0.0
    assert res.n_resamples == 500
    assert res.confidence == pytest.approx(0.95)


def test_bootstrap_ci_median_statistic_on_binary_data() -> None:
    data = [0.0] * 50 + [1.0] * 50
    res = bootstrap_ci(data, lambda x: median(list(x)), n_resamples=300, seed=21)
    # Median of an even-size binary resample is exactly 0.0, 0.5 or 1.0.
    assert res.estimate == pytest.approx(0.5)
    assert {res.lower, res.upper} <= {0.0, 0.5, 1.0}
    assert res.lower <= res.upper


def test_custom_statistic_is_respected() -> None:
    data = [3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0]
    res = bootstrap_ci(data, lambda x: max(x), n_resamples=100, seed=11)
    assert res.estimate == pytest.approx(max(data))
    assert res.lower >= min(data)


def test_constant_data_gives_degenerate_zero_width_interval() -> None:
    data = [7.5] * 30
    res = bootstrap_ci(data, n_resamples=100, seed=3)
    assert res.estimate == pytest.approx(7.5)
    assert res.lower == pytest.approx(7.5)
    assert res.upper == pytest.approx(7.5)
    assert res.se == 0.0


def test_single_resample_collapses_bounds_and_se() -> None:
    data = [1.0, 2.0, 3.0]
    res = bootstrap_ci(data, n_resamples=1, seed=42)
    assert res.lower == res.upper
    assert res.se == 0.0


def test_quantile_hits_exact_order_statistic_positions() -> None:
    # confidence=0.5 → q in {0.25, 0.75}; n=41 → positions 10 and 30 exactly,
    # exercising the integer-position (lo == hi) quantile branch.
    data = [float(v) for v in range(1, 31)]
    res = bootstrap_ci(data, n_resamples=41, confidence=0.5, seed=5)
    assert res.confidence == pytest.approx(0.5)
    assert res.lower <= res.upper


@pytest.mark.parametrize("bad_confidence", [0.0, 1.0, -0.1, 1.5])
def test_invalid_confidence_raises(bad_confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence must be strictly between"):
        bootstrap_ci([1.0, 2.0], confidence=bad_confidence)
    with pytest.raises(ValueError, match="confidence must be strictly between"):
        bootstrap_diff_ci([1.0], [2.0], confidence=bad_confidence)


@pytest.mark.parametrize("bad_n", [0, -5])
def test_nonpositive_n_resamples_raises(bad_n: int) -> None:
    with pytest.raises(ValueError, match="n_resamples must be at least 1"):
        bootstrap_ci([1.0, 2.0], n_resamples=bad_n)
    with pytest.raises(ValueError, match="n_resamples must be at least 1"):
        bootstrap_diff_ci([1.0], [2.0], n_resamples=bad_n)


def test_empty_data_raises() -> None:
    with pytest.raises(ValueError, match="data must be non-empty"):
        bootstrap_ci([], n_resamples=10)


@pytest.mark.parametrize("empty_side", ["a", "b"])
def test_empty_group_in_diff_raises(empty_side: str) -> None:
    full = [1.0, 2.0, 3.0]
    a, b = ([], full) if empty_side == "a" else (full, [])
    with pytest.raises(ValueError, match="both samples must be non-empty"):
        bootstrap_diff_ci(a, b, n_resamples=10)


def test_bootstrap_diff_ci_fixed_seed_reproduces_exact_bounds() -> None:
    a = [2.0, 4.0, 6.0, 8.0, 10.0]
    b = [1.0, 3.0, 5.0, 7.0, 9.0]
    first = bootstrap_diff_ci(a, b, n_resamples=200, seed=777)
    second = bootstrap_diff_ci(a, b, n_resamples=200, seed=777)
    assert first == second


def test_bootstrap_diff_ci_shifted_groups_exclude_zero() -> None:
    a = [10.0, 12.0, 11.5, 13.0, 12.5, 10.5, 11.0, 12.0, 13.5, 11.0]
    b = [1.0, 2.5, 1.5, 2.0, 3.0, 1.0, 2.0, 1.5, 2.5, 3.0]
    res = bootstrap_diff_ci(a, b, n_resamples=300, seed=99)
    assert res.estimate == pytest.approx(sum(a) / len(a) - sum(b) / len(b))
    assert res.lower > 0.0
    assert res.lower <= res.upper


def test_bootstrap_diff_ci_identical_groups_include_zero() -> None:
    data = [float(v % 7) + 0.5 for v in range(24)]
    res = bootstrap_diff_ci(data, list(data), n_resamples=400, seed=13)
    assert res.estimate == pytest.approx(0.0)
    assert res.lower <= 0.0 <= res.upper


def test_bootstrap_diff_ci_custom_statistic_is_respected() -> None:
    a = [5.0, 6.0, 7.0, 8.0, 9.0]
    b = [1.0, 2.0, 3.0, 4.0, 5.0]
    res = bootstrap_diff_ci(a, b, lambda x: max(x), n_resamples=150, seed=8)
    assert res.estimate == pytest.approx(max(a) - max(b))
