"""Tests for :mod:`cds.stats.nonparametric` (Mann–Whitney U + Wilcoxon)."""

from __future__ import annotations

import pytest

from cds.stats import mann_whitney_u, wilcoxon_signed_rank


def test_mann_whitney_separated_samples_give_tiny_p() -> None:
    a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    b = [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
    res = mann_whitney_u(a, b)
    assert res.statistic == pytest.approx(0.0)  # U = 0: complete separation
    assert res.p_value < 0.001
    assert res.z < -3.0


def test_mann_whitney_identical_distributions_not_significant() -> None:
    a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    b = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
    res = mann_whitney_u(a, b)
    assert res.p_value > 0.05
    assert abs(res.z) < 2.0


def test_mann_whitney_tie_correction_path() -> None:
    # Heavy ties exercise the Σ(t³−t) variance correction.
    a = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]
    b = [2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0]
    res = mann_whitney_u(a, b)
    assert 0.0 < res.p_value <= 1.0


def test_mann_whitney_direction_of_effect() -> None:
    a = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]
    b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    res = mann_whitney_u(a, b)
    # U is reported as min(U1, U2), so complete separation in either
    # direction drives the statistic to 0 and z far below the null.
    assert res.statistic == pytest.approx(0.0)
    assert res.p_value < 0.001


def test_mann_whitney_empty_sample_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        mann_whitney_u([], [1.0])
    with pytest.raises(ValueError, match="non-empty"):
        mann_whitney_u([1.0], [])


def test_wilcoxon_positive_shift_significant() -> None:
    diffs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    res = wilcoxon_signed_rank(diffs)
    assert res.statistic == pytest.approx(36.0)  # all positive → max W+
    # n=8 with the normal approximation: two-sided p ≈ 0.0117.
    assert res.p_value < 0.05
    assert res.n_effective == 8


def test_wilcoxon_symmetric_noise_not_significant() -> None:
    diffs = [-3.0, 1.0, -1.0, 3.0, 2.0, -2.0, 4.0, -4.0]
    res = wilcoxon_signed_rank(diffs)
    assert res.p_value > 0.05


def test_wilcoxon_zero_differences_dropped() -> None:
    diffs = [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    res = wilcoxon_signed_rank(diffs)
    assert res.n_effective == 8
    assert res.p_value < 0.05


def test_wilcoxon_ties_in_absolute_differences() -> None:
    diffs = [1.5, -1.5, 2.0, -2.0, 3.0, 3.0, -3.0, 4.0]
    res = wilcoxon_signed_rank(diffs)
    assert 0.0 < res.p_value <= 1.0


def test_wilcoxon_all_zero_raises() -> None:
    with pytest.raises(ValueError, match="all differences are zero"):
        wilcoxon_signed_rank([0.0, 0.0, 0.0])


def test_wilcoxon_empty_input_raises() -> None:
    with pytest.raises(ValueError, match="nothing to test"):
        wilcoxon_signed_rank([])
