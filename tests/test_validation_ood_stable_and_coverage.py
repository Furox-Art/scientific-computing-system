"""Reference tests for extreme finite inputs and scientific CI branches."""

from __future__ import annotations

import math

import pytest

from cds.validation import (
    CheckStatus,
    check_distribution_drift,
    check_residual_diagnostics,
    screen_ood,
)


def test_constant_extreme_reference_does_not_certify_different_observation() -> None:
    report = screen_ood([[1e308], [1e308]], [[0.0], [1e308]])
    assert report.observations[0].out_of_distribution
    assert report.observations[0].max_abs_z == math.inf
    assert report.observations[1].max_abs_z == 0.0
    assert not report.observations[1].out_of_distribution


def test_large_balanced_reference_has_finite_ood_standard_deviation() -> None:
    report = screen_ood([[1e308], [-1e308]], [[1e308]], z_threshold=4.0)
    assert report.observations[0].max_abs_z == pytest.approx(1.0)
    assert not report.any_ood


def test_extreme_mixed_reference_refuses_unrepresentable_centered_values() -> None:
    reference = [[1e308]] * 9 + [[-1e308]]
    with pytest.raises(ValueError, match="reference deviations overflowed"):
        screen_ood(reference, [[0.0]])


def test_ordinary_multifeature_ood_detects_shift() -> None:
    report = screen_ood(
        [[-1.0, 4.0], [1.0, 4.0], [0.0, 4.0]],
        [[0.0, 4.0], [5.0, 4.0], [0.0, 5.0]],
        z_threshold=3.0,
    )
    assert [row.out_of_distribution for row in report.observations] == [False, True, True]


def test_drift_extreme_mean_shift_overflow_is_fail_not_pass() -> None:
    report = check_distribution_drift([1e307] * 2, [-1e307] * 2)
    assert report.status is CheckStatus.FAIL
    assert "overflow" in report.message


def test_drift_extreme_variance_ratio_overflow_is_fail_not_pass() -> None:
    report = check_distribution_drift([-1e-11, 1e-11], [-1e153, 1e153])
    assert report.status is CheckStatus.FAIL
    assert "overflow" in report.message


def test_residual_segment_variance_ratio_overflow_is_fail_not_pass() -> None:
    report = check_residual_diagnostics([1e-10, -1e-10, 1e150, -1e150])
    assert report.status is CheckStatus.FAIL
    assert "overflow" in report.message
