"""Validation must not certify overflowed scientific diagnostics as PASS."""

from __future__ import annotations

import math

from cds.validation import CheckStatus, check_distribution_drift, check_residual_diagnostics


def test_drift_rejects_overflowed_finite_means() -> None:
    check = check_distribution_drift([1e308, 1e308], [-1e308, -1e308])
    assert check.status is CheckStatus.FAIL
    assert "numerical overflow" in check.message
    assert "standardized_mean_shift" not in check.details


def test_drift_rejects_overflowed_finite_variances() -> None:
    check = check_distribution_drift([1e308, -1e308], [1e308, -1e308])
    assert check.status is CheckStatus.FAIL
    assert "numerical overflow" in check.message


def test_drift_keeps_well_defined_zero_variance_mismatch_as_warning() -> None:
    check = check_distribution_drift([1.0, 1.0], [1.0, 2.0])
    assert check.status is CheckStatus.WARNING
    assert check.details["variance_ratio"] == math.inf


def test_drift_keeps_identical_finite_sample_as_pass() -> None:
    check = check_distribution_drift([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert check.status is CheckStatus.PASS


def test_residuals_reject_overflowed_finite_means() -> None:
    check = check_residual_diagnostics([1e308] * 4)
    assert check.status is CheckStatus.FAIL
    assert "numerical overflow" in check.message


def test_residuals_reject_overflowed_finite_squares_without_raising() -> None:
    check = check_residual_diagnostics([1e308, -1e308, 1e308, -1e308])
    assert check.status is CheckStatus.FAIL
    assert "numerical overflow" in check.message


def test_residuals_reject_overflowed_variances() -> None:
    check = check_residual_diagnostics([1e154, -1e154, 1e154, -1e154])
    assert check.status is CheckStatus.FAIL


def test_residuals_keep_ordinary_centered_sample_as_pass() -> None:
    check = check_residual_diagnostics([-1.0, 0.5, 1.0, -0.5, 0.5, -1.0, -0.5, 1.0])
    assert check.status is CheckStatus.PASS


def test_nonfinite_raw_samples_still_fail() -> None:
    assert check_distribution_drift([0.0, float("inf")], [1.0, 2.0]).status is CheckStatus.FAIL
    assert check_residual_diagnostics([0.0, float("nan"), 0.0, 0.0]).status is CheckStatus.FAIL
