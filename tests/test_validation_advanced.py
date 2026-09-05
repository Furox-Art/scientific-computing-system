"""Tests for residual, drift, OOD, and grouped-leakage validation."""

from __future__ import annotations

import math

import pytest

from cds.validation import (
    CheckStatus,
    check_distribution_drift,
    check_group_leakage,
    check_ood_ranges,
    check_residual_diagnostics,
)


def test_group_leakage_detects_cross_split_entities() -> None:
    clean = check_group_leakage(["a", "b"], ["c", "d"])
    assert clean.status is CheckStatus.PASS
    leaked = check_group_leakage(["a", "b", "c"], ["c", "d"])
    assert leaked.status is CheckStatus.FAIL
    assert "'c'" in leaked.details["overlap"]


def test_ood_range_check_passes_and_fails_interpretable_bounds() -> None:
    training = [[0.0, 10.0], [1.0, 12.0], [2.0, 14.0]]
    inside = check_ood_ranges(training, [[0.5, 11.0], [1.5, 13.0]])
    assert inside.status is CheckStatus.PASS
    outside = check_ood_ranges(training, [[3.0, 13.0], [1.0, 20.0]])
    assert outside.status is CheckStatus.FAIL
    assert outside.details["outside_indices"] == [0, 1]


def test_ood_range_margin_and_allowed_fraction() -> None:
    training = [[0.0], [10.0]]
    check = check_ood_ranges(
        training,
        [[10.5], [30.0]],
        margin_fraction=0.1,
        allowed_fraction=0.5,
    )
    assert check.status is CheckStatus.PASS
    assert check.details["outside_fraction"] == pytest.approx(0.5)


def test_ood_range_validation_errors_and_nonfinite() -> None:
    with pytest.raises(ValueError, match="allowed_fraction"):
        check_ood_ranges([[0.0]], [[1.0]], allowed_fraction=1.1)
    with pytest.raises(ValueError, match="margin_fraction"):
        check_ood_ranges([[0.0]], [[1.0]], margin_fraction=-0.1)
    with pytest.raises(ValueError, match="training_rows"):
        check_ood_ranges([], [[1.0]])
    with pytest.raises(ValueError, match="at least one feature"):
        check_ood_ranges([[]], [[]])
    with pytest.raises(ValueError, match="same feature count"):
        check_ood_ranges([[0.0, 1.0]], [[1.0]])
    assert check_ood_ranges([[0.0]], [[math.inf]]).status is CheckStatus.FAIL


def test_distribution_drift_detects_mean_or_variance_shift() -> None:
    reference = [-1.0, -0.5, 0.0, 0.5, 1.0]
    stable = check_distribution_drift(reference, [-0.9, -0.4, 0.1, 0.4, 0.9])
    assert stable.status is CheckStatus.PASS
    shifted = check_distribution_drift(reference, [4.0, 4.5, 5.0, 5.5, 6.0])
    assert shifted.status is CheckStatus.WARNING
    spread = check_distribution_drift(reference, [-10.0, -5.0, 0.0, 5.0, 10.0])
    assert spread.status is CheckStatus.WARNING


def test_distribution_drift_handles_constant_and_invalid_inputs() -> None:
    assert check_distribution_drift([1.0, 1.0], [1.0, 1.0]).status is CheckStatus.PASS
    assert check_distribution_drift([1.0, 1.0], [1.0, 2.0]).status is CheckStatus.WARNING
    assert check_distribution_drift([1.0, math.nan], [1.0, 2.0]).status is CheckStatus.FAIL
    with pytest.raises(ValueError, match="at least two"):
        check_distribution_drift([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="mean_shift"):
        check_distribution_drift([1.0, 2.0], [1.0, 2.0], max_standardized_mean_shift=-1)
    with pytest.raises(ValueError, match="variance_ratio"):
        check_distribution_drift([1.0, 2.0], [1.0, 2.0], max_variance_ratio=0.5)


def test_residual_diagnostics_flags_bias_dependence_and_scale_change() -> None:
    clean = check_residual_diagnostics([-1.0, 0.5, 1.0, -0.5, 0.5, -1.0, -0.5, 1.0])
    assert clean.status is CheckStatus.PASS

    biased = check_residual_diagnostics([2.0, 2.1, 1.9, 2.0, 2.1, 1.9])
    assert biased.status is CheckStatus.WARNING

    dependent = check_residual_diagnostics([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    assert dependent.status is CheckStatus.WARNING

    hetero = check_residual_diagnostics([0.1, -0.1, 0.1, -4.0, 4.0, -4.0, 4.0, -4.0])
    assert hetero.status is CheckStatus.WARNING


def test_residual_diagnostics_small_nonfinite_and_threshold_errors() -> None:
    assert check_residual_diagnostics([0.0, 0.0, 0.0]).status is CheckStatus.WARNING
    assert check_residual_diagnostics([0.0, math.inf, 0.0, 0.0]).status is CheckStatus.FAIL
    with pytest.raises(ValueError, match="thresholds"):
        check_residual_diagnostics([0.0] * 4, max_abs_lag1=-1)


def test_residual_diagnostics_zero_variance_is_well_defined() -> None:
    check = check_residual_diagnostics([0.0, 0.0, 0.0, 0.0])
    assert check.status is CheckStatus.PASS
    assert check.details["bias_z"] == 0.0
    assert check.details["variance_ratio"] == 1.0
