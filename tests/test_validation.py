"""Tests for scientific validation and final audit helpers."""

from __future__ import annotations

import math

import pytest

from cds.validation import (
    CheckStatus,
    ValidationCheck,
    ValidationReport,
    check_bounds,
    check_duplicate_rows,
    check_finite,
    check_numerical_stability,
    cross_method_agreement,
    final_audit,
)


def test_check_finite_pass_and_fail() -> None:
    passed = check_finite([1.0, 2.0])
    assert passed.status is CheckStatus.PASS
    assert passed.details["count"] == 2

    failed = check_finite([1.0, math.inf, math.nan])
    assert failed.status is CheckStatus.FAIL
    assert failed.details["bad_indices"] == [1, 2]


def test_cross_method_agreement_validation_and_outcomes() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        cross_method_agreement(1.0, 1.0, rtol=-1.0)
    with pytest.raises(ValueError, match="non-negative"):
        cross_method_agreement(1.0, 1.0, atol=-1.0)

    bad = cross_method_agreement(math.inf, 1.0)
    assert bad.status is CheckStatus.FAIL

    good = cross_method_agreement(10.0, 10.0 + 1e-7)
    assert good.status is CheckStatus.PASS

    disagree = cross_method_agreement(1.0, 2.0, rtol=1e-9, atol=0.0)
    assert disagree.status is CheckStatus.FAIL
    assert disagree.details["delta"] == 1.0


def test_bounds_and_data_leakage_checks() -> None:
    with pytest.raises(ValueError, match="same length"):
        check_bounds([1.0], [(0.0, 2.0), (0.0, 3.0)])

    assert check_bounds([1.0, 2.0], [(0.0, 1.0), (1.0, 3.0)]).status is CheckStatus.PASS
    violated = check_bounds([2.0], [(0.0, 1.0)])
    assert violated.status is CheckStatus.FAIL
    assert violated.details["violations"] == [0]

    clean = check_duplicate_rows([[1, 2]], [[3, 4]])
    assert clean.status is CheckStatus.PASS

    leaked = check_duplicate_rows([[1, 2], ["a"]], [[1, 2], ["b"]])
    assert leaked.status is CheckStatus.FAIL
    assert leaked.details["test_indices"] == [0]


def test_numerical_stability_validation_and_outcomes() -> None:
    with pytest.raises(ValueError, match="relative_step"):
        check_numerical_stability(lambda x: x, 1.0, relative_step=0.0)
    with pytest.raises(ValueError, match="max_relative_change"):
        check_numerical_stability(lambda x: x, 1.0, max_relative_change=-1.0)

    stable = check_numerical_stability(lambda x: 2.0 * x, 1.0)
    assert stable.status is CheckStatus.PASS

    sensitive = check_numerical_stability(
        lambda x: 1e8 * x,
        0.0,
        relative_step=1e-7,
        max_relative_change=1e-6,
    )
    assert sensitive.status is CheckStatus.WARNING

    nonfinite = check_numerical_stability(lambda _x: math.inf, 1.0)
    assert nonfinite.status is CheckStatus.FAIL


def test_validation_report_and_final_audit() -> None:
    warning = ValidationCheck("w", CheckStatus.WARNING, "warning")
    failure = ValidationCheck("f", CheckStatus.FAIL, "failure")
    report = final_audit([warning])
    assert report.passed
    assert report.warnings == (warning,)
    assert report.failures == ()

    report.add(failure)
    assert not report.passed
    assert report.failures == (failure,)

    empty = ValidationReport()
    assert empty.passed
