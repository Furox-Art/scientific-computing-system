"""Never turn finite user tolerances into false-positive scientific passes."""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from cds.validation.checks import check_conservation, check_ood_ranges, cross_method_agreement
from cds.validation.report import CheckStatus, ValidationCheck


@pytest.mark.parametrize(
    ("check", "message"),
    [
        (cross_method_agreement, "derived cross-method tolerance"),
        (check_conservation, "derived conservation tolerance"),
    ],
)
def test_finite_user_tolerance_must_not_overflow_to_false_pass(
    check: Callable[..., ValidationCheck], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        check(1e200, -1e200, rtol=1e308)


@pytest.mark.parametrize("margin_fraction", [1e308, 9e307])
def test_ood_finite_margin_overflow_never_masks_extreme_outliers(
    margin_fraction: float,
) -> None:
    with pytest.raises(ValueError, match="derived OOD margins"):
        check_ood_ranges([[0.0], [2.0]], [[1e300]], margin_fraction=margin_fraction)


def test_regular_tolerances_still_have_expected_outcomes() -> None:
    assert cross_method_agreement(1.0, 1.0 + 1e-10).status is CheckStatus.PASS
    assert cross_method_agreement(1.0, 2.0).status is CheckStatus.FAIL
    assert check_conservation(3.0, 3.0).status is CheckStatus.PASS
    assert check_conservation(3.0, 4.0).status is CheckStatus.FAIL
    assert check_ood_ranges([[0.0], [2.0]], [[100.0]]).status is CheckStatus.FAIL


def test_extreme_finite_outlier_is_not_certified_by_a_normal_margin() -> None:
    result = check_ood_ranges([[0.0], [2.0]], [[1e300]], margin_fraction=0.1)
    assert result.status is CheckStatus.FAIL


def test_positive_infinite_margin_is_already_rejected() -> None:
    with pytest.raises(ValueError, match="margin_fraction"):
        check_ood_ranges([[0.0], [2.0]], [[1e300]], margin_fraction=math.inf)
