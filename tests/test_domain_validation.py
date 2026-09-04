"""Tests for generic scientific-domain validation checks."""

from __future__ import annotations

import math

import pytest

from cds.validation import CheckStatus, check_conservation, check_monotonic, check_positive


def test_check_positive_strict_and_allow_zero() -> None:
    passed = check_positive([1.0, 2.0, 3.0])
    assert passed.status is CheckStatus.PASS
    assert passed.details == {"allow_zero": False}

    strict = check_positive([1.0, 0.0, -1.0, math.inf])
    assert strict.status is CheckStatus.FAIL
    assert strict.details["violations"] == [1, 2, 3]

    zero_allowed = check_positive([0.0, 1.0], allow_zero=True)
    assert zero_allowed.status is CheckStatus.PASS

    negative = check_positive([0.0, -0.1], allow_zero=True)
    assert negative.status is CheckStatus.FAIL
    assert negative.details["violations"] == [1]


def test_check_conservation_pass_fail_nonfinite_and_validation() -> None:
    passed = check_conservation(10.0, 10.0 + 1e-11)
    assert passed.status is CheckStatus.PASS
    assert passed.message == "conservation law satisfied"

    failed = check_conservation(10.0, 11.0)
    assert failed.status is CheckStatus.FAIL
    assert failed.message == "conservation law violated"
    assert failed.details["delta"] == 1.0

    nonfinite = check_conservation(math.inf, 1.0)
    assert nonfinite.status is CheckStatus.FAIL
    assert "non-finite" in nonfinite.message

    with pytest.raises(ValueError, match="rtol and atol"):
        check_conservation(1.0, 1.0, rtol=-1.0)
    with pytest.raises(ValueError, match="rtol and atol"):
        check_conservation(1.0, 1.0, atol=-1.0)


def test_check_monotonic_all_directions_and_strictness() -> None:
    assert check_monotonic([1.0, 1.0, 2.0]).status is CheckStatus.PASS
    assert check_monotonic([1.0, 1.0, 2.0], strict=True).status is CheckStatus.FAIL
    assert check_monotonic([3.0, 2.0, 2.0], increasing=False).status is CheckStatus.PASS
    assert (
        check_monotonic([3.0, 2.0, 2.0], increasing=False, strict=True).status is CheckStatus.FAIL
    )
    assert check_monotonic([1.0, 3.0, 2.0]).details["violations"] == [1]

    decreasing_strict = check_monotonic([3.0, 2.0, 1.0], increasing=False, strict=True)
    assert decreasing_strict.status is CheckStatus.PASS

    nonfinite = check_monotonic([1.0, math.nan])
    assert nonfinite.status is CheckStatus.FAIL
    assert "non-finite" in nonfinite.message

    empty = check_monotonic([])
    assert empty.status is CheckStatus.PASS
