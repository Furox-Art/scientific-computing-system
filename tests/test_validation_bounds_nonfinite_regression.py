"""Bounds validation must fail closed on non-finite data or invalid intervals."""

from __future__ import annotations

import math

from cds.validation import CheckStatus, check_bounds


def test_nonfinite_values_do_not_pass_domain_bounds() -> None:
    result = check_bounds([math.nan, math.inf, -math.inf, 1.0], [(0.0, 2.0)] * 4)
    assert result.status is CheckStatus.FAIL
    assert result.details["violations"] == [0, 1, 2]


def test_nonfinite_bound_and_reversed_interval_do_not_pass() -> None:
    result = check_bounds(
        [1.0, 1.0, 1.0, 1.0],
        [(math.nan, 2.0), (0.0, math.inf), (2.0, 0.0), (0.0, 2.0)],
    )
    assert result.status is CheckStatus.FAIL
    assert result.details["violations"] == [0, 1, 2]


def test_finite_boundary_values_remain_inclusive() -> None:
    result = check_bounds([0.0, 2.0], [(0.0, 2.0)] * 2)
    assert result.status is CheckStatus.PASS
