"""Direct unit tests for :mod:`cds.math_utils.special` kernels.

These complement the indirect coverage the stats tests provide by hitting
edge paths (like betacf's zero initial denominator) with exact inputs.
"""

from __future__ import annotations

import math

import pytest

from cds.math_utils.special import betacf, betai, gammln, gammp, gammq, gser


def test_gammln_known_values() -> None:
    assert gammln(1.0) == pytest.approx(0.0)
    assert gammln(5.0) == pytest.approx(math.log(24.0))


def test_gammp_gammq_sum_to_one() -> None:
    for a, x in [(2.0, 1.5), (0.5, 0.3), (5.0, 4.0)]:
        assert gammp(a, x) + gammq(a, x) == pytest.approx(1.0)


def test_gser_returns_zero_for_nonpositive_x() -> None:
    assert gser(2.0, 0.0) == 0.0
    assert gser(2.0, -1.0) == 0.0


def test_betacf_zero_initial_denominator_guard() -> None:
    # a=b=x=1 makes the initial d exactly 1 - qab*x/qap = 0, forcing the
    # FPMIN guard on the very first line of the continued fraction.
    result = betacf(1.0, 1.0, 1.0)
    assert math.isfinite(result)


def test_betai_midpoint_is_one_half() -> None:
    assert betai(0.5, 0.5, 0.5) == pytest.approx(0.5)


def test_betai_rejects_out_of_domain() -> None:
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        betai(2.0, 3.0, -0.1)
