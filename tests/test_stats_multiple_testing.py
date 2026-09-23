from __future__ import annotations

import math

import pytest

from cds.stats import (
    MultipleTestingMethod,
    adjust_p_values,
    benjamini_hochberg,
    benjamini_yekutieli,
    bonferroni,
    holm,
)


def test_bonferroni_preserves_order_and_caps_adjusted_values() -> None:
    result = bonferroni([0.01, 0.2, 0.9], alpha=0.05)
    assert result.method is MultipleTestingMethod.BONFERRONI
    assert result.p_values == (0.01, 0.2, 0.9)
    assert result.adjusted_p_values == pytest.approx((0.03, 0.6, 1.0))
    assert result.rejected == (True, False, False)
    assert result.rejection_count == 1


def test_holm_matches_step_down_reference_and_handles_empty_input() -> None:
    result = holm([0.01, 0.04, 0.03, 0.8], alpha=0.05)
    assert result.adjusted_p_values == pytest.approx((0.04, 0.09, 0.09, 0.8))
    assert result.rejected == (True, False, False, False)

    empty = holm([], alpha=0.05)
    assert empty.p_values == ()
    assert empty.adjusted_p_values == ()
    assert empty.rejected == ()


def test_benjamini_hochberg_reference_values_and_tie_stability() -> None:
    result = benjamini_hochberg([0.01, 0.04, 0.03, 0.002, 0.04], alpha=0.05)
    assert result.method is MultipleTestingMethod.BENJAMINI_HOCHBERG
    assert result.adjusted_p_values == pytest.approx((0.025, 0.04, 0.04, 0.01, 0.04))
    assert result.rejected == (True, True, True, True, True)

    ties = benjamini_hochberg([0.02, 0.02, 0.5], alpha=0.05)
    assert ties.adjusted_p_values[0] == ties.adjusted_p_values[1]


def test_benjamini_yekutieli_is_at_least_as_conservative_as_bh() -> None:
    p_values = [0.001, 0.01, 0.04, 0.2]
    bh = benjamini_hochberg(p_values)
    by = benjamini_yekutieli(p_values)
    assert by.method is MultipleTestingMethod.BENJAMINI_YEKUTIELI
    assert all(
        by_value >= bh_value
        for by_value, bh_value in zip(by.adjusted_p_values, bh.adjusted_p_values, strict=True)
    )
    assert by.rejection_count <= bh.rejection_count

    empty = benjamini_yekutieli([])
    assert empty.adjusted_p_values == ()


def test_dispatch_accepts_enum_and_string_and_rejects_unknown_method() -> None:
    direct = adjust_p_values([0.01, 0.2], method=MultipleTestingMethod.HOLM)
    string = adjust_p_values([0.01, 0.2], method="holm")
    assert direct == string

    with pytest.raises(ValueError, match="unsupported multiple-testing method"):
        adjust_p_values([0.1], method="made-up")


def test_multiple_testing_input_validation() -> None:
    for alpha in (0.0, 1.0, -0.1, math.inf, math.nan):
        with pytest.raises(ValueError, match="alpha"):
            bonferroni([0.1], alpha=alpha)

    for bad in (-0.01, 1.01, math.inf, math.nan):
        with pytest.raises(ValueError, match="p-values"):
            benjamini_hochberg([0.1, bad])


def test_zero_p_values_are_valid_and_rejected() -> None:
    result = adjust_p_values([0.0, 1.0], method="benjamini-yekutieli")
    assert result.adjusted_p_values[0] == 0.0
    assert result.adjusted_p_values[1] == 1.0
    assert result.rejected == (True, False)
