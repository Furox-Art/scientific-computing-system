"""Tests for family-wise and false-discovery-rate p-value adjustments."""

from __future__ import annotations

import pytest

from cds.stats.multiple_testing import adjust_p_values, rejected


def test_empty_and_none_preserve_input_order() -> None:
    assert adjust_p_values(()) == ()
    assert adjust_p_values((0.2, 0.01), method="none") == (0.2, 0.01)


def test_bonferroni_caps_at_one() -> None:
    assert adjust_p_values((0.01, 0.2, 0.8), method="bonferroni") == pytest.approx(
        (0.03, 0.6, 1.0)
    )


def test_holm_matches_step_down_reference_and_restores_order() -> None:
    raw = (0.04, 0.01, 0.03, 0.20)
    adjusted = adjust_p_values(raw, method="holm")
    assert adjusted == pytest.approx((0.09, 0.04, 0.09, 0.20))


def test_benjamini_hochberg_matches_reference_and_is_monotone_in_rank() -> None:
    raw = (0.01, 0.04, 0.03, 0.002)
    adjusted = adjust_p_values(raw, method="fdr_bh")
    assert adjusted == pytest.approx((0.02, 0.04, 0.04, 0.008))


def test_rejected_uses_adjusted_p_values() -> None:
    assert rejected((0.01, 0.03, 0.2), alpha=0.05, method="bonferroni") == (
        True,
        False,
        False,
    )


@pytest.mark.parametrize("bad", [-0.01, 1.01, float("inf"), float("nan")])
def test_invalid_p_values_fail_closed(bad: float) -> None:
    with pytest.raises(ValueError, match="p-values"):
        adjust_p_values((0.1, bad))


def test_invalid_method_and_alpha_fail_closed() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        adjust_p_values((0.1,), method="invalid")  # type: ignore[arg-type]
    for alpha in (0.0, 1.0, -0.1, float("inf")):
        with pytest.raises(ValueError, match="alpha"):
            rejected((0.01,), alpha=alpha)
