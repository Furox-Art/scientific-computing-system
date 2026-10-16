"""Non-finite tolerance settings must never certify invalid scientific results."""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from cds.validation.checks import (
    check_conservation,
    check_distribution_drift,
    check_ood_ranges,
    check_residual_diagnostics,
    cross_method_agreement,
)


@pytest.mark.parametrize("threshold", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("parameter", ["rtol", "atol"])
@pytest.mark.parametrize("check", [cross_method_agreement, check_conservation])
def test_tolerances_reject_nonfinite_values(
    check: Callable[..., object], parameter: str, threshold: float
) -> None:
    with pytest.raises(ValueError, match="rtol and atol"):
        check(1.0, 2.0, **{parameter: threshold})


@pytest.mark.parametrize("margin", [math.nan, math.inf, -math.inf])
def test_ood_margin_must_be_finite(margin: float) -> None:
    with pytest.raises(ValueError, match="margin_fraction"):
        check_ood_ranges([[1.0], [2.0]], [[100.0]], margin_fraction=margin)


@pytest.mark.parametrize("threshold", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("parameter", ["max_standardized_mean_shift", "max_variance_ratio"])
def test_drift_thresholds_reject_nonfinite(parameter: str, threshold: float) -> None:
    with pytest.raises(ValueError, match=parameter):
        if parameter == "max_standardized_mean_shift":
            check_distribution_drift(
                [1.0, 2.0], [10.0, 11.0], max_standardized_mean_shift=threshold
            )
        else:
            check_distribution_drift([1.0, 2.0], [10.0, 11.0], max_variance_ratio=threshold)


@pytest.mark.parametrize("threshold", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("parameter", ["max_bias_z", "max_abs_lag1", "max_variance_ratio"])
def test_residual_thresholds_reject_nonfinite(parameter: str, threshold: float) -> None:
    with pytest.raises(ValueError, match="residual diagnostic thresholds"):
        residuals = [1.0, -1.0, 2.0, -2.0]
        if parameter == "max_bias_z":
            check_residual_diagnostics(residuals, max_bias_z=threshold)
        elif parameter == "max_abs_lag1":
            check_residual_diagnostics(residuals, max_abs_lag1=threshold)
        else:
            check_residual_diagnostics(residuals, max_variance_ratio=threshold)
