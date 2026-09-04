"""Tests for measurement-uncertainty propagation."""

from __future__ import annotations

import math

import pytest

from cds.uncertainty import UncertainValue, propagate_linear, propagate_monte_carlo
from cds.uncertainty.propagation import _cholesky, _quantile


def test_uncertain_value_validation() -> None:
    measurement = UncertainValue(2.0, 0.1)
    assert measurement.value == 2.0
    assert measurement.standard_uncertainty == 0.1

    with pytest.raises(ValueError, match="value must be finite"):
        UncertainValue(math.inf, 0.1)
    with pytest.raises(ValueError, match="standard_uncertainty"):
        UncertainValue(1.0, -0.1)
    with pytest.raises(ValueError, match="standard_uncertainty"):
        UncertainValue(1.0, math.nan)


def test_linear_propagation_independent_and_correlated_inputs() -> None:
    independent = propagate_linear(
        lambda x, y: x + 2.0 * y,
        [3.0, 4.0],
        standard_uncertainties=[0.1, 0.2],
    )
    assert independent.value == pytest.approx(11.0)
    assert independent.sensitivities == pytest.approx((1.0, 2.0), rel=1e-7)
    assert independent.variance == pytest.approx(0.17, rel=1e-6)
    assert independent.standard_uncertainty == pytest.approx(math.sqrt(0.17), rel=1e-6)

    correlated = propagate_linear(
        lambda x, y: x + y,
        [1.0, 2.0],
        covariance=[[0.04, 0.01], [0.01, 0.09]],
    )
    assert correlated.variance == pytest.approx(0.15, rel=1e-6)

    exact = propagate_linear(lambda x: x * 3.0, [2.0])
    assert exact.standard_uncertainty == 0.0


def test_linear_propagation_validation_paths() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        propagate_linear(lambda: 1.0, [])
    with pytest.raises(ValueError, match="finite values"):
        propagate_linear(lambda x: x, [math.inf])
    with pytest.raises(ValueError, match="either standard_uncertainties or covariance"):
        propagate_linear(
            lambda x: x,
            [1.0],
            standard_uncertainties=[0.1],
            covariance=[[0.01]],
        )
    with pytest.raises(ValueError, match="match the number of means"):
        propagate_linear(lambda x: x, [1.0], standard_uncertainties=[0.1, 0.2])
    with pytest.raises(ValueError, match="finite and non-negative"):
        propagate_linear(lambda x: x, [1.0], standard_uncertainties=[-0.1])
    with pytest.raises(ValueError, match="finite and non-negative"):
        propagate_linear(lambda x: x, [1.0], standard_uncertainties=[math.inf])
    with pytest.raises(ValueError, match="square matrix"):
        propagate_linear(lambda x: x, [1.0], covariance=[[1.0, 0.0]])
    with pytest.raises(ValueError, match="square matrix"):
        propagate_linear(lambda x, y: x + y, [1.0, 2.0], covariance=[[1.0], [0.0]])
    with pytest.raises(ValueError, match="finite values"):
        propagate_linear(lambda x: x, [1.0], covariance=[[math.nan]])
    with pytest.raises(ValueError, match="diagonal"):
        propagate_linear(lambda x: x, [1.0], covariance=[[-1.0]])
    with pytest.raises(ValueError, match="symmetric"):
        propagate_linear(lambda x, y: x + y, [1.0, 2.0], covariance=[[1.0, 0.2], [0.1, 1.0]])
    with pytest.raises(ValueError, match="relative_step"):
        propagate_linear(lambda x: x, [1.0], relative_step=0.0)
    with pytest.raises(ValueError, match="relative_step"):
        propagate_linear(lambda x: x, [1.0], relative_step=math.inf)
    with pytest.raises(ValueError, match="non-finite result"):
        propagate_linear(lambda _x: math.inf, [1.0])
    with pytest.raises(ValueError, match="non-finite result"):
        propagate_linear(lambda x: math.inf if x > 1.0 else x, [1.0])


def test_linear_propagation_rejects_indefinite_covariance() -> None:
    with pytest.raises(ValueError, match="negative propagated variance"):
        propagate_linear(
            lambda x, y: x - y,
            [1.0, 1.0],
            covariance=[[1.0, 2.0], [2.0, 1.0]],
        )


def test_cholesky_psd_paths() -> None:
    lower = _cholesky([[4.0, 2.0], [2.0, 2.0]])
    assert lower[0][0] == 2.0
    assert lower[1][0] == 1.0
    assert lower[1][1] == 1.0

    singular = _cholesky([[0.0, 0.0], [0.0, 1.0]])
    assert singular[0][0] == 0.0
    assert singular[1][0] == 0.0

    with pytest.raises(ValueError, match="positive semidefinite"):
        _cholesky([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(ValueError, match="positive semidefinite"):
        _cholesky([[0.0, 1.0], [1.0, 1.0]])


def test_quantile_integer_and_interpolated_positions() -> None:
    values = [0.0, 10.0, 20.0, 30.0, 40.0]
    assert _quantile(values, 0.25) == 10.0
    assert _quantile(values, 0.125) == 5.0


def test_monte_carlo_propagation_is_reproducible_and_correlated() -> None:
    first = propagate_monte_carlo(
        lambda x, y: x + y,
        [1.0, 2.0],
        standard_uncertainties=[0.2, 0.3],
        samples=2_000,
        seed=42,
        confidence=0.90,
    )
    second = propagate_monte_carlo(
        lambda x, y: x + y,
        [1.0, 2.0],
        standard_uncertainties=[0.2, 0.3],
        samples=2_000,
        seed=42,
        confidence=0.90,
    )
    assert first == second
    assert first.mean == pytest.approx(3.0, abs=0.04)
    assert first.standard_deviation == pytest.approx(math.sqrt(0.13), abs=0.03)
    assert first.lower < first.mean < first.upper
    assert first.samples == 2_000
    assert first.seed == 42
    assert first.confidence == 0.90

    correlated = propagate_monte_carlo(
        lambda x, y: x + y,
        [0.0, 0.0],
        covariance=[[1.0, 0.5], [0.5, 1.0]],
        samples=1_000,
        seed=None,
    )
    assert correlated.standard_deviation > 1.0


def test_monte_carlo_validation_paths() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        propagate_monte_carlo(lambda x: x, [1.0], samples=1)
    with pytest.raises(ValueError, match="confidence"):
        propagate_monte_carlo(lambda x: x, [1.0], confidence=0.0)
    with pytest.raises(ValueError, match="confidence"):
        propagate_monte_carlo(lambda x: x, [1.0], confidence=1.0)
    with pytest.raises(ValueError, match="positive semidefinite"):
        propagate_monte_carlo(
            lambda x, y: x + y,
            [0.0, 0.0],
            covariance=[[1.0, 2.0], [2.0, 1.0]],
            samples=2,
        )
    with pytest.raises(ValueError, match="non-finite result"):
        propagate_monte_carlo(lambda _x: math.nan, [0.0], samples=2)
