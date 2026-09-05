"""Tests for bounded-memory Monte Carlo uncertainty propagation."""

from __future__ import annotations

import math

import pytest

from cds.uncertainty import propagate_monte_carlo, propagate_monte_carlo_streaming


def test_streaming_exact_reservoir_matches_exact_method() -> None:
    kwargs = {
        "standard_uncertainties": [0.2, 0.3],
        "samples": 500,
        "seed": 17,
        "confidence": 0.9,
    }
    exact = propagate_monte_carlo(lambda x, y: x + y, [1.0, 2.0], **kwargs)
    streaming = propagate_monte_carlo_streaming(
        lambda x, y: x + y,
        [1.0, 2.0],
        reservoir_size=500,
        **kwargs,
    )
    assert streaming.quantiles_exact
    assert streaming.mean == pytest.approx(exact.mean, rel=1e-12, abs=1e-12)
    assert streaming.standard_deviation == pytest.approx(
        exact.standard_deviation, rel=1e-12, abs=1e-12
    )
    assert streaming.lower == pytest.approx(exact.lower)
    assert streaming.upper == pytest.approx(exact.upper)


def test_streaming_large_run_is_bounded_and_reproducible() -> None:
    first = propagate_monte_carlo_streaming(
        lambda x: x * x,
        [2.0],
        standard_uncertainties=[0.4],
        samples=20_000,
        reservoir_size=257,
        seed=91,
    )
    second = propagate_monte_carlo_streaming(
        lambda x: x * x,
        [2.0],
        standard_uncertainties=[0.4],
        samples=20_000,
        reservoir_size=257,
        seed=91,
    )
    assert first == second
    assert first.reservoir_size == 257
    assert not first.quantiles_exact
    assert first.lower < first.mean < first.upper
    assert math.isfinite(first.standard_deviation)


def test_streaming_validation_and_nonfinite_function() -> None:
    with pytest.raises(ValueError, match="reservoir_size"):
        propagate_monte_carlo_streaming(lambda x: x, [1.0], samples=2, reservoir_size=1)
    with pytest.raises(ValueError, match="at least 2"):
        propagate_monte_carlo_streaming(lambda x: x, [1.0], samples=1)
    with pytest.raises(ValueError, match="non-finite"):
        propagate_monte_carlo_streaming(lambda _x: math.nan, [1.0], samples=2)


def test_streaming_correlated_covariance() -> None:
    result = propagate_monte_carlo_streaming(
        lambda x, y: x + y,
        [0.0, 0.0],
        covariance=[[1.0, 0.7], [0.7, 1.0]],
        samples=5_000,
        reservoir_size=100,
        seed=4,
    )
    assert result.standard_deviation > 1.5
