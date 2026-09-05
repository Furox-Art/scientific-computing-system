"""Tests for dependency-free local and global sensitivity analysis."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from cds.sensitivity import (
    local_sensitivity,
    morris_screening,
    pairwise_interactions,
    sobol_indices,
)


def test_local_sensitivity_derivatives_normalization_and_ranking() -> None:
    def model(parameters: Sequence[float]) -> float:
        return 2.0 * parameters[0] + 5.0 * parameters[1]

    report = local_sensitivity(model, [3.0, 4.0])
    assert report.output == 26.0
    assert len(report.parameters) == 2
    assert math.isclose(report.parameters[0].derivative, 2.0, rel_tol=1e-9)
    assert math.isclose(report.parameters[1].derivative, 5.0, rel_tol=1e-9)
    assert math.isclose(report.parameters[0].normalized or 0.0, 6.0 / 26.0, rel_tol=1e-9)
    assert math.isclose(report.parameters[1].normalized or 0.0, 20.0 / 26.0, rel_tol=1e-9)
    assert report.most_influential() is report.parameters[1]


def test_zero_output_and_empty_parameter_report() -> None:
    zero = local_sensitivity(lambda values: values[0] - values[0], [2.0])
    assert zero.parameters[0].normalized is None
    assert zero.most_influential() is zero.parameters[0]
    empty = local_sensitivity(lambda _values: 7.0, [])
    assert empty.parameters == ()
    assert empty.most_influential() is None


def test_local_sensitivity_validation() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        local_sensitivity(lambda _values: 1.0, [1.0], relative_step=0.0)
    with pytest.raises(ValueError, match="must be positive"):
        local_sensitivity(lambda _values: 1.0, [1.0], absolute_step=-1.0)
    with pytest.raises(ValueError, match="finite values"):
        local_sensitivity(lambda _values: 1.0, [math.inf])
    with pytest.raises(ValueError, match="model output"):
        local_sensitivity(lambda _values: math.nan, [1.0])

    def unstable(values: Sequence[float]) -> float:
        return 1.0 if values[0] == 1.0 else math.inf

    with pytest.raises(ValueError, match="model output"):
        local_sensitivity(unstable, [1.0])


def test_morris_screening_ranks_dominant_parameter_and_is_reproducible() -> None:
    def model(values: Sequence[float]) -> float:
        return values[0] + 10.0 * values[1]

    first = morris_screening(model, [(0.0, 1.0), (0.0, 1.0)], trajectories=12, seed=7)
    second = morris_screening(model, [(0.0, 1.0), (0.0, 1.0)], trajectories=12, seed=7)
    assert first == second
    assert first.most_influential() is first.parameters[1]
    assert first.parameters[0].mean_absolute == pytest.approx(1.0)
    assert first.parameters[1].mean_absolute == pytest.approx(10.0)
    assert first.parameters[0].standard_deviation == pytest.approx(0.0, abs=1e-12)


def test_sobol_additive_model_separates_first_and_total_effects() -> None:
    def model(values: Sequence[float]) -> float:
        return values[0] + 3.0 * values[1]

    report = sobol_indices(model, [(0.0, 1.0), (0.0, 1.0)], samples=5000, seed=4)
    first, second = report.parameters
    assert report.variance > 0
    assert 0.05 < first.first_order < 0.2
    assert 0.05 < first.total_order < 0.2
    assert 0.8 < second.first_order < 1.05
    assert 0.8 < second.total_order < 1.05


def test_pairwise_interaction_detects_product_term() -> None:
    def model(values: Sequence[float]) -> float:
        return 2.0 * values[0] * values[1] + values[2]

    report = pairwise_interactions(model, [2.0, 3.0, 1.0])
    assert len(report.effects) == 3
    product = next(effect for effect in report.effects if (effect.first, effect.second) == (0, 1))
    assert product.derivative == pytest.approx(2.0, rel=1e-5)
    assert report.strongest() is product

    single = pairwise_interactions(lambda values: values[0], [1.0])
    assert single.effects == ()
    assert single.strongest() is None


def test_global_sensitivity_validation_paths() -> None:
    with pytest.raises(ValueError, match="bounds"):
        morris_screening(lambda _values: 1.0, [])
    with pytest.raises(ValueError, match="lower < upper"):
        morris_screening(lambda _values: 1.0, [(1.0, 1.0)])
    with pytest.raises(ValueError, match="trajectories"):
        morris_screening(lambda _values: 1.0, [(0.0, 1.0)], trajectories=1)
    for levels in (3, 5):
        with pytest.raises(ValueError, match="levels"):
            morris_screening(lambda _values: 1.0, [(0.0, 1.0)], levels=levels)
    with pytest.raises(ValueError, match="samples"):
        sobol_indices(lambda values: values[0], [(0.0, 1.0)], samples=1)
    with pytest.raises(ValueError, match="zero-variance"):
        sobol_indices(lambda _values: 1.0, [(0.0, 1.0)], samples=10, seed=1)
    with pytest.raises(ValueError, match="must be positive"):
        pairwise_interactions(lambda values: values[0], [1.0], relative_step=0.0)
    with pytest.raises(ValueError, match="finite values"):
        pairwise_interactions(lambda values: values[0], [math.inf])
