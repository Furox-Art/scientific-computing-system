"""Tests for global range-wide sensitivity screening."""

from __future__ import annotations

import math

import pytest

from cds.sensitivity import global_sensitivity


def test_global_sensitivity_ranks_additive_coefficients() -> None:
    report = global_sensitivity(
        lambda values: 5.0 * values[0] + 0.5 * values[1],
        [(0.0, 1.0), (0.0, 1.0)],
        trajectories=20,
        seed=7,
    )
    assert report.evaluations == 60
    assert report.parameters[0].mean_effect == pytest.approx(5.0)
    assert report.parameters[1].mean_effect == pytest.approx(0.5)
    assert report.parameters[0].mean_absolute_effect > report.parameters[1].mean_absolute_effect
    assert report.parameters[0].monotonicity == pytest.approx(1.0)
    assert report.most_influential() == report.parameters[0]


def test_global_sensitivity_detects_interaction_effect_spread() -> None:
    report = global_sensitivity(
        lambda values: values[0] * values[1],
        [(-1.0, 1.0), (-1.0, 1.0)],
        trajectories=64,
        seed=11,
    )
    assert report.parameters[0].effect_std > 0.1
    assert report.parameters[1].effect_std > 0.1
    assert report.parameters[0].monotonicity < 0.8


def test_global_sensitivity_is_seed_reproducible() -> None:
    kwargs = {"trajectories": 10, "levels": 5, "seed": 123}
    first = global_sensitivity(lambda values: values[0] ** 2 + values[1], [(0, 2), (1, 3)], **kwargs)
    second = global_sensitivity(
        lambda values: values[0] ** 2 + values[1], [(0, 2), (1, 3)], **kwargs
    )
    assert first == second


def test_global_sensitivity_validates_configuration_and_outputs() -> None:
    with pytest.raises(ValueError, match="at least one"):
        global_sensitivity(lambda _values: 0.0, [])
    with pytest.raises(ValueError, match="trajectories"):
        global_sensitivity(lambda _values: 0.0, [(0.0, 1.0)], trajectories=0)
    with pytest.raises(ValueError, match="levels"):
        global_sensitivity(lambda _values: 0.0, [(0.0, 1.0)], levels=1)
    with pytest.raises(ValueError, match="finite"):
        global_sensitivity(lambda _values: 0.0, [(0.0, math.inf)])
    with pytest.raises(ValueError, match="lower < upper"):
        global_sensitivity(lambda _values: 0.0, [(1.0, 1.0)])
    with pytest.raises(ValueError, match="model output"):
        global_sensitivity(lambda _values: math.inf, [(0.0, 1.0)], trajectories=1)


def test_global_sensitivity_detects_nonfinite_perturbation() -> None:
    def model(values: list[float] | tuple[float, ...]) -> float:
        if values[0] > 0.5:
            return math.nan
        return values[0]

    with pytest.raises(ValueError, match="model"):
        global_sensitivity(model, [(0.0, 1.0)], trajectories=8, seed=2)
