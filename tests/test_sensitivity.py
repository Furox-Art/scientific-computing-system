"""Tests for dependency-free local/global sensitivity and identifiability."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from cds.sensitivity import global_sensitivity, local_identifiability, local_sensitivity


def test_local_sensitivity_derivatives_normalization_and_ranking() -> None:
    def model(parameters: Sequence[float]) -> float:
        return 2.0 * parameters[0] + 5.0 * parameters[1]

    report = local_sensitivity(model, [3.0, 4.0])
    assert report.output == 26.0
    assert len(report.parameters) == 2
    assert math.isclose(report.parameters[0].derivative, 2.0, rel_tol=1e-9)
    assert math.isclose(report.parameters[1].derivative, 5.0, rel_tol=1e-9)
    assert report.parameters[0].normalized is not None
    assert report.parameters[1].normalized is not None
    assert math.isclose(report.parameters[0].normalized or 0.0, 6.0 / 26.0, rel_tol=1e-9)
    assert math.isclose(report.parameters[1].normalized or 0.0, 20.0 / 26.0, rel_tol=1e-9)
    influential = report.most_influential()
    assert influential is not None
    assert influential.index == 1


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
    with pytest.raises(ValueError, match="baseline output"):
        local_sensitivity(lambda _values: math.nan, [1.0])

    def unstable(values: Sequence[float]) -> float:
        return 1.0 if values[0] == 1.0 else math.inf

    with pytest.raises(ValueError, match="perturbation output"):
        local_sensitivity(unstable, [1.0])


def test_global_sensitivity_ranks_dominant_parameter_and_is_reproducible() -> None:
    def model(parameters: Sequence[float]) -> float:
        return 4.0 * parameters[0] + 0.25 * parameters[1]

    first = global_sensitivity(model, [(-1.0, 1.0), (-1.0, 1.0)], samples=2048, seed=7)
    second = global_sensitivity(model, [(-1.0, 1.0), (-1.0, 1.0)], samples=2048, seed=7)

    assert first == second
    assert first.samples == 2048
    assert first.variance > 0.0
    assert len(first.parameters) == 2
    assert first.most_influential().index == 0
    assert first.parameters[0].total_order > 0.9
    assert first.parameters[1].total_order < 0.02
    assert first.parameters[0].first_order > first.parameters[1].first_order


def test_global_sensitivity_detects_interaction_as_total_order_effect() -> None:
    def interaction(parameters: Sequence[float]) -> float:
        return parameters[0] * parameters[1]

    report = global_sensitivity(interaction, [(0.0, 1.0), (0.0, 1.0)], samples=4096, seed=11)
    assert all(parameter.total_order > 0.4 for parameter in report.parameters)
    assert all(parameter.first_order < parameter.total_order for parameter in report.parameters)


def test_global_sensitivity_validation_and_nonfinite_output() -> None:
    with pytest.raises(ValueError, match="samples must be at least 2"):
        global_sensitivity(lambda values: values[0], [(0.0, 1.0)], samples=1)
    with pytest.raises(ValueError, match="variance_tolerance"):
        global_sensitivity(
            lambda values: values[0],
            [(0.0, 1.0)],
            variance_tolerance=0.0,
        )
    with pytest.raises(ValueError, match="at least one parameter"):
        global_sensitivity(lambda _values: 1.0, [])
    with pytest.raises(ValueError, match="finite values"):
        global_sensitivity(lambda values: values[0], [(0.0, math.inf)])
    with pytest.raises(ValueError, match="smaller"):
        global_sensitivity(lambda values: values[0], [(1.0, 1.0)])
    with pytest.raises(ValueError, match="too small"):
        global_sensitivity(lambda _values: 3.0, [(0.0, 1.0)], samples=8)
    with pytest.raises(ValueError, match="outputs must be finite"):
        global_sensitivity(lambda _values: math.nan, [(0.0, 1.0)], samples=8)


def test_local_identifiability_full_rank_and_singular_models() -> None:
    def identifiable(parameters: Sequence[float]) -> Sequence[float]:
        return (2.0 * parameters[0], 3.0 * parameters[1])

    full = local_identifiability(identifiable, [1.0, 2.0])
    assert full.identifiable
    assert full.rank == 2
    assert full.parameter_count == 2
    assert full.output_count == 2
    assert full.independent_parameters == (True, True)
    assert math.isclose(full.jacobian[0][0], 2.0, rel_tol=1e-9)
    assert math.isclose(full.jacobian[1][1], 3.0, rel_tol=1e-9)

    def confounded(parameters: Sequence[float]) -> Sequence[float]:
        combined = parameters[0] + parameters[1]
        return (combined, 2.0 * combined)

    singular = local_identifiability(confounded, [1.0, 1.0])
    assert not singular.identifiable
    assert singular.rank == 1
    assert singular.independent_parameters == (True, False)


def test_local_identifiability_zero_column_is_not_independent() -> None:
    def model(parameters: Sequence[float]) -> Sequence[float]:
        return (parameters[0], 7.0)

    report = local_identifiability(model, [1.0, 5.0])
    assert report.rank == 1
    assert report.independent_parameters == (True, False)


def test_identifiability_validation_and_model_contracts() -> None:
    model = lambda values: (values[0],)
    with pytest.raises(ValueError, match="must be positive"):
        local_identifiability(model, [1.0], relative_step=0.0)
    with pytest.raises(ValueError, match="must be positive"):
        local_identifiability(model, [1.0], absolute_step=0.0)
    with pytest.raises(ValueError, match="rank_tolerance"):
        local_identifiability(model, [1.0], rank_tolerance=0.0)
    with pytest.raises(ValueError, match="at least one value"):
        local_identifiability(lambda _values: (1.0,), [])
    with pytest.raises(ValueError, match="finite values"):
        local_identifiability(model, [math.inf])
    with pytest.raises(ValueError, match="at least one output"):
        local_identifiability(lambda _values: (), [1.0])
    with pytest.raises(ValueError, match="outputs must be finite"):
        local_identifiability(lambda _values: (math.nan,), [1.0])

    def changing_length(values: Sequence[float]) -> Sequence[float]:
        return (1.0,) if values[0] <= 1.0 else (1.0, 2.0)

    with pytest.raises(ValueError, match="length must remain constant"):
        local_identifiability(changing_length, [1.0])
