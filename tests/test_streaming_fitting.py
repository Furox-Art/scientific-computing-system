"""Regression tests for replayable out-of-core fitting."""

from __future__ import annotations

import pytest

from cds.modeling import MathModel, Variable, fit_parameters_advanced


def _linear_model() -> MathModel:
    model = MathModel(name="stream-linear", parameters={"a": 0.0, "b": 0.0}, variables=["x"])
    model.add_equation("y", Variable("a") * Variable("x") + Variable("b"))
    return model


def test_replayable_factory_is_used_without_residual_retention() -> None:
    rows = [({"x": float(index)}, 2.0 * index + 1.0) for index in range(12)]
    passes = 0

    def source():
        nonlocal passes
        passes += 1
        yield from rows

    result = fit_parameters_advanced(
        _linear_model(),
        source,
        ["a", "b"],
        x0=[1.7, 0.8],
        optimizer="adam",
        lr=0.03,
        max_iter=2500,
        store_residuals=False,
    )
    assert passes > 2
    assert result.parameters["a"] == pytest.approx(2.0, abs=2e-2)
    assert result.parameters["b"] == pytest.approx(1.0, abs=3e-2)
    assert result.diagnostics.observations == len(rows)
    assert result.diagnostics.residuals == ()
    assert not result.diagnostics.residuals_stored
    assert result.diagnostics.identifiable
    assert result.diagnostics.condition_number is not None


def test_replayable_source_must_keep_stable_row_count() -> None:
    calls = 0

    def changing_source():
        nonlocal calls
        calls += 1
        stop = 4 if calls == 1 else 3
        for index in range(stop):
            yield ({"x": float(index)}, float(index))

    with pytest.raises(ValueError, match="changed row count"):
        fit_parameters_advanced(
            _linear_model(),
            changing_source,
            ["a", "b"],
            x0=[1.0, 0.0],
            optimizer="gradient_descent",
            max_iter=1,
            store_residuals=False,
        )


def test_near_collinear_jacobian_is_practically_non_identifiable() -> None:
    model = MathModel(name="near-collinear", parameters={"a": 0.0, "b": 0.0}, variables=["x"])
    epsilon = 1e-7
    model.add_equation(
        "y",
        Variable("a") * Variable("x") + Variable("b") * (Variable("x") + epsilon),
    )
    rows = [({"x": float(index)}, 2.0 * index) for index in range(1, 8)]
    result = fit_parameters_advanced(
        model,
        rows,
        ["a", "b"],
        x0=[1.0, 1.0],
        optimizer="nelder_mead",
        max_iter=300,
        identifiability_condition_limit=1e6,
    )
    assert not result.diagnostics.identifiable
    assert result.diagnostics.condition_number is not None
    assert result.diagnostics.condition_number > 1e6
    assert "condition number" in result.diagnostics.identifiability_reason


def test_invalid_condition_limit_and_nonfinite_rows_fail_closed() -> None:
    model = _linear_model()
    rows = [({"x": 1.0}, 3.0), ({"x": 2.0}, 5.0), ({"x": 3.0}, 7.0)]
    with pytest.raises(ValueError, match="condition_limit"):
        fit_parameters_advanced(model, rows, ["a", "b"], identifiability_condition_limit=1.0)
    with pytest.raises(ValueError, match="finite"):
        fit_parameters_advanced(model, [({"x": float("inf")}, 1.0)], ["a"])
