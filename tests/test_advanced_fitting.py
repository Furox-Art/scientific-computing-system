"""Tests for advanced model fitting and diagnostics."""

from __future__ import annotations

import math

import pytest

from cds.modeling import MathModel, Variable, fit_parameters_advanced
from cds.modeling.fitting import (
    FitOptimizer,
    _invert_matrix,
    _loss_value,
    _make_starts,
    _select_optimizer,
    _uncertainty,
)


def _linear_model() -> MathModel:
    model = MathModel(name="linear", parameters={"a": 0.0, "b": 0.0}, variables=["x"])
    model.add_equation("y", Variable("a") * Variable("x") + Variable("b"))
    return model


def _linear_observations() -> list[tuple[dict[str, float], float]]:
    return [
        ({"x": 0.0}, 1.0),
        ({"x": 1.0}, 3.0),
        ({"x": 2.0}, 5.0),
        ({"x": 3.0}, 7.0),
        ({"x": 4.0}, 9.0),
    ]


def test_loss_and_optimizer_selection_helpers() -> None:
    assert _loss_value(-2.0, "squared", 1.0) == 4.0
    assert _loss_value(-2.0, "absolute", 1.0) == 2.0
    assert _loss_value(0.5, "huber", 1.0) == pytest.approx(0.125)
    assert _loss_value(2.0, "huber", 1.0) == pytest.approx(1.5)

    assert _select_optimizer("gradient_descent", "squared", None)[0] == "gradient_descent"
    assert _select_optimizer("auto", "squared", [(0.0, 1.0)])[0] == "projected_gradient"
    assert _select_optimizer("auto", "absolute", None)[0] == "nelder_mead"
    assert _select_optimizer("auto", "huber", None)[0] == "adam"


def test_make_starts_is_seeded_and_respects_bounds() -> None:
    assert _make_starts([1.0], 1, None, 3) == [[1.0]]

    unbounded_a = _make_starts([1.0, 2.0], 3, None, 7)
    unbounded_b = _make_starts([1.0, 2.0], 3, None, 7)
    assert unbounded_a == unbounded_b
    assert len(unbounded_a) == 3

    bounded = _make_starts([0.5], 4, [(0.0, 1.0)], 11)
    assert bounded[0] == [0.5]
    assert all(0.0 <= row[0] <= 1.0 for row in bounded)


def test_local_matrix_inverse_validation_and_pivoting() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _invert_matrix([])
    with pytest.raises(ValueError, match="square"):
        _invert_matrix([[1.0, 2.0], [3.0]])
    with pytest.raises(ValueError, match="singular"):
        _invert_matrix([[1.0, 2.0], [2.0, 4.0]])

    identity = _invert_matrix([[1.0, 0.0], [0.0, 1.0]])
    assert identity == [[1.0, 0.0], [0.0, 1.0]]

    swapped = _invert_matrix([[0.0, 1.0], [1.0, 0.0]])
    assert swapped == [[0.0, 1.0], [1.0, 0.0]]

    inverse = _invert_matrix([[2.0, 1.0], [1.0, 2.0]])
    assert inverse[0] == pytest.approx([2.0 / 3.0, -1.0 / 3.0])
    assert inverse[1] == pytest.approx([-1.0 / 3.0, 2.0 / 3.0])


def test_advanced_fit_input_validation() -> None:
    model = _linear_model()
    observed = _linear_observations()

    with pytest.raises(ValueError, match="at least one parameter"):
        fit_parameters_advanced(model, observed, [])
    with pytest.raises(ValueError, match="unique"):
        fit_parameters_advanced(model, observed, ["a", "a"])
    with pytest.raises(ValueError, match="at least one equation"):
        fit_parameters_advanced(MathModel(name="empty"), observed, ["a"])
    with pytest.raises(ValueError, match="at least one"):
        fit_parameters_advanced(model, [], ["a"])
    with pytest.raises(ValueError, match="x0 length"):
        fit_parameters_advanced(model, observed, ["a", "b"], x0=[0.0])
    with pytest.raises(ValueError, match="multi_start"):
        fit_parameters_advanced(model, observed, ["a"], multi_start=0)
    with pytest.raises(ValueError, match="huber_delta"):
        fit_parameters_advanced(model, observed, ["a"], huber_delta=0.0)
    with pytest.raises(ValueError, match="confidence"):
        fit_parameters_advanced(model, observed, ["a"], confidence=1.0)
    with pytest.raises(ValueError, match="same length"):
        fit_parameters_advanced(model, observed, ["a"], bounds=[(0.0, 1.0), (0.0, 2.0)])
    with pytest.raises(ValueError, match="lower <= upper"):
        fit_parameters_advanced(model, observed, ["a"], bounds=[(2.0, 1.0)])
    with pytest.raises(ValueError, match="inside bounds"):
        fit_parameters_advanced(model, observed, ["a"], x0=[3.0], bounds=[(0.0, 2.0)])
    with pytest.raises(ValueError, match="bounded fitting"):
        fit_parameters_advanced(
            model,
            observed,
            ["a"],
            x0=[1.0],
            bounds=[(0.0, 2.0)],
            optimizer="adam",
        )
    with pytest.raises(KeyError):
        fit_parameters_advanced(model, observed, ["a"], target_label="missing")
    with pytest.raises(ValueError, match="projected_gradient requires bounds"):
        fit_parameters_advanced(
            model,
            observed,
            ["a"],
            optimizer="projected_gradient",
            max_iter=1,
        )


@pytest.mark.parametrize("optimizer", ["gradient_descent", "adam", "nelder_mead"])
def test_unconstrained_optimizers_recover_linear_model(optimizer: FitOptimizer) -> None:
    model = _linear_model()
    fit_lr = 0.01 if optimizer == "gradient_descent" else 0.05
    result = fit_parameters_advanced(
        model,
        _linear_observations(),
        ["a", "b"],
        x0=[1.5, 0.5],
        optimizer=optimizer,
        multi_start=2,
        seed=4,
        lr=fit_lr,
        max_iter=3000,
    )
    assert result.optimizer == optimizer
    assert result.starts_tried == 2
    assert result.parameters["a"] == pytest.approx(2.0, abs=2e-2)
    assert result.parameters["b"] == pytest.approx(1.0, abs=3e-2)
    assert result.diagnostics.rmse < 0.05
    assert result.diagnostics.r_squared > 0.999
    assert result.diagnostics.identifiable
    assert result.diagnostics.standard_errors is not None
    assert result.diagnostics.confidence_intervals is not None


def test_auto_bounds_and_robust_loss_routes() -> None:
    slope = MathModel(name="slope", parameters={"a": 0.0}, variables=["x"])
    slope.add_equation("y", Variable("a") * Variable("x"))
    observed = [({"x": 1.0}, 2.0), ({"x": 2.0}, 4.0), ({"x": 3.0}, 6.0)]

    bounded = fit_parameters_advanced(
        slope,
        observed,
        ["a"],
        x0=[1.0],
        bounds=[(0.0, 3.0)],
        optimizer="auto",
        lr=0.05,
        max_iter=1000,
    )
    assert bounded.optimizer == "projected_gradient"
    assert bounded.parameters["a"] == pytest.approx(2.0, abs=1e-3)

    robust = fit_parameters_advanced(
        slope,
        observed + [({"x": 4.0}, 100.0)],
        ["a"],
        x0=[1.0],
        loss="absolute",
        optimizer="auto",
        max_iter=1000,
    )
    assert robust.optimizer == "nelder_mead"
    assert not robust.diagnostics.identifiable
    assert robust.diagnostics.standard_errors is None
    assert robust.diagnostics.confidence_intervals is None


def test_uncertainty_detects_underdetermined_and_singular_models() -> None:
    slope = MathModel(name="slope", parameters={"a": 1.0}, variables=["x"])
    slope.add_equation("y", Variable("a") * Variable("x"))
    under = _uncertainty(
        slope,
        [({"x": 1.0}, 1.0)],
        ["a"],
        [1.0],
        "y",
        [0.0],
        0.95,
    )
    assert under == (None, None, False)

    singular = MathModel(name="singular", parameters={"a": 0.0, "b": 0.0})
    singular.add_equation("y", Variable("a") + Variable("b"))
    result = _uncertainty(
        singular,
        [({}, 1.0), ({}, 1.0), ({}, 1.0)],
        ["a", "b"],
        [0.5, 0.5],
        "y",
        [0.0, 0.0, 0.0],
        0.95,
    )
    assert result == (None, None, False)


def test_constant_target_r_squared_branches_and_nonfinite_attempts() -> None:
    constant = MathModel(name="constant", parameters={"a": 2.0})
    constant.add_equation("y", Variable("a"))
    observed: list[tuple[dict[str, float], float]] = [({}, 2.0), ({}, 2.0)]

    exact = fit_parameters_advanced(
        constant,
        observed,
        ["a"],
        x0=[2.0],
        optimizer="gradient_descent",
        max_iter=2,
    )
    assert exact.diagnostics.r_squared == 1.0

    constrained = fit_parameters_advanced(
        constant,
        observed,
        ["a"],
        x0=[0.0],
        bounds=[(0.0, 0.0)],
        max_iter=2,
    )
    assert constrained.diagnostics.r_squared == 0.0

    with pytest.raises(ArithmeticError, match="non-finite"):
        fit_parameters_advanced(
            constant,
            [({}, math.nan), ({}, math.nan)],
            ["a"],
            x0=[0.0],
            optimizer="nelder_mead",
            max_iter=2,
        )
