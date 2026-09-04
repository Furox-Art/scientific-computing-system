"""Regression tests for model-fitting input validation."""

from __future__ import annotations

import pytest

from cds.modeling import MathModel, Variable, fit_parameters


def _linear_model() -> MathModel:
    model = MathModel(name="linear")
    model.add_equation("y", Variable("m") * Variable("x"))
    return model


def test_fit_rejects_duplicate_parameter_names() -> None:
    model = _linear_model()
    data = [({"x": 1.0}, 2.0)]

    with pytest.raises(ValueError, match="unique"):
        fit_parameters(model, data, ["m", "m"], x0=[0.0, 0.0])


def test_fit_rejects_empty_model() -> None:
    model = MathModel(name="empty")

    with pytest.raises(ValueError, match="at least one equation"):
        fit_parameters(model, [({"x": 1.0}, 2.0)], ["m"], x0=[0.0])


@pytest.mark.parametrize("x0", [[], [0.0, 1.0]])
def test_fit_rejects_x0_length_mismatch(x0: list[float]) -> None:
    model = _linear_model()
    data = [({"x": 1.0}, 2.0)]

    with pytest.raises(ValueError, match="exactly match parameter_names"):
        fit_parameters(model, data, ["m"], x0=x0)


def test_fit_default_x0_still_works() -> None:
    model = _linear_model()
    data = [({"x": 1.0}, 2.0), ({"x": 2.0}, 4.0)]

    result = fit_parameters(model, data, ["m"])

    assert abs(result.parameters["m"] - 2.0) < 1e-3
