"""Tests for nonparametric uncertainty in advanced model fitting."""

from __future__ import annotations

import math

import pytest

from cds.modeling import MathModel, Variable, fit_parameters_advanced
from cds.modeling.fitting import _percentile


def _model() -> MathModel:
    model = MathModel(name="linear-bootstrap", parameters={"a": 0.0, "b": 0.0}, variables=["x"])
    model.add_equation("y", Variable("a") * Variable("x") + Variable("b"))
    return model


def _observations() -> list[tuple[dict[str, float], float]]:
    noise = [0.0, 0.2, -0.1, 0.1, -0.2]
    return [
        ({"x": float(index)}, 2.0 * index + 1.0 + noise[index % len(noise)])
        for index in range(20)
    ]


def test_bootstrap_uncertainty_is_seeded_and_contains_fit() -> None:
    kwargs = dict(
        x0=[1.8, 1.0],
        optimizer="nelder_mead",
        uncertainty="bootstrap",
        bootstrap_samples=24,
        bootstrap_seed=17,
        max_iter=1500,
    )
    first = fit_parameters_advanced(_model(), _observations(), ["a", "b"], **kwargs)
    second = fit_parameters_advanced(_model(), _observations(), ["a", "b"], **kwargs)
    assert first.diagnostics.uncertainty_method == "bootstrap"
    assert first.diagnostics.bootstrap_successes >= 20
    assert first.diagnostics.confidence_intervals == first.diagnostics.bootstrap_confidence_intervals
    assert first.diagnostics.confidence_intervals == second.diagnostics.confidence_intervals
    assert first.diagnostics.standard_errors is not None
    assert first.diagnostics.identifiable
    for name, estimate in first.parameters.items():
        low, high = first.diagnostics.confidence_intervals[name]
        assert low <= estimate <= high


def test_both_uncertainty_keeps_normal_and_bootstrap_intervals() -> None:
    result = fit_parameters_advanced(
        _model(),
        _observations(),
        ["a", "b"],
        x0=[1.8, 1.0],
        optimizer="nelder_mead",
        uncertainty="both",
        bootstrap_samples=20,
        bootstrap_seed=3,
        max_iter=1500,
    )
    assert result.diagnostics.uncertainty_method == "both"
    assert result.diagnostics.confidence_intervals is not None
    assert result.diagnostics.bootstrap_confidence_intervals is not None
    assert result.diagnostics.identifiable


def test_bootstrap_supports_robust_absolute_loss() -> None:
    observed = _observations() + [({"x": 10.0}, 200.0)]
    result = fit_parameters_advanced(
        _model(),
        observed,
        ["a", "b"],
        x0=[2.0, 1.0],
        loss="absolute",
        optimizer="nelder_mead",
        uncertainty="bootstrap",
        bootstrap_samples=20,
        bootstrap_seed=9,
        min_bootstrap_success_fraction=0.5,
        max_iter=1500,
    )
    assert result.diagnostics.bootstrap_confidence_intervals is not None
    assert result.diagnostics.standard_errors is not None


def test_uncertainty_none_and_auto_preserve_fast_behavior() -> None:
    none_result = fit_parameters_advanced(
        _model(), _observations(), ["a", "b"], x0=[2.0, 1.0], uncertainty="none"
    )
    assert none_result.diagnostics.uncertainty_method == "none"
    assert none_result.diagnostics.confidence_intervals is None

    robust_auto = fit_parameters_advanced(
        _model(),
        _observations(),
        ["a", "b"],
        x0=[2.0, 1.0],
        loss="absolute",
        uncertainty="auto",
        max_iter=500,
    )
    assert robust_auto.diagnostics.uncertainty_method == "none"


def test_uncertainty_configuration_validation() -> None:
    model = _model()
    observed = _observations()
    with pytest.raises(ValueError, match="unknown uncertainty"):
        fit_parameters_advanced(model, observed, ["a", "b"], uncertainty="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="bootstrap_samples"):
        fit_parameters_advanced(model, observed, ["a", "b"], bootstrap_samples=1)
    with pytest.raises(ValueError, match="min_bootstrap"):
        fit_parameters_advanced(
            model, observed, ["a", "b"], min_bootstrap_success_fraction=0.0
        )
    with pytest.raises(ValueError, match="squared loss"):
        fit_parameters_advanced(
            model,
            observed,
            ["a", "b"],
            loss="absolute",
            uncertainty="normal",
        )


def test_percentile_interpolates_and_validates() -> None:
    assert _percentile([0.0, 10.0], 0.25) == pytest.approx(2.5)
    assert _percentile([3.0], 0.5) == 3.0
    with pytest.raises(ValueError, match="at least one"):
        _percentile([], 0.5)
    with pytest.raises(ValueError, match="probability"):
        _percentile([1.0], -0.1)


def test_bootstrap_failure_is_reported_not_fabricated(monkeypatch: pytest.MonkeyPatch) -> None:
    import cds.modeling.fitting as fitting

    original = fitting.fit_parameters_advanced
    calls = 0

    def flaky(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        nonlocal calls
        if kwargs.get("uncertainty") == "none":
            calls += 1
            raise ArithmeticError("forced bootstrap failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(fitting, "fit_parameters_advanced", flaky)
    result = original(
        _model(),
        _observations(),
        ["a", "b"],
        x0=[2.0, 1.0],
        optimizer="nelder_mead",
        uncertainty="bootstrap",
        bootstrap_samples=4,
        min_bootstrap_success_fraction=0.5,
        max_iter=500,
    )
    assert calls == 4
    assert result.diagnostics.bootstrap_successes == 0
    assert result.diagnostics.bootstrap_confidence_intervals is None
    assert result.diagnostics.standard_errors is None
    assert not result.diagnostics.identifiable


def test_bootstrap_ignores_nonfinite_parameter_replications(monkeypatch: pytest.MonkeyPatch) -> None:
    import cds.modeling.fitting as fitting

    class FakeResult:
        parameters = {"a": math.inf, "b": 0.0}

    monkeypatch.setattr(fitting, "fit_parameters_advanced", lambda *args, **kwargs: FakeResult())
    errors, intervals, successes = fitting._bootstrap_uncertainty(
        _model(),
        _observations(),
        ["a", "b"],
        [2.0, 1.0],
        target_label="y",
        optimizer="nelder_mead",
        loss="squared",
        bounds=None,
        huber_delta=1.0,
        confidence=0.95,
        bootstrap_samples=3,
        bootstrap_seed=1,
        min_success_fraction=0.5,
        lr=0.01,
        tol=1e-8,
        max_iter=10,
    )
    assert (errors, intervals, successes) == (None, None, 0)
