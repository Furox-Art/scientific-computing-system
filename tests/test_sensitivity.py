"""Tests for dependency-free local sensitivity analysis."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from cds.sensitivity import local_sensitivity


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
