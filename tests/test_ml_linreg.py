"""Tests for :class:`cds.ml.linear_models.LinearRegression` (vector OLS)."""

from __future__ import annotations

import pytest

from cds.ml import LinearRegression


def test_exact_linear_relationship_recovered() -> None:
    # Columns chosen non-collinear with each other and the intercept.
    X = [[1.0, 0.0], [2.0, 1.0], [4.0, 2.0], [5.0, 4.0]]
    y = [2.0 * a + 0.5 * b + 1.0 for a, b in X]
    model = LinearRegression().fit(X, y)
    assert model.weights[0] == pytest.approx(2.0, abs=1e-8)
    assert model.weights[1] == pytest.approx(0.5, abs=1e-8)
    assert model.intercept == pytest.approx(1.0, abs=1e-8)
    assert model.score(X, y) == pytest.approx(1.0, abs=1e-9)


def test_predict_before_fit_raises() -> None:
    with pytest.raises(ValueError, match="not fitted"):
        LinearRegression().predict([1.0])


def test_predict_width_mismatch() -> None:
    model = LinearRegression().fit([[1.0], [2.0]], [1.0, 2.0])
    with pytest.raises(ValueError, match="features"):
        model.predict([1.0, 2.0])


def test_no_intercept_forces_zero_bias() -> None:
    X = [[1.0], [2.0], [3.0]]
    y = [2.0 * x + 10.0 for (x,) in [(1.0,), (2.0,), (3.0,)]]
    model = LinearRegression(fit_intercept=False).fit(X, y)
    assert model.intercept == 0.0
    # Best intercept-free slope for offset data is attenuated, not 2.
    assert model.weights[0] != pytest.approx(2.0)
    assert model.predict([4.0]) == pytest.approx(model.weights[0] * 4.0)


def test_rank_deficient_design_raises() -> None:
    X = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]  # duplicate column
    y = [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="rank-deficient"):
        LinearRegression().fit(X, y)


def test_score_zero_variance_target_is_zero() -> None:
    X = [[1.0], [2.0], [3.0]]
    y = [5.0, 5.0, 5.0]
    model = LinearRegression().fit(X, y)
    assert model.score(X, y) == 0.0


def test_fit_validations() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        LinearRegression().fit([], [])
    with pytest.raises(ValueError, match="same length"):
        LinearRegression().fit([[1.0]], [1.0, 2.0])
    with pytest.raises(ValueError, match="same length"):
        LinearRegression().fit([[1.0, 2.0], [3.0]], [1.0, 2.0])


def test_imperfect_fit_positive_r_squared() -> None:
    X = [[1.0], [2.0], [3.0], [4.0]]
    y = [2.1, 3.9, 6.2, 7.8]
    model = LinearRegression().fit(X, y)
    r2 = model.score(X, y)
    assert 0.99 < r2 < 1.0
