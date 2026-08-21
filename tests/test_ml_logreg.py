"""Tests for :mod:`cds.ml.linear_models` (logistic regression)."""

from __future__ import annotations

import pytest

from cds.ml import LogisticRegression


def _separable() -> tuple[list[list[float]], list[int]]:
    X = [[1.0], [2.0], [-1.0], [-2.0]]
    y = [1, 1, 0, 0]
    return X, y


def test_fit_separates_training_data() -> None:
    X, y = _separable()
    model = LogisticRegression(lr=0.5, epochs=300).fit(X, y)
    assert all(model.predict(row) == label for row, label in zip(X, y))
    assert model.loss_history[-1] < model.loss_history[0]
    assert len(model.loss_history) == 300


def test_predict_proba_extremes_cover_both_sigmoid_branches() -> None:
    X, y = _separable()
    model = LogisticRegression(lr=0.5, epochs=300).fit(X, y)
    high = model.predict_proba([10.0])
    low = model.predict_proba([-10.0])
    assert 0.99 < high <= 1.0
    assert 0.0 <= low < 0.01
    assert model.predict([10.0]) == 1
    assert model.predict([-10.0]) == 0


def test_l2_penalty_shrinks_weights() -> None:
    X, y = _separable()
    plain = LogisticRegression(lr=0.5, epochs=200).fit(X, y)
    regularized = LogisticRegression(lr=0.5, epochs=200, l2=1.0).fit(X, y)
    plain_norm = sum(w * w for w in plain.weights)
    reg_norm = sum(w * w for w in regularized.weights)
    assert reg_norm < plain_norm


def test_two_feature_problem() -> None:
    X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [3.0, 3.0]]
    y = [0, 0, 0, 1]
    model = LogisticRegression(lr=0.8, epochs=500).fit(X, y)
    assert model.predict([0.0, 0.5]) == 0
    assert model.predict([3.0, 3.0]) == 1


def test_init_validates_hyperparameters() -> None:
    with pytest.raises(ValueError, match="lr must be positive"):
        LogisticRegression(lr=0.0)
    with pytest.raises(ValueError, match="epochs must be >= 1"):
        LogisticRegression(epochs=0)
    with pytest.raises(ValueError, match="l2 must be >= 0"):
        LogisticRegression(l2=-0.5)


def test_fit_validates_inputs() -> None:
    model = LogisticRegression()
    with pytest.raises(ValueError, match="non-empty"):
        model.fit([], [])
    with pytest.raises(ValueError, match="same length"):
        model.fit([[1.0]], [0, 1])
    with pytest.raises(ValueError, match="same length"):
        model.fit([[1.0, 2.0], [3.0]], [0, 1])
    with pytest.raises(ValueError, match="only 0 and 1"):
        model.fit([[1.0], [2.0]], [0, 2])
    with pytest.raises(ValueError, match="both classes"):
        model.fit([[1.0], [2.0]], [1, 1])


def test_predict_before_fit_raises() -> None:
    with pytest.raises(ValueError, match="not fitted"):
        LogisticRegression().predict_proba([1.0])


def test_predict_rejects_wrong_query_width() -> None:
    X, y = _separable()
    model = LogisticRegression().fit(X, y)
    with pytest.raises(ValueError, match="features"):
        model.predict([1.0, 2.0])
