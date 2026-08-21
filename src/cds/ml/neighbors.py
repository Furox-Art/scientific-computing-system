"""Brute-force k-nearest-neighbour classification and regression.

Both estimators share the same core machinery: Euclidean neighbour search with
deterministic ``(distance, row_index)`` ordering so results never depend on
dict/set iteration order. Everything runs on plain ``list[float]`` vectors,
matching the rest of CDS.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def _validate_training_data(X: list[list[float]], y: Sequence[object]) -> None:
    """Check shared fit-time preconditions for both k-NN estimators."""
    if not X:
        raise ValueError("X must be non-empty")
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    width = len(X[0])
    if any(len(row) != width for row in X):
        raise ValueError("all rows must have the same length")


def _squared_distance(a: list[float], b: list[float]) -> float:
    """Squared Euclidean distance between two equal-length vectors."""
    return sum((ai - bi) ** 2 for ai, bi in zip(a, b))


def _k_nearest(
    X: list[list[float]],
    query: list[float],
    k: int,
    *,
    context: str,
) -> list[int]:
    """Indices of the ``k`` training rows nearest to ``query``.

    Ties in distance are broken by ascending row index, so the returned
    neighbour set is fully deterministic.
    """
    if not X:
        raise ValueError("model is not fitted")
    if len(query) != len(X[0]):
        raise ValueError(f"{context} must have {len(X[0])} features")
    ranked = sorted(range(len(X)), key=lambda i: (_squared_distance(X[i], query), i))
    return ranked[:k]


class KNeighborsClassifier:
    """k-nearest-neighbour majority-vote classifier.

    Votes are counted in neighbour order (nearest first) and a label keeps the
    lead only with a *strictly* greater count, so ties resolve to the label
    seen earliest among the neighbours — a stable, input-independent rule.
    """

    def __init__(self, k: int = 3) -> None:
        """Store hyperparameters; call :meth:`fit` before predicting.

        Args:
            k: Number of neighbours consulted per prediction (>= 1).

        Raises:
            ValueError: if ``k < 1``.
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self._X: list[list[float]] = []
        self._y: list[str | int] = []

    def fit(self, X: list[list[float]], y: Sequence[str | int]) -> KNeighborsClassifier:
        """Memorise the training set.

        Args:
            X: Feature rows.
            y: Class labels (strings or ints).

        Returns:
            ``self``, so ``fit`` calls can be chained.

        Raises:
            ValueError: if ``X`` is empty, lengths mismatch, or rows are ragged.
        """
        _validate_training_data(X, y)
        if self.k > len(X):
            raise ValueError("k must not exceed the number of training samples")
        self._X = [list(row) for row in X]
        self._y = list(y)
        return self

    def predict_proba(self, x: list[float]) -> dict[str | int, float]:
        """Class-membership fractions over the ``k`` nearest neighbours."""
        neighbours = _k_nearest(self._X, x, self.k, context="query")
        counts: dict[str | int, int] = {}
        for idx in neighbours:
            label = self._y[idx]
            counts[label] = counts.get(label, 0) + 1
        total = float(len(neighbours))
        return {label: c / total for label, c in counts.items()}

    def predict(self, x: list[float]) -> str | int:
        """Majority label among the ``k`` nearest training rows."""
        counts: dict[str | int, int] = {}
        best_label: str | int = ""
        best_count = -1
        for idx in _k_nearest(self._X, x, self.k, context="query"):
            label = self._y[idx]
            c = counts.get(label, 0) + 1
            counts[label] = c
            # A strictly-greater comparison makes the earliest-seen label win
            # ties, independent of hash order — same rule as predict_proba.
            if c > best_count:
                best_count = c
                best_label = label
        return best_label


class KNeighborsRegressor:
    """k-nearest-neighbour regression: mean target of the ``k`` neighbours."""

    def __init__(self, k: int = 3) -> None:
        """Store hyperparameters; call :meth:`fit` before predicting.

        Args:
            k: Number of neighbours averaged per prediction (>= 1).

        Raises:
            ValueError: if ``k < 1``.
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self._X: list[list[float]] = []
        self._y: list[float] = []

    def fit(self, X: list[list[float]], y: list[float]) -> KNeighborsRegressor:
        """Memorise the training set.

        Args:
            X: Feature rows.
            y: Continuous targets.

        Returns:
            ``self``, so ``fit`` calls can be chained.

        Raises:
            ValueError: if ``X`` is empty, lengths mismatch, or rows are ragged.
        """
        _validate_training_data(X, y)
        if self.k > len(X):
            raise ValueError("k must not exceed the number of training samples")
        self._X = [list(row) for row in X]
        self._y = list(y)
        return self

    def predict(self, x: list[float]) -> float:
        """Mean target value across the ``k`` nearest training rows."""
        neighbours = _k_nearest(self._X, x, self.k, context="query")
        return math.fsum(self._y[i] for i in neighbours) / len(neighbours)
