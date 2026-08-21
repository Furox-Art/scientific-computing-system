"""Preprocessing utilities: feature scaling and train/test splitting.

Sklearn-flavored ``fit``/``transform`` API on plain lists — zero dependencies,
deterministic given a seed.
"""

from __future__ import annotations

import random


class StandardScaler:
    """Zero-mean, unit-variance scaling per column (population std, ddof=0).

    Zero-variance columns are mapped to 0.0 instead of dividing by zero —
    matching sklearn's behavior of leaving constant features unscaled-but-
    centered.
    """

    def __init__(self) -> None:
        """Create an unfitted scaler; call :meth:`fit` first."""
        self._means: list[float] = []
        self._stds: list[float] = []
        self._fitted = False

    def fit(self, X: list[list[float]]) -> StandardScaler:
        """Learn per-column means and standard deviations.

        Raises:
            ValueError: if ``X`` is empty or rows are ragged.
        """
        if not X:
            raise ValueError("X must be non-empty")
        width = len(X[0])
        if any(len(row) != width for row in X):
            raise ValueError("all rows must have the same length")
        n = float(len(X))
        self._means = [sum(row[d] for row in X) / n for d in range(width)]
        self._stds = [
            (sum((row[d] - self._means[d]) ** 2 for row in X) / n) ** 0.5 for d in range(width)
        ]
        self._fitted = True
        return self

    def transform(self, X: list[list[float]]) -> list[list[float]]:
        """Scale ``X`` with the fitted statistics.

        Raises:
            ValueError: if called before :meth:`fit`, rows are ragged, or the
                column count differs from the fitted data.
        """
        if not self._fitted:
            raise ValueError("scaler is not fitted")
        if not X:
            raise ValueError("X must be non-empty")
        if any(len(row) != len(self._means) for row in X):
            raise ValueError(f"rows must have {len(self._means)} features to match the fitted data")
        return [
            [
                (row[d] - self._means[d]) / self._stds[d] if self._stds[d] > 0 else 0.0
                for d in range(len(self._means))
            ]
            for row in X
        ]

    def inverse_transform(self, X: list[list[float]]) -> list[list[float]]:
        """Undo the scaling applied by :meth:`transform`.

        Raises:
            ValueError: if called before :meth:`fit`.
        """
        if not self._fitted:
            raise ValueError("scaler is not fitted")
        return [
            [
                row[d] * self._stds[d] + self._means[d] if self._stds[d] > 0 else self._means[d]
                for d in range(len(self._means))
            ]
            for row in X
        ]

    def fit_transform(self, X: list[list[float]]) -> list[list[float]]:
        """Fit on ``X`` then return its scaled version."""
        return self.fit(X).transform(X)


def train_test_split(
    X: list[list[float]],
    y: list[float],
    *,
    test_size: float = 0.25,
    seed: int | None = None,
) -> tuple[list[list[float]], list[list[float]], list[float], list[float]]:
    """Shuffle and split ``(X, y)`` into train/test partitions.

    Uses a Fisher–Yates shuffle over row indices driven by a seeded
    :class:`random.Random`, so a fixed ``seed`` reproduces the exact split.

    Args:
        X: feature rows
        y: targets aligned with ``X``
        test_size: fraction of rows assigned to the test set, in ``(0, 1)``
        seed: RNG seed; ``None`` uses OS entropy

    Returns:
        ``(X_train, X_test, y_train, y_test)``.

    Raises:
        ValueError: on empty input, length mismatch, ragged rows, or
            ``test_size`` outside ``(0, 1)``.
    """
    if not X:
        raise ValueError("X must be non-empty")
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    if any(len(row) != len(X[0]) for row in X):
        raise ValueError("all rows must have the same length")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be in (0, 1)")

    n_test = max(1, round(len(X) * test_size))
    indices = list(range(len(X)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    test_idx = set(indices[:n_test])
    X_train = [X[i] for i in indices[n_test:]]
    X_test = [X[i] for i in sorted(test_idx)]
    y_train = [y[i] for i in indices[n_test:]]
    y_test = [y[i] for i in sorted(test_idx)]
    return X_train, X_test, y_train, y_test
