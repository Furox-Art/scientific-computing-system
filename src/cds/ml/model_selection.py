"""K-fold cross-validation utilities — pure Python, zero dependencies.

Sklearn-flavored helpers built on plain lists and the :mod:`cds.ml`
estimator interface (``fit(X, y)`` / ``predict(x)``). Everything is
deterministic given a seed.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from cds.ml.metrics import Label, accuracy


class SupervisedModel(Protocol):
    """Structural interface shared by cds.ml estimators used in CV."""

    def fit(self, X: list[list[float]], y: Sequence[Label]) -> object:
        """Fit on ``(X, y)`` and return ``self``."""
        ...

    def predict(self, x: list[float]) -> Label:
        """Predict a single row."""
        ...


ModelFactory = Callable[[], SupervisedModel]
Scorer = Callable[[Sequence[Label], Sequence[Label]], float]


@dataclass
class CVResult:
    """Fold scores produced by :func:`cross_val_score`.

    Attributes:
        scores: one score per fold, in fold order.
        mean_score: unweighted mean of ``scores``.
    """

    scores: list[float]
    mean_score: float


def k_fold_indices(
    n: int,
    k: int,
    *,
    shuffle: bool = False,
    seed: int | None = None,
) -> list[tuple[list[int], list[int]]]:
    """Partition ``n`` row indices into ``k`` folds.

    Each element is ``(train_indices, test_indices)`` for one fold. Fold sizes
    differ by at most one; when they do, the *earlier* folds are larger. With
    ``shuffle=False`` folds are contiguous index blocks (deterministic); with
    ``shuffle=True`` a seeded Fisher-Yates shuffle runs first.

    Args:
        n: number of rows (must satisfy ``n >= k``).
        k: number of folds (``k >= 2``).
        shuffle: whether to shuffle indices before splitting.
        seed: RNG seed used only when ``shuffle`` is true.

    Returns:
        List of ``k`` ``(train, test)`` index pairs.

    Raises:
        ValueError: if ``k < 2``, ``n < k``, or ``n < 1``.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if k < 2:
        raise ValueError("k must be >= 2")
    if n < k:
        raise ValueError("n must be >= k")

    indices = list(range(n))
    if shuffle:
        random.Random(seed).shuffle(indices)

    base, extra = divmod(n, k)
    folds: list[tuple[list[int], list[int]]] = []
    start = 0
    for fold in range(k):
        size = base + 1 if fold < extra else base
        test = indices[start : start + size]
        train = [i for i in indices if i not in set(test)]
        folds.append((train, test))
        start += size
    return folds


def cross_val_score(
    make_model: ModelFactory,
    X: list[list[float]],
    y: Sequence[Label],
    *,
    k: int = 5,
    shuffle: bool = False,
    seed: int | None = None,
    scorer: Scorer = accuracy,
) -> CVResult:
    """Evaluate a model factory with k-fold cross-validation.

    A fresh model is constructed per fold via ``make_model()``, fitted on the
    fold's training partition and scored with ``scorer(y_true, y_pred)`` on
    the held-out rows. Deterministic whenever ``shuffle=False`` or a fixed
    ``seed`` plus deterministic factory/scorer are supplied.

    Args:
        make_model: zero-argument callable producing an unfitted estimator.
        X: feature rows.
        y: labels aligned with ``X``.
        k: number of folds (``2 <= k <= len(X)``).
        shuffle: whether to shuffle before folding (recommended for sorted
            datasets).
        seed: RNG seed used only when ``shuffle`` is true.
        scorer: callable ``(y_true, y_pred) -> float``; defaults to accuracy.

    Returns:
        :class:`CVResult` with per-fold scores and their mean.

    Raises:
        ValueError: if ``X`` is empty, lengths mismatch, rows are ragged, or
            the fold geometry is invalid.
    """
    if not X:
        raise ValueError("X must be non-empty")
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    if any(len(row) != len(X[0]) for row in X):
        raise ValueError("all rows must have the same length")

    scores: list[float] = []
    for train_idx, test_idx in k_fold_indices(len(X), k, shuffle=shuffle, seed=seed):
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        y_test = [y[i] for i in test_idx]

        model = make_model()
        model.fit(X_train, y_train)
        y_pred = [model.predict(row) for row in X_test]
        scores.append(scorer(y_test, y_pred))

    mean = sum(scores) / len(scores)
    return CVResult(scores=scores, mean_score=mean)
