"""Gradient-boosted trees for binary classification — pure Python.

Logistic-loss gradient boosting over shallow regression trees. Each stage
fits a minimal CART regressor (:class:`_RegressionTree`) to the negative
gradients ``y - sigmoid(F)`` and refines every leaf with a Newton step
``sum(residual) / sum(p·(1-p))`` over its rows, shrunk by ``learning_rate``.
No subsampling and no randomness: identical inputs yield bit-identical models.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from cds.ml.metrics import Label

_EPS = 1e-12


def _probability(z: float) -> float:
    """Numerically stable logistic function, clamped away from 0 and 1.

    Clamping keeps Newton denominators ``p·(1-p)`` strictly positive even
    when the additive model saturates.
    """
    if z >= 0.0:
        p = 1.0 / (1.0 + math.exp(-z))
    else:
        e = math.exp(z)
        p = e / (1.0 + e)
    return min(1.0 - _EPS, max(p, _EPS))


class _Leaf:
    """Terminal node holding a single Newton-step value."""

    __slots__ = ("value",)

    def __init__(self, value: float) -> None:
        self.value = value


class _Split:
    """Internal node routing on ``x[feature] <= threshold``."""

    __slots__ = ("feature", "threshold", "left", "right")

    def __init__(
        self,
        feature: int,
        threshold: float,
        left: _Leaf | _Split,
        right: _Leaf | _Split,
    ) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right


class _RegressionTree:
    """Minimal CART regressor fitting residual vectors for boosting.

    Splits minimise weighted child SSE, with candidates taken as midpoints
    between consecutive distinct sorted values of each feature. A node becomes
    a leaf when it reaches ``max_depth``, has fewer than ``min_samples_leaf``
    rows, or no feature varies. Ties keep the earliest candidate (lowest
    feature index, then lowest threshold), so the tree depends only on the
    data. Leaf values are Newton steps ``sum(residual) / sum(weight)``.
    """

    def __init__(self, *, max_depth: int = 2, min_samples_leaf: int = 2) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self._X: list[list[float]] = []
        self._residual: list[float] = []
        self._weight: list[float] = []
        self._n_features = 0
        self._root: _Leaf | _Split = _Leaf(0.0)

    def fit(
        self,
        X: list[list[float]],
        residual: list[float],
        weight: list[float],
    ) -> _RegressionTree:
        """Grow the tree on feature rows, gradients, and Newton weights."""
        self._X = X
        self._residual = residual
        self._weight = weight
        self._n_features = len(X[0])
        self._root = self._build(list(range(len(X))), depth=0)
        return self

    def leaf_value(self, x: list[float]) -> float:
        """Newton-step value stored in the leaf reached by ``x``."""
        node = self._root
        while isinstance(node, _Split):
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    # ------------------------------------------------------------------ #
    # Tree induction                                                      #
    # ------------------------------------------------------------------ #

    def _newton_value(self, idxs: list[int]) -> float:
        """``sum(residual) / sum(weight)`` over the rows indexed by ``idxs``."""
        numerator = sum(self._residual[i] for i in idxs)
        denominator = sum(self._weight[i] for i in idxs)
        return numerator / denominator

    def _sse(self, idxs: list[int]) -> float:
        """Sum of squared residuals around the mean of the indexed rows."""
        mean = sum(self._residual[i] for i in idxs) / len(idxs)
        return sum((self._residual[i] - mean) ** 2 for i in idxs)

    def _build(self, idxs: list[int], *, depth: int) -> _Leaf | _Split:
        """Recursively split ``idxs`` (indices into the training data)."""
        if depth >= self.max_depth or len(idxs) < self.min_samples_leaf:
            return _Leaf(self._newton_value(idxs))

        best = self._best_split(idxs)
        if best is None:
            return _Leaf(self._newton_value(idxs))  # no feature varies

        feature, threshold = best
        left_ids = [i for i in idxs if self._X[i][feature] <= threshold]
        right_ids = [i for i in idxs if self._X[i][feature] > threshold]
        return _Split(
            feature,
            threshold,
            self._build(left_ids, depth=depth + 1),
            self._build(right_ids, depth=depth + 1),
        )

    def _best_split(self, idxs: list[int]) -> tuple[int, float] | None:
        """Exhaustive best ``(feature, threshold)`` by weighted SSE, or None.

        Candidates are midpoints between consecutive *distinct* sorted values
        of each feature. Ties keep the earliest candidate, mirroring
        :meth:`cds.ml.tree.DecisionTreeClassifier._best_split`.
        """
        n = len(idxs)
        best_score = self._sse(idxs) + 1.0  # anything strictly better wins
        best: tuple[int, float] | None = None

        for feature in range(self._n_features):
            ordered = sorted((self._X[i][feature], i) for i in idxs)
            for pos in range(len(ordered) - 1):
                va, _ = ordered[pos]
                vb, _ = ordered[pos + 1]
                if va == vb:
                    continue  # duplicate values: no midpoint between them
                threshold = (va + vb) / 2.0
                left_ids = [i for i in idxs if self._X[i][feature] <= threshold]
                right_ids = [i for i in idxs if self._X[i][feature] > threshold]
                score = (self._sse(left_ids) + self._sse(right_ids)) / n
                if score < best_score:
                    best_score = score
                    best = (feature, threshold)
        return best


class GradientBoostingClassifier:
    """Gradient-boosted trees for binary classification.

    Fits an initial log-odds constant from the base rate, then adds
    ``n_estimators`` shallow regression trees, each trained on the logistic
    loss's negative gradients. Every leaf contributes a Newton step scaled by
    ``learning_rate``. Training is fully deterministic — there is no
    subsampling and no seed.

    The *second* distinct label encountered in ``y`` is the positive class:
    :meth:`predict_proba_one` reports its probability.

    Args:
        n_estimators: Number of boosting stages (>= 1).
        learning_rate: Shrinkage applied to each tree's output (> 0).
        max_depth: Maximum depth of each tree (>= 0), passed through.
        min_samples_leaf: Minimum rows for a node to be split (>= 1),
            passed through.

    Raises:
        ValueError: if any hyperparameter is out of range.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 2,
        min_samples_leaf: int = 2,
    ) -> None:
        """Validate hyperparameters and create an unfitted booster."""
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if max_depth < 0:
            raise ValueError("max_depth must be >= 0")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self._trees: list[_RegressionTree] = []
        self._labels: list[Label] = []
        self._f0 = 0.0
        self._n_features = 0
        self._fitted = False

    def fit(self, X: list[list[float]], y: Sequence[Label]) -> GradientBoostingClassifier:
        """Fit the additive logistic model on ``(X, y)``.

        Args:
            X: Feature rows.
            y: Binary labels (exactly two distinct strings or ints). The
                second distinct label is treated as the positive class.

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: if ``X`` is empty, lengths mismatch, rows are ragged,
                or the labels are not binary.
        """
        if not X:
            raise ValueError("X must be non-empty")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        width = len(X[0])
        if any(len(row) != width for row in X):
            raise ValueError("all rows must have the same length")

        labels: list[Label] = []
        counts: dict[Label, int] = {}
        for label in y:
            if label not in labels:
                labels.append(label)
            counts[label] = counts.get(label, 0) + 1
        if len(labels) != 2:
            raise ValueError("labels must be binary")
        positive = labels[1]

        rate = counts[positive] / len(y)
        rate = min(1.0 - _EPS, max(rate, _EPS))
        self._f0 = math.log(rate / (1.0 - rate))
        targets = [1.0 if label == positive else 0.0 for label in y]
        scores = [self._f0] * len(y)

        trees: list[_RegressionTree] = []
        for _ in range(self.n_estimators):
            probs = [_probability(score) for score in scores]
            residuals = [t - p for t, p in zip(targets, probs)]
            weights = [p * (1.0 - p) for p in probs]
            tree = _RegressionTree(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf)
            tree.fit(X, residuals, weights)
            for i, row in enumerate(X):
                scores[i] += self.learning_rate * tree.leaf_value(row)
            trees.append(tree)

        self._trees = trees
        self._labels = labels
        self._n_features = width
        self._fitted = True
        return self

    def predict_proba_one(self, x: list[float]) -> float:
        """Probability that ``x`` belongs to the second-seen label.

        Returns:
            The additive model's logistic output, in ``(0, 1)``.

        Raises:
            ValueError: if called before :meth:`fit` or on a feature-count
                mismatch.
        """
        if not self._fitted:
            raise ValueError("model is not fitted")
        if len(x) != self._n_features:
            raise ValueError(f"query must have {self._n_features} features")
        score = self._f0
        for tree in self._trees:
            score += self.learning_rate * tree.leaf_value(x)
        return _probability(score)

    def predict_proba(self, x: list[float]) -> dict[Label, float]:
        """Class-membership probabilities for ``x``, keyed by both labels.

        Labels appear in first-appearance order; the values sum to 1.0.

        Raises:
            ValueError: if called before :meth:`fit` or on a feature-count
                mismatch.
        """
        p = self.predict_proba_one(x)
        first, second = self._labels[0], self._labels[1]
        return {first: 1.0 - p, second: p}

    def predict(self, x: list[float]) -> Label:
        """Higher-probability class for ``x`` (ties → first-seen label)."""
        return self._labels[1] if self.predict_proba_one(x) > 0.5 else self._labels[0]
