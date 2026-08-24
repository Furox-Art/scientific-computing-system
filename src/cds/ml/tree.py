"""CART decision-tree classifier with Gini impurity — pure Python.

A shallow, dependency-free CART implementation for numeric features:
exhaustive binary-threshold splits chosen by weighted Gini decrease, with
deterministic tie-breaking (lowest feature index, then lowest threshold).
Depth is bounded by ``max_depth``, so recursion depth is never a concern.
"""

from __future__ import annotations

import random
from collections.abc import Sequence


def _gini(labels: list[str | int]) -> float:
    """Gini impurity ``1 - Σ p_k²`` of a label list (0.0 when empty or pure)."""
    n = len(labels)
    if n == 0:
        return 0.0
    counts: dict[str | int, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return 1.0 - sum((c / n) ** 2 for c in counts.values())


def _class_probabilities(labels: list[str | int]) -> dict[str | int, float]:
    """Empirical label distribution, keys in first-appearance order."""
    counts: dict[str | int, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    total = float(len(labels))
    return {label: c / total for label, c in counts.items()}


class _Leaf:
    """Terminal node holding the empirical class distribution."""

    __slots__ = ("probs",)

    def __init__(self, probs: dict[str | int, float]) -> None:
        self.probs = probs


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


class DecisionTreeClassifier:
    """Binary CART tree over numeric features.

    Splits maximise weighted Gini decrease. A node becomes a leaf when it is
    pure, reaches ``max_depth``, has fewer than ``min_samples_split`` rows, or
    no feature varies at all — every stopping rule keeps the tree honest on
    tiny/degenerate datasets.
    """

    def __init__(
        self,
        *,
        max_depth: int = 5,
        min_samples_split: int = 2,
        max_features: int | None = None,
        seed: int | None = None,
    ) -> None:
        """Store hyperparameters; call :meth:`fit` before predicting.

        Args:
            max_depth: Maximum number of split levels (>= 0).
            min_samples_split: Minimum rows required to consider a split (>= 2).
            max_features: Number of features considered at each split
                (``None`` uses all features). Subsets are drawn without
                replacement from a seeded RNG, enabling random-forest-style
                split randomization.
            seed: RNG seed used only when ``max_features`` is set.

        Raises:
            ValueError: if a hyperparameter is out of range.
        """
        if max_depth < 0:
            raise ValueError("max_depth must be >= 0")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")
        if max_features is not None and max_features < 1:
            raise ValueError("max_features must be >= 1")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self._rng = random.Random(seed)
        self._fitted = False
        self._root: _Leaf | _Split = _Leaf({})
        self._n_features = 0

    def fit(self, X: list[list[float]], y: Sequence[str | int]) -> DecisionTreeClassifier:
        """Grow the tree on ``(X, y)``.

        Args:
            X: Feature rows.
            y: Class labels (strings or ints).

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: if ``X`` is empty, lengths mismatch, or rows are ragged.
        """
        if not X:
            raise ValueError("X must be non-empty")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        width = len(X[0])
        if any(len(row) != width for row in X):
            raise ValueError("all rows must have the same length")
        if self.max_features is not None and self.max_features > width:
            raise ValueError("max_features must be <= the number of features")
        self._n_features = width
        self._X = [list(row) for row in X]
        self._y = list(y)
        self._root = self._build(list(range(len(X))), depth=0)
        self._fitted = True
        return self

    def predict_proba(self, x: list[float]) -> dict[str | int, float]:
        """Class-membership probabilities from the leaf reached by ``x``.

        Raises:
            ValueError: if called before :meth:`fit` or on a feature-count
                mismatch.
        """
        if not self._fitted:
            raise ValueError("model is not fitted")
        node = self._root
        while isinstance(node, _Split):
            if len(x) != self._n_features:
                raise ValueError(f"query must have {self._n_features} features")
            node = node.left if x[node.feature] <= node.threshold else node.right
        return dict(node.probs)

    def predict(self, x: list[float]) -> str | int:
        """Most probable class for ``x`` (ties → earliest-seen label)."""
        best_label: str | int = ""
        best_p = -1.0
        for label, p in self.predict_proba(x).items():
            if p > best_p:
                best_p = p
                best_label = label
        return best_label

    # ------------------------------------------------------------------ #
    # Tree induction                                                      #
    # ------------------------------------------------------------------ #

    def _build(self, idxs: list[int], *, depth: int) -> _Leaf | _Split:
        """Recursively split ``idxs`` (indices into the training data)."""
        labels_here = [self._y[i] for i in idxs]
        probs = _class_probabilities(labels_here)
        if depth >= self.max_depth or len(idxs) < self.min_samples_split or len(probs) == 1:
            return _Leaf(probs)

        best = self._best_split(idxs)
        if best is None:
            return _Leaf(probs)  # no feature varies — nothing to split on

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
        """Exhaustive best ``(feature, threshold)`` by weighted Gini, or None.

        Candidates are midpoints between consecutive *distinct* sorted values
        of each feature. Ties keep the earliest candidate, so the induced tree
        depends only on the data.
        """
        n = len(idxs)
        parent_gini = _gini([self._y[i] for i in idxs])
        best_score = parent_gini + 1.0  # anything strictly better wins
        best: tuple[int, float] | None = None

        if self.max_features is None:
            feature_pool = list(range(self._n_features))
        else:
            # Random-forest-style split randomization: draw a fresh subset
            # per node. The shared per-tree RNG keeps each tree deterministic.
            feature_pool = self._rng.sample(range(self._n_features), self.max_features)

        for feature in feature_pool:
            ordered = sorted((self._X[i][feature], i) for i in idxs)
            for pos in range(len(ordered) - 1):
                va, _ = ordered[pos]
                vb, _ = ordered[pos + 1]
                if va == vb:
                    continue  # duplicate values: no midpoint between them
                threshold = (va + vb) / 2.0
                left_ids = [i for i in idxs if self._X[i][feature] <= threshold]
                right_ids = [i for i in idxs if self._X[i][feature] > threshold]
                score = (
                    len(left_ids) * _gini([self._y[i] for i in left_ids])
                    + len(right_ids) * _gini([self._y[i] for i in right_ids])
                ) / n
                if score < best_score:
                    best_score = score
                    best = (feature, threshold)
        return best

    # Fitted-data accessors used by _build; assigned during fit(). Declared as
    # instance attributes here so mypy --strict sees concrete types.
    _X: list[list[float]]
    _y: list[str | int]
