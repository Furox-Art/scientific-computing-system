"""Random Forest classifier — bagged CART trees with split randomization.

Averages many :class:`~cds.ml.tree.DecisionTreeClassifier` instances, each
grown on a bootstrap resample of the rows and restricted to a random feature
subset at every split. Pure Python and deterministic given a seed.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Literal

from cds.ml.metrics import Label
from cds.ml.tree import DecisionTreeClassifier

MaxFeatures = Literal["sqrt", "log2"] | int | None


class RandomForestClassifier:
    """Bagged CART forest over numeric features.

    Each tree trains on a bootstrap resample (rows drawn with replacement)
    and considers only ``max_features`` randomly chosen columns per split.
    Prediction averages the per-tree class probabilities; ties resolve to the
    earliest-encountered label, matching :meth:`DecisionTreeClassifier.predict`.

    Args:
        n_trees: Number of trees in the forest (>= 1).
        max_depth: Maximum depth of each tree (>= 0), passed through.
        min_samples_split: Minimum rows for a split (>= 2), passed through.
        max_features: Features considered per split — ``"sqrt"`` (default),
            ``"log2"``, an explicit count in ``[1, n_features]``, or ``None``
            for all features.
        seed: RNG seed; ``None`` uses OS entropy. Controls both bootstrap
            resamples and per-tree split randomization.

    Raises:
        ValueError: if ``n_trees < 1``, ``max_features`` is an out-of-range
            integer, or an unrecognized string.
    """

    def __init__(
        self,
        *,
        n_trees: int = 100,
        max_depth: int = 5,
        min_samples_split: int = 2,
        max_features: MaxFeatures = "sqrt",
        seed: int | None = None,
    ) -> None:
        """Validate hyperparameters and create an unfitted forest."""
        if n_trees < 1:
            raise ValueError("n_trees must be >= 1")
        if isinstance(max_features, int) and max_features < 1:
            raise ValueError("max_features must be >= 1")
        if isinstance(max_features, str) and max_features not in ("sqrt", "log2"):
            raise ValueError("max_features must be 'sqrt', 'log2', an int, or None")
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features: MaxFeatures = max_features
        self.seed = seed
        self._trees: list[DecisionTreeClassifier] = []
        self._fitted = False

    def fit(self, X: list[list[float]], y: Sequence[Label]) -> RandomForestClassifier:
        """Grow the forest on ``(X, y)``.

        A master RNG seeded from ``seed`` derives each tree's own seed and its
        bootstrap row sample, so a fixed seed reproduces the exact forest.

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: if ``X`` is empty, lengths mismatch, rows are ragged,
                or an integer ``max_features`` exceeds the feature count.
        """
        if not X:
            raise ValueError("X must be non-empty")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        width = len(X[0])
        if any(len(row) != width for row in X):
            raise ValueError("all rows must have the same length")

        m = self._resolve_max_features(width)
        rng = random.Random(self.seed)
        trees: list[DecisionTreeClassifier] = []
        for _ in range(self.n_trees):
            tree_seed = rng.randrange(2**32)
            boot = [rng.randrange(len(X)) for _ in range(len(X))]
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=m,
                seed=tree_seed,
            )
            tree.fit([X[i] for i in boot], [y[i] for i in boot])
            trees.append(tree)

        self._trees = trees
        self._fitted = True
        return self

    def predict_proba(self, x: list[float]) -> dict[Label, float]:
        """Forest-averaged class-membership probabilities for ``x``.

        Labels appear in first-encounter order across trees. The values sum
        to 1.0 because every tree's distribution does.

        Raises:
            ValueError: if called before :meth:`fit` or on a feature-count
                mismatch.
        """
        if not self._fitted or not self._trees:
            raise ValueError("model is not fitted")
        totals: dict[Label, float] = {}
        for tree in self._trees:
            for label, p in tree.predict_proba(x).items():
                totals[label] = totals.get(label, 0.0) + p
        n = float(len(self._trees))
        return {label: total / n for label, total in totals.items()}

    def predict(self, x: list[float]) -> Label:
        """Most probable class for ``x`` (ties → earliest-seen label)."""
        best_label: Label = ""
        best_p = -1.0
        for label, p in self.predict_proba(x).items():
            if p > best_p:
                best_p = p
                best_label = label
        return best_label

    def _resolve_max_features(self, width: int) -> int:
        """Translate the ``max_features`` spec into a concrete count.

        Raises:
            ValueError: if an explicit integer count is out of range.
        """
        spec = self.max_features
        if spec is None:
            return width
        if isinstance(spec, int):
            if spec > width:
                raise ValueError("max_features must be <= the number of features")
            return spec
        if spec == "sqrt":
            return max(1, round(math.sqrt(width)))
        return max(1, round(math.log2(width)))
