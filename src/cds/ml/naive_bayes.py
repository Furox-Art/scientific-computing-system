"""Gaussian Naive Bayes classifier — pure Python.

Models each feature as conditionally independent per class with a
maximum-likelihood Gaussian: the per-class *population* variance
(``ddof=0``) and mean are used. Scoring runs entirely in log space and the
posterior comes from a softmax over log-joints, so extreme inputs cannot
overflow or underflow to zero.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from cds.ml.metrics import Label

_LOG_2PI = math.log(2.0 * math.pi)

#: Absolute variance floor used only when the global feature variance is 0.0
#: (i.e. every training value of every feature is identical), which would
#: otherwise produce ``epsilon == 0.0`` and a division by zero.
_ABSOLUTE_FLOOR = 1e-12


class GaussianNaiveBayes:
    """Gaussian Naive Bayes over plain numeric feature rows.

    Variance smoothing rule
    -----------------------
    For every (class, feature) pair a *population* variance (``ddof=0``) is
    estimated. Any variance that equals ``0.0`` is replaced by::

        epsilon = var_smoothing * max_j Var_global(feature j)

    where ``Var_global(feature j)`` is the population variance of feature
    ``j`` computed over **all** training rows regardless of class, and the
    maximum is taken over the features. ``var_smoothing`` defaults to
    ``1e-9``. When that maximum is itself ``0.0`` (the whole training matrix
    is constant), ``epsilon`` falls back to the absolute floor ``1e-12``.
    The resulting value is exposed as :attr:`epsilon_` after :meth:`fit`.
    """

    def __init__(self, *, var_smoothing: float = 1e-9) -> None:
        """Store hyperparameters; call :meth:`fit` before predicting.

        Args:
            var_smoothing: Fraction of the largest global feature variance
                used as the zero-variance floor (see above). Must be > 0.

        Raises:
            ValueError: if ``var_smoothing`` is not strictly positive.
        """
        if var_smoothing <= 0.0:
            raise ValueError("var_smoothing must be > 0")
        self.var_smoothing = var_smoothing
        self._fitted = False
        self._classes: list[Label] = []
        self._priors: dict[Label, float] = {}
        self._means: dict[Label, list[float]] = {}
        self._variances: dict[Label, list[float]] = {}
        self._n_features = 0
        #: Effective zero-variance floor; assigned during :meth:`fit`.
        self.epsilon_ = 0.0

    def fit(self, X: list[list[float]], y: Sequence[Label]) -> GaussianNaiveBayes:
        """Estimate per-class priors, means and population variances.

        Classes are indexed in first-appearance order, keeping all outputs
        deterministic for repeated fits on the same data.

        Args:
            X: Feature rows.
            y: Class labels (strings or ints), one per row.

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: if ``X`` is empty, lengths mismatch, or rows are
                ragged.
        """
        if not X:
            raise ValueError("X must be non-empty")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        width = len(X[0])
        if any(len(row) != width for row in X):
            raise ValueError("all rows must have the same length")

        n = len(X)
        counts: dict[Label, int] = {}
        for label in y:
            counts[label] = counts.get(label, 0) + 1
        self._classes = list(counts)
        self._priors = {label: c / n for label, c in counts.items()}
        self._n_features = width

        # Global (class-agnostic) population variance of each feature.
        global_means = [sum(row[j] for row in X) / n for j in range(width)]
        global_vars = [sum((row[j] - global_means[j]) ** 2 for row in X) / n for j in range(width)]
        peak_variance = max(global_vars, default=0.0)
        self.epsilon_ = (
            self.var_smoothing * peak_variance if peak_variance > 0.0 else _ABSOLUTE_FLOOR
        )

        for label in self._classes:
            rows = [row for row, lbl in zip(X, y) if lbl == label]
            k = len(rows)
            means: list[float] = []
            variances: list[float] = []
            for j in range(width):
                column = [row[j] for row in rows]
                mean = sum(column) / k
                variance = sum((value - mean) ** 2 for value in column) / k
                if variance == 0.0:
                    variance = self.epsilon_
                means.append(mean)
                variances.append(variance)
            self._means[label] = means
            self._variances[label] = variances

        self._fitted = True
        return self

    def predict_proba(self, x: list[float]) -> dict[Label, float]:
        """Posterior class probabilities for one sample.

        Computed as a softmax over ``log(prior) + Σ_j log N(x_j | μ_cj, σ²_cj)``,
        returned with keys in first-appearance class order.

        Raises:
            ValueError: if called before :meth:`fit` or on a feature-count
                mismatch.
        """
        return self.predict_proba_one(x)

    def predict_proba_one(self, x: list[float]) -> dict[Label, float]:
        """Posterior class probabilities for a single sample.

        Raises:
            ValueError: if called before :meth:`fit` or ``len(x)`` differs
                from the number of fitted features.
        """
        if not self._fitted:
            raise ValueError("model is not fitted")
        if len(x) != self._n_features:
            raise ValueError(f"query must have {self._n_features} features")

        log_joints: dict[Label, float] = {}
        for label in self._classes:
            log_joint = math.log(self._priors[label])
            means = self._means[label]
            variances = self._variances[label]
            for j, value in enumerate(x):
                diff = value - means[j]
                log_joint -= 0.5 * (_LOG_2PI + math.log(variances[j]))
                log_joint -= (diff * diff) / (2.0 * variances[j])
            log_joints[label] = log_joint

        peak = max(log_joints.values())
        unscaled = {label: math.exp(log_joint - peak) for label, log_joint in log_joints.items()}
        total = sum(unscaled.values())
        return {label: weight / total for label, weight in unscaled.items()}

    def predict(self, x: list[float]) -> Label:
        """Most probable class for ``x`` (ties resolve to the earliest-seen
        class).

        Raises:
            ValueError: if called before :meth:`fit` or on a feature-count
                mismatch.
        """
        best_label: Label = ""
        best_p = -1.0
        for label, p in self.predict_proba_one(x).items():
            if p > best_p:
                best_p = p
                best_label = label
        return best_label
