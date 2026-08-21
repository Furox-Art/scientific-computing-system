"""Binary logistic regression trained by full-batch gradient descent.

Loss is binary cross-entropy computed in a numerically stable form
(``softplus(z) - y·z`` with ``softplus(z) = log1p(exp(-|z|)) + max(z, 0)``),
so no probability clipping is needed and the loss never overflows even for
saturated logits. Optional L2 regularisation shrinks weights but never the
bias term.
"""

from __future__ import annotations

import math


def _sigmoid(z: float) -> float:
    """Numerically stable logistic sigmoid (argument kept non-positive)."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _softplus(z: float) -> float:
    """Stable ``log(1 + e^z)``: exact branch-free form via ``log1p``/``abs``."""
    return math.log1p(math.exp(-abs(z))) + max(z, 0.0)


class LogisticRegression:
    """Binary classifier with a linear decision boundary.

    Labels must be exactly 0 and 1. Training minimises BCE (plus an optional
    L2 penalty on weights) with fixed-learning-rate gradient descent for a
    fixed number of epochs — fully deterministic, no RNG involved.
    """

    def __init__(
        self,
        *,
        lr: float = 0.1,
        epochs: int = 1000,
        l2: float = 0.0,
    ) -> None:
        """Store hyperparameters; call :meth:`fit` before predicting.

        Args:
            lr: Gradient-descent learning rate (> 0).
            epochs: Number of full-batch passes over the data (>= 1).
            l2: L2 penalty coefficient applied to weights, not bias (>= 0).

        Raises:
            ValueError: if any hyperparameter is out of range.
        """
        if lr <= 0:
            raise ValueError("lr must be positive")
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        if l2 < 0:
            raise ValueError("l2 must be >= 0")
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self._fitted = False
        self.weights: list[float] = []
        self.bias = 0.0
        self.loss_history: list[float] = []

    def fit(self, X: list[list[float]], y: list[int]) -> LogisticRegression:
        """Fit weights by gradient descent; records per-epoch loss history.

        Args:
            X: Feature rows.
            y: Binary labels — every entry must be 0 or 1, and both classes
                must appear.

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: if ``X`` is empty, lengths mismatch, rows are ragged,
                labels fall outside {0, 1}, or only one class is present.
        """
        if not X:
            raise ValueError("X must be non-empty")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        width = len(X[0])
        if any(len(row) != width for row in X):
            raise ValueError("all rows must have the same length")
        if any(label not in (0, 1) for label in y):
            raise ValueError("y must contain only 0 and 1")
        if set(y) != {0, 1}:
            raise ValueError("y must contain both classes")

        self._fitted = True
        n = float(len(X))
        self.weights = [0.0] * width
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.epochs):
            # Forward pass: probabilities and stable BCE loss.
            probs: list[float] = []
            loss = 0.0
            for row, label in zip(X, y):
                z = sum(w * xi for w, xi in zip(self.weights, row)) + self.bias
                probs.append(_sigmoid(z))
                loss += _softplus(z) - label * z
            loss = loss / n
            # Unconditional add is safe: the term vanishes when l2 == 0.
            loss += self.l2 * sum(w * w for w in self.weights) / 2.0
            self.loss_history.append(loss)

            # Gradients of BCE w.r.t. w and b (+ L2 term on weights only).
            grad_w = [
                sum((probs[i] - y[i]) * row[d] for i, row in enumerate(X)) / n
                + self.l2 * self.weights[d]
                for d in range(width)
            ]
            grad_b = sum(p - t for p, t in zip(probs, y)) / n
            for d in range(width):
                self.weights[d] -= self.lr * grad_w[d]
            self.bias -= self.lr * grad_b
        return self

    def predict_proba(self, x: list[float]) -> float:
        """P(class = 1) for one feature row.

        Raises:
            ValueError: if called before :meth:`fit` or on a feature-count
                mismatch.
        """
        if not self._fitted:
            raise ValueError("model is not fitted")
        if len(x) != len(self.weights):
            raise ValueError(f"query must have {len(self.weights)} features")
        z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return _sigmoid(z)

    def predict(self, x: list[float]) -> int:
        """Predicted class: 1 when P(class = 1) >= 0.5, else 0."""
        return 1 if self.predict_proba(x) >= 0.5 else 0
