"""Principal Component Analysis via cyclic Jacobi eigenvalue iteration.

The covariance matrix of the centered data is symmetric, so the classic
Jacobi rotation method applies: repeatedly annihilate the largest off-diagonal
entry with a Givens rotation until the matrix is diagonal. Robust, pure
Python, and deterministic — no RNG anywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PCAResult:
    """Fitted PCA model.

    Attributes:
        components_: principal axes as rows, length ``n_components``
        explained_variance_: variance captured by each kept component
        explained_variance_ratio_: fraction of total variance per component
        mean_: per-column mean subtracted before projection
    """

    components_: list[list[float]]
    explained_variance_: list[float]
    explained_variance_ratio_: list[float]
    mean_: list[float]

    def transform(self, X: list[list[float]]) -> list[list[float]]:
        """Project rows onto the kept components.

        Raises:
            ValueError: on column-count mismatch with the fitted data.
        """
        if any(len(row) != len(self.mean_) for row in X):
            raise ValueError(f"rows must have {len(self.mean_)} features to match the fitted data")
        return [
            [
                sum((row[d] - self.mean_[d]) * comp[d] for d in range(len(self.mean_)))
                for comp in self.components_
            ]
            for row in X
        ]

    def inverse_transform(self, Z: list[list[float]]) -> list[list[float]]:
        """Reconstruct (approximate) original-space rows from projections.

        Raises:
            ValueError: if ``Z`` width differs from the number of components.
        """
        if any(len(z) != len(self.components_) for z in Z):
            raise ValueError(f"rows must have {len(self.components_)} components to invert")
        n_features = len(self.mean_)
        return [
            [
                self.mean_[d] + sum(z[k] * self.components_[k][d] for k in range(len(z)))
                for d in range(n_features)
            ]
            for z in Z
        ]


def _jacobi_eigen(
    matrix: list[list[float]], *, tol: float = 1e-12, max_sweeps: int = 100
) -> tuple[list[float], list[list[float]]]:
    """Eigenvalues and eigenvectors of a symmetric matrix (cyclic Jacobi).

    Returns ``(eigenvalues, eigenvectors)`` where eigenvectors are the COLUMNS
    of the returned matrix, in arbitrary order (callers sort).
    """
    n = len(matrix)
    a = [row[:] for row in matrix]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    # Quadratic convergence retires every off-diagonal long before max_sweeps
    # for any well-conditioned symmetric input, so natural exhaustion of the
    # sweep loop is unreachable (same numerical-recipes arc as cds.stats._gser).
    for _ in range(max_sweeps):  # pragma: no branch
        off = math.sqrt(sum(a[i][j] ** 2 for i in range(n) for j in range(n) if i != j))
        if off < tol:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(a[p][q]) < 1e-30:
                    continue  # already annihilated
                theta = 0.5 * math.atan2(2.0 * a[p][q], a[q][q] - a[p][p])
                cos = math.cos(theta)
                sin = math.sin(theta)
                # Apply the Givens rotation to rows/cols p,q of A and columns p,q of V.
                for k in range(n):
                    akp = a[k][p]
                    akq = a[k][q]
                    a[k][p] = cos * akp - sin * akq
                    a[k][q] = sin * akp + cos * akq
                for k in range(n):
                    apk = a[p][k]
                    aqk = a[q][k]
                    a[p][k] = cos * apk - sin * aqk
                    a[q][k] = sin * apk + cos * aqk
                for k in range(n):
                    vkp = v[k][p]
                    vkq = v[k][q]
                    v[k][p] = cos * vkp - sin * vkq
                    v[k][q] = sin * vkp + cos * vkq

    return [a[i][i] for i in range(n)], v


class PCA:
    """Principal component analysis for numeric feature matrices.

    Fits all principal components; ``n_components`` selects how many are kept
    for :meth:`PCAResult.transform` / :meth:`PCAResult.inverse_transform`.
    """

    def __init__(self, n_components: int) -> None:
        """Store how many components to keep.

        Args:
            n_components: number of leading components retained (>= 1).

        Raises:
            ValueError: if ``n_components < 1``.
        """
        if n_components < 1:
            raise ValueError("n_components must be >= 1")
        self.n_components = n_components

    def fit(self, X: list[list[float]]) -> PCAResult:
        """Center ``X``, diagonalize its covariance matrix, keep top components.

        Raises:
            ValueError: if ``X`` has fewer than 2 rows, is ragged, or has
                fewer columns than ``n_components``.
        """
        if not X or len(X) < 2:
            raise ValueError("need at least 2 samples to estimate covariance")
        width = len(X[0])
        if any(len(row) != width for row in X):
            raise ValueError("all rows must have the same length")
        if self.n_components > width:
            raise ValueError("n_components must not exceed the feature count")

        n = float(len(X))
        col_means = [sum(row[d] for row in X) / n for d in range(width)]
        centered = [[row[d] - col_means[d] for d in range(width)] for row in X]

        cov = [
            [sum(row[i] * row[j] for row in centered) / (n - 1.0) for j in range(width)]
            for i in range(width)
        ]

        eigenvalues, eigenvector_columns = _jacobi_eigen(cov)

        order = sorted(range(width), key=lambda i: -eigenvalues[i])
        kept = order[: self.n_components]
        total = sum(eigenvalues)
        ratio_base = total if total > 0 else 1.0

        return PCAResult(
            components_=[[eigenvector_columns[r][c] for r in range(width)] for c in kept],
            explained_variance_=[eigenvalues[c] for c in kept],
            explained_variance_ratio_=[max(eigenvalues[c], 0.0) / ratio_base for c in kept],
            mean_=col_means,
        )

    def fit_transform(self, X: list[list[float]]) -> tuple[PCAResult, list[list[float]]]:
        """Fit on ``X`` and return ``(model, projected_rows)``."""
        model = self.fit(X)
        return model, model.transform(X)
