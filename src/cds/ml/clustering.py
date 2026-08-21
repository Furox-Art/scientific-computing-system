"""k-means clustering with k-means++ seeding — pure Python.

Lloyd's algorithm over plain ``list[list[float]]`` data. Seeding uses the
k-means++ recipe (spread initial centres apart with probability proportional
to squared distance) driven by a caller-seeded :class:`random.Random`, so a
fixed ``seed`` reproduces the exact same clustering.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class KMeansResult:
    """Outcome of a k-means fit.

    Attributes:
        labels: Cluster index (``0 .. n_clusters-1``) for every training row.
        centroids: Final centroid coordinates, one row per cluster.
        inertia: Sum of squared distances from each point to its centroid.
        n_iter: Lloyd iterations actually executed (including the final,
            converged one).
    """

    labels: list[int]
    centroids: list[list[float]]
    inertia: float
    n_iter: int


def _squared_distance(a: list[float], b: list[float]) -> float:
    """Squared Euclidean distance between two equal-length vectors."""
    return sum((ai - bi) ** 2 for ai, bi in zip(a, b))


def _validate_rows(X: list[list[float]]) -> None:
    """Reject empty datasets and ragged feature matrices."""
    if not X:
        raise ValueError("X must be non-empty")
    width = len(X[0])
    if any(len(row) != width for row in X):
        raise ValueError("all rows must have the same length")


class KMeans:
    """k-means clustering (Lloyd's algorithm, k-means++ init).

    Assignment ties resolve to the *lowest* cluster index; an empty cluster
    keeps its previous centroid instead of collapsing. Both rules keep runs
    deterministic for a fixed ``seed``.
    """

    def __init__(
        self,
        n_clusters: int,
        *,
        max_iter: int = 300,
        tol: float = 1e-6,
        seed: int | None = None,
    ) -> None:
        """Store hyperparameters; call :meth:`fit` to cluster.

        Args:
            n_clusters: Number of clusters ``k`` (>= 1).
            max_iter: Upper bound on Lloyd iterations (>= 1).
            tol: Convergence threshold on the largest centroid shift (>= 0).
            seed: Seed for the k-means++ RNG; ``None`` uses OS entropy.

        Raises:
            ValueError: if any hyperparameter is out of range.
        """
        if n_clusters < 1:
            raise ValueError("n_clusters must be >= 1")
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        if tol < 0:
            raise ValueError("tol must be >= 0")
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self._centroids: list[list[float]] = []

    def fit(self, X: list[list[float]]) -> KMeansResult:
        """Cluster ``X`` and return labels, centroids, inertia, iterations.

        Args:
            X: Feature rows.

        Returns:
            A frozen :class:`KMeansResult`.

        Raises:
            ValueError: if ``X`` is empty, ragged, or has fewer rows than
                ``n_clusters``.
        """
        _validate_rows(X)
        n = len(X)
        if self.n_clusters > n:
            raise ValueError("n_clusters must not exceed the number of samples")

        centroids = self._init_plus_plus(X)
        dim = len(X[0])
        labels = [0] * n
        n_iter = 0

        for iteration in range(1, self.max_iter + 1):
            sums = [[0.0] * dim for _ in range(self.n_clusters)]
            counts = [0] * self.n_clusters
            for i, x in enumerate(X):
                best_j = self._nearest(x, centroids)
                labels[i] = best_j
                counts[best_j] += 1
                for d in range(dim):
                    sums[best_j][d] += x[d]

            # An empty cluster keeps its previous centre rather than collapsing.
            new_centroids = [
                [s / counts[j] for s in row] if counts[j] > 0 else list(centroids[j])
                for j, row in enumerate(sums)
            ]
            shift = max(
                _squared_distance(new_centroids[j], centroids[j]) for j in range(self.n_clusters)
            )
            centroids = new_centroids
            n_iter = iteration
            if shift <= self.tol:
                break

        self._centroids = centroids
        return KMeansResult(
            labels=labels,
            centroids=centroids,
            inertia=sum(_squared_distance(x, centroids[labels[i]]) for i, x in enumerate(X)),
            n_iter=n_iter,
        )

    def predict(self, x: list[float]) -> int:
        """Index of the nearest fitted centroid for a fresh point.

        Raises:
            ValueError: if called before :meth:`fit` or on a dimension mismatch.
        """
        if not self._centroids:
            raise ValueError("model is not fitted")
        if len(x) != len(self._centroids[0]):
            raise ValueError(f"query must have {len(self._centroids[0])} features")
        return self._nearest(x, self._centroids)

    def _nearest(self, x: list[float], centroids: list[list[float]]) -> int:
        """Argmin over centroids; strict ``<`` keeps the lowest index on ties."""
        best_j = 0
        best_d = _squared_distance(x, centroids[0])
        for j in range(1, len(centroids)):
            d = _squared_distance(x, centroids[j])
            if d < best_d:
                best_d = d
                best_j = j
        return best_j

    def _init_plus_plus(self, X: list[list[float]]) -> list[list[float]]:
        """k-means++ seeding: each next centre is sampled ∝ squared distance."""
        rng = random.Random(self.seed)
        n = len(X)
        centroids = [list(X[rng.randrange(n)])]
        # Squared distance from every point to its nearest chosen centre.
        d2 = [_squared_distance(x, centroids[0]) for x in X]
        for _ in range(1, self.n_clusters):
            total = sum(d2)
            if total == 0.0:
                # All points coincide with existing centres — any pick works;
                # fall back to a uniform draw so duplicate data never hangs us.
                idx = rng.randrange(n)
            else:
                r = rng.random() * total
                cum = 0.0
                idx = n - 1
                # The final cum equals total, and r < total by construction,
                # so the scan always breaks before exhausting d2 under exact
                # arithmetic (same numerical-recipes arc as cds.stats._gser).
                for i, d in enumerate(d2):  # pragma: no branch
                    cum += d
                    if cum >= r:
                        idx = i
                        break
            centroids.append(list(X[idx]))
            d2 = [min(d2[i], _squared_distance(X[i], centroids[-1])) for i in range(n)]
        return centroids
