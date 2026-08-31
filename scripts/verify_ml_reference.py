"""Verify ``cds.ml`` estimators against scikit-learn reference values.

One command:

    python scripts/verify_ml_reference.py

The dataset is generated in-process from a fixed seed, so the run is fully
deterministic. Reference values were computed ONCE with scikit-learn 1.9.0
on the same generated data and are hard-coded below, keeping this script
dependency-free (pure CDS). To re-derive the reference values, see the
snippet at the bottom of ``docs/ml_reference.md``.

Exits 0 when every metric lands within its tolerance, 1 otherwise.
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from math import comb

from cds.ml import (
    PCA,
    DecisionTreeClassifier,
    KMeans,
    LinearRegression,
    LogisticRegression,
    StandardScaler,
    accuracy,
    r2_score,
)

SEED = 42
N_SAMPLES = 200

# Reference outputs computed with scikit-learn 1.9.0 (see docs/ml_reference.md).
REFERENCE = {
    "logreg_accuracy": 0.930000,
    "tree_accuracy": 0.980000,
    "kmeans_ari": 0.738293,
    "pca_var_ratio_2": 0.712253,
    "linreg_r2": 0.981701975,
}

TOLERANCE = {
    "logreg_accuracy": 0.02,  # gradient descent vs LBFGS optimum
    "tree_accuracy": 0.02,  # tie-breaking on equal-Gini splits differs
    "kmeans_ari": 0.03,  # k-means++ seeding differs from sklearn's
    "pca_var_ratio_2": 1e-6,  # Jacobi vs LAPACK eigensolver numerics
    "linreg_r2": 1e-9,  # both are exact closed-form solves
}


def make_dataset(seed: int = SEED) -> tuple[list[list[float]], list[int], list[float]]:
    """Two Gaussian blobs (binary labels) plus a linear regression target.

    Feature scales are comparable; the regression target is a noisy linear
    combination of the features, so a correct OLS fit scores near-perfect r2.
    """
    rng = random.Random(seed)
    X: list[list[float]] = []
    y: list[int] = []
    y_reg: list[float] = []
    centers = ((-1.0, -0.5, 0.5, 1.0), (1.0, 0.5, -0.5, -1.0))
    for i in range(N_SAMPLES):
        label = i % 2
        row = [centers[label][d] + rng.gauss(0.0, 1.0) for d in range(4)]
        X.append(row)
        y.append(label)
        y_reg.append(3.0 * row[0] - 2.0 * row[1] + 0.5 * row[2] + rng.gauss(0.0, 0.5))
    return X, y, y_reg


def adjusted_rand_index(a: list[int], b: list[int]) -> float:
    """Hubert-Arabie adjusted Rand index (same definition as sklearn)."""
    table: Counter[tuple[int, int]] = Counter(zip(a, b))
    sum_comb_c = sum(comb(n, 2) for n in table.values())
    row_sums: Counter[int] = Counter()
    col_sums: Counter[int] = Counter()
    for (ai, bi), n in table.items():
        row_sums[ai] += n
        col_sums[bi] += n
    sum_comb_a = sum(comb(n, 2) for n in row_sums.values())
    sum_comb_b = sum(comb(n, 2) for n in col_sums.values())
    total_comb = comb(len(a), 2)
    expected = sum_comb_a * sum_comb_b / total_comb
    max_index = 0.5 * (sum_comb_a + sum_comb_b)
    if max_index == expected:
        return 1.0
    return (sum_comb_c - expected) / (max_index - expected)


def run_cds() -> dict[str, float]:
    """Run every estimator once and collect the measured metrics."""
    X, y, y_reg = make_dataset()

    Xs = StandardScaler().fit_transform(X)

    logreg = LogisticRegression(lr=0.5, epochs=2000).fit(Xs, y)
    logreg_acc = accuracy(y, [logreg.predict(row) for row in Xs])

    tree = DecisionTreeClassifier(max_depth=5, seed=SEED).fit(X, y)
    tree_acc = accuracy(y, [tree.predict(row) for row in X])

    kmeans = KMeans(2, seed=SEED).fit(X)
    kmeans_ari = adjusted_rand_index(kmeans.labels, y)

    pca = PCA(n_components=2).fit(Xs)
    pca_ratio = sum(pca.explained_variance_ratio_)

    linreg = LinearRegression().fit(X, y_reg)
    linreg_r2 = r2_score(y_reg, [linreg.predict(row) for row in X])

    return {
        "logreg_accuracy": logreg_acc,
        "tree_accuracy": tree_acc,
        "kmeans_ari": kmeans_ari,
        "pca_var_ratio_2": pca_ratio,
        "linreg_r2": linreg_r2,
    }


DESCRIPTIONS = {
    "logreg_accuracy": "LogisticRegression accuracy (scaled features)",
    "tree_accuracy": "DecisionTreeClassifier accuracy (max_depth=5)",
    "kmeans_ari": "KMeans(k=2) adjusted Rand index vs true labels",
    "pca_var_ratio_2": "PCA cumulative explained variance (2 components)",
    "linreg_r2": "LinearRegression r2 on noisy linear target",
}


def main() -> int:
    measured = run_cds()
    print(f"# cds.ml vs scikit-learn reference (seed={SEED}, n={N_SAMPLES})")
    print()
    print("| Check | CDS | Reference (sklearn 1.9.0) | |diff| | Tol | Status |")
    print("|---|---|---|---|---|---|")
    failed = False
    for key in REFERENCE:
        got = measured[key]
        ref = REFERENCE[key]
        tol = TOLERANCE[key]
        diff = abs(got - ref)
        ok = diff <= tol
        failed |= not ok
        status = "PASS" if ok else "FAIL"
        print(
            f"| {DESCRIPTIONS[key]} | {got:.6f} | {ref:.6f} | {diff:.2e} | {tol:.0e} | {status} |"
        )
    print()
    if failed:
        print("RESULT: FAIL — at least one metric drifted outside its tolerance.")
        return 1
    print("RESULT: PASS — all estimators agree with the reference values.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
