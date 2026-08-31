"""
Demo: end-to-end PCA with preprocessing — scaling, variance analysis,
2-D projection, reconstruction, and train/test discipline.
Everything is 100% Pure Python.
"""

import random

from cds.data_analysis import plot_bar
from cds.ml import PCA, PCAResult, StandardScaler, train_test_split

SEED = 42
FEATURE_NAMES = ["sensor_mm", "load_kg", "temp_diff", "noise", "bias_const"]


def make_dataset(n_per_cluster: int = 80) -> tuple[list[list[float]], list[float]]:
    """Two latent clusters embedded in 5 features with deliberate pitfalls.

    - sensor_mm is measured in millimeters (huge scale vs the rest)
    - noise is pure random information
    - bias_const is a zero-variance column
    """
    rng = random.Random(SEED)
    X: list[list[float]] = []
    y: list[float] = []
    for cluster in (0, 1):
        center = -1.0 if cluster == 0 else 1.0
        for _ in range(n_per_cluster):
            latent = center + rng.gauss(0.0, 0.8)
            row = [
                1000.0 * (latent + 0.3 * rng.gauss(0.0, 1.0)),
                1.5 * latent + 0.3 * rng.gauss(0.0, 1.0),
                -latent + 0.3 * rng.gauss(0.0, 1.0),
                rng.gauss(0.0, 1.0),
                7.0,
            ]
            X.append(row)
            y.append(float(cluster))
    return X, y


def ascii_scatter(points: list[list[float]], labels: list[float]) -> str:
    """Bin 2-D points into a character grid: cluster 0 -> 'A', 1 -> 'B'."""
    width, height = 56, 16
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_span = (x_max - x_min) or 1.0
    y_span = (y_max - y_min) or 1.0
    grid = [[" "] * width for _ in range(height)]
    for (px, py), label in zip(points, labels):
        col = int((px - x_min) / x_span * (width - 1))
        row = height - 1 - int((py - y_min) / y_span * (height - 1))
        mark = "A" if label == 0 else "B"
        if grid[row][col] != " ":
            mark = "#"
        grid[row][col] = mark
    return "\n".join("".join(r) for r in grid)


def explained_variance_report(model: PCAResult, prefix: str) -> None:
    ratios = model.explained_variance_ratio_
    bars = {f"PC{i + 1}": round(r, 4) for i, r in enumerate(ratios)}
    print(plot_bar(bars, title=f"{prefix}: explained variance ratio"))
    cumulative = 0.0
    for i, r in enumerate(ratios):
        cumulative += r
        print(f"  PC{i + 1}: {r:7.4f}   cumulative {cumulative:7.4f}")
    k90 = next(i + 1 for i, _ in enumerate(ratios) if sum(ratios[: i + 1]) >= 0.90)
    print(f"  -> {k90} component(s) capture >= 90% of the variance")


def mean_abs_error(original: list[list[float]], reconstructed: list[list[float]]) -> float:
    diffs = [
        abs(original[i][d] - reconstructed[i][d])
        for i in range(len(original))
        for d in range(len(original[i]))
    ]
    return sum(diffs) / len(diffs)


def run_demo() -> None:
    X, y = make_dataset()
    n_samples, n_features = len(X), len(X[0])
    print("--- End-to-End PCA + Preprocessing Demo ---")
    print(f"Dataset: {n_samples} samples x {n_features} features, two hidden clusters")
    print(f"Features: {', '.join(FEATURE_NAMES)}")

    print("\n[1] Why scaling matters — PCA on RAW data")
    raw_model = PCA(n_components=n_features).fit(X)
    explained_variance_report(raw_model, "RAW")
    print("  The millimeter-scale column dominates; cluster structure is invisible.")

    print("\n[2] StandardScaler -> PCA on SCALED data")
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    scaled_model = PCA(n_components=n_features).fit(Xs)
    explained_variance_report(scaled_model, "SCALED")
    print("  Zero-variance 'bias_const' became a flat 0 column and contributes nothing.")

    print("\n[3] Project onto the top 2 components")
    model2 = PCA(n_components=2).fit(Xs)
    Z = model2.transform(Xs)
    print(ascii_scatter(Z, y))
    print("  A/B = true clusters — PC1 alone separates them.")

    print("\n[4] Round-trip: 2-D projection -> inverse_transform")
    X_rec = model2.inverse_transform(Z)
    print(f"  mean |x - reconstruct(x)| on scaled data: {mean_abs_error(Xs, X_rec):.4f}")

    print("\n[5] Train/test discipline — fit on train only, then transform test")
    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.25, seed=SEED)
    scaler_t = StandardScaler().fit(X_train)
    pca_t = PCA(n_components=2).fit(scaler_t.transform(X_train))
    Z_test = pca_t.transform(scaler_t.transform(X_test))
    X_test_rec = pca_t.inverse_transform(Z_test)
    test_err = mean_abs_error(scaler_t.transform(X_test), X_test_rec)
    print(f"  test samples: {len(X_test)}, reconstruction error: {test_err:.4f}")
    print("  No statistics leak from the test set into the fitted pipeline.")


if __name__ == "__main__":
    run_demo()
