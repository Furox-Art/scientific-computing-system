# PCA + Preprocessing Tutorial

`cds.ml` ships a from-scratch `PCA` (cyclic Jacobi eigendecomposition —
deterministic, no RNG) alongside sklearn-flavored `StandardScaler` and
`train_test_split`. This tutorial walks the full pipeline: why scaling
matters, how to read explained variance, projecting to 2-D, reconstructing,
and keeping test data out of the fitted statistics.

## 1. Why scaling matters

PCA maximizes variance, so any column measured on a larger scale hijacks the
first component. The demo builds a two-cluster dataset where `sensor_mm` is
in millimeters (×1000 larger than the other features):

```python
from cds.ml import PCA

raw_model = PCA(n_components=5).fit(X)
print(raw_model.explained_variance_ratio_)
# [1.0, 0.0, 0.0, 0.0, 0.0]  <- one column owns ALL the variance
```

The cluster structure is invisible — PC1 just re-states the millimeter column.

## 2. Scale first, then decompose

`StandardScaler` learns per-column means and standard deviations
(zero-variance columns are safely mapped to 0.0, matching sklearn):

```python
from cds.ml import StandardScaler

scaler = StandardScaler().fit(X)
Xs = scaler.transform(X)
model = PCA(n_components=5).fit(Xs)
```

Now the variance is spread across real structure:

```
PC1:  0.7279   cumulative  0.7279
PC2:  0.2481   cumulative  0.9761
PC3:  0.0160   cumulative  0.9921
-> 2 component(s) capture >= 90% of the variance
```

A common recipe: fit with all components, read the cumulative ratios, then
refit with the smallest `n_components` that clears your threshold (90–95% is
typical).

## 3. Project, inspect, reconstruct

`fit` returns a `PCAResult` with `transform` / `inverse_transform`:

```python
model2 = PCA(n_components=2).fit(Xs)
Z = model2.transform(Xs)          # 2-D projections
X_rec = model2.inverse_transform(Z)  # approximate reconstruction
```

Plotting `Z` against the true labels shows PC1 separating the two clusters —
ASCII scatter, no matplotlib needed. The mean reconstruction error
(`|x - reconstruct(x)|`) quantifies exactly how much detail the 2-D view kept.

## 4. Train/test discipline

Fit the scaler and PCA on the training split only, then reuse both to
transform held-out data. This keeps test-set statistics from leaking into the
pipeline:

```python
from cds.ml import train_test_split

X_train, X_test, _, _ = train_test_split(X, y, test_size=0.25, seed=42)
scaler_t = StandardScaler().fit(X_train)
pca_t = PCA(n_components=2).fit(scaler_t.transform(X_train))
Z_test = pca_t.transform(scaler_t.transform(X_test))
```

The reconstruction error on the test split should be close to the training
one — a quick sanity check that the projection generalizes.

Run the full demo:

```bash
python examples/pca_demo.py
```
