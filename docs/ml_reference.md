# ML Reference Values

`cds.ml` estimators are written from scratch in pure Python. To prove they
are *correct*, not just runnable, `scripts/verify_ml_reference.py` runs
each v1.6 estimator on a fixed seeded dataset and compares against reference
outputs computed with **scikit-learn 1.9.0**:

| Check | CDS | Reference (sklearn 1.9.0) | \|diff\| | Tol | Status |
|---|---|---|---|---|---|
| LogisticRegression accuracy (scaled features) | 0.930000 | 0.930000 | 0.00e+00 | 2e-02 | PASS |
| DecisionTreeClassifier accuracy (max_depth=5) | 0.980000 | 0.980000 | 0.00e+00 | 2e-02 | PASS |
| KMeans(k=2) adjusted Rand index vs true labels | 0.738285 | 0.738293 | 8.15e-06 | 3e-02 | PASS |
| PCA cumulative explained variance (2 components) | 0.712253 | 0.712253 | 2.80e-07 | 1e-06 | PASS |
| LinearRegression r2 on noisy linear target | 0.981702 | 0.981702 | 3.21e-10 | 1e-09 | PASS |

Reproduce with one command (no dependencies beyond CDS itself):

```bash
python scripts/verify_ml_reference.py
```

The dataset is generated in-process from `seed=42` (200 samples, 4 features:
two Gaussian blobs with binary labels, plus a noisy linear regression
target), so every run is bit-for-bit deterministic.

## How the reference values were made

Each metric was computed once with scikit-learn 1.9.0 on the *identical*
generated data and hard-coded into the script, keeping the verification
dependency-free:

```python
# Reference derivation (requires scikit-learn) — run once, then freeze the
# numbers into scripts/verify_ml_reference.py:REFERENCE.
from scripts.verify_ml_reference import make_dataset  # seed=42

X, y, y_reg = make_dataset()

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, adjusted_rand_score, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

Xs = StandardScaler().fit_transform(X)

logreg = LogisticRegression(max_iter=2000).fit(Xs, y)
tree = DecisionTreeClassifier(max_depth=5, random_state=42).fit(X, y)
km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(X)
pca = PCA(n_components=2).fit(Xs)
lin = LinearRegression().fit(X, y_reg)

print(accuracy_score(y, logreg.predict(Xs)))  # 0.930000
print(accuracy_score(y, tree.predict(X)))  # 0.980000
print(adjusted_rand_score(y, km.labels_))  # 0.738293
print(pca.explained_variance_ratio_.sum())  # 0.712253
print(r2_score(y_reg, lin.predict(X)))  # 0.981701975
```

## Honest differences between CDS and sklearn

The numbers agree, but the implementations get there differently. Where
behavior can legitimately differ:

- **LogisticRegression**: CDS uses fixed-learning-rate full-batch gradient
  descent (`lr=0.5`, `epochs=2000`); sklearn uses LBFGS. Both land on the
  same decision boundary here, but on barely-separable data the optima (and
  therefore a handful of predictions) can differ. Tolerance: ±0.02 accuracy.
- **DecisionTreeClassifier**: both are greedy Gini CART. Candidates tied on
  Gini improvement are broken by scan order, which differs between the two
  implementations; accuracy-level agreement is expected, identical trees are
  not.
- **KMeans**: CDS uses k-means++ seeding from a seeded RNG, assigns
  distance ties to the *lowest* cluster index, and keeps the previous
  centroid for an empty cluster. sklearn runs `n_init=10` restarts and
  breaks ties differently. Both converge to the same basin on this dataset
  (ARI difference ≈ 8×10⁻⁶), but label *indices* are arbitrary, compare
  with ARI, never raw labels.
- **PCA**: CDS diagonalizes the covariance matrix with cyclic Jacobi
  rotations; sklearn uses LAPACK SVD. Explained-variance ratios are
  sign-invariant and match to ~10⁻⁷; individual component signs may be
  flipped between the two (a valid PCA convention difference).
- **LinearRegression**: both solve the normal equations in closed form;
  agreement is at floating-point rounding level (~10⁻¹⁰).

## When the table needs refreshing

Update the frozen numbers only when the dataset generator, an estimator's
algorithm, or the pinned sklearn version changes, and always re-run the
full script afterwards. If a check starts failing, treat it as a regression
signal first, not a tolerance problem.
