# Advanced ML Tutorial — PCA, Scaling, and Classic Models

v1.6.0 rounds out `cds.ml` with the classic scikit-learn-style toolkit:
preprocessing, dimensionality reduction, and four from-scratch estimators.
Everything is pure Python, deterministic, and typed.

## 1. StandardScaler + train_test_split

```python
from cds.ml import StandardScaler, train_test_split

X = [[1.0, 100.0], [2.0, 200.0], [3.0, 150.0], [4.0, 50.0],
     [5.0, 120.0], [6.0, 90.0], [7.0, 180.0], [8.0, 60.0]]
y = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25, seed=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)      # same statistics, no refit
```

Zero-variance columns map to `0.0` instead of dividing by zero, and
`inverse_transform` undoes everything.

## 2. PCA (cyclic Jacobi eigen-solver)

```python
from cds.ml import PCA

model = PCA(n_components=1).fit(X_train)
Z = model.transform(X_train)             # projected rows
ratio = model.explained_variance_ratio_[0]
print(f"PC1 keeps {ratio:.0%} of the variance")
```

With all components kept, `inverse_transform(transform(X))` reconstructs the
data to machine precision; with fewer, it is a denoised approximation.

## 3. Linear & Logistic Regression

Closed-form OLS via the normal equations:

```python
from cds.ml import LinearRegression

reg = LinearRegression().fit(X_train, [r[0] * 2 + r[1] * 0.01 for r in X_train])
print(reg.predict([2.5, 110.0]))
print(reg.score(X_test, [r[0] * 2 for r in X_test]))   # R²
```

Binary logistic regression by gradient descent:

```python
from cds.ml import LogisticRegression

clf = LogisticRegression(lr=0.3, epochs=500).fit(X_train_s, y_train)
print(clf.predict_proba(X_test_s[0]))
```

## 4. k-NN, k-Means, Decision Tree

```python
from cds.ml import DecisionTreeClassifier, KMeans, KNeighborsClassifier

knn = KNeighborsClassifier(k=3).fit(X_train_s, y_train)
tree = DecisionTreeClassifier(max_depth=3).fit(X_train_s, y_train)

km_model = KMeans(2, seed=0)
res = km_model.fit(X_train_s)
print(res.labels, res.inertia_, res.n_iter)
```

- **k-NN**: ties break deterministically to the earliest-seen neighbour label.
- **KMeans**: k-means++ seeding driven by a seeded `random.Random`; empty
  clusters keep their centroid instead of collapsing.
- **DecisionTree**: CART with Gini impurity; stops on purity, depth, or
  `min_samples_split`.

## 5. Choosing between them

| Situation | Reach for |
|---|---|
| Tabular, few features, need interpretability | `DecisionTreeClassifier` |
| Continuous target | `LinearRegression` |
| Binary outcome with probabilities | `LogisticRegression` |
| Tiny dataset, low noise | `KNeighborsClassifier` |
| Unknown cluster count exploration | `KMeans` |
| Correlated features → smaller model | `PCA` first |
