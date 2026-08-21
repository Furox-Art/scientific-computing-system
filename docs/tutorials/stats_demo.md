# Statistics Tutorial

`cds.stats` covers descriptive statistics, linear regression, and a set of
classic hypothesis tests (t-tests, ANOVA, chi-square). Tests return a
`TestResult` dataclass carrying the test statistic, p-value, and degrees of
freedom.

## 1. Descriptive statistics

```python
from cds.stats import mean, median, variance, stdev

data = [23.1, 27.5, 19.8, 31.2, 25.6, 28.3, 22.0, 30.1]
print(f"Mean={mean(data):.2f}  Median={median(data):.2f}")  # 25.95  26.55
print(f"Var={variance(data):.2f}  StdDev={stdev(data):.2f}")  # 16.34  4.04
```

## 2. Correlation and linear regression

Pearson correlation between two samples, and ordinary least-squares regression.
The `RegressionResult` exposes `slope`, `intercept`, `r_squared`, and a
`predict(x)` helper.

```python
from cds.stats import correlation, linear_regression

x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
y = [2.1, 4.3, 5.8, 8.1, 9.7, 12.2, 13.8, 16.1]

print(f"r = {correlation(x, y):.4f}")  # r = 0.9991

reg = linear_regression(x, y)
print(f"y = {reg.slope:.2f}x + {reg.intercept:.2f}")  # y = 1.98x + 0.10
print(f"R² = {reg.r_squared:.4f}")  # R² = 0.9981
print(f"predict(10) = {reg.predict(10):.2f}")  # 19.90
```

## 3. Hypothesis tests

All tests return a `TestResult` with `.statistic`, `.p_value`, and `.df`.

### One-sample t-test

Is the sample mean different from a hypothesized population mean?

```python
from cds.stats import one_sample_ttest

r = one_sample_ttest([2.1, 2.4, 1.9, 2.2, 2.0, 2.3], popmean=2.0)
print(f"t={r.statistic:.4f}, p={r.p_value:.4f}, df={r.df}")  # t=1.9640, p=0.1067, df=5
# p > 0.05 → do not reject H0 (mean ≈ 2.0)
```

### Two-sample t-test

Do two independent samples have equal means?

```python
from cds.stats import two_sample_ttest

r = two_sample_ttest([1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0])
print(f"t={r.statistic:.4f}, p={r.p_value:.4f}")  # t=-1.0954, p=0.3153
```

### One-way ANOVA

Do three or more groups share the same mean? Pass each group as a positional
argument.

```python
from cds.stats import one_way_anova

r = one_way_anova([1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0])
print(f"F={r.statistic:.4f}, p={r.p_value:.4f}")  # F=3.0000, p=0.1250
```

### Chi-square tests

Goodness-of-fit (observed vs expected counts) and independence (a 2-D
contingency table).

```python
from cds.stats import chi_square_gof, chi_square_independence

r = chi_square_gof([10.0, 20.0, 30.0], [15.0, 20.0, 25.0])
print(f"chi2={r.statistic:.4f}, p={r.p_value:.4f}")  # chi2=2.6667, p=0.2636

r = chi_square_independence([[10.0, 20.0], [30.0, 40.0]])
print(f"chi2={r.statistic:.4f}, p={r.p_value:.4f}")  # chi2=0.7937, p=0.3730
```

---

Run the full demo with `python examples/stats_demo.py`.
