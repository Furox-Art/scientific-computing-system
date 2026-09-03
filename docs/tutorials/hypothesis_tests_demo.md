# Hypothesis Testing Tutorial

`cds.stats` ships classical inferential tests: one-sample and two-sample
t-tests, chi-square (goodness-of-fit and independence), and one-way ANOVA.
Each returns a result object with `statistic`, `df`, and `p_value`.

## 1. One-sample t-test

Is the sample mean different from a reference value? The t-statistic compares
the deviation of the mean to its standard error.

```python
from cds.stats import one_sample_ttest

sample = [5.1, 4.8, 5.3, 4.9, 5.0, 5.2, 4.7]
res = one_sample_ttest(sample, popmean=4.5)
print(f"H0: mean == 4.5  ->  t = {res.statistic:.3f}, df = {res.df}, p = {res.p_value:.4f}")
#   t = 6.124, df = 6, p = 0.0009
```

A tiny p-value rejects the null hypothesis at the 5% level.

## 2. Two-sample t-test (pooled vs Welch)

Compare two groups. `equal_var=True` uses the pooled variance (Student's t);
`equal_var=False` uses Welch's correction for unequal variances.

```python
from cds.stats import two_sample_ttest

group_a = [20.0, 22.0, 19.0, 24.0, 23.0, 21.0]
group_b = [28.0, 31.0, 26.0, 30.0, 29.0, 27.0]

pooled = two_sample_ttest(group_a, group_b, equal_var=True)
welch = two_sample_ttest(group_a, group_b, equal_var=False)
print(f"Pooled: t = {pooled.statistic:.3f}, df = {pooled.df:.1f}, p = {pooled.p_value:.5f}")
print(f"Welch : t = {welch.statistic:.3f}, df = {welch.df:.2f}, p = {welch.p_value:.5f}")
```

```
Pooled: t = -6.481, df = 10.0, p = 0.00007
Welch : t = -6.481, df = 10.00, p = 0.00007
```

## 3. Chi-square goodness-of-fit

Do observed counts match expected proportions?

```python
from cds.stats import chi_square_gof

observed = [16.0, 18.0, 16.0, 14.0, 12.0, 12.0]
expected = [16.0, 16.0, 16.0, 16.0, 16.0, 8.0]
gof = chi_square_gof(observed, expected)
print(f"chi2 = {gof.statistic:.3f}, df = {gof.df}, p = {gof.p_value:.4f}")
#   chi2 = 3.500, df = 5, p = 0.6234
```

A large p-value here means the fit is plausible, fail to reject the null.

## 4. Chi-square test of independence

Are two categorical variables associated? Pass a contingency table.

```python
from cds.stats import chi_square_independence

table = [[10.0, 20.0, 30.0], [6.0, 9.0, 17.0]]
ind = chi_square_independence(table)
print(f"chi2 = {ind.statistic:.3f}, df = {ind.df}, p = {ind.p_value:.4f}")
#   chi2 = 0.272, df = 2, p = 0.8730
```

## 5. One-way ANOVA

Do three or more groups share the same mean? ANOVA partitions variance into
between-group and within-group components.

```python
from cds.stats import one_way_anova

g1 = [8.0, 9.0, 7.0, 8.0, 9.0]
g2 = [10.0, 11.0, 9.0, 12.0, 10.0]
g3 = [13.0, 12.0, 14.0, 13.0, 15.0]
anova = one_way_anova(g1, g2, g3)
print(f"F = {anova.statistic:.3f}, df_between = {anova.df}, p = {anova.p_value:.6f}")
#   F = 30.970, df_between = 2, p = 0.000018
```

The very small p-value indicates at least one group mean differs.

## 6. Effect sizes (how big, not just how significant)

A small p-value tells you an effect exists; it does **not** tell you how large it
is. `cds.stats` ships four standardized effect-size measures, each the natural
companion to a significance test above.

### Cohen's d: companion to the two-sample t-test

Standardized mean difference using the pooled standard deviation. Conventional
reading (Cohen, 1988): |d| ~0.2 small, ~0.5 medium, ~0.8 large.

```python
from cds.stats import cohens_d

group_a = [20.0, 22.0, 19.0, 24.0, 23.0, 21.0]
group_b = [28.0, 31.0, 26.0, 30.0, 29.0, 27.0]
print(f"Cohen's d = {cohens_d(group_a, group_b):.3f}")
#   d = -3.742  (large; group_b mean is higher)
```

The negative sign means `group_a` has the smaller mean, the magnitude is what
you read against the cutoffs.

### η² (eta-squared): companion to one-way ANOVA

The proportion of total variance accounted for by group membership, derived from
the F statistic. Range [0, 1]; ~0.01 small, ~0.06 medium, ~0.14 large.

```python
from cds.stats import eta_squared_from_f, one_way_anova

g1 = [8.0, 9.0, 7.0, 8.0, 9.0]
g2 = [10.0, 11.0, 9.0, 12.0, 10.0]
g3 = [13.0, 12.0, 14.0, 13.0, 15.0]
anova = one_way_anova(g1, g2, g3)
eta2 = eta_squared_from_f(anova.statistic, anova.df, 12)  # df1=k-1=2, df2=N-k=12
print(f"eta^2 = {eta2:.3f}")
#   eta^2 = 0.838  (very large; group membership explains ~84% of variance)
```

### Cramér's V: companion to chi-square independence

Normalizes the chi-square statistic to [0, 1] using sample size and the smaller
table dimension. 0 = no association, 1 = perfect; ~0.1 small, ~0.3 medium,
~0.5 large.

```python
from cds.stats import cramers_v

table = [[10.0, 20.0, 30.0], [6.0, 9.0, 17.0]]
print(f"Cramer's V = {cramers_v(table):.3f}")
#   V = 0.054  (negligible association, matching the large p-value in §4)
```

### Bonferroni correction: for multiple comparisons

When you run several tests on the same data, the chance of a false positive
grows. The Bonferroni correction divides your family-wise alpha by the number of
comparisons so each test must beat a stricter threshold.

```python
from cds.stats import bonferroni_corrected_alpha

alpha_corrected = bonferroni_corrected_alpha(0.05, 5)  # 5 comparisons
print(f"corrected alpha = {alpha_corrected:.4f}")
#   0.0100  — each p-value must be < 0.01, not < 0.05
```

## Reading the results

All tests return the same shape of result, so you can decide uniformly:
**p < α (typically 0.05) → reject H₀.** Pair these with descriptive stats
(`mean`, `stdev`) from the same module for a complete analysis.

Run the full demo:

```bash
python examples/hypothesis_tests_demo.py
```
