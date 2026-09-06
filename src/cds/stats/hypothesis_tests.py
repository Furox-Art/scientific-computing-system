"""Classical statistical hypothesis tests in pure Python.

The public tests in this module validate finite numeric inputs explicitly and
fail closed on statistically invalid count tables instead of propagating NaN or
silently producing nominal degrees of freedom for empty margins.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from cds.stats._distributions import (  # re-exported for compatibility
    _EPS,
    _FPMIN,
    _MAXIT,
    _betacf,
    _betai,
    _gammln,
    _gammp,
    _gammq,
    _gcf,
    _gser,
    chi2_sf,
    f_sf,
    t_sf,
)
from cds.stats.descriptive import mean, variance

__all__ = [
    "TestResult",
    "one_sample_ttest",
    "two_sample_ttest",
    "paired_ttest",
    "chi_square_gof",
    "chi_square_independence",
    "one_way_anova",
    "cohens_d",
    "paired_cohens_d",
    "eta_squared_from_f",
    "cramers_v",
    "bonferroni_corrected_alpha",
    "t_sf",
    "chi2_sf",
    "f_sf",
    "_EPS",
    "_FPMIN",
    "_MAXIT",
    "_betacf",
    "_betai",
    "_gammln",
    "_gammp",
    "_gammq",
    "_gcf",
    "_gser",
]


@dataclass
class TestResult:
    """Result of a hypothesis test: statistic, degrees of freedom, and p-value."""

    statistic: float
    df: float
    p_value: float


def _validate_finite_sample(values: list[float], name: str, *, minimum: int) -> None:
    if len(values) < minimum:
        raise ValueError(f"{name} needs at least {minimum} observations")
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{name} observations must be finite")


def _validate_count_values(values: list[float], name: str, *, positive: bool) -> None:
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{name} counts must be finite")
    if positive:
        if any(value <= 0 for value in values):
            raise ValueError(f"{name} counts must be positive")
    elif any(value < 0 for value in values):
        raise ValueError(f"{name} counts must be non-negative")


def one_sample_ttest(data: list[float], popmean: float = 0.0) -> TestResult:
    """Two-sided one-sample Student t-test against ``popmean``."""
    _validate_finite_sample(data, "sample", minimum=2)
    if not math.isfinite(popmean):
        raise ValueError("population mean must be finite")
    n = len(data)
    df = n - 1
    se = math.sqrt(variance(data, ddof=1) / n)
    if se == 0.0:
        raise ValueError("zero variance; t-test undefined")
    statistic = (mean(data) - popmean) / se
    return TestResult(statistic=statistic, df=df, p_value=t_sf(statistic, df))


def two_sample_ttest(
    a: list[float],
    b: list[float],
    equal_var: bool = True,
) -> TestResult:
    """Two-sided independent-samples Student or Welch t-test."""
    _validate_finite_sample(a, "first sample", minimum=2)
    _validate_finite_sample(b, "second sample", minimum=2)
    na, nb = len(a), len(b)
    va, vb = variance(a, ddof=1), variance(b, ddof=1)
    difference = mean(a) - mean(b)

    if equal_var:
        df = na + nb - 2
        pooled_variance = ((na - 1) * va + (nb - 1) * vb) / df
        se = math.sqrt(pooled_variance * (1.0 / na + 1.0 / nb))
        df_effective = float(df)
    else:
        variance_term = va / na + vb / nb
        se = math.sqrt(variance_term)
        if se == 0.0:
            raise ValueError("zero variance; t-test undefined")
        denominator = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
        df_effective = variance_term**2 / denominator

    if se == 0.0:
        raise ValueError("zero variance; t-test undefined")
    statistic = difference / se
    return TestResult(
        statistic=statistic,
        df=df_effective,
        p_value=t_sf(statistic, df_effective),
    )


def paired_ttest(a: list[float], b: list[float]) -> TestResult:
    """Two-sided paired t-test using within-pair differences.

    Equal-length matched observations are reduced to ``a_i - b_i`` and tested
    against zero. When every difference is identical, the sampling variance is
    zero: an exactly zero difference yields ``t=0, p=1``; a non-zero constant
    difference yields the limiting ``|t|=inf, p=0`` result explicitly.
    """
    if len(a) != len(b):
        raise ValueError("paired samples must have the same length")
    _validate_finite_sample(a, "first paired sample", minimum=2)
    _validate_finite_sample(b, "second paired sample", minimum=2)
    differences = [left - right for left, right in zip(a, b)]
    difference_mean = mean(differences)
    df = len(differences) - 1
    difference_variance = variance(differences, ddof=1)
    if difference_variance == 0.0:
        if difference_mean == 0.0:
            return TestResult(statistic=0.0, df=df, p_value=1.0)
        return TestResult(
            statistic=math.copysign(math.inf, difference_mean),
            df=df,
            p_value=0.0,
        )
    se = math.sqrt(difference_variance / len(differences))
    statistic = difference_mean / se
    return TestResult(statistic=statistic, df=df, p_value=t_sf(statistic, df))


def chi_square_gof(observed: list[float], expected: list[float]) -> TestResult:
    """Pearson chi-square goodness-of-fit test for count vectors."""
    if len(observed) != len(expected):
        raise ValueError("observed and expected must have same length")
    if len(observed) < 2:
        raise ValueError("need at least 2 categories")
    _validate_count_values(observed, "observed", positive=False)
    _validate_count_values(expected, "expected", positive=True)
    observed_total = sum(observed)
    expected_total = sum(expected)
    if not math.isclose(observed_total, expected_total, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("observed and expected counts must have equal totals")
    statistic = sum((obs - exp) ** 2 / exp for obs, exp in zip(observed, expected))
    df = len(observed) - 1
    return TestResult(statistic=statistic, df=df, p_value=chi2_sf(statistic, df))


def chi_square_independence(table: list[list[float]]) -> TestResult:
    """Pearson chi-square independence test for a non-degenerate count table."""
    rows = len(table)
    if rows < 2:
        raise ValueError("need at least 2 rows")
    cols = len(table[0])
    if cols < 2 or any(len(row) != cols for row in table):
        raise ValueError("need a rectangular table with at least 2 columns")
    for row in table:
        _validate_count_values(row, "table", positive=False)

    row_totals = [sum(row) for row in table]
    column_totals = [sum(table[i][j] for i in range(rows)) for j in range(cols)]
    grand_total = sum(row_totals)
    if grand_total <= 0.0:
        raise ValueError("table total must be positive")
    if any(total <= 0.0 for total in row_totals) or any(total <= 0.0 for total in column_totals):
        raise ValueError("every row and column must have a positive marginal total")

    statistic = 0.0
    for i in range(rows):
        for j in range(cols):
            expected = row_totals[i] * column_totals[j] / grand_total
            statistic += (table[i][j] - expected) ** 2 / expected
    df = (rows - 1) * (cols - 1)
    return TestResult(statistic=statistic, df=df, p_value=chi2_sf(statistic, df))


def one_way_anova(*groups: list[float]) -> TestResult:
    """Fisher one-way ANOVA F-test."""
    group_count = len(groups)
    if group_count < 2:
        raise ValueError("need at least 2 groups")
    for index, group in enumerate(groups, start=1):
        _validate_finite_sample(group, f"group {index}", minimum=1)
    total_count = sum(len(group) for group in groups)
    if total_count <= group_count:
        raise ValueError("need more observations than groups")

    grand_mean = sum(sum(group) for group in groups) / total_count
    ss_between = sum(len(group) * (mean(group) - grand_mean) ** 2 for group in groups)
    ss_within = sum(sum((value - mean(group)) ** 2 for value in group) for group in groups)
    df_between = group_count - 1
    df_within = total_count - group_count
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    if ms_within == 0.0:
        raise ValueError("zero within-group variance; F undefined")
    statistic = ms_between / ms_within
    return TestResult(
        statistic=statistic,
        df=df_between,
        p_value=f_sf(statistic, df_between, df_within),
    )


def cohens_d(group_a: list[float], group_b: list[float]) -> float:
    """Pooled-standard-deviation Cohen d for independent samples."""
    _validate_finite_sample(group_a, "first sample", minimum=2)
    _validate_finite_sample(group_b, "second sample", minimum=2)
    na, nb = len(group_a), len(group_b)
    va, vb = variance(group_a, ddof=1), variance(group_b, ddof=1)
    pooled_variance = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)
    if pooled_variance == 0.0:
        raise ValueError("zero pooled variance; Cohen's d undefined")
    return (mean(group_a) - mean(group_b)) / math.sqrt(pooled_variance)


def paired_cohens_d(a: list[float], b: list[float]) -> float:
    """Paired-sample Cohen dz based on the standard deviation of differences."""
    if len(a) != len(b):
        raise ValueError("paired samples must have the same length")
    _validate_finite_sample(a, "first paired sample", minimum=2)
    _validate_finite_sample(b, "second paired sample", minimum=2)
    differences = [left - right for left, right in zip(a, b)]
    difference_mean = mean(differences)
    difference_variance = variance(differences, ddof=1)
    if difference_variance == 0.0:
        if difference_mean == 0.0:
            return 0.0
        return math.copysign(math.inf, difference_mean)
    return difference_mean / math.sqrt(difference_variance)


def eta_squared_from_f(f: float, df1: int, df2: int) -> float:
    """Eta-squared effect size derived from an ANOVA F statistic."""
    if not math.isfinite(f) or f < 0.0:
        raise ValueError("F statistic must be finite and non-negative")
    if df1 < 1 or df2 < 1:
        raise ValueError("df1 and df2 must be >= 1")
    return (f * df1) / (f * df1 + df2)


def cramers_v(table: list[list[float]]) -> float:
    """Cramer V effect size for a valid contingency table."""
    rows = len(table)
    if rows < 2:
        raise ValueError("need at least 2 rows")
    cols = len(table[0])
    if cols < 2 or any(len(row) != cols for row in table):
        raise ValueError("need a rectangular table with at least 2 columns")
    total = sum(sum(row) for row in table)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("table total must be finite and positive")
    statistic = chi_square_independence(table).statistic
    return math.sqrt(statistic / (total * min(rows - 1, cols - 1)))


def bonferroni_corrected_alpha(alpha: float, k: int) -> float:
    """Return the Bonferroni per-test alpha for ``k`` comparisons."""
    if k < 1:
        raise ValueError("number of comparisons k must be >= 1")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and in the open interval (0, 1)")
    return alpha / k
