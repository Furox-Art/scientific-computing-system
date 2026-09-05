"""Descriptive statistics — mean, median, variance, stdev, and correlations."""

from __future__ import annotations

import math

from cds.core._numeric import NEAR_ZERO


def mean(data: list[float]) -> float:
    """Calculate the arithmetic mean of a non-empty list."""
    if not data:
        raise ValueError("mean requires at least one data point")
    return sum(data) / len(data)


def median(data: list[float]) -> float:
    """Calculate the median of a non-empty list.

    Empty input has no statistical median; returning a numeric sentinel would
    conflate an undefined statistic with a genuine measurement of zero.
    """
    if not data:
        raise ValueError("median requires at least one data point")
    sorted_data = sorted(data)
    n = len(sorted_data)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_data[mid - 1] + sorted_data[mid]) / 2
    return float(sorted_data[mid])


def variance(data: list[float], ddof: int = 1) -> float:
    """Calculate sample/population variance using ``ddof``."""
    if len(data) <= ddof:
        raise ValueError(f"variance requires more than {ddof} data points")
    m = mean(data)
    return sum((x - m) ** 2 for x in data) / (len(data) - ddof)


def stdev(data: list[float], ddof: int = 1) -> float:
    """Calculate the standard deviation."""
    return math.sqrt(variance(data, ddof))


def correlation(x: list[float], y: list[float]) -> float:
    """Calculate the Pearson correlation coefficient.

    Raises:
        ValueError: If lengths differ, fewer than two pairs are supplied, a
            value is non-finite, or either input has effectively zero variance.
            Pearson correlation is undefined in the zero-variance case and is
            never converted to the numeric value ``0.0``.
    """
    if len(x) != len(y):
        raise ValueError("lists must be the same length")
    if len(x) < 2:
        raise ValueError("correlation requires at least two data points")
    if any(not math.isfinite(value) for value in x) or any(not math.isfinite(value) for value in y):
        raise ValueError("correlation requires only finite values")

    mx, my = mean(x), mean(y)
    x_ss = sum((xi - mx) ** 2 for xi in x)
    y_ss = sum((yi - my) ** 2 for yi in y)
    denominator = math.sqrt(x_ss * y_ss)
    if denominator <= NEAR_ZERO:
        raise ValueError("correlation is undefined when either input has zero variance")
    numerator = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    return numerator / denominator


def average_ranks(values: list[float]) -> list[float]:
    """Return average ranks (1-based) with midranks for ties."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        mid = 0.5 * ((i + 1) + (j + 1))
        for k in range(i, j + 1):
            ranks[order[k]] = mid
        i = j + 1
    return ranks


_average_ranks = average_ranks


def spearman_correlation(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation using average ranks for ties.

    As with Pearson correlation, a constant rank series makes the statistic
    undefined and therefore raises ``ValueError``.
    """
    if len(x) != len(y):
        raise ValueError("lists must be the same length")
    if len(x) < 2:
        raise ValueError("spearman_correlation requires at least two data points")
    return correlation(_average_ranks(x), _average_ranks(y))


def percentile(data: list[float], p: float) -> float:
    """Linear-interpolation percentile (``p`` in ``[0, 100]``)."""
    if not data:
        raise ValueError("percentile requires at least one data point")
    if not 0.0 <= p <= 100.0:
        raise ValueError("p must be in [0, 100]")
    xs = sorted(float(v) for v in data)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * (p / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def z_scores(data: list[float], ddof: int = 1) -> list[float]:
    """Standardize ``data`` to z-scores ``(x - mean) / stdev``."""
    if not data:
        raise ValueError("z_scores requires at least one data point")
    s = stdev(data, ddof=ddof)
    if s <= NEAR_ZERO:
        raise ValueError("z_scores requires non-zero standard deviation")
    m = mean(data)
    return [(x - m) / s for x in data]
