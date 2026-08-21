"""Descriptive statistics — mean, median, variance, stdev."""

from __future__ import annotations

import math

from cds.core._numeric import NEAR_ZERO


def mean(data: list[float]) -> float:
    """Calculate the arithmetic mean of a list of numbers.

    Args:
        data: List of numeric values.

    Returns:
        Arithmetic mean (sum / N).

    Raises:
        ValueError: if data is empty.
    """
    if not data:
        raise ValueError("mean requires at least one data point")
    return sum(data) / len(data)


def median(data: list[float]) -> float:
    """Calculate the median (middle value) of a list of numbers.

    Args:
        data: List of numeric values.

    Returns:
        Median value.
    """
    if not data:
        return 0.0
    sorted_data = sorted(data)
    n = len(sorted_data)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_data[mid - 1] + sorted_data[mid]) / 2
    return float(sorted_data[mid])


def variance(data: list[float], ddof: int = 1) -> float:
    """Calculate the sample variance of a list of numbers.

    Args:
        data: List of numeric values.
        ddof: Delta Degrees of Freedom (1 for sample, 0 for population).

    Returns:
        Sample or population variance.

    Raises:
        ValueError: if data size is <= ddof.
    """
    if len(data) <= ddof:
        raise ValueError(f"variance requires more than {ddof} data points")
    m = mean(data)
    return sum((x - m) ** 2 for x in data) / (len(data) - ddof)


def stdev(data: list[float], ddof: int = 1) -> float:
    """Calculate the standard deviation of a list of numbers.

    Args:
        data: List of numeric values.
        ddof: Delta Degrees of Freedom.

    Returns:
        Standard deviation.
    """
    return math.sqrt(variance(data, ddof))


def correlation(x: list[float], y: list[float]) -> float:
    """Calculate the Pearson correlation coefficient between two lists.

    Args:
        x: first list of values
        y: second list of values

    Returns:
        Pearson correlation coefficient.

    Raises:
        ValueError: if lengths mismatch or lists are too short.
    """
    if len(x) != len(y):
        raise ValueError("lists must be the same length")
    if len(x) < 2:
        raise ValueError("correlation requires at least two data points")

    mx, my = mean(x), mean(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den = math.sqrt(sum((xi - mx) ** 2 for xi in x) * sum((yi - my) ** 2 for yi in y))

    return num / den if den > NEAR_ZERO else 0.0


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
        # ranks i..j (0-based in sorted order) → midrank of (i+1)..(j+1)
        mid = 0.5 * ((i + 1) + (j + 1))
        for k in range(i, j + 1):
            ranks[order[k]] = mid
        i = j + 1
    return ranks


# Historical private name kept as an alias for backwards compatibility.
_average_ranks = average_ranks


def spearman_correlation(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation (Pearson correlation of average ranks).

    Handles ties via midranks. Returns 0.0 if either rank series is constant.
    """
    if len(x) != len(y):
        raise ValueError("lists must be the same length")
    if len(x) < 2:
        raise ValueError("spearman_correlation requires at least two data points")
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    return correlation(rx, ry)


def percentile(data: list[float], p: float) -> float:
    """Linear-interpolation percentile (``p`` in ``[0, 100]``).

    Uses the "inclusive" method: position ``(n-1) * p/100``.
    """
    if not data:
        raise ValueError("percentile requires at least one data point")
    if not (0.0 <= p <= 100.0):
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
    """Standardize ``data`` to z-scores ``(x - mean) / stdev``.

    Raises:
        ValueError: if the sample is empty/too short for ``ddof``, or stdev is 0.
    """
    if not data:
        raise ValueError("z_scores requires at least one data point")
    s = stdev(data, ddof=ddof)
    if s <= NEAR_ZERO:
        raise ValueError("z_scores requires non-zero standard deviation")
    m = mean(data)
    return [(x - m) / s for x in data]
