"""Percentile bootstrap confidence intervals for arbitrary statistics.

The percentile method resamples the data with replacement, recomputes the
statistic on every resample, and reads the confidence bounds off the empirical
distribution of those resample statistics. No distributional assumptions are
made, so any real-valued sample statistic works — medians, ratios, extremes,
trimmed means — including ones with no tractable analytic sampling variance.

References:
    - Efron, B. (1979). Ann. Statist. 7(1), 1-26.
    - Efron, B. & Tibshirani, R.J. (1993). An Introduction to the Bootstrap,
      Chapman & Hall, ch. 13.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from cds.stats.descriptive import mean

#: Type of a statistic: maps a sample to a single real number.
StatFunc = Callable[[Sequence[float]], float]


def _mean_stat(values: Sequence[float]) -> float:
    """Arithmetic mean over any sequence (delegates to descriptive.mean)."""
    return mean(list(values))


@dataclass(frozen=True)
class BootstrapResult:
    """Outcome of a percentile bootstrap confidence-interval computation.

    Attributes:
        estimate: statistic evaluated on the original sample(s).
        lower: lower confidence bound (alpha/2 quantile of the bootstrap
            distribution).
        upper: upper confidence bound (1 - alpha/2 quantile).
        n_resamples: number of bootstrap resamples drawn.
        confidence: requested confidence level, in (0, 1).
        se: bootstrap standard error (standard deviation of the bootstrap
            distribution of the statistic).
    """

    estimate: float
    lower: float
    upper: float
    n_resamples: int
    confidence: float
    se: float


def _quantile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation quantile ``q`` in ``[0, 1]`` of pre-sorted values.

    Position ``(n - 1) * q`` between adjacent order statistics, matching the
    "inclusive" convention used by :func:`cds.stats.descriptive.percentile`.
    """
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    pos = (n - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    weight = pos - lo
    return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight


def _check_args(n_resamples: int, confidence: float) -> None:
    """Validate the keyword arguments shared by both entry points."""
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")


def _summarize(
    estimate: float,
    boot_stats: list[float],
    n_resamples: int,
    confidence: float,
) -> BootstrapResult:
    """Build bounds + SE from the bootstrap distribution of the statistic."""
    boot_stats.sort()
    alpha = 1.0 - confidence
    lower = _quantile(boot_stats, alpha / 2.0)
    upper = _quantile(boot_stats, 1.0 - alpha / 2.0)
    center = sum(boot_stats) / n_resamples
    se = math.sqrt(sum((s - center) ** 2 for s in boot_stats) / n_resamples)
    return BootstrapResult(
        estimate=estimate,
        lower=lower,
        upper=upper,
        n_resamples=n_resamples,
        confidence=confidence,
        se=se,
    )


def bootstrap_ci(
    data: list[float],
    stat: StatFunc = _mean_stat,
    *,
    n_resamples: int = 5000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> BootstrapResult:
    """Percentile bootstrap confidence interval for ``stat`` on one sample.

    Draws ``n_resamples`` resamples of size ``len(data)`` with replacement
    from ``data``, recomputes ``stat`` on each, and takes the alpha/2 and
    1 - alpha/2 quantiles of the resulting distribution as the bounds.

    Args:
        data: observed sample (non-empty).
        stat: statistic mapping a sample to a real number; defaults to the
            arithmetic mean.
        n_resamples: number of bootstrap resamples (at least 1).
        confidence: confidence level, strictly between 0 and 1.
        seed: optional seed for a dedicated :class:`random.Random` instance;
            passing a fixed integer makes the interval exactly reproducible.

    Returns:
        A :class:`BootstrapResult` with the point estimate, percentile
        bounds, and bootstrap standard error.

    Raises:
        ValueError: if ``data`` is empty, ``n_resamples < 1``, or
            ``confidence`` is not strictly between 0 and 1.
    """
    if not data:
        raise ValueError("data must be non-empty")
    _check_args(n_resamples, confidence)
    rng = random.Random(seed)
    estimate = stat(data)
    boot = [stat(rng.choices(data, k=len(data))) for _ in range(n_resamples)]
    return _summarize(estimate, boot, n_resamples, confidence)


def bootstrap_diff_ci(
    a: list[float],
    b: list[float],
    stat: StatFunc = _mean_stat,
    *,
    n_resamples: int = 5000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> BootstrapResult:
    """Percentile bootstrap CI for ``stat(a) - stat(b)`` on two samples.

    Each iteration draws independent resamples of ``a`` and of ``b`` (sizes
    preserved) from a single seeded RNG, computes the difference of the
    statistic on the pair, and accumulates the bootstrap distribution.

    Args:
        a: first sample (non-empty).
        b: second sample (non-empty).
        stat: statistic mapping a sample to a real number; defaults to the
            arithmetic mean, giving a CI for the difference of means.
        n_resamples: number of paired bootstrap resamples (at least 1).
        confidence: confidence level, strictly between 0 and 1.
        seed: optional seed; both groups are resampled from one shared
            seeded :class:`random.Random`, so results are reproducible.

    Returns:
        A :class:`BootstrapResult` whose ``estimate`` is ``stat(a) - stat(b)``
        and whose bounds come from the percentile method on the differences.

    Raises:
        ValueError: if either sample is empty, ``n_resamples < 1``, or
            ``confidence`` is not strictly between 0 and 1.
    """
    if not a or not b:
        raise ValueError("both samples must be non-empty")
    _check_args(n_resamples, confidence)
    rng = random.Random(seed)
    estimate = stat(a) - stat(b)
    boot: list[float] = []
    for _ in range(n_resamples):
        resampled_a = rng.choices(a, k=len(a))
        resampled_b = rng.choices(b, k=len(b))
        boot.append(stat(resampled_a) - stat(resampled_b))
    return _summarize(estimate, boot, n_resamples, confidence)
