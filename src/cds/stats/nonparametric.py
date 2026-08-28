"""Distribution-free hypothesis tests: Mann–Whitney U and Wilcoxon signed-rank.

Both use the large-sample normal approximation with tie corrections, which is
accurate for group sizes ≳ 8 per the usual rule of thumb. Exact small-sample
distributions are deliberately out of scope — they add combinatorial machinery
without changing what the tests teach.

References:
    - Mann, H.B. & Whitney, D.R. (1947). Ann. Math. Statist. 18(1), 50-60.
    - Wilcoxon, F. (1945). Biometrics Bulletin 1(6), 80-83.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from cds.stats.descriptive import average_ranks


@dataclass(frozen=True)
class RankTestResult:
    """Outcome of a rank-based test.

    Attributes:
        statistic: U (Mann–Whitney) or W+ (Wilcoxon), as documented per test.
        z: normal-approximation score.
        p_value: two-sided p-value from the standard normal.
        n_effective: number of observations entering the statistic.
    """

    statistic: float
    z: float
    p_value: float
    n_effective: int


def mann_whitney_u(a: list[float], b: list[float]) -> RankTestResult:
    """Two-sided Mann–Whitney U test: do two independent samples differ?

    Pools both samples, ranks with midranks for ties, and applies the
    tie-corrected variance of the normal approximation.

    Args:
        a: first sample (non-empty)
        b: second sample (non-empty)

    Returns:
        A :class:`RankTestResult`; ``statistic`` is the smaller of U₁ and U₂.

    Raises:
        ValueError: if either sample is empty.
    """
    if not a or not b:
        raise ValueError("both samples must be non-empty")
    pooled = [*a, *b]
    ranks = average_ranks(pooled)
    n1 = float(len(a))
    n2 = len(b)
    r1 = sum(ranks[: len(a)])
    # Classic identity: U1 counts (a,b) pairs with a > b (+ half the ties).
    u1 = r1 - n1 * (n1 + 1.0) / 2.0
    u2 = n1 * n2 - u1
    u = min(u1, u2)

    mu = n1 * n2 / 2.0
    # Tie correction: Σ t³ − t over each tie group size t.
    counts: dict[float, int] = {}
    for value in pooled:
        counts[value] = counts.get(value, 0) + 1
    tie_term = sum(t**3 - t for t in counts.values() if t > 1)
    n = n1 + n2
    var = (n1 * n2 / 12.0) * ((n + 1.0) - tie_term / (n * (n - 1.0)))
    sigma = math.sqrt(var) if var > 0 else 1e-12

    z = (u - mu) / sigma
    return RankTestResult(
        statistic=u,
        z=z,
        p_value=_two_sided_normal_p(z),
        n_effective=len(pooled),
    )


def wilcoxon_signed_rank(differences: list[float]) -> RankTestResult:
    """Two-sided Wilcoxon signed-rank test on paired differences.

    Zero differences are dropped; remaining |d| values are ranked with
    midranks, and W+ is the rank sum of positive differences. The normal
    approximation includes both the zero-drop and tie corrections.

    Args:
        differences: paired differences ``x_i − y_i`` (non-empty after
            dropping zeros).

    Returns:
        A :class:`RankTestResult`; ``statistic`` is W+.

    Raises:
        ValueError: if every difference is zero or the input is empty.
    """
    nonzero = [d for d in differences if d != 0]
    if not nonzero:
        raise ValueError("all differences are zero; nothing to test")

    abs_ranks = average_ranks([abs(d) for d in nonzero])
    w_plus = sum(rank for rank, d in zip(abs_ranks, nonzero) if d > 0)
    n = float(len(nonzero))
    mu = n * (n + 1.0) / 4.0

    # Variance with tie correction over |d| groups (zeros already removed).
    #
    # var = n(n+1)(2n+1)/24 - (sum over tie groups of t^3 - t) / 48
    #
    # The divisor is 48, not 2. Both terms come from the same derivation: the
    # untied variance is sum of squared ranks / 4, and a tie group of size t
    # replaces t distinct squared ranks by their midrank, removing
    # (t^3 - t)/48 from the total. Using /2 inflates the correction 24-fold
    # and shrinks the variance, so z is too large and p too small — at n = 12
    # with six pairs of tied |d| it reported p = 0.0012 where the correct
    # value is 0.0022. [Hollander & Wolfe 1999, sec. 3.1]
    counts: dict[float, int] = {}
    for d in nonzero:
        counts[abs(d)] = counts.get(abs(d), 0) + 1
    tie_term = sum(t**3 - t for t in counts.values() if t > 1)
    var = n * (n + 1.0) * (2.0 * n + 1.0) / 24.0 - tie_term / 48.0
    sigma = math.sqrt(var) if var > 0 else 1e-12

    z = (w_plus - mu) / sigma
    return RankTestResult(
        statistic=w_plus,
        z=z,
        p_value=_two_sided_normal_p(z),
        n_effective=len(nonzero),
    )


def _two_sided_normal_p(z: float) -> float:
    """Two-sided p-value P(|Z| >= |z|) under the standard normal."""
    erfc_arg = abs(z) / math.sqrt(2.0)
    return math.erfc(erfc_arg)
