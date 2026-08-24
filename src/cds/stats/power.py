"""Statistical power analysis for classical tests, in pure Python.

Prospective power analysis built on the machinery already present in
:mod:`cds.stats.hypothesis_tests`: every t-based quantity derives from the
two-tailed Student t survival function ``t_sf``, and the normal quantities
used by the two-proportion z-test derive from ``math.erfc``. No new special
functions are introduced and the module keeps the package dependency-free.

The t-test power uses the classical noncentrality ("shift") approximation:
under the alternative, the pooled two-sample t statistic behaves like a
central t variate with ``df = 2 * n_per_group - 2`` degrees of freedom,
shifted by the noncentrality ``delta = effect_size * sqrt(n / 2)`` for two
equal-sized groups. Rejection probabilities under the shifted law are
integrated exactly with the t tail routines, which reproduces noncentral-t
power tables (Cohen, 1988) to within the usual shift-approximation error.

All critical values (Student t and standard normal alike) are obtained by
numerically inverting the corresponding strictly decreasing survival
function with the shared bisection helper :func:`_invert_decreasing`.

References:
    - Cohen, J. (1988). "Statistical Power Analysis for the Behavioral
      Sciences," 2nd ed., Lawrence Erlbaum. (power conventions, effect-size
      benchmarks, noncentral-t tables)
    - Student [W. S. Gosset] (1908). "The probable error of a mean."
      Biometrika, 6(1), 1-25. (t-distribution underlying ``t_sf``)
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from cds.stats.hypothesis_tests import t_sf

__all__ = [
    "PowerResult",
    "power_t_test",
    "required_n_per_group",
    "power_proportion_test",
]

_SQRT2 = math.sqrt(2.0)
_BISECT_ITERATIONS = 60
_MIN_N_PER_GROUP = 2
_MAX_N_PER_GROUP = 10_000


@dataclass
class PowerResult:
    """Bundles the inputs and outcome of a power analysis.

    Attributes:
        power: achieved rejection probability in [0, 1].
        alpha: significance level the power was computed at.
        effect_size: standardized effect (Cohen's d for t analyses, raw
            p1 - p2 scale difference for proportion analyses).
        n_per_group: observations per group, when group-structured.
        n: total observations, when a single-sample analysis.
    """

    power: float
    alpha: float
    effect_size: float
    n_per_group: int | None = None
    n: int | None = None


def _validate_alpha(alpha: float) -> None:
    """Raise unless ``alpha`` lies in the open interval (0, 1)."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in the open interval (0, 1)")


def _tail_upper_t(x: float, df: float) -> float:
    """Upper-tail probability P(T >= x) for a central Student t variate.

    Rebuilt from the two-tailed :func:`t_sf`, which depends only on ``x * x``:
    for ``x >= 0``, ``P(T >= x) = t_sf(x, df) / 2``; symmetry gives
    ``P(T >= x) = 1 - t_sf(x, df) / 2`` otherwise.
    """
    half = 0.5 * t_sf(x, df)
    return half if x >= 0.0 else 1.0 - half


def _norm_sf(x: float) -> float:
    """Standard normal upper-tail probability, ``P(Z >= x) = erfc(x/sqrt(2))/2``."""
    return 0.5 * math.erfc(x / _SQRT2)


def _invert_decreasing(func: Callable[[float], float], target: float) -> float:
    """Solve ``func(x) = target`` for a continuous, strictly decreasing ``func``.

    Generic numerical inversion shared by every critical value in this module
    (Student t and standard normal). The root is first bracketed by stepping
    both endpoints away from zero geometrically until
    ``func(left) >= target >= func(right)`` holds; the bracket is then bisected
    a fixed ``_BISECT_ITERATIONS`` (= 60) times. The iteration count is fixed
    rather than tolerance-driven deliberately: each pass halves the bracket,
    and after 60 halvings any double-precision bracket has collapsed below the
    width of a single ULP, so further iterations provably cannot change the
    answer. A fixed count therefore yields machine-precision roots at a
    bounded, identical cost for every call site.

    Args:
        func: strictly decreasing function on the reals with
            ``func(-inf) > target > func(+inf)``.
        target: value to invert to.

    Returns:
        x such that ``func(x)`` equals ``target`` to machine precision.
    """
    left, right = -1.0, 1.0
    while func(left) < target:
        left *= 2.0
    while func(right) > target:
        right *= 2.0
    for _ in range(_BISECT_ITERATIONS):
        mid = 0.5 * (left + right)
        if func(mid) > target:
            left = mid
        else:
            right = mid
    return 0.5 * (left + right)


def _critical_t(df: float, alpha: float, two_sided: bool) -> float:
    """Critical value of the central t distribution at level ``alpha``."""
    if two_sided:
        return _invert_decreasing(lambda x: 2.0 * _tail_upper_t(x, df), alpha)
    return _invert_decreasing(lambda x: _tail_upper_t(x, df), alpha)


def _critical_z(alpha: float, two_sided: bool) -> float:
    """Critical value of the standard normal distribution at level ``alpha``."""
    if two_sided:
        return _invert_decreasing(lambda x: 2.0 * _norm_sf(x), alpha)
    return _invert_decreasing(_norm_sf, alpha)


def power_t_test(
    effect_size: float,
    n_per_group: int,
    *,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> float:
    """Power of the pooled two-sample t-test (noncentral-t shift method).

    Approximates the rejection probability of Student's pooled two-sample
    t-test with ``n_per_group`` observations in each arm at significance
    ``alpha`` for standardized effect ``effect_size`` (Cohen's d). Under the
    alternative the statistic is modeled as a central t with
    ``df = 2 * n_per_group - 2`` degrees of freedom shifted by the
    noncentrality ``delta = effect_size * sqrt(n / 2)``; the rejection
    probability under this shifted law is evaluated exactly with ``t_sf``.
    A one-tailed test targets the positive-effect direction (group 1 larger).

    Args:
        effect_size: Cohen's d; must be non-zero (d = 0 has power == alpha).
        n_per_group: observations per group (>= 2).
        alpha: significance level in the open interval (0, 1).
        two_sided: two-tailed test if True, one-tailed otherwise.

    Returns:
        Power in [0, 1]: the probability the test rejects H0 under the
        alternative.

    Raises:
        ValueError: if effect_size == 0, n_per_group < 2, or alpha lies
            outside (0, 1).
    """
    if effect_size == 0.0:
        raise ValueError("effect_size must be non-zero")
    if n_per_group < 2:
        raise ValueError("n_per_group must be >= 2")
    _validate_alpha(alpha)
    n = float(n_per_group)
    df = 2.0 * n - 2.0
    delta = effect_size * math.sqrt(n / 2.0)
    crit = _critical_t(df, alpha, two_sided)
    if two_sided:
        return _tail_upper_t(crit - delta, df) + (1.0 - _tail_upper_t(-crit - delta, df))
    return _tail_upper_t(crit - delta, df)


def required_n_per_group(
    effect_size: float,
    target_power: float = 0.8,
    *,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> int:
    """Smallest per-group sample size reaching ``target_power``.

    Power under :func:`power_t_test` is non-decreasing in ``n_per_group``, so
    the smallest qualifying size is found by binary search over
    ``[2, 10_000]`` after checking the two boundary candidates directly.

    Args:
        effect_size: Cohen's d; must be non-zero.
        target_power: desired minimum power, in the open interval (0, 1).
        alpha: significance level in the open interval (0, 1).
        two_sided: two-tailed test if True, one-tailed otherwise.

    Returns:
        Smallest integer n in [2, 10_000] whose power is >= target_power.

    Raises:
        ValueError: if effect_size == 0, target_power or alpha lies outside
            its open unit interval, or even 10_000 observations per group
            fall short of target_power.
    """
    if effect_size == 0.0:
        raise ValueError("effect_size must be non-zero")
    if not 0.0 < target_power < 1.0:
        raise ValueError("target_power must be in the open interval (0, 1)")
    _validate_alpha(alpha)

    def power_of(n: int) -> float:
        return power_t_test(effect_size, n, alpha=alpha, two_sided=two_sided)

    if power_of(_MIN_N_PER_GROUP) >= target_power:
        return _MIN_N_PER_GROUP
    if power_of(_MAX_N_PER_GROUP) < target_power:
        raise ValueError(
            f"effect size {effect_size} unattainable: even "
            f"n_per_group={_MAX_N_PER_GROUP} yields power "
            f"{power_of(_MAX_N_PER_GROUP):.4f} < target_power={target_power}"
        )
    low, high = _MIN_N_PER_GROUP, _MAX_N_PER_GROUP
    while high - low > 1:
        mid = (low + high) // 2
        if power_of(mid) >= target_power:
            high = mid
        else:
            low = mid
    return high


def power_proportion_test(
    p1: float,
    p2: float,
    n_per_group: int,
    *,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> float:
    """Power of the pooled two-proportion z-test (normal approximation).

    Models the difference of sample proportions as Gaussian with mean
    ``p1 - p2`` and the *unpooled* alternative standard error
    ``sqrt((p1 (1 - p1) + p2 (1 - p2)) / n)``, while the critical value comes
    from the *pooled* null standard error with ``p_bar = (p1 + p2) / 2``:
    ``se_null = sqrt(2 p_bar (1 - p_bar) / n)``. Equal proportions make the
    power collapse to the size of the test (power == alpha); that degenerate
    case is returned as the computed value, not treated as an error. A
    one-tailed test targets the p1 > p2 direction.

    Args:
        p1: baseline group proportion, in [0, 1].
        p2: comparison group proportion, in [0, 1].
        n_per_group: observations per group (>= 2).
        alpha: significance level in the open interval (0, 1).
        two_sided: two-tailed test if True, one-tailed otherwise.

    Returns:
        Power in [0, 1].

    Raises:
        ValueError: if p1 or p2 lies outside [0, 1], n_per_group < 2, or
            alpha lies outside (0, 1).
    """
    for name, p in (("p1", p1), ("p2", p2)):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")
    if n_per_group < 2:
        raise ValueError("n_per_group must be >= 2")
    _validate_alpha(alpha)
    n = float(n_per_group)
    p_bar = 0.5 * (p1 + p2)
    se_null = math.sqrt(2.0 * p_bar * (1.0 - p_bar) / n)
    se_alt = math.sqrt((p1 * (1.0 - p1) + p2 * (1.0 - p2)) / n)
    diff = p1 - p2
    threshold = _critical_z(alpha, two_sided) * se_null
    upper = _norm_sf((threshold - diff) / se_alt)
    if two_sided:
        return upper + 1.0 - _norm_sf((-threshold - diff) / se_alt)
    return upper
