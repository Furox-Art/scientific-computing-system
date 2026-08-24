"""Additional discrete distributions: geometric, hypergeometric, negative binomial.

Complements :mod:`cds.probability.distributions` with three discrete workhorses.
Every CDF is computed as an explicit deterministic summation of ``pmf`` terms
over the feasible support (no closed-form shortcuts), so ``cdf`` and ``pmf``
can never disagree numerically.

Boundary conventions for ``p`` (deliberately wider than the continuous
distributions elsewhere in this package):

* ``p == 1`` collapses to a degenerate distribution concentrated at the
  smallest support point (``k = 1`` for geometric, ``k = 0`` for negative
  binomial) and the formulas return those masses exactly.
* ``p == 0`` means success never occurs, so every ``pmf`` value is ``0.0``;
  the corresponding CDF is ``0.0`` for every finite ``k``.

Out-of-support rules: distribution *parameters* (``p``, ``r``, population
counts, ``draws``) raise ``ValueError`` when out of domain. The evaluation
point ``k`` raises ``ValueError`` in a ``pmf`` when it falls outside the
feasible support, while every ``cdf`` — being defined on the whole integer
line — clamps instead (``0.0`` below the support, full mass at or above it).
"""

from __future__ import annotations

import math


def geometric_pmf(k: int, p: float) -> float:
    """Geometric PMF counting trials until the first success (``k >= 1``).

    ``P(K=k) = (1 - p)**(k - 1) * p``.

    Edge conventions: ``p == 1`` gives ``pmf(1) == 1.0`` and ``pmf(k) == 0.0``
    for ``k >= 2``; ``p == 0`` gives ``pmf(k) == 0.0`` for every finite ``k``.

    Args:
        k: trial on which the first success occurs; must satisfy ``k >= 1``.
        p: per-trial success probability; must be in ``[0, 1]``.

    Raises:
        ValueError: if ``p`` is outside ``[0, 1]`` or ``k < 1``.
    """
    if not (0 <= p <= 1):
        raise ValueError("p must be in [0, 1]")
    if k < 1:
        raise ValueError("k must be at least 1")
    return ((1.0 - p) ** (k - 1)) * p


def geometric_cdf(k: int, p: float) -> float:
    """``P(K <= k)`` for the trials-until-first-success geometric law.

    Computed as the explicit sum ``pmf(1) + ... + pmf(k)``.

    Args:
        k: evaluation point; any integer is accepted (values below the support
            evaluate to ``0.0``).
        p: per-trial success probability; must be in ``[0, 1]``.

    Raises:
        ValueError: if ``p`` is outside ``[0, 1]``.
    """
    if not (0 <= p <= 1):
        raise ValueError("p must be in [0, 1]")
    if k < 1:
        return 0.0
    return sum(geometric_pmf(i, p) for i in range(1, k + 1))


def _hypergeometric_support(
    population_successes: int,
    population_failures: int,
    draws: int,
) -> tuple[int, int]:
    """Validate hypergeometric parameters, return feasible ``(lo, hi)`` bounds.

    Raises:
        ValueError: if any count is negative or ``draws`` exceeds the
            population size ``population_successes + population_failures``.
    """
    if population_successes < 0:
        raise ValueError("population_successes must be non-negative")
    if population_failures < 0:
        raise ValueError("population_failures must be non-negative")
    if draws < 0:
        raise ValueError("draws must be non-negative")
    if draws > population_successes + population_failures:
        raise ValueError("draws must not exceed the population size")
    return max(0, draws - population_failures), min(draws, population_successes)


def hypergeometric_pmf(
    k: int,
    population_successes: int,
    population_failures: int,
    draws: int,
) -> float:
    """Hypergeometric PMF: successes in ``draws`` without replacement.

    ``P(K=k) = C(S, k) * C(F, d - k) / C(S + F, d)`` with ``S`` successes,
    ``F`` failures in the population and ``d`` draws.

    Feasible support: ``max(0, d - F) <= k <= min(d, S)``. Drawing zero times
    from an empty population puts unit mass on ``k = 0``; drawing the whole
    population puts unit mass on ``k = S``.

    Args:
        k: number of drawn successes; must lie in the feasible support.
        population_successes: successes ``S`` in the population; ``>= 0``.
        population_failures: failures ``F`` in the population; ``>= 0``.
        draws: items drawn ``d``; ``0 <= d <= S + F``.

    Raises:
        ValueError: if any count is negative, ``draws`` exceeds the population
            size, or ``k`` falls outside the feasible support.
    """
    lo, hi = _hypergeometric_support(population_successes, population_failures, draws)
    if k < lo or k > hi:
        raise ValueError(f"k must be in [{lo}, {hi}]")
    numerator = math.comb(population_successes, k) * math.comb(population_failures, draws - k)
    return numerator / math.comb(population_successes + population_failures, draws)


def hypergeometric_cdf(
    k: int,
    population_successes: int,
    population_failures: int,
    draws: int,
) -> float:
    """``P(K <= k)`` for the hypergeometric law, by explicit summation.

    Args:
        k: evaluation point; any integer is accepted (``0.0`` below the
            feasible support, total mass at or above its upper end).
        population_successes: successes ``S`` in the population; ``>= 0``.
        population_failures: failures ``F`` in the population; ``>= 0``.
        draws: items drawn ``d``; ``0 <= d <= S + F``.

    Raises:
        ValueError: if any count is negative or ``draws`` exceeds the
            population size.
    """
    lo, hi = _hypergeometric_support(population_successes, population_failures, draws)
    if k < lo:
        return 0.0
    return sum(
        hypergeometric_pmf(j, population_successes, population_failures, draws)
        for j in range(lo, min(k, hi) + 1)
    )


def negative_binomial_pmf(k: int, r: int, p: float) -> float:
    """Negative binomial PMF: failures before the ``r``-th success (``k >= 0``).

    ``P(K=k) = C(k + r - 1, k) * p**r * (1 - p)**k``.

    Edge conventions: ``p == 1`` gives ``pmf(0) == 1.0`` (the ``r``-th success
    happens immediately, zero failures) and ``pmf(k) == 0.0`` for ``k >= 1``;
    ``p == 0`` gives ``pmf(k) == 0.0`` for every finite ``k``.

    Args:
        k: number of failures observed; must satisfy ``k >= 0``.
        r: target number of successes; must be ``>= 1``.
        p: per-trial success probability; must be in ``[0, 1]``.

    Raises:
        ValueError: if ``p`` is outside ``[0, 1]``, ``r < 1`` or ``k < 0``.
    """
    if not (0 <= p <= 1):
        raise ValueError("p must be in [0, 1]")
    if r < 1:
        raise ValueError("r must be a positive integer")
    if k < 0:
        raise ValueError("k must be non-negative")
    return math.comb(k + r - 1, k) * (p**r) * ((1.0 - p) ** k)


def negative_binomial_cdf(k: int, r: int, p: float) -> float:
    """``P(K <= k)`` for the failures-before-rth-success negative binomial.

    Computed as the explicit sum ``pmf(0) + ... + pmf(k)``.

    Args:
        k: evaluation point; any integer is accepted (values below the support
            evaluate to ``0.0``).
        r: target number of successes; must be ``>= 1``.
        p: per-trial success probability; must be in ``[0, 1]``.

    Raises:
        ValueError: if ``p`` is outside ``[0, 1]`` or ``r < 1``.
    """
    if not (0 <= p <= 1):
        raise ValueError("p must be in [0, 1]")
    if r < 1:
        raise ValueError("r must be a positive integer")
    if k < 0:
        return 0.0
    return sum(negative_binomial_pmf(j, r, p) for j in range(0, k + 1))
