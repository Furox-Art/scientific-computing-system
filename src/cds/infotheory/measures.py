"""Information-theory measures — entropy, divergence and mutual information.

Pure-Python, zero-dependency implementation of Shannon information measures
using only :mod:`math`. All logarithms are computed via :func:`math.log`
with an explicit change-of-base, so no ``numpy`` is required.

References:
    Shannon, C. E. (1948). A Mathematical Theory of Communication.
    Bell System Technical Journal, 27, 379–423, 623–656.
"""

from __future__ import annotations

import math

__all__ = [
    "cross_entropy",
    "entropy",
    "js_divergence",
    "kl_divergence",
    "mutual_information",
]


def _validate_base(base: float) -> None:
    """Validate that ``base`` is positive and not equal to 1.

    Raises:
        ValueError: if ``base <= 0`` or ``base == 1`` (within floating tolerance).
    """
    if base <= 0 or math.isclose(base, 1.0):
        raise ValueError("base must be positive and not equal to 1")


def _validate_distribution(probabilities: list[float]) -> None:
    """Validate a discrete probability distribution.

    Checks that ``probabilities`` is non-empty, all entries are non-negative,
    and the entries sum to 1 within ``abs_tol=1e-9``.

    Raises:
        ValueError: if any check fails. Messages contain ``empty``,
            ``non-negative`` or ``sum to 1`` so callers can match on substrings.
    """
    if not probabilities:
        raise ValueError("probabilities must be non-empty")
    for p in probabilities:
        if p < 0:
            raise ValueError("probabilities must be non-negative and sum to 1")
        if not math.isfinite(p):
            raise ValueError("probabilities must be non-negative and sum to 1")
    if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-9):
        raise ValueError("probabilities must be non-negative and sum to 1")


def _validate_joint(joint: list[list[float]]) -> None:
    """Validate a joint probability table.

    Checks that ``joint`` is a non-empty rectangular 2-D table with
    non-negative finite entries summing to 1 within ``abs_tol=1e-9``.

    Raises:
        ValueError: if any check fails. Messages contain ``2-D``,
            ``non-empty``, ``non-negative`` or ``sum to 1``.
    """
    if not joint or not joint[0]:
        raise ValueError("joint distribution must be a non-empty 2-D table")
    n_cols = len(joint[0])
    for row in joint:
        if len(row) != n_cols:
            raise ValueError("joint distribution must be a non-empty 2-D table")
    flat: list[float] = []
    for row in joint:
        for v in row:
            if v < 0 or not math.isfinite(v):
                raise ValueError("joint probabilities must be non-negative and sum to 1")
            flat.append(v)
    if not math.isclose(sum(flat), 1.0, abs_tol=1e-9):
        raise ValueError("joint probabilities must sum to 1")


def entropy(probabilities: list[float], base: float = 2.0) -> float:
    """Shannon entropy ``H(P) = -sum p log_b p`` (Shannon 1948).

    Uses the convention ``0 log 0 := 0`` so zero-probability events contribute
    nothing. Units are determined by ``base`` (``2`` → bits, ``e`` → nats).

    Args:
        probabilities: discrete distribution (non-negative, sums to 1 within 1e-9).
        base: logarithm base, must be positive and not equal to 1.

    Returns:
        Entropy ``H(P)`` in units of ``base``.

    Raises:
        ValueError: if ``probabilities`` is empty, contains a negative value,
            does not sum to 1, or ``base`` is invalid.

    References:
        Shannon, C. E. (1948). A Mathematical Theory of Communication.
    """
    _validate_base(base)
    _validate_distribution(probabilities)
    log_base = math.log(base)
    total = 0.0
    for p in probabilities:
        if p > 0:
            total -= p * math.log(p) / log_base
    return total


def kl_divergence(p: list[float], q: list[float], base: float = 2.0) -> float:
    """Kullback-Leibler divergence ``D_KL(P || Q)`` (Shannon 1948).

    ``D_KL(P || Q) = sum p log_b(p/q)`` with ``0 log 0 := 0``. The divergence
    is non-negative and zero iff ``P == Q``.

    Args:
        p: first distribution (non-negative, sums to 1).
        q: second distribution (non-negative, sums to 1, same length as ``p``).
        base: logarithm base, must be positive and not equal to 1.

    Returns:
        KL divergence in units of ``base``. Returns ``float('inf')`` when
        ``q`` assigns zero mass where ``p`` is positive.

    Raises:
        ValueError: if either distribution is empty, contains a negative value,
            does not sum to 1, lengths differ, or ``base`` is invalid.

    References:
        Shannon, C. E. (1948). A Mathematical Theory of Communication.
    """
    _validate_base(base)
    _validate_distribution(p)
    _validate_distribution(q)
    if len(p) != len(q):
        raise ValueError("p and q must have the same length")
    log_base = math.log(base)
    total = 0.0
    for pi, qi in zip(p, q):
        if pi == 0:
            continue
        if qi == 0:
            return float("inf")
        total += pi * (math.log(pi) - math.log(qi)) / log_base
    return total


def cross_entropy(p: list[float], q: list[float], base: float = 2.0) -> float:
    """Cross-entropy ``H(P, Q) = -sum p log_b q`` (Shannon 1948).

    Satisfies ``H(P, Q) = H(P) + D_KL(P || Q)``. Uses ``0 log q := 0`` when
    ``p == 0`` regardless of ``q``.

    Args:
        p: true distribution (non-negative, sums to 1).
        q: predicted distribution (non-negative, sums to 1, same length as ``p``).
        base: logarithm base, must be positive and not equal to 1.

    Returns:
        Cross-entropy in units of ``base``. Returns ``float('inf')`` when
        ``q`` assigns zero mass where ``p`` is positive.

    Raises:
        ValueError: if either distribution is empty, contains a negative value,
            does not sum to 1, lengths differ, or ``base`` is invalid.

    References:
        Shannon, C. E. (1948). A Mathematical Theory of Communication.
    """
    _validate_base(base)
    _validate_distribution(p)
    _validate_distribution(q)
    if len(p) != len(q):
        raise ValueError("p and q must have the same length")
    log_base = math.log(base)
    total = 0.0
    for pi, qi in zip(p, q):
        if pi == 0:
            continue
        if qi == 0:
            return float("inf")
        total -= pi * math.log(qi) / log_base
    return total


def js_divergence(p: list[float], q: list[float], base: float = 2.0) -> float:
    """Jensen-Shannon divergence (symmetric, bounded) (Shannon 1948).

    ``JSD(P || Q) = 0.5 D_KL(P || M) + 0.5 D_KL(Q || M)`` where ``M = (P+Q)/2``.
    Unlike KL, it is symmetric and bounded in ``[0, log_b 2]`` (``[0, 1]`` for
    ``base=2``).

    Args:
        p: first distribution (non-negative, sums to 1).
        q: second distribution (non-negative, sums to 1, same length as ``p``).
        base: logarithm base, must be positive and not equal to 1.

    Returns:
        Jensen-Shannon divergence in units of ``base``.

    Raises:
        ValueError: if either distribution is empty, contains a negative value,
            does not sum to 1, lengths differ, or ``base`` is invalid.

    References:
        Shannon, C. E. (1948). A Mathematical Theory of Communication.
    """
    _validate_base(base)
    _validate_distribution(p)
    _validate_distribution(q)
    if len(p) != len(q):
        raise ValueError("p and q must have the same length")
    mixture: list[float] = [(pi + qi) * 0.5 for pi, qi in zip(p, q)]
    # Mixture is guaranteed valid (non-negative, sums to 1) so KL will not hit
    # the infinite case; inline the raw sum to avoid double-validation.
    log_base = math.log(base)
    kl_pm = 0.0
    kl_qm = 0.0
    for pi, mi in zip(p, mixture):
        if pi == 0:
            continue
        # mi > 0 whenever pi > 0 because mi = (pi+qi)/2 >= pi/2
        kl_pm += pi * (math.log(pi) - math.log(mi)) / log_base
    for qi, mi in zip(q, mixture):
        if qi == 0:
            continue
        kl_qm += qi * (math.log(qi) - math.log(mi)) / log_base
    return 0.5 * kl_pm + 0.5 * kl_qm


def mutual_information(joint: list[list[float]], base: float = 2.0) -> float:
    """Mutual information ``I(X; Y)`` from a joint table (Shannon 1948).

    ``I(X; Y) = sum_{x,y} p(x,y) log_b(p(x,y) / (p(x) p(y)))`` with
    ``0 log 0 := 0``. Zero for independent variables, equal to the entropy
    when the variables are perfectly correlated.

    Args:
        joint: 2-D joint distribution ``p(x, y)`` where rows index ``X`` and
            columns index ``Y``. Must be rectangular, non-negative and sum to 1
            within 1e-9.
        base: logarithm base, must be positive and not equal to 1.

    Returns:
        Mutual information in units of ``base``.

    Raises:
        ValueError: if ``joint`` is empty, ragged, contains a negative value,
            does not sum to 1, or ``base`` is invalid.

    References:
        Shannon, C. E. (1948). A Mathematical Theory of Communication.
    """
    _validate_base(base)
    _validate_joint(joint)
    n_rows = len(joint)
    n_cols = len(joint[0])
    row_sums: list[float] = [sum(row) for row in joint]
    col_sums: list[float] = [0.0] * n_cols
    for row in joint:
        for j, v in enumerate(row):
            col_sums[j] += v
    log_base = math.log(base)
    mi = 0.0
    for i in range(n_rows):
        for j in range(n_cols):
            p_xy = joint[i][j]
            if p_xy == 0:
                continue
            p_x = row_sums[i]
            p_y = col_sums[j]
            # p_x and p_y are >0 whenever p_xy>0 (marginals dominate the cell)
            mi += p_xy * math.log(p_xy / (p_x * p_y)) / log_base
    return mi
