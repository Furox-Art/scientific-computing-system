"""Common probability distributions — pure Python."""

from __future__ import annotations

import math
import random

# Smallest log-probability that still yields a non-zero double. Below it the
# probability is a subnormal-then-zero float, and returning 0.0 is both
# cheaper and more predictable than letting ``math.exp`` underflow silently.
_LOG_UNDERFLOW_LIMIT = -745.0


def _log_factorial(n: int) -> float:
    """Return ``ln(n!)`` for ``n >= 0`` (``0.0`` for ``n <= 1``).

    ``math.lgamma(n + 1)`` is the log of ``math.factorial(n)`` computed
    without ever materialising the factorial, so ``k`` in the thousands costs
    the same as ``k == 3``. Building the factorial first and taking the log
    last is exactly what overflowed in :func:`poisson_pmf`.
    """
    if n <= 1:
        return 0.0
    return math.lgamma(n + 1.0)


def _exp_or_zero(log_probability: float) -> float:
    """``exp(log_probability)`` with an explicit underflow-to-zero rule.

    ``math.exp`` already returns ``0.0`` far below this limit, but the branch
    is spelled out so the intent is documented rather than inferred from a
    magic return value. NaN is passed through untouched: a NaN
    log-probability must not be quietly rounded into a probability of zero.
    """
    if math.isnan(log_probability):
        return math.nan
    if log_probability < _LOG_UNDERFLOW_LIMIT:
        return 0.0
    return math.exp(log_probability)


def gaussian_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Gaussian (normal) probability density function.

    Args:
        x: point to evaluate
        mu: mean
        sigma: standard deviation

    Raises:
        ValueError: if sigma <= 0
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    coeff = 1 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)


def uniform_pdf(
    x: float,
    a: float = 0.0,
    b: float = 1.0,
) -> float:
    """Uniform distribution PDF on [a, b].

    Args:
        x: point to evaluate
        a: lower bound
        b: upper bound

    Raises:
        ValueError: if a >= b
    """
    if a >= b:
        raise ValueError("a must be less than b")
    if a <= x <= b:
        return 1 / (b - a)
    return 0.0


def exponential_pdf(x: float, lam: float = 1.0) -> float:
    """Exponential distribution PDF.

    Args:
        x: point to evaluate (must be >= 0)
        lam: rate parameter (lambda)

    Raises:
        ValueError: if lam <= 0
    """
    if lam <= 0:
        raise ValueError("lambda must be positive")
    if x < 0:
        return 0.0
    return lam * math.exp(-lam * x)


def binomial_pmf(k: int, n: int, p: float) -> float:
    """Binomial distribution probability mass function.

    ``P(X=k) = C(n,k) * p^k * (1-p)^(n-k)``

    Evaluated in log space, ``exp(lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)
    + k·ln p + (n-k)·ln(1-p))``. The direct product overflows for any
    ``n`` whose binomial coefficient exceeds the largest float
    (``binomial_pmf(1000, 2000, 0.5)`` used to raise ``OverflowError``) and
    *underflows* to a silent ``0.0`` when one factor underflows while the
    product stays representable (``binomial_pmf(500, 1000, 0.99)`` used to
    return ``0.0`` instead of ``0.025…``). Log space fixes both: the terms
    are summed first, and only the final magnitude is exponentiated.

    Args:
        k: number of successes
        n: number of trials
        p: probability of success per trial

    Returns:
        ``P(X = k)``; exactly ``0.0`` outside the support ``0 <= k <= n`` and
        exactly ``1.0``/``0.0`` at the two degenerate endpoints.

    Raises:
        ValueError: if ``p`` is outside ``[0, 1]``.
    """
    if not (0 <= p <= 1):
        raise ValueError("p must be in [0, 1]")
    if k < 0 or k > n:
        return 0.0
    # Degenerate endpoints carry exact integer semantics: p == 1 puts the
    # whole mass at k == n, p == 0 at k == 0. Both are exact by construction
    # rather than by limit, so ln(p) (which is -inf) never enters.
    if p == 0.0:
        return 1.0 if k == 0 else 0.0
    if p == 1.0:
        return 1.0 if k == n else 0.0
    log_coefficient = _log_factorial(n) - _log_factorial(k) - _log_factorial(n - k)
    log_probability = log_coefficient + k * math.log(p) + (n - k) * math.log1p(-p)
    return _exp_or_zero(log_probability)


def poisson_pmf(k: int, lam: float) -> float:
    """Poisson distribution probability mass function.

    ``P(X=k) = (lambda^k * e^-lambda) / k!``

    Evaluated in log space, ``exp(k·ln λ − λ − ln k!)`` with the factorial
    contributed by :func:`math.lgamma`. The direct form raised
    ``OverflowError: int too large to convert to float`` for the perfectly
    ordinary ``poisson_pmf(999, 1000)`` because ``λ^k`` and ``k!`` both
    exceed the float range long before their ratio does — the true answer is
    about ``0.0126``.

    Args:
        k: number of events
        lam: expected rate (lambda)

    Returns:
        ``P(X = k)``; exactly ``0.0`` for ``k < 0`` and exactly
        ``1.0`` for ``lam == 0, k == 0``.

    Raises:
        ValueError: if ``lam < 0``.
    """
    if lam < 0:
        raise ValueError("lambda must be non-negative")
    if k < 0:
        return 0.0
    # A Poisson(0) variable is identically zero; 0**0 would also give 1.0
    # but only by accident of Python's power convention.
    if lam == 0.0:
        return 1.0 if k == 0 else 0.0
    log_probability = k * math.log(lam) - lam - _log_factorial(k)
    return _exp_or_zero(log_probability)


def geometric_pmf(k: int, p: float) -> float:
    """Geometric PMF (number of trials until first success, ``k >= 1``).

    ``P(X=k) = (1-p)^{k-1} * p``.
    """
    if not (0 < p <= 1):
        raise ValueError("p must be in (0, 1]")
    if k < 1:
        return 0.0
    return ((1.0 - p) ** (k - 1)) * p


def gaussian_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Gaussian CDF via ``math.erf`` (accurate enough for teaching/numerics)."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    z = (x - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def uniform_cdf(x: float, a: float = 0.0, b: float = 1.0) -> float:
    """Uniform CDF on ``[a, b]``."""
    if a >= b:
        raise ValueError("a must be less than b")
    if x < a:
        return 0.0
    if x >= b:
        return 1.0
    return (x - a) / (b - a)


def exponential_cdf(x: float, lam: float = 1.0) -> float:
    """Exponential CDF: ``1 - exp(-λx)`` for ``x >= 0``."""
    if lam <= 0:
        raise ValueError("lambda must be positive")
    if x < 0:
        return 0.0
    return 1.0 - math.exp(-lam * x)


def uniform_sample(
    a: float,
    b: float,
    n: int,
    seed: int | None = None,
) -> list[float]:
    """Generate n uniform random samples from [a, b].

    Args:
        a: lower bound
        b: upper bound
        n: number of samples
        seed: optional random seed
    """
    if a >= b:
        raise ValueError("a must be less than b")
    if n < 0:
        raise ValueError("n must be non-negative")
    rng = random.Random(seed)
    return [rng.uniform(a, b) for _ in range(n)]


def gaussian_sample(
    n: int,
    mu: float = 0.0,
    sigma: float = 1.0,
    seed: int | None = None,
) -> list[float]:
    """Generate ``n`` i.i.d. Gaussian samples (Box–Muller)."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if n < 0:
        raise ValueError("n must be non-negative")
    rng = random.Random(seed)
    out: list[float] = []
    while len(out) < n:
        u1 = rng.random()
        u2 = rng.random()
        # Avoid log(0)
        u1 = max(u1, 1e-12)
        r = math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * math.pi * u2
        out.append(mu + sigma * r * math.cos(theta))
        if len(out) < n:
            out.append(mu + sigma * r * math.sin(theta))
    return out[:n]


def exponential_sample(
    n: int,
    lam: float = 1.0,
    seed: int | None = None,
) -> list[float]:
    """Generate ``n`` exponential samples via inverse transform."""
    if lam <= 0:
        raise ValueError("lambda must be positive")
    if n < 0:
        raise ValueError("n must be non-negative")
    rng = random.Random(seed)
    out: list[float] = []
    for _ in range(n):
        u = max(rng.random(), 1e-12)
        out.append(-math.log(u) / lam)
    return out


def poisson_sample(
    n: int,
    lam: float = 1.0,
    seed: int | None = None,
) -> list[int]:
    """Generate ``n`` Poisson samples (Knuth algorithm)."""
    if lam < 0:
        raise ValueError("lambda must be non-negative")
    if n < 0:
        raise ValueError("n must be non-negative")
    rng = random.Random(seed)
    if lam == 0.0:
        return [0] * n
    out: list[int] = []
    l_exp = math.exp(-lam)
    for _ in range(n):
        k = 0
        p = 1.0
        while True:
            k += 1
            p *= rng.random()
            if p <= l_exp:
                out.append(k - 1)
                break
    return out
