"""Advanced distributions: chi-square, Student-t, gamma and beta.

Complements :mod:`cds.probability.distributions` with the two sampling
distributions every hypothesis-testing curriculum needs plus the gamma/beta
family behind them. The heavy special-function lifting (regularized
incomplete gamma/beta) live in their single home, ``cds.math_utils.special``, so no lower-level package depends on a higher-level one.
"""

from __future__ import annotations

import math
import random

from cds.math_utils.special import betai, gammln, gammp

_BISECT_ITERS = 200
_BISECT_TOL = 1e-12


# --------------------------------------------------------------------- #
# Chi-square                                                             #
# --------------------------------------------------------------------- #


def chi2_pdf(x: float, df: float) -> float:
    """Chi-square density with ``df`` degrees of freedom.

    For ``x == 0`` the limit is returned exactly (0.5 when ``df == 2``, +inf
    below, 0.0 above); negative ``x`` has zero density.

    Raises:
        ValueError: if ``df <= 0``.
    """
    if df <= 0:
        raise ValueError("df must be positive")
    if x < 0:
        return 0.0
    if x == 0:
        if df == 2:
            return 0.5
        return math.inf if df < 2 else 0.0
    log_norm = -math.lgamma(df / 2) - (df / 2) * math.log(2.0)
    return math.exp(log_norm + (df / 2 - 1) * math.log(x) - x / 2)


def chi2_cdf(x: float, df: float) -> float:
    """P(X <= x) for X ~ chi-square(df), via P(df/2, x/2).

    Raises:
        ValueError: if ``df <= 0``.
    """
    if df <= 0:
        raise ValueError("df must be positive")
    if x <= 0:
        return 0.0
    return gammp(df / 2, x / 2)


def chi2_ppf(p: float, df: float) -> float:
    """Inverse CDF (quantile) of the chi-square distribution.

    Raises:
        ValueError: if ``p`` outside ``(0, 1)`` or ``df <= 0``.
    """
    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")
    if df <= 0:
        raise ValueError("df must be positive")
    hi = 1.0
    while chi2_cdf(hi, df) < p:
        hi *= 2.0
    lo = hi * 0.5
    # Halving shrinks the bracket below _BISECT_TOL within ~60 iterations,
    # so all _BISECT_ITERS rounds never run out — exhaustion arc unreachable.
    for _ in range(_BISECT_ITERS):  # pragma: no branch
        mid = (lo + hi) / 2
        if chi2_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < _BISECT_TOL:
            break
    return (lo + hi) / 2


# --------------------------------------------------------------------- #
# Student's t                                                            #
# --------------------------------------------------------------------- #


def t_pdf(x: float, df: float) -> float:
    """Student-t density with ``df`` degrees of freedom.

    Raises:
        ValueError: if ``df <= 0``.
    """
    if df <= 0:
        raise ValueError("df must be positive")
    log_norm = math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - 0.5 * math.log(df * math.pi)
    return math.exp(log_norm - ((df + 1) / 2) * math.log(1 + x * x / df))


def t_cdf(x: float, df: float) -> float:
    """P(T <= x) for T ~ t(df), via the regularized incomplete beta.

    Uses ``F(t) = I_{df/(df+t²)}(df/2, 1/2)`` on the left half and mirrors it
    on the right; ``t = 0`` falls out as exactly 0.5.

    Raises:
        ValueError: if ``df <= 0``.
    """
    if df <= 0:
        raise ValueError("df must be positive")
    half_tail = 0.5 * betai(df / 2, 0.5, df / (df + x * x))
    if x <= 0:
        return half_tail
    return 1.0 - half_tail


def t_ppf(p: float, df: float) -> float:
    """Inverse CDF (quantile) of Student's t distribution.

    Raises:
        ValueError: if ``p`` outside ``(0, 1)`` or ``df <= 0``.
    """
    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")
    if df <= 0:
        raise ValueError("df must be positive")
    lo, hi = -1.0, 1.0
    while t_cdf(lo, df) > p:
        lo *= 2.0
    while t_cdf(hi, df) < p:
        hi *= 2.0
    # Same halving argument as chi2_ppf: the bracket always beats the
    # tolerance long before _BISECT_ITERS — exhaustion arc unreachable.
    for _ in range(_BISECT_ITERS):  # pragma: no branch
        mid = (lo + hi) / 2
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < _BISECT_TOL:
            break
    return (lo + hi) / 2


# --------------------------------------------------------------------- #
# Gamma                                                                  #
# --------------------------------------------------------------------- #


def gamma_pdf(x: float, shape: float, scale: float = 1.0) -> float:
    """Gamma density parameterized by ``shape`` and ``scale`` (mean = a·scale).

    For ``x == 0`` the exact limit is returned (``1/scale`` when
    ``shape == 1``, +inf below, 0.0 above); negative ``x`` has zero density.

    Raises:
        ValueError: if ``shape <= 0`` or ``scale <= 0``.
    """
    if shape <= 0:
        raise ValueError("shape must be positive")
    if scale <= 0:
        raise ValueError("scale must be positive")
    if x < 0:
        return 0.0
    if x == 0:
        if shape == 1:
            return 1.0 / scale
        return math.inf if shape < 1 else 0.0
    log_norm = -math.lgamma(shape) - shape * math.log(scale)
    return math.exp(log_norm + (shape - 1) * math.log(x) - x / scale)


def _std_normal(rng: random.Random) -> float:
    """One standard-normal draw via Box–Muller (same recipe as gaussian_sample)."""
    u1 = max(rng.random(), 1e-12)
    u2 = rng.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _std_gamma(rng: random.Random, shape: float) -> float:
    """One unit-scale Gamma(shape) draw via Marsaglia–Tsang (shape >= 1)."""
    d = shape - 1.0 / 3.0
    c = 1.0 / (3.0 * math.sqrt(d))
    while True:
        x = _std_normal(rng)
        v = (1.0 + c * x) ** 3
        if v <= 0:
            continue  # rare squeeze failure — redraw
        u = rng.random()
        # Log-domain squeeze test avoids exp() overflow for large |x|.
        if math.log(u) < 0.5 * x * x + d - d * v + d * math.log(v):
            return d * v


def sample_gamma(
    n: int,
    shape: float,
    scale: float = 1.0,
    seed: int | None = None,
) -> list[float]:
    """Generate ``n`` Gamma(shape, scale) samples (Marsaglia–Tsang, seeded).

    For ``shape < 1`` uses the boost identity
    ``Gamma(a) ~ Gamma(a+1) · U^(1/a)``, which preserves exactness without a
    rejection loop of its own.

    Raises:
        ValueError: if ``n < 0``, ``shape <= 0``, or ``scale <= 0``.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if shape <= 0:
        raise ValueError("shape must be positive")
    if scale <= 0:
        raise ValueError("scale must be positive")
    rng = random.Random(seed)
    out: list[float] = []
    for _ in range(n):
        out.append(_boosted_gamma(rng, shape) * scale)
    return out


# --------------------------------------------------------------------- #
# Beta                                                                   #
# --------------------------------------------------------------------- #


def beta_pdf(x: float, a: float, b: float) -> float:
    """Beta density on (0, 1); endpoints and exterior clamp to 0.0.

    Raises:
        ValueError: if ``a <= 0`` or ``b <= 0``.
    """
    if a <= 0:
        raise ValueError("a must be positive")
    if b <= 0:
        raise ValueError("b must be positive")
    if x <= 0 or x >= 1:
        return 0.0
    log_norm = gammln(a + b) - gammln(a) - gammln(b)
    return math.exp(log_norm + (a - 1) * math.log(x) + (b - 1) * math.log(1 - x))


def sample_beta(
    n: int,
    a: float,
    b: float,
    seed: int | None = None,
) -> list[float]:
    """Generate ``n`` Beta(a, b) samples via the exact gamma-ratio identity.

    ``X/(X+Y)`` with ``X ~ Gamma(a)`` and ``Y ~ Gamma(b)`` is exactly
    Beta(a, b). Jöhnk's algorithm was rejected deliberately: its accept/retry
    loop adds RNG-path branches without any accuracy gain over the ratio.

    Raises:
        ValueError: if ``n < 0``, ``a <= 0``, or ``b <= 0``.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if a <= 0:
        raise ValueError("a must be positive")
    if b <= 0:
        raise ValueError("b must be positive")
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        x = _boosted_gamma(rng, a)
        y = _boosted_gamma(rng, b)
        out.append(x / (x + y))
    return out


def _boosted_gamma(rng: random.Random, shape: float) -> float:
    """Gamma(shape) draw handling the shape < 1 boost internally."""
    if shape < 1:
        # math.pow keeps the boost off random()'s legacy Any typing.
        boost: float = math.pow(float(rng.random()), 1.0 / shape)
        return _std_gamma(rng, shape + 1) * boost
    return _std_gamma(rng, shape)
