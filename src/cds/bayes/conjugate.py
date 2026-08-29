"""Conjugate Bayesian updates — pure Python, ``math`` only.

Implements the three textbook conjugate families that cover most
introductory Bayesian inference:

* **Beta–Binomial** for a Bernoulli rate ``p`` in ``[0, 1]``
* **Gamma–Poisson** for a Poisson rate ``lambda > 0``
* **Normal–Normal** (known observation variance) for a Gaussian mean

Every posterior is available in closed form, so no MCMC is required.
Credible intervals are obtained by numerically inverting the corresponding
CDF (regularized incomplete beta / gamma, or ``erf``) with a bisection
root-finder — the same strategy used in :mod:`cds.probability._advanced`
and :mod:`cds.stats.power`. No ``numpy`` or ``scipy`` is imported.

References
----------
* Bernardo, J. M. & Smith, A. F. M. (1994). Bayesian Theory. Wiley.
* Gelman, A. et al. (2013). Bayesian Data Analysis, 3rd ed. Chapman & Hall.
* Numerical Recipes §6.2–6.4 for the incomplete gamma/beta kernels.
"""

from __future__ import annotations

import math

from cds.math_utils.special import betai, gammp

__all__ = [
    "bayes_factor",
    "beta_binomial_update",
    "beta_credible_interval",
    "credible_interval",
    "gamma_credible_interval",
    "gamma_poisson_update",
    "normal_credible_interval",
    "normal_normal_update",
]

_BISECT_ITERS = 200
_BISECT_TOL = 1e-12
_SQRT2 = math.sqrt(2.0)


# ---------------------------------------------------------------------------
# Internal quantile helpers (bisection on the CDF)
# ---------------------------------------------------------------------------


def _beta_ppf(p: float, a: float, b: float) -> float:
    """Quantile (inverse CDF) of ``Beta(a, b)`` via bisection on ``betai``."""
    if p <= 0.0:  # pragma: no cover - defensive; public callers use 0<p<1
        return 0.0
    if p >= 1.0:  # pragma: no cover - defensive; public callers use 0<p<1
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(_BISECT_ITERS):  # pragma: no branch
        mid = 0.5 * (lo + hi)
        if betai(a, b, mid) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < _BISECT_TOL:
            break
    return 0.5 * (lo + hi)


def _gamma_cdf(x: float, shape: float, rate: float) -> float:
    """CDF of ``Gamma(shape, rate)`` at ``x`` (rate parametrization)."""
    if x <= 0.0:  # pragma: no cover - defensive; x>0 for all public paths
        return 0.0
    return gammp(shape, rate * x)


def _gamma_ppf(p: float, shape: float, rate: float) -> float:
    """Quantile of ``Gamma(shape, rate)`` via bisection on ``gammp``."""
    if p <= 0.0:  # pragma: no cover - defensive
        return 0.0
    if p >= 1.0:  # pragma: no cover - defensive
        return math.inf
    # bracket hi: start at mean, double until CDF exceeds p
    hi = shape / rate if rate > 0 else 1.0  # pragma: no cover - rate>0 by caller
    if hi <= 0:  # pragma: no cover - shape/rate>0 by caller
        hi = 1.0
    while _gamma_cdf(hi, shape, rate) < p:
        hi *= 2.0
        if hi > 1e12:  # pragma: no cover - defensive overflow guard
            break
    lo = 0.0
    for _ in range(_BISECT_ITERS):  # pragma: no branch
        mid = 0.5 * (lo + hi)
        if _gamma_cdf(mid, shape, rate) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < _BISECT_TOL:
            break
    return 0.5 * (lo + hi)


def _normal_cdf(x: float, mu: float, sigma: float) -> float:
    """Gaussian CDF via ``math.erf``."""
    z = (x - mu) / (sigma * _SQRT2)
    return 0.5 * (1.0 + math.erf(z))


def _normal_ppf(p: float, mu: float, sigma: float) -> float:
    """Gaussian quantile via bisection on ``_normal_cdf``."""
    if not 0.0 < p < 1.0:  # pragma: no cover - defensive; callers use valid p
        raise ValueError("p must be in (0, 1)")
    # bracket
    lo, hi = mu - 10.0 * sigma, mu + 10.0 * sigma
    # expand if p is extreme (outside 10 sigma)
    while _normal_cdf(lo, mu, sigma) > p:  # pragma: no cover - extreme tail only
        lo -= 10.0 * sigma  # pragma: no cover
    while _normal_cdf(hi, mu, sigma) < p:  # pragma: no cover - extreme tail only
        hi += 10.0 * sigma  # pragma: no cover
    for _ in range(_BISECT_ITERS):  # pragma: no branch
        mid = 0.5 * (lo + hi)
        if _normal_cdf(mid, mu, sigma) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < _BISECT_TOL:
            break
    return 0.5 * (lo + hi)


def _standard_normal_ppf(p: float) -> float:
    """Standard normal quantile ``N(0,1)``."""
    return _normal_ppf(p, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Public credible-interval helpers
# ---------------------------------------------------------------------------


def beta_credible_interval(
    alpha: float,
    beta: float,
    level: float = 0.95,
) -> tuple[float, float]:
    """Equal-tailed credible interval for ``Beta(alpha, beta)``.

    Args:
        alpha: shape ``a > 0``.
        beta: shape ``b > 0``.
        level: credible mass in ``(0, 1)`` (default 0.95).

    Returns:
        ``(lower, upper)`` quantiles at ``(1-level)/2`` and ``1-(1-level)/2``.

    Raises:
        ValueError: if ``alpha <= 0``, ``beta <= 0``, or ``level`` not in
            ``(0, 1)``.
    """
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be positive")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    tail = (1.0 - level) / 2.0
    return _beta_ppf(tail, alpha, beta), _beta_ppf(1.0 - tail, alpha, beta)


def gamma_credible_interval(
    alpha: float,
    beta: float,
    level: float = 0.95,
) -> tuple[float, float]:
    """Equal-tailed credible interval for ``Gamma(alpha, beta)`` (rate).

    The ``Gamma(alpha, beta)`` density with shape ``alpha`` and rate ``beta``
    has mean ``alpha / beta`` and variance ``alpha / beta**2`` — the
    parametrization used for the Gamma–Poisson conjugate prior.

    Args:
        alpha: shape ``> 0``.
        beta: rate ``> 0``.
        level: credible mass in ``(0, 1)``.

    Returns:
        ``(lower, upper)`` quantiles.

    Raises:
        ValueError: if ``alpha <= 0``, ``beta <= 0``, or ``level`` not in
            ``(0, 1)``.
    """
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be positive")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    tail = (1.0 - level) / 2.0
    return _gamma_ppf(tail, alpha, beta), _gamma_ppf(1.0 - tail, alpha, beta)


def normal_credible_interval(
    mu: float,
    sigma: float,
    level: float = 0.95,
) -> tuple[float, float]:
    """Equal-tailed credible interval for ``Normal(mu, sigma**2)``.

    Args:
        mu: posterior mean.
        sigma: posterior standard deviation (``> 0``).
        level: credible mass in ``(0, 1)``.

    Returns:
        ``(lower, upper)`` as ``mu +/- z * sigma``.

    Raises:
        ValueError: if ``sigma <= 0`` or ``level`` not in ``(0, 1)``.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    tail = (1.0 - level) / 2.0
    z_low = _standard_normal_ppf(tail)
    z_high = _standard_normal_ppf(1.0 - tail)
    return mu + z_low * sigma, mu + z_high * sigma


def credible_interval(
    alpha: float,
    beta: float,
    level: float = 0.95,
) -> tuple[float, float]:
    """Generic equal-tailed credible interval for ``Beta(alpha, beta)``.

    Convenience alias for :func:`beta_credible_interval` — the most common
    credible interval in Beta–Binomial workflows. For Gamma or Normal
    intervals use :func:`gamma_credible_interval` or
    :func:`normal_credible_interval` directly.

    Args:
        alpha: Beta shape ``a > 0``.
        beta: Beta shape ``b > 0``.
        level: credible mass in ``(0, 1)``.

    Returns:
        ``(lower, upper)`` quantiles.
    """
    return beta_credible_interval(alpha, beta, level=level)


# ---------------------------------------------------------------------------
# Conjugate updates
# ---------------------------------------------------------------------------


def beta_binomial_update(
    alpha: float,
    beta: float,
    k: int,
    n: int,
    *,
    credible_level: float = 0.95,
) -> dict[str, float]:
    """Beta–Binomial conjugate update.

    Prior ``p ~ Beta(alpha, beta)``, likelihood ``k ~ Binomial(n, p)``.
    Posterior is ``Beta(alpha + k, beta + n - k)``.

    Args:
        alpha: prior Beta shape ``a > 0``.
        beta: prior Beta shape ``b > 0``.
        k: observed successes (``0 <= k <= n``).
        n: number of trials (``n >= 0``; ``n == 0`` means no data).
        credible_level: credible mass for the equal-tailed interval,
            in ``(0, 1)``.

    Returns:
        Dictionary with posterior parameters and summaries::

            {
                "alpha_post": ...,
                "beta_post": ...,
                "alpha": ...,        # alias for alpha_post
                "beta": ...,         # alias for beta_post
                "mean": ...,
                "variance": ...,
                "var": ...,          # alias for variance
                "ci_lower": ...,
                "ci_upper": ...,
                "credible_level": ...,
            }

        ``mean`` is the posterior mean ``a'/(a'+b')`` and ``ci_*`` are the
        equal-tailed credible bounds at ``credible_level``.

    Raises:
        ValueError: if ``alpha <= 0``, ``beta <= 0``, ``n < 0``,
            ``k`` outside ``[0, n]``, or ``credible_level`` not in
            ``(0, 1)``.
    """
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be positive")
    if n < 0:
        raise ValueError("n must be non-negative")
    if k < 0 or k > n:
        raise ValueError("k must be in [0, n]")
    if not 0.0 < credible_level < 1.0:
        raise ValueError("credible_level must be in (0, 1)")

    a_post = alpha + float(k)
    b_post = beta + float(n - k)
    total = a_post + b_post
    mean = a_post / total
    variance = (a_post * b_post) / (total * total * (total + 1.0))
    lo, hi = beta_credible_interval(a_post, b_post, level=credible_level)

    return {
        "alpha_post": a_post,
        "beta_post": b_post,
        "alpha": a_post,
        "beta": b_post,
        "mean": mean,
        "variance": variance,
        "var": variance,
        "ci_lower": lo,
        "ci_upper": hi,
        "credible_level": credible_level,
    }


def gamma_poisson_update(
    alpha: float,
    beta: float,
    k: int,
    exposure: float = 1.0,
    *,
    credible_level: float = 0.95,
) -> dict[str, float]:
    """Gamma–Poisson conjugate update (rate parametrization).

    Prior ``lambda ~ Gamma(alpha, beta)`` with shape ``alpha`` and rate
    ``beta`` (mean ``alpha / beta``). Likelihood
    ``k ~ Poisson(lambda * exposure)``. Posterior is
    ``Gamma(alpha + k, beta + exposure)``.

    Args:
        alpha: prior Gamma shape ``> 0``.
        beta: prior Gamma rate ``> 0``.
        k: observed count (``>= 0``).
        exposure: exposure / time window (``> 0``; default 1.0).
        credible_level: credible mass for the equal-tailed interval.

    Returns:
        Dictionary with posterior parameters and summaries::

            {
                "alpha_post": ...,
                "beta_post": ...,
                "alpha": ...,
                "beta": ...,
                "mean": ...,
                "variance": ...,
                "var": ...,
                "ci_lower": ...,
                "ci_upper": ...,
                "credible_level": ...,
            }

        ``mean = alpha_post / beta_post``; credible bounds via
        :func:`gamma_credible_interval`.

    Raises:
        ValueError: if ``alpha <= 0``, ``beta <= 0``, ``k < 0``,
            ``exposure <= 0``, or ``credible_level`` not in ``(0, 1)``.
    """
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be positive")
    if k < 0:
        raise ValueError("k must be non-negative")
    if exposure <= 0:
        raise ValueError("exposure must be positive")
    if not 0.0 < credible_level < 1.0:
        raise ValueError("credible_level must be in (0, 1)")

    a_post = alpha + float(k)
    b_post = beta + exposure
    mean = a_post / b_post
    variance = a_post / (b_post * b_post)
    lo, hi = gamma_credible_interval(a_post, b_post, level=credible_level)

    return {
        "alpha_post": a_post,
        "beta_post": b_post,
        "alpha": a_post,
        "beta": b_post,
        "mean": mean,
        "variance": variance,
        "var": variance,
        "ci_lower": lo,
        "ci_upper": hi,
        "credible_level": credible_level,
    }


def normal_normal_update(
    mu0: float,
    sigma0: float,
    data: list[float],
    sigma: float,
    *,
    credible_level: float = 0.95,
) -> dict[str, float]:
    """Normal–Normal conjugate update with known observation variance.

    Prior ``mu ~ Normal(mu0, sigma0**2)``, likelihood
    ``x_i ~ Normal(mu, sigma**2)`` i.i.d. Posterior is
    ``Normal(mu_n, sigma_n**2)`` where::

        sigma_n^2 = 1 / (1/sigma0^2 + n/sigma^2)
        mu_n      = sigma_n^2 * (mu0/sigma0^2 + n*xbar/sigma^2)

    With no data (``data == []``) the posterior equals the prior.

    Args:
        mu0: prior mean.
        sigma0: prior standard deviation (``> 0``).
        data: observations (``list[float]``; may be empty).
        sigma: known observation standard deviation (``> 0``).
        credible_level: credible mass for the equal-tailed interval.

    Returns:
        Dictionary with posterior summaries::

            {
                "mu_post": ...,
                "sigma_post": ...,
                "mean": ...,         # alias for mu_post
                "variance": ...,     # sigma_post**2
                "var": ...,          # alias
                "ci_lower": ...,
                "ci_upper": ...,
                "credible_level": ...,
                "n": ...,            # number of observations as float
            }

    Raises:
        ValueError: if ``sigma0 <= 0``, ``sigma <= 0``, or
            ``credible_level`` not in ``(0, 1)``.
    """
    if sigma0 <= 0:
        raise ValueError("sigma0 must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if not 0.0 < credible_level < 1.0:
        raise ValueError("credible_level must be in (0, 1)")

    n = len(data)
    if n == 0:
        mu_post = mu0
        sigma_post = sigma0
    else:
        xbar = sum(data) / n
        prec0 = 1.0 / (sigma0 * sigma0)
        prec_n = prec0 + n / (sigma * sigma)
        sigma_post = math.sqrt(1.0 / prec_n)
        mu_post = (1.0 / prec_n) * (mu0 * prec0 + n * xbar / (sigma * sigma))

    variance = sigma_post * sigma_post
    lo, hi = normal_credible_interval(mu_post, sigma_post, level=credible_level)

    return {
        "mu_post": mu_post,
        "sigma_post": sigma_post,
        "mean": mu_post,
        "variance": variance,
        "var": variance,
        "ci_lower": lo,
        "ci_upper": hi,
        "credible_level": credible_level,
        "n": float(n),
    }


def bayes_factor(
    k: int,
    n: int,
    p0: float,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> float:
    """Bayes factor ``BF10`` for a binomial rate vs a point null.

    Compares

    * ``H0: p = p0``  (point null), and
    * ``H1: p ~ Beta(alpha, beta)`` (composite alternative)

    on data ``k ~ Binomial(n, p)``. The marginal likelihood under ``H1``
    is the beta-binomial; under ``H0`` it is the binomial point mass.
    The binomial coefficient ``C(n,k)`` cancels:

    .. math::

        BF_{10} = \\frac{B(\\alpha+k, \\beta+n-k)}{B(\\alpha,\\beta)}
                  \\Big/ \\left(p_0^k (1-p_0)^{n-k}\\right)

    ``BF10 > 1`` favours ``H1``; ``BF10 < 1`` favours ``H0``.
    ``BF01 = 1 / BF10``. With the default ``Beta(1,1)`` (uniform) prior
    this is the standard Bayesian test of a proportion.

    Args:
        k: observed successes (``0 <= k <= n``).
        n: number of trials (``n >= 0``).
        p0: null rate in ``(0, 1)`` (use ``1e-12`` / ``1-1e-12`` for
            boundary cases instead of exact 0/1).
        alpha: Beta ``a`` for ``H1`` (``> 0``).
        beta: Beta ``b`` for ``H1`` (``> 0``).

    Returns:
        ``BF10`` as a positive ``float`` (``math.inf`` if ``H0``
        likelihood is zero).

    Raises:
        ValueError: if ``alpha <= 0``, ``beta <= 0``, ``n < 0``,
            ``k`` outside ``[0, n]``, or ``p0`` outside ``[0, 1]``.
    """
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be positive")
    if n < 0:
        raise ValueError("n must be non-negative")
    if k < 0 or k > n:
        raise ValueError("k must be in [0, n]")
    if not 0.0 <= p0 <= 1.0:
        raise ValueError("p0 must be in [0, 1]")

    # Handle boundary p0 exactly to avoid log(0).
    if p0 == 0.0:
        if k == 0:
            log_h0 = n * math.log(1.0)  # 0.0
        else:
            return math.inf  # H0 gives zero likelihood, BF10 infinite
    elif p0 == 1.0:
        if k == n:
            log_h0 = n * math.log(1.0)
        else:
            return math.inf
    else:
        log_h0 = k * math.log(p0) + (n - k) * math.log(1.0 - p0)

    # log Beta(alpha, beta) = lgamma(a)+lgamma(b)-lgamma(a+b)
    log_beta_prior = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    log_beta_post = (
        math.lgamma(alpha + k) + math.lgamma(beta + n - k) - math.lgamma(alpha + beta + n)
    )
    log_marginal_h1 = log_beta_post - log_beta_prior
    # Both H0 and H1 share C(n,k); cancellation already reflected.
    log_bf10 = log_marginal_h1 - log_h0
    # Guard against overflow
    try:
        return math.exp(log_bf10)
    except OverflowError:  # pragma: no cover - extreme BF only
        return math.inf  # pragma: no cover
