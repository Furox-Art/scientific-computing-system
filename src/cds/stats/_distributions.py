"""Probability tail functions for the classical hypothesis tests.

The heavy special-function kernels (incomplete gamma/beta) live in their
single home, :mod:`cds.math_utils.special`; this module keeps the historical
``_``-prefixed names as aliases so existing imports
(``from cds.stats.hypothesis_tests import _gser`` and the test-suite's
coverage imports) keep working unchanged.

Provides the survival functions ``t_sf`` (Student's t), ``chi2_sf``
(chi-square) and ``f_sf`` (Fisher F) that the classical tests need to turn a
statistic into a p-value.

References:
    - Abramowitz, M., & Stegun, I. A. (1964). "Handbook of Mathematical
      Functions," §6.5, §26.
    - Student [W. S. Gosset] (1908). "The probable error of a mean."
      Biometrika, 6(1), 1-25. (t-distribution)
    - Pearson, K. (1900). Philosophical Magazine, 50(302), 157-175.
      (chi-square)
    - Fisher, R. A. (1925). "Statistical Methods for Research Workers."
      Oliver & Boyd. (F-distribution)
"""

from __future__ import annotations

from cds.math_utils.special import (
    EPS,
    FPMIN,
    MAX_ITER,
    betacf,
    betai,
    gammln,
    gammp,
    gammq,
    gcf,
    gser,
)

# Historical private names — kept as aliases for backwards compatibility.
_gammln = gammln
_gser = gser
_gcf = gcf
_gammp = gammp
_gammq = gammq
_betacf = betacf
_betai = betai
_MAXIT = MAX_ITER
_EPS = EPS
_FPMIN = FPMIN

__all__ = [
    "_gammln",
    "_gser",
    "_gcf",
    "_gammp",
    "_gammq",
    "_betacf",
    "_betai",
    "_MAXIT",
    "_EPS",
    "_FPMIN",
    "t_sf",
    "chi2_sf",
    "f_sf",
]


def t_sf(t: float, df: float) -> float:
    """Two-tailed survival probability for Student's t distribution.

    Returns P(|T| >= |t|) for T ~ t(df), via the incomplete beta function:
    p = I_{df/(df+t^2)}(df/2, 1/2).

    Reference: Student (1908); Numerical Recipes §6.14.
    """
    x = df / (df + t * t)
    return _betai(df / 2.0, 0.5, x)


def chi2_sf(x: float, df: float) -> float:
    """Upper-tail probability for the chi-square distribution: P(X >= x).

    Equals Q(df/2, x/2) with the regularized upper incomplete gamma.

    Reference: Pearson (1900); Abramowitz & Stegun §26.4.
    """
    if x <= 0.0:
        return 1.0
    return _gammq(df / 2.0, x / 2.0)


def f_sf(f: float, df1: float, df2: float) -> float:
    """Upper-tail probability for the F distribution: P(F >= f).

    Equals I_{df2/(df2+df1 f)}(df2/2, df1/2).

    Reference: Fisher (1925); Numerical Recipes §6.14.
    """
    if f <= 0.0:
        return 1.0
    x = df2 / (df2 + df1 * f)
    return _betai(df2 / 2.0, df1 / 2.0, x)
