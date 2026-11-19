"""Time-series analysis — autocorrelation, moving averages, stationarity.

Complements the descriptive and hypothesis-testing tools in
:mod:`cds.stats` with the classical toolkit for ordered, equally-spaced
observations. All routines are pure Python with no external dependencies.

The stationarity test implements the KPSS intuition (testing the null of
*stationarity* against a unit-root alternative) via a simplified
lag-window long-run variance estimator. The KPSS asymptotic null distribution
is non-standard (a functional of a Brownian bridge), so this module does NOT
return a p-value: ``StationarityResult.normal_tail_probability`` is the
standard-normal upper-tail probability of the statistic, exposed only as a
teaching/heuristic indicator. For inference, compare ``statistic`` against
published KPSS critical values (level model: 0.347 at 10%, 0.463 at 5%,
0.739 at 1%; Kwiatkowski et al. 1992, Table 1). The former ``p_value`` field
name survives as a deprecated alias because a field called "p_value" invited
downstream misuse of a quantity that is not a valid p-value.

References:
    - Box, G.E.P. & Jenkins, G.M. (1970). Time Series Analysis: Forecasting & Control.
    - KPSS: Kwiatkowski et al. (1992). Testing the null of stationarity.
    - Hyndman, R.J. & Athanasopoulos, G. Forecasting: Principles and Practice.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

from cds.core._numeric import NEAR_ZERO
from cds.stats.descriptive import mean
from cds.stats.hypothesis_tests import chi2_sf


@dataclass
class StationarityResult:
    """Outcome of a stationarity (KPSS-style) test.

    Attributes:
        statistic: the KPSS LM-type statistic (eta). Compare it against
            published asymptotic critical values for formal inference
            (level model: 0.463 at the 5% level).
        is_stationary: heuristic decision at the 5% level (``eta < 0.463``).
        lags: number of lags used for the long-run variance estimator.
        normal_tail_probability: ``P(Z > statistic)`` for ``Z ~ N(0, 1)``.
            This is a teaching/heuristic indicator, NOT a valid KPSS p-value:
            the true asymptotic null distribution of the KPSS statistic is
            non-standard. The name says exactly what the quantity is so it
            cannot be mistaken for a p-value.

    The former field name ``p_value`` is kept as a deprecated read/write alias
    and as a deprecated constructor keyword; both emit ``DeprecationWarning``.
    """

    statistic: float
    is_stationary: bool
    lags: int
    normal_tail_probability: float

    def __init__(
        self,
        statistic: float,
        is_stationary: bool,
        lags: int,
        normal_tail_probability: float | None = None,
        p_value: float | None = None,
    ) -> None:
        # A custom __init__ (the dataclass decorator keeps an explicitly
        # defined one) so the deprecated ``p_value=`` keyword keeps working
        # for existing callers while new code uses the honest field name.
        if p_value is not None:
            warnings.warn(
                "StationarityResult(p_value=...) is deprecated: the value is a "
                "normal-tail heuristic, not a valid KPSS p-value. Pass "
                "normal_tail_probability=... instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if normal_tail_probability is not None and p_value is not None:
            raise ValueError(
                "pass either normal_tail_probability or the deprecated p_value, not both"
            )
        value = normal_tail_probability if normal_tail_probability is not None else p_value
        if value is None:
            raise TypeError(
                "StationarityResult requires normal_tail_probability "
                "(the deprecated p_value keyword is also accepted)"
            )
        self.statistic = statistic
        self.is_stationary = is_stationary
        self.lags = lags
        self.normal_tail_probability = value

    @property
    def p_value(self) -> float:
        """Deprecated alias for :attr:`normal_tail_probability`.

        Not a valid KPSS p-value: the KPSS null distribution is non-standard.
        Use ``normal_tail_probability`` for the heuristic indicator and the
        published critical values for inference.
        """
        warnings.warn(
            "StationarityResult.p_value is deprecated and is not a valid KPSS "
            "p-value (the KPSS null distribution is non-standard). Read "
            "normal_tail_probability instead and compare statistic against "
            "published KPSS critical values.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.normal_tail_probability

    @p_value.setter
    def p_value(self, value: float) -> None:
        warnings.warn(
            "StationarityResult.p_value is deprecated and is not a valid KPSS "
            "p-value (the KPSS null distribution is non-standard). Write "
            "normal_tail_probability instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.normal_tail_probability = value


_SQRT2 = math.sqrt(2.0)


def _normal_cdf(z: float) -> float:
    """Standard normal CDF, ``P(Z <= z)``.

    Uses ``math.erfc`` and the identity ``Phi(z) = erfc(-z / sqrt(2)) / 2``,
    which is exact to double precision and keeps the left tail accurate (the
    ``0.5 * (1 + erf(z / sqrt(2)))`` form cancels catastrophically there).

    This previously hand-rolled Abramowitz & Stegun 7.1.26 and got it wrong
    twice over: 7.1.26 approximates ``erf``, not ``Phi``, so its argument must
    be ``z / sqrt(2)`` rather than ``z``, and two of the five polynomial
    coefficients were missing. The result was off by up to 0.196 —
    ``Phi(0)`` returned 0.304 instead of 0.5 — which silently corrupted every
    tail probability from :func:`kpss_statistic`. There is no reason to
    approximate a function the standard library computes exactly.
    """
    return 0.5 * math.erfc(-z / _SQRT2)


def autocorrelation(data: list[float], lag: int = 1) -> float:
    """Sample autocorrelation at a given ``lag``.

    Computes ``r_k = sum_{t=k+1}^{n} (x_t - x̄)(x_{t-k} - x̄) / sum_{t=1}^{n} (x_t - x̄)²``,
    the biased estimator normalized by the full-sample variance. Returns ``0.0``
    for a constant series (zero variance). [Box & Jenkins 1970]

    Args:
        data: ordered, equally-spaced observations.
        lag: number of steps back to correlate (``0 <= lag < n``).

    Returns:
        Autocorrelation in ``[-1, 1]`` (clamped).

    Raises:
        ValueError: if ``lag`` is out of range or ``data`` too short.
    """
    n = len(data)
    if n < 2:
        raise ValueError("autocorrelation requires at least 2 observations")
    if lag < 0:
        raise ValueError("lag must be non-negative")
    if lag >= n:
        raise ValueError(f"lag must be < n (got lag={lag}, n={n})")

    m = mean(data)
    den = sum((x - m) ** 2 for x in data)
    if den <= NEAR_ZERO:
        return 0.0
    num = sum((data[t] - m) * (data[t - lag] - m) for t in range(lag, n))
    r = num / den
    # Guard against tiny floating overshoot beyond [-1, 1].
    return max(-1.0, min(1.0, r))


def autocorrelation_function(data: list[float], max_lag: int | None = None) -> list[float]:
    """Sample ACF for lags ``0..max_lag`` inclusive.

    Args:
        data: ordered observations (``n >= 2``).
        max_lag: largest lag to evaluate (defaults to ``n // 2``, capped at ``n-1``).

    Returns:
        List ``[r_0, r_1, ..., r_{max_lag}]``; ``r_0`` is always ``1.0``.
    """
    n = len(data)
    if n < 2:
        raise ValueError("autocorrelation_function requires at least 2 observations")
    upper = n - 1 if max_lag is None else min(max_lag, n - 1)
    if upper < 0:
        raise ValueError("max_lag must be non-negative")
    return [autocorrelation(data, lag) for lag in range(upper + 1)]


def partial_autocorrelation(data: list[float], max_lag: int | None = None) -> list[float]:
    """Sample partial autocorrelation function (PACF) via Durbin-Levinson.

    Recursively solves the Yule-Walker system for successive AR orders,
    returning the last coefficient of each order — the PACF. [Box & Jenkins 1970]

    Args:
        data: ordered observations (``n >= 2``).
        max_lag: largest lag (defaults to ``n // 2``, capped at ``n-1``).

    Returns:
        List ``[phi_00, phi_11, ..., phi_{kk}]`` with ``phi_00 = 1.0``.
    """
    n = len(data)
    if n < 2:
        raise ValueError("partial_autocorrelation requires at least 2 observations")
    upper = (n // 2) if max_lag is None else min(max_lag, n - 1)
    if upper < 0:
        raise ValueError("max_lag must be non-negative")

    gamma = autocorrelation_function(data, upper)  # gamma[0..upper]
    pacf = [1.0]
    # phi[k][j] holds the j-th AR(k) coefficient.
    prev_phi: list[float] = []
    for k in range(1, upper + 1):
        # Numerator: gamma[k] - sum_{j=1}^{k-1} phi[k-1][j] * gamma[k-j]
        num = gamma[k] - sum(prev_phi[j - 1] * gamma[k - j] for j in range(1, k))
        den = 1.0 - sum(prev_phi[j - 1] * gamma[j] for j in range(1, k))
        phi_kk = num / den if abs(den) > NEAR_ZERO else 0.0
        curr_phi = [0.0] * k
        curr_phi[k - 1] = phi_kk
        for j in range(1, k):
            curr_phi[j - 1] = prev_phi[j - 1] - phi_kk * prev_phi[k - j - 1]
        pacf.append(phi_kk)
        prev_phi = curr_phi
    return pacf


def moving_average(data: list[float], window: int = 3) -> list[float]:
    """Centered simple moving average (SMA) of fixed ``window`` size.

    Uses a centered window so the smoothed series stays time-aligned. For even
    ``window`` the result is placed at the left-of-center index. Boundary points
    shrink the window symmetrically, so the output has the same length as the
    input.

    Args:
        data: ordered observations (``n >= 1``).
        window: half-extent controls; must be ``>= 1``.

    Returns:
        Smoothed series of the same length as ``data``.

    Raises:
        ValueError: if ``window < 1``.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    n = len(data)
    if n == 0:
        return []
    out: list[float] = []
    for i in range(n):
        lo = max(0, i - window + 1)
        hi = min(n, i + window)
        segment = data[lo:hi]
        out.append(sum(segment) / len(segment))
    return out


def exponential_smoothing(data: list[float], alpha: float = 0.5) -> list[float]:
    """Simple exponential smoothing (SES).

    Recurrence ``s_0 = x_0``; ``s_t = alpha * x_t + (1 - alpha) * s_{t-1}``.
    Equivalent to an exponentially weighted moving average; a common
    zero-parameter forecast baseline. [Hyndman & Athanasopoulos]

    Args:
        data: ordered observations (``n >= 1``).
        alpha: smoothing factor in ``(0, 1]`` (higher = faster adaptation).

    Returns:
        Smoothed series of the same length as ``data``.

    Raises:
        ValueError: if ``alpha`` is outside ``(0, 1]``.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    if not data:
        return []
    out = [data[0]]
    for t in range(1, len(data)):
        out.append(alpha * data[t] + (1.0 - alpha) * out[t - 1])
    return out


def difference(data: list[float], lag: int = 1, order: int = 1) -> list[float]:
    """Discrete differencing to remove trend / seasonality.

    ``order=1``: ``(x_t - x_{t-lag})``; ``order=k`` applies the difference
    recursively ``k`` times. The classic Box-Jenkins tool for rendering a
    non-stationary series approximately stationary before ARIMA fitting.

    Args:
        data: ordered observations.
        lag: seasonal lag (``>= 1``).
        order: number of times to apply the difference (``>= 1``).

    Returns:
        Differenced series (shorter than the input by ``lag * order``).

    Raises:
        ValueError: if ``lag`` or ``order`` are ``< 1``.
    """
    if lag < 1:
        raise ValueError("lag must be >= 1")
    if order < 1:
        raise ValueError("order must be >= 1")
    series = list(data)
    for _ in range(order):
        if len(series) <= lag:
            return []
        series = [series[t] - series[t - lag] for t in range(lag, len(series))]
    return series


def kpss_statistic(data: list[float], lags: int | None = None) -> StationarityResult:
    """KPSS-style stationarity test (null: the series is stationary).

    Computes the LM statistic from detrended (level) residuals using a
    Bartlett (Newey-West) long-run variance estimator. Tests the null of
    *stationarity* (opposite of the ADF test). [Kwiatkowski et al. 1992]

    No p-value is returned. The KPSS asymptotic null distribution is
    non-standard (a functional of a Brownian bridge), so referring the
    statistic to N(0,1) does NOT produce a valid p-value. The result carries
    ``normal_tail_probability`` — the standard-normal upper-tail probability
    of the statistic — purely as a teaching/heuristic indicator, and
    ``is_stationary`` applies the published 5% critical value (0.463 for the
    level model). For formal inference compare ``statistic`` against the
    published asymptotic critical values (0.347 at 10%, 0.463 at 5%, 0.739 at
    1%). The old ``p_value`` attribute remains as a deprecated alias that
    warns on access.

    Args:
        data: ordered observations (``n >= 3``).
        lags: truncation lag for the long-run variance (defaults to
            ``floor(4 * (n/100)^(1/4))``, the Schwert rule).

    Returns:
        :class:`StationarityResult`.

    Raises:
        ValueError: if the series is too short.
    """
    n = len(data)
    if n < 3:
        raise ValueError("kpss_statistic requires at least 3 observations")

    if lags is None:
        # Schwert (1989) truncation-lag rule.
        lags = int(4.0 * (n / 100.0) ** 0.25)
        lags = max(1, lags)

    # Level model: residual around the mean.
    m = mean(data)
    residuals = [x - m for x in data]
    cum = 0.0
    s = 0.0
    for r in residuals:
        cum += r
        s += cum * cum

    # Long-run variance with Bartlett kernel.
    n2 = n * n
    # Unreachable: n >= 3 is enforced above, so n^2 >= 9 >> NEAR_ZERO.
    if n2 <= NEAR_ZERO:  # pragma: no cover
        return StationarityResult(
            statistic=0.0, is_stationary=True, lags=lags, normal_tail_probability=1.0
        )
    gamma0 = sum(r * r for r in residuals) / n
    if gamma0 <= NEAR_ZERO:
        # Zero-variance (constant) series is trivially stationary.
        return StationarityResult(
            statistic=0.0, is_stationary=True, lags=lags, normal_tail_probability=1.0
        )

    lr_var = gamma0
    for k in range(1, lags + 1):
        weight = 1.0 - k / (lags + 1.0)
        gamma_k = sum(residuals[t] * residuals[t - k] for t in range(k, n)) / n
        lr_var += 2.0 * weight * gamma_k
    # Defensive fallback: lr_var starts at gamma0 (> 0 here) and only adds
    # Bartlett-weighted autocovariances. Reaching ~0 requires the weighted
    # autocovariances to almost exactly cancel gamma0, which cannot occur for
    # the non-negative-definite spectral estimate — hence unreachable.
    if lr_var <= NEAR_ZERO:  # pragma: no cover
        lr_var = gamma0

    eta = s / (n2 * lr_var)
    # Heuristic indicator only: the upper tail of N(0,1) at eta. The real KPSS
    # asymptotic null is non-standard, so this is NOT a valid p-value -- the
    # field name says what it is, and the decision below uses the published
    # 5% critical value instead.
    p = max(0.0, min(1.0, 1.0 - _normal_cdf(eta)))
    # KPSS rejects stationarity for large eta; critical ~0.463 at 5% level.
    is_stat = eta < 0.463
    return StationarityResult(
        statistic=eta, is_stationary=is_stat, lags=lags, normal_tail_probability=p
    )


def seasonal_decompose(
    data: list[float], period: int
) -> tuple[list[float], list[float], list[float]]:
    """Classical additive seasonal decomposition (trend, seasonal, residual).

    Estimates the trend via a centered moving average whose window matches the
    seasonal ``period``, the seasonal pattern as the period-averaged detrended
    mean, and the residual as ``data - trend - seasonal``. [Hyndman & Athanasopoulos]

    Args:
        data: ordered observations (``n >= 2 * period``).
        period: seasonal cycle length (``>= 2``).

    Returns:
        Tuple ``(trend, seasonal, residual)`` each of length ``n``.

    Raises:
        ValueError: if ``period`` is invalid or the series too short.
    """
    n = len(data)
    if period < 2:
        raise ValueError("period must be >= 2")
    if n < 2 * period:
        raise ValueError(f"need at least 2 full periods (got n={n}, period={period})")

    half = period // 2
    trend = moving_average(data, window=half if half >= 1 else 1)

    # Detrended seasonal indices.
    detrended = [data[i] - trend[i] for i in range(n)]
    seasonal_idx: list[float] = [0.0] * period
    counts = [0] * period
    for i in range(n):
        s = i % period
        seasonal_idx[s] += detrended[i]
        counts[s] += 1
    seasonal_idx = [(seasonal_idx[s] / counts[s]) if counts[s] > 0 else 0.0 for s in range(period)]
    # Normalize seasonal pattern to sum to zero (additive model).
    seasonal_mean = sum(seasonal_idx) / period
    seasonal_idx = [s - seasonal_mean for s in seasonal_idx]

    seasonal = [seasonal_idx[i % period] for i in range(n)]
    residual = [data[i] - trend[i] - seasonal[i] for i in range(n)]
    return trend, seasonal, residual


@dataclass
class LjungBoxResult:
    """Outcome of the Ljung-Box portmanteau test for autocorrelation.

    Attributes:
        statistic: the Ljung-Box Q statistic.
        p_value: approximate p-value from a chi-square distribution with
            ``lags`` degrees of freedom.
        df: degrees of freedom (``= lags``).
        has_autocorrelation: heuristic decision at 5% (True = reject independence).
    """

    statistic: float
    p_value: float
    df: int
    has_autocorrelation: bool


def ljung_box(data: list[float], lags: int = 10) -> LjungBoxResult:
    """Ljung-Box portmanteau test for autocorrelation.

    Tests whether any of the first ``lags`` autocorrelations differ from zero.
    Null hypothesis: the data are independently distributed (no autocorrelation).
    ``Q = n(n+2) * sum_{k=1}^{lags} r_k^2 / (n - k)`` referred to a chi-square
    with ``lags`` degrees of freedom. [Ljung & Box 1978]

    Args:
        data: ordered observations (``n >= lags + 2``).
        lags: number of lags to test (``>= 1``).

    Returns:
        :class:`LjungBoxResult`.

    Raises:
        ValueError: if ``lags`` is invalid or the series too short.
    """
    n = len(data)
    if lags < 1:
        raise ValueError("lags must be >= 1")
    if n < lags + 2:
        raise ValueError(f"need n >= lags + 2 (got n={n}, lags={lags})")

    q = 0.0
    for k in range(1, lags + 1):
        r = autocorrelation(data, k)
        q += (r * r) / (n - k)
    q *= n * (n + 2)

    p = chi2_sf(q, lags)
    return LjungBoxResult(
        statistic=q,
        p_value=p,
        df=lags,
        has_autocorrelation=p < 0.05,
    )
