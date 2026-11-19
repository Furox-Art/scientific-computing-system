"""Tests for the time-series analysis tools in cds.stats."""

from __future__ import annotations

import dataclasses
import math

import pytest

from cds.stats.time_series import (
    LjungBoxResult,
    StationarityResult,
    autocorrelation,
    autocorrelation_function,
    difference,
    exponential_smoothing,
    kpss_statistic,
    ljung_box,
    moving_average,
    partial_autocorrelation,
    seasonal_decompose,
)


# --------------------------------------------------------------------------- #
# autocorrelation
# --------------------------------------------------------------------------- #
def test_autocorrelation_lag0_is_one() -> None:
    assert autocorrelation([1.0, 2.0, 3.0, 4.0], lag=0) == pytest.approx(1.0)


def test_autocorrelation_linear_trend_positive() -> None:
    # Strongly trending series -> high positive lag-1 autocorrelation.
    data = [float(i) for i in range(20)]
    assert autocorrelation(data, lag=1) > 0.8


def test_autocorrelation_alternating_negative() -> None:
    # Alternating series [1,-1,...,n=6]: biased estimator r_1 = -5/6 (num=-5, den=6).
    data = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    assert autocorrelation(data, lag=1) == pytest.approx(-5.0 / 6.0, abs=1e-9)
    # n=200 alternating: biased ACF = -(n-1)/n = -199/200 = -0.995.
    long_alt = [1.0 if i % 2 == 0 else -1.0 for i in range(200)]
    assert autocorrelation(long_alt, lag=1) == pytest.approx(-199.0 / 200.0, abs=1e-9)


def test_autocorrelation_constant_series_zero() -> None:
    # Zero-variance series -> 0.0 (divide-by-zero guard).
    assert autocorrelation([5.0, 5.0, 5.0, 5.0], lag=1) == 0.0


def test_autocorrelation_too_short() -> None:
    with pytest.raises(ValueError):
        autocorrelation([1.0], lag=1)


def test_autocorrelation_lag_out_of_range() -> None:
    with pytest.raises(ValueError):
        autocorrelation([1.0, 2.0, 3.0], lag=3)
    with pytest.raises(ValueError):
        autocorrelation([1.0, 2.0, 3.0], lag=-1)


def test_autocorrelation_value_matches_definition() -> None:
    # Hand-checked: mean=2, den = (1+0+1+0)=2, num(lag1)=(1*0)+(0*1)=0 -> 0.0
    assert autocorrelation([1.0, 2.0, 3.0, 2.0], lag=1) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# autocorrelation_function
# --------------------------------------------------------------------------- #
def test_acf_length_and_r0() -> None:
    data = [float(i) for i in range(10)]
    acf = autocorrelation_function(data, max_lag=4)
    assert len(acf) == 5
    assert acf[0] == pytest.approx(1.0)


def test_acf_default_max_lag() -> None:
    data = [1.0, 2.0, 3.0, 4.0]
    acf = autocorrelation_function(data)
    assert len(acf) == 4  # lags 0..n-1


def test_acf_too_short() -> None:
    with pytest.raises(ValueError):
        autocorrelation_function([1.0])


def test_acf_negative_max_lag() -> None:
    with pytest.raises(ValueError):
        autocorrelation_function([1.0, 2.0, 3.0], max_lag=-1)


# --------------------------------------------------------------------------- #
# partial_autocorrelation
# --------------------------------------------------------------------------- #
def test_pacf_first_value_is_one() -> None:
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    pacf = partial_autocorrelation(data, max_lag=3)
    assert pacf[0] == pytest.approx(1.0)


def test_pacf_pure_ar1_cuttoff() -> None:
    # Deterministic AR(1) with phi=0.8 and a non-zero seed: x_t = 0.8 * x_{t-1}.
    # PACF should spike at lag 1 (~0.8) then cut off toward 0.
    data = [1.0]
    for _ in range(200):
        data.append(0.8 * data[-1])  # deterministic AR(1), phi=0.8
    pacf = partial_autocorrelation(data, max_lag=4)
    # phi_11 ~ 0.8 for a pure AR(1).
    assert pacf[1] == pytest.approx(0.8, abs=0.05)
    # Higher-order PACF should be near 0 (AR(1) cutoff property).
    assert abs(pacf[2]) < 0.1
    assert abs(pacf[3]) < 0.1


def test_pacf_too_short() -> None:
    with pytest.raises(ValueError):
        partial_autocorrelation([1.0])


def test_pacf_negative_max_lag() -> None:
    with pytest.raises(ValueError):
        partial_autocorrelation([1.0, 2.0, 3.0], max_lag=-1)


# --------------------------------------------------------------------------- #
# moving_average
# --------------------------------------------------------------------------- #
def test_moving_average_basic() -> None:
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    out = moving_average(data, window=1)
    assert len(out) == 5
    # Center value (index 2) with window=1 averages [2,3,4] = 3.
    assert out[2] == pytest.approx(3.0)


def test_moving_average_constant_preserved() -> None:
    out = moving_average([7.0, 7.0, 7.0], window=2)
    assert all(abs(v - 7.0) < 1e-9 for v in out)


def test_moving_average_empty() -> None:
    assert moving_average([], window=3) == []


def test_moving_average_invalid_window() -> None:
    with pytest.raises(ValueError):
        moving_average([1.0, 2.0], window=0)


def test_moving_average_smooths_noise() -> None:
    # Noisy up-trend: MA should reduce variance.
    data = [1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0]
    smoothed = moving_average(data, window=2)
    var_raw = sum((x - (sum(data) / len(data))) ** 2 for x in data) / len(data)
    var_sm = sum((x - (sum(smoothed) / len(smoothed))) ** 2 for x in smoothed) / len(smoothed)
    assert var_sm < var_raw


# --------------------------------------------------------------------------- #
# exponential_smoothing
# --------------------------------------------------------------------------- #
def test_exponential_smoothing_first_value_preserved() -> None:
    out = exponential_smoothing([5.0, 3.0, 4.0], alpha=0.5)
    assert out[0] == pytest.approx(5.0)


def test_exponential_smoothing_alpha_one_is_identity() -> None:
    data = [3.0, 1.0, 4.0, 1.0, 5.0]
    out = exponential_smoothing(data, alpha=1.0)
    assert out == pytest.approx(data)


def test_exponential_smoothing_empty() -> None:
    assert exponential_smoothing([], alpha=0.3) == []


def test_exponential_smoothing_invalid_alpha() -> None:
    with pytest.raises(ValueError):
        exponential_smoothing([1.0, 2.0], alpha=0.0)
    with pytest.raises(ValueError):
        exponential_smoothing([1.0, 2.0], alpha=1.5)


def test_exponential_smoothing_reduces_variance() -> None:
    data = [10.0, 0.0, 10.0, 0.0, 10.0, 0.0]
    out = exponential_smoothing(data, alpha=0.3)
    m = sum(out) / len(out)
    var = sum((x - m) ** 2 for x in out) / len(out)
    assert var < 25.0  # well below raw oscillation variance


# --------------------------------------------------------------------------- #
# difference
# --------------------------------------------------------------------------- #
def test_difference_linear_trend_is_constant() -> None:
    data = [1.0, 3.0, 5.0, 7.0, 9.0]  # y = 2t + 1
    diff = difference(data, lag=1, order=1)
    assert diff == pytest.approx([2.0, 2.0, 2.0, 2.0])


def test_difference_order_two() -> None:
    # Quadratic: second difference is constant 2.
    data = [float(i * i) for i in range(6)]  # 0,1,4,9,16,25
    diff = difference(data, lag=1, order=2)
    # Length = n - lag*order = 6 - 2 = 4; each second difference of i^2 is 2.
    assert diff == pytest.approx([2.0, 2.0, 2.0, 2.0])


def test_difference_seasonal_lag() -> None:
    data = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]  # period 3
    diff = difference(data, lag=3, order=1)
    assert diff == pytest.approx([0.0, 0.0, 0.0])


def test_difference_too_short_returns_empty() -> None:
    assert difference([1.0], lag=1, order=1) == []


def test_difference_invalid_params() -> None:
    with pytest.raises(ValueError):
        difference([1.0, 2.0], lag=0)
    with pytest.raises(ValueError):
        difference([1.0, 2.0], order=0)


# --------------------------------------------------------------------------- #
# kpss_statistic
# --------------------------------------------------------------------------- #
def test_kpss_stationary_white_noise() -> None:
    # A simple oscillating-around-mean series should be flagged stationary.
    data = [
        0.1,
        -0.2,
        0.15,
        -0.1,
        0.2,
        -0.15,
        0.05,
        -0.1,
        0.1,
        -0.2,
        0.15,
        -0.1,
        0.2,
        -0.15,
        0.05,
        -0.1,
        0.1,
        -0.2,
        0.15,
        -0.1,
        0.2,
        -0.15,
        0.05,
        -0.1,
    ]
    res = kpss_statistic(data)
    assert isinstance(res, StationarityResult)
    assert res.is_stationary is True
    assert res.lags >= 1
    assert 0.0 <= res.normal_tail_probability <= 1.0


def test_kpss_random_walk_nonstationary() -> None:
    # Cumulative sum -> a random walk (classic unit root).
    data = [0.0]
    for _ in range(200):
        data.append(data[-1] + (1.0 if (_ % 2) else -1.0) + 0.3)
    res = kpss_statistic(data)
    assert res.statistic > 0.0


def test_kpss_constant_series_stationary() -> None:
    res = kpss_statistic([5.0] * 10)
    assert res.is_stationary is True
    assert res.statistic == pytest.approx(0.0, abs=1e-9)


def test_kpss_too_short() -> None:
    with pytest.raises(ValueError):
        kpss_statistic([1.0, 2.0])


def test_kpss_explicit_lags() -> None:
    res = kpss_statistic([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], lags=2)
    assert res.lags == 2


def test_kpss_normal_tail_matches_erfc_closed_form() -> None:
    # Oracle: the indicator must be exactly P(Z > eta) = erfc(eta/sqrt(2))/2,
    # computed independently from the returned statistic.
    data = [0.1, -0.2, 0.15, -0.1, 0.2, -0.15] * 8
    res = kpss_statistic(data)
    expected = 0.5 * math.erfc(res.statistic / math.sqrt(2.0))
    assert res.normal_tail_probability == pytest.approx(expected, rel=1e-12, abs=1e-15)


def test_kpss_constant_series_tail_is_one() -> None:
    res = kpss_statistic([5.0] * 10)
    assert res.normal_tail_probability == 1.0


def test_kpss_decision_uses_published_critical_value() -> None:
    # is_stationary must track eta < 0.463 (KPSS 5% level-model critical
    # value), not the normal-tail heuristic.
    stationary = [0.1, -0.2, 0.15, -0.1, 0.2, -0.15] * 8
    walk = [0.0]
    for i in range(200):
        walk.append(walk[-1] + (1.0 if i % 2 else -1.0) + 0.3)
    for data in (stationary, walk):
        res = kpss_statistic(data)
        assert res.is_stationary == (res.statistic < 0.463)


def test_kpss_deprecated_p_value_read_warns_and_matches() -> None:
    res = kpss_statistic([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    with pytest.warns(DeprecationWarning, match="not a valid KPSS"):
        legacy = res.p_value
    assert legacy == res.normal_tail_probability


def test_kpss_deprecated_p_value_write_warns_and_updates() -> None:
    res = kpss_statistic([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    with pytest.warns(DeprecationWarning, match="deprecated"):
        res.p_value = 0.25
    assert res.normal_tail_probability == 0.25


def test_kpss_deprecated_constructor_keyword_warns() -> None:
    with pytest.warns(DeprecationWarning, match="p_value"):
        res = StationarityResult(statistic=0.1, is_stationary=True, lags=1, p_value=0.4)
    assert res.normal_tail_probability == 0.4


def test_kpss_constructor_positional_and_new_keyword() -> None:
    positional = StationarityResult(0.1, True, 1, 0.4)
    keyword = StationarityResult(
        statistic=0.1, is_stationary=True, lags=1, normal_tail_probability=0.4
    )
    assert positional == keyword
    assert positional.normal_tail_probability == 0.4


def test_kpss_constructor_requires_a_tail_value() -> None:
    with pytest.raises(TypeError, match="normal_tail_probability"):
        StationarityResult(0.1, True, 1)


def test_kpss_constructor_rejects_both_keywords() -> None:
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="not both"):
        StationarityResult(0.1, True, 1, normal_tail_probability=0.4, p_value=0.4)


def test_kpss_result_still_a_dataclass() -> None:
    # dataclasses.replace must keep working after the field rename.
    res = kpss_statistic([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    replaced = dataclasses.replace(res, lags=5)
    assert replaced.lags == 5
    assert replaced.normal_tail_probability == res.normal_tail_probability
    assert "normal_tail_probability" in {f.name for f in dataclasses.fields(res)}


# --------------------------------------------------------------------------- #
# seasonal_decompose
# --------------------------------------------------------------------------- #
def test_seasonal_decompose_recovers_pattern() -> None:
    # Pure seasonal signal with no trend: seasonal should dominate.
    base = [1.0, 2.0, 3.0, 4.0]
    data = base * 6  # 6 full periods, period=4
    trend, seasonal, residual = seasonal_decompose(data, period=4)
    assert len(trend) == len(data)
    assert len(seasonal) == len(data)
    assert len(residual) == len(data)
    # Seasonal pattern sums to ~0 across one period (additive model).
    one_period = seasonal[:4]
    assert sum(one_period) == pytest.approx(0.0, abs=1e-9)


def test_seasonal_decompose_invalid_period() -> None:
    with pytest.raises(ValueError):
        seasonal_decompose([1.0, 2.0, 3.0], period=1)


def test_seasonal_decompose_too_short() -> None:
    with pytest.raises(ValueError):
        seasonal_decompose([1.0, 2.0], period=4)


def test_seasonal_decompose_residual_small_for_clean_signal() -> None:
    # Clean additive signal -> residuals near zero.
    base = [10.0, 20.0, 30.0, 40.0]
    data = base * 5
    _, _, residual = seasonal_decompose(data, period=4)
    # The centered MA trend will not perfectly reproduce a sawtooth, but
    # residuals should stay bounded.
    assert all(abs(r) < 20.0 for r in residual)


# --------------------------------------------------------------------------- #
# ljung_box
# --------------------------------------------------------------------------- #
def test_ljung_box_white_noise_not_autocorrelated() -> None:
    # Deterministic, low-autocorrelation series.
    data = [float(((-1) ** i) * (i % 3)) for i in range(50)]
    res = ljung_box(data, lags=5)
    assert isinstance(res, LjungBoxResult)
    assert res.df == 5
    assert res.statistic >= 0.0
    assert 0.0 <= res.p_value <= 1.0


def test_ljung_box_trending_series_flagged() -> None:
    # Strongly trending -> significant autocorrelation expected.
    data = [float(i) for i in range(100)]
    res = ljung_box(data, lags=10)
    assert res.has_autocorrelation is True


def test_ljung_box_invalid_lags() -> None:
    with pytest.raises(ValueError):
        ljung_box([1.0, 2.0, 3.0], lags=0)


def test_ljung_box_too_short() -> None:
    with pytest.raises(ValueError):
        ljung_box([1.0, 2.0], lags=5)
