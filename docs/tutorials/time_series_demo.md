# Time-Series Analysis Tutorial

`cds.stats` provides a classical toolkit for ordered, equally-spaced
observations — autocorrelation, partial autocorrelation, moving-average
smoothing, differencing, a KPSS-style stationarity test, seasonal
decomposition, and the Ljung-Box portmanteau test. All routines are pure
Python with no external dependencies.

```python
from cds.stats import (
    autocorrelation,
    autocorrelation_function,
    partial_autocorrelation,
    moving_average,
    exponential_smoothing,
    difference,
    kpss_statistic,
    seasonal_decompose,
    ljung_box,
)
```

## 1. Autocorrelation (ACF)

The sample autocorrelation at lag `k` measures how correlated a series is
with a delayed copy of itself. A trending series shows high positive
autocorrelation; an alternating series shows strong negative autocorrelation.

```python
# A linear ramp: each point is close to its predecessor -> r_1 near 1.
trend = [float(i) for i in range(20)]
print(f"r_1 = {autocorrelation(trend, lag=1):.3f}")  # ~0.95
print(f"r_5 = {autocorrelation(trend, lag=5):.3f}")  # still high

# Full ACF for lags 0..4 (r_0 is always 1.0).
acf = autocorrelation_function(trend, max_lag=4)
print(acf)  # [1.0, 0.95, 0.90, 0.85, 0.79]
```

## 2. Partial autocorrelation (PACF)

The PACF isolates the *direct* correlation at each lag, removing the effect
of intermediate lags via the Durbin-Levinson recursion. For a pure AR(1)
process `x_t = 0.8·x_{t-1}`, the PACF spikes at lag 1 then cuts off to ~0 —
the hallmark signature for choosing ARIMA order `p`.

```python
# Deterministic AR(1) with phi = 0.8.
ar1 = [1.0]
for _ in range(200):
    ar1.append(0.8 * ar1[-1])

pacf = partial_autocorrelation(ar1, max_lag=4)
print(f"phi_11 = {pacf[1]:.3f}")  # ~0.80
print(f"phi_22 = {pacf[2]:.3f}")  # ~0.00 (cutoff)
```

## 3. Smoothing: moving average & exponential smoothing

Smoothing reveals the underlying signal in noisy data. The centered moving
average uses a fixed window; exponential smoothing weights recent
observations more heavily via `alpha ∈ (0, 1]`.

```python
noisy = [10.0, 0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 0.0]

ma = moving_average(noisy, window=2)  # centered SMA
ses = exponential_smoothing(noisy, alpha=0.3)  # weight recent points
print(f"SMA variance  = {sum((x - 5) ** 2 for x in ma) / len(ma):.2f}")
print(f"SES variance  = {sum((x - 5) ** 2 for x in ses) / len(ses):.2f}")
```

## 4. Differencing

Differencing removes trend and seasonality — the standard Box-Jenkins
preprocessing step before fitting ARIMA models. `order=k` applies the
difference repeatedly.

```python
ramp = [1.0, 3.0, 5.0, 7.0, 9.0]  # y = 2t + 1
print(difference(ramp, lag=1, order=1))  # [2.0, 2.0, 2.0, 2.0]  (constant)

quad = [float(i * i) for i in range(6)]  # 0,1,4,9,16,25
print(difference(quad, lag=1, order=2))  # [2.0, 2.0, 2.0, 2.0]  (2nd diff constant)

seasonal = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]  # period 3
print(difference(seasonal, lag=3, order=1))  # [0.0, 0.0, 0.0]
```

## 5. Stationarity test (KPSS-style)

The KPSS statistic tests the **null of stationarity** (the opposite of the
ADF test). A small statistic means the series looks stationary; a large one
suggests a unit root. `kpss_statistic` returns a heuristic decision and an
approximate p-value — for formal inference consult the published asymptotic
critical values.

```python
# Oscillating around a mean -> stationary.
stationary = [0.1, -0.2, 0.15, -0.1, 0.2, -0.15] * 4
res = kpss_statistic(stationary)
print(f"eta = {res.statistic:.4f}, stationary = {res.is_stationary}")  # True

# A random walk (cumulative sum) -> non-stationary.
rw = [0.0]
for _ in range(200):
    rw.append(rw[-1] + (1.0 if _ % 2 else -1.0) + 0.3)
res2 = kpss_statistic(rw)
print(f"eta = {res2.statistic:.4f}")  # large
```

## 6. Seasonal decomposition

Classical additive decomposition splits a series into **trend + seasonal +
residual**. Pass the seasonal `period` (e.g. 12 for monthly data with annual
cycle). The seasonal component is normalized to sum to zero over one period.

```python
base = [10.0, 20.0, 30.0, 40.0]  # one seasonal period
data = base * 5  # 5 full periods
trend, seasonal, residual = seasonal_decompose(data, period=4)

print(f"seasonal sums to ~0: {sum(seasonal[:4]):.2f}")  # 0.00
print(f"max |residual| = {max(abs(r) for r in residual):.2f}")
```

## 7. Ljung-Box test for autocorrelation

The Ljung-Box Q statistic tests whether *any* of the first `lags`
autocorrelations differ from zero. A low p-value rejects independence —
useful for checking whether ARIMA residuals are white noise.

```python
# A clean trend has strong autocorrelation.
strong = [float(i) for i in range(100)]
lb = ljung_box(strong, lags=10)
print(f"Q = {lb.statistic:.2f}, p = {lb.p_value:.4f}")
print(f"autocorrelated? {lb.has_autocorrelation}")  # True
```

---

These tools compose naturally: smooth a noisy signal, difference away a
trend, test the result for stationarity, then decompose any remaining
seasonality — all from `cds.stats`, no NumPy required.
