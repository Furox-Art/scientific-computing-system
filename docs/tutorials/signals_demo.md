# Signal Processing Tutorial

`cds.signals` provides a from-scratch Fourier toolkit (DFT/FFT, forward and
inverse), convolution, power-spectrum estimation, a frequency-domain low-pass
filter, and a classical digital-filter *design* suite (Butterworth low/high/
band-pass/band-stop) with time-domain application — all pure Python with no
NumPy dependency.

## 1. DFT and inverse

Compute the discrete Fourier transform of a complex signal; `idft` inverts it.

```python
import math
from cds.signals import dft, idft

n = 16
signal = [complex(math.cos(2 * math.pi * 2 * k / n)) for k in range(n)]  # 2 Hz cosine
spectrum = dft(signal)

print("Frequency magnitudes:")
for k, x in enumerate(spectrum):
    if abs(x) > 0.5:
        print(f"  Bin {k}: {abs(x):.2f}")
#   Bin 2: 8.00
#   Bin 14: 8.00   (negative-frequency mirror)
```

## 2. Radix-2 FFT and 2-D FFT

`fft_radix2` is the fast path for power-of-two lengths; `fft2` / `ifft2` handle
2-D signals (matrices).

```python
from cds.signals import fft_radix2

fft_result = fft_radix2(signal)
print(all(abs(a - b) < 1e-9 for a, b in zip(spectrum, fft_result)))  # True
```

## 3. Convolution

Linear convolution of two real sequences.

```python
from cds.signals import convolve

a = [1.0, 2.0, 3.0, 4.0]
b = [0.5, 0.5]  # 2-point averaging kernel
print(convolve(a, b))  # [0.5, 1.5, 2.5, 3.5, 2.0]
```

## 4. Power spectrum and low-pass filtering

`power_spectrum` returns the magnitude at each frequency bin. `low_pass_filter`
zeroes out bins above a cutoff frequency to smooth a noisy signal.

```python
import math
from cds.signals import low_pass_filter

noisy = [1 + 0.3 * math.sin(2 * math.pi * 7 * k / n) for k in range(n)]  # high-freq noise on DC
filtered = low_pass_filter([complex(x) for x in noisy], cutoff=3)
print("Original (first 4):", [f"{x:.2f}" for x in noisy[:4]])
print("Filtered (first 4):", [f"{x.real:.2f}" for x in filtered[:4]])
# Original (first 4): ['1.00', '1.11', '0.79', '1.28']
# Filtered (first 4):  ['1.00', '1.00', '1.00', '1.00']  — noise removed
```

> The frequency-domain `low_pass_filter` above is convenient for offline
> cleaning of a whole block. For real-time, sample-by-sample processing — or
> when you need steeper rolloff, a band-pass, or a notch — use the IIR filter
> *design* suite below.

## 5. Butterworth filter design (IIR)

`cds.signals.filters` designs classical Butterworth filters from scratch
(analog prototype + bilinear transform, the same recipe scipy uses) and
applies them in the time domain via a direct-form II difference equation.
Each function returns a `FilterCoefficients` object that you pass to
`apply_filter`.

Cutoffs are **normalised to Nyquist**: `1.0` is half the sample rate, so
`cutoff=0.25` means a quarter of Nyquist (one-eighth of `fs`).

```python
import math
from cds.signals import butter_lowpass, apply_filter

n = 512
# A slow trend (1 cycle over the whole record) plus high-frequency jitter.
fs = 1.0  # normalised
clean = [math.sin(2 * math.pi * 1 * k / n) for k in range(n)]
jitter = [0.5 * math.sin(2 * math.pi * 0.45 * k) for k in range(n)]  # near Nyquist
noisy = [a + b for a, b in zip(clean, jitter)]

# Design a 4th-order low-pass at 0.1 (one-tenth of Nyquist) and apply it.
coef = butter_lowpass(order=4, cutoff=0.1)
smoothed = apply_filter(noisy, coef)

tail = slice(n // 2, n)  # skip the startup transient
print("input  amplitude:", round((max(noisy[tail]) - min(noisy[tail])) / 2, 3))
print("output amplitude:", round((max(smoothed[tail]) - min(smoothed[tail])) / 2, 3))
# input  amplitude: ~1.05  (trend + jitter)
# output amplitude: ~1.00  (jitter removed, slow trend preserved)
```

High-pass is the mirror image — it removes the slow drift and keeps the fast
component:

```python
from cds.signals import butter_highpass

hp = butter_highpass(order=4, cutoff=0.3)  # keep only the upper third of the band
isolated = apply_filter(noisy, hp)
# 'isolated' now holds the near-Nyquist jitter, with the slow trend gone.
```

### Band-pass and band-stop

Band responses are built from the low/high sections: a band-pass is a
**cascade** (high-pass then low-pass) and a band-stop is the **parallel sum**
of a low- and high-pass. This keeps the -3 dB edges exact at `low` and `high`.

```python
from cds.signals import butter_bandpass, butter_bandstop

# Isolate the band between 0.2 and 0.5 of Nyquist.
bp = butter_bandpass(order=4, low=0.2, high=0.5)

# Or carve a notch between 0.2 and 0.5 — pass everything except that band.
bs = butter_bandstop(order=4, low=0.2, high=0.5)

band_only = apply_filter(mixed_signal, bp)
notched = apply_filter(mixed_signal, bs)
```

Each band filter has an overall order of `2 * order` (one `order`-section for
each edge). Both accept the same `apply_filter` call as the single-section
filters — the dispatch is handled for you.

## 6. Robust denoising with a moving median

When the noise is *impulsive* (occasional large spikes, sensor glitches) a
linear filter smears the spikes instead of removing them. `moving_median`
replaces each sample with the median of a centred window — the classical
order-statistic denoiser that erases salt-and-pepper outliers while keeping
genuine step edges sharp.

```python
from cds.signals import moving_median

clean = [1.0] * 8
corrupted = clean[:3] + [25.0] + clean[3:6] + [-18.0] + clean[6:]  # two spikes
# corrupted = [1, 1, 1, 25, 1, 1, 1, -18, 1]

restored = moving_median(corrupted, window=3)
print([round(x, 1) for x in restored])
# [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  — both spikes gone
```

A moving *average* of the same window would turn the `25.0` spike into a
`~9.0` bump spread across three samples; the median drops it entirely.

---

Run the full demo with `python examples/signals_demo.py`.
