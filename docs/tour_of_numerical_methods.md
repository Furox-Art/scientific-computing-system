# Tour of Numerical Methods

A guided walk through the classical numerical algorithms CDS implements from scratch — *readable pure Python, no NumPy/SciPy*. Each stop pairs the **idea** with a tiny **experiment** you can reproduce in a few lines, so you see *why* the algorithm matters before you ever read its source.

> Want to tinker live instead of reading? [Launch the interactive notebook on Binder](https://mybinder.org/v2/gh/Furox-Art/scientific-computing-system/main?urlpath=lab/tree/examples/tour_of_numerical_methods.ipynb) — no install required.

---

## Why a tour?

Most libraries hand you a function and a result. CDS hands you the **algorithm**: every routine lives in `src/cds/` as plain Python you can step through line by line. This tour is the bridge — it picks the algorithms where the *design choice* is the lesson (a faster rule, a higher order, a clever factorization) and shows that choice paying off.

The recurring theme: **intelligence beats brute force.** Picking the right algorithm often matters more than any micro-optimization.

---

## Stop 1 — Numerical Integration: the ladder of accuracy

We want $\int_0^1 e^x\,dx = e - 1 \approx 1.7182818\ldots$ Every rule approximates the area under the curve with a different trade-off between evaluations and accuracy.

```python
import math
from cds.numerical_integration import trapezoid, simpson, romberg, gaussian_quadrature

f, a, b = math.exp, 0.0, 1.0
true_value = math.e - 1

rules = {
    "Trapezoid (n=100)": trapezoid(f, a, b, n=100),
    "Simpson 1/3 (n=100)": simpson(f, a, b, n=100),
    "Romberg (max_iter=64)": romberg(f, a, b, max_iter=64).value,
    "Gauss-Legendre (n=5)": gaussian_quadrature(f, a, b, n=5),
}
for name, est in rules.items():
    print(f"{name:<24} {est:.8f}  err={est - true_value:+.2e}")
```

```
Trapezoid (n=100)         1.71829615  err=+1.43e-05
Simpson 1/3 (n=100)       1.71828183  err=+9.55e-11
Romberg (max_iter=64)     1.71828183  err=+8.88e-16
Gauss-Legendre (n=5)      1.71828183  err=-6.53e-13
```

**The lesson.** Gauss–Legendre reaches near-machine precision with **5 evaluations**, where the trapezoid needs 100 to barely clear five decimals. The rules climb a ladder of increasing *cleverness per evaluation*: trapezoid fits straight lines, Simpson fits parabolas, Romberg stacks trapezoid estimates and extrapolates the limit, and Gauss places its evaluation points *optimally* rather than evenly. Read [`numerical_integration/quadrature.py`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/numerical_integration/quadrature.py) to see each rung built.

---

## Stop 2 — Monte Carlo: accuracy from randomness

Deterministic rules shine in low dimensions, but their cost explodes as dimensions grow. Monte Carlo sidesteps this: the error of a random-sample estimate falls as $1/\sqrt{N}$, **regardless of dimension**.

```python
from cds.montecarlo import estimate_pi

for n in [1_000, 10_000, 100_000]:
    r = estimate_pi(n_samples=n, seed=42)
    print(f"n={n:>7,}  π≈{r.estimate:.4f}  err={r.estimate - math.pi:+.2e}  σ={r.std_error:.2e}")
```

```
n=  1,000  π≈3.1600  err=+1.84e-02  σ=5.15e-02
n= 10,000  π≈3.1372  err=-4.39e-03  σ=1.65e-02
n=100,000  π≈3.1292  err=-1.24e-02  σ=5.22e-03
```

**The lesson.** Halving the standard error takes a **4×** larger sample — slow convergence. But notice the *standard error* tracks the actual error, and crucially the same recipe works in 100 dimensions where grid rules would need $100^{N}$ points. Monte Carlo trades speed for **generality**. CDS also parallelizes the sampling across CPU cores ([`montecarlo/methods.py`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/montecarlo/methods.py)).

---

## Stop 3 — Linear Algebra: determinants scale as $O(N^3)$

The determinant via PLU decomposition factorizes a matrix into a permutation, lower, and upper triangular matrix; the determinant is then a product of pivots. The work is dominated by the elimination, which scales as $O(N^3)$. Let's check that empirically by doubling $N$:

```python
import time
from cds.math_utils import determinant


def make_matrix(n):
    # diagonally dominant so it stays well-conditioned at any n
    return [[1.0 / (i + j + 1) if i != j else float(n) for j in range(n)] for i in range(n)]


t50 = sum(determinant(make_matrix(50)) or 0 for _ in range(1))  # warm-up shape
```

```python
t0 = time.perf_counter()
determinant(make_matrix(50))
t50 = time.perf_counter() - t0
t0 = time.perf_counter()
determinant(make_matrix(100))
t100 = time.perf_counter() - t0
print(f"ratio (doubling N): {t100 / t50:.1f}x   expected ≈ 8x for O(N³)")
```

```
ratio (doubling N): 7.6x   expected ≈ 8x for O(N³)
```

**The lesson.** Doubling $N$ multiplies the cost by $\approx 2^3 = 8$. That single number is the fingerprint of cubic complexity — and it's why numerical linear algebra obsesses over reducing the *constant* and the *exponent* (Cholesky halves work for symmetric matrices, Strassen beats $O(N^3)$ for multiplication at huge $N$). See [`math_utils/linalg.py`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/math_utils/linalg.py) for `lu_decomposition`, `qr_decomposition`, `cholesky`, and more.

---

## Stop 4 — Signal Processing: FFT vs the naive DFT

The Discrete Fourier Transform is textbook $O(N^2)$: for each of $N$ output frequencies, sum over $N$ samples. The Fast Fourier Transform (Cooley–Tukey, radix-2) reorganizes that same computation to exploit symmetry, dropping the cost to $O(N\log N)$.

```python
import time, math
from cds.signals import dft, fft_radix2, power_spectrum

N = 1024
signal = [
    math.sin(2 * math.pi * 5 * k / N) + 0.5 * math.sin(2 * math.pi * 13 * k / N) for k in range(N)
]

t0 = time.perf_counter()
dft(signal)
t_dft = time.perf_counter() - t0
t0 = time.perf_counter()
fft_radix2(signal)
t_fft = time.perf_counter() - t0
print(f"DFT {t_dft * 1000:.1f}ms  vs  FFT {t_fft * 1000:.1f}ms  →  {t_dft / t_fft:.0f}x faster")
```

```
DFT 230.3ms  vs  FFT 1.8ms  →  128x faster
```

**The lesson.** Same answer, $128\times$ less work — for $N=1024$ that's the gap between ~1 million and ~10 thousand operations, and the gap *widens* with $N$. This is the canonical example of algorithmic intelligence, and it underpins everything from MP3 compression to MRI reconstruction. The 2-D variant `fft2` is in [`signals/processing.py`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/signals/processing.py).

---

## Stop 5 — Differential Equations: order matters

Solve the harmonic oscillator $y'' + y = 0$ (true solution $\cos t$) as a first-order system $[y, y']' = [y', -y]$. Compare first-order **Euler** to fourth-order **RK4** at the *same* step size:

```python
from cds.diffeq import solve_system


def harmonic(t, y):  # state [position, velocity], derivative [vel, -pos]
    return [y[1], -y[0]]


ts, ys = solve_system(harmonic, t0=0.0, y0=[1.0, 0.0], t_end=6.283, dt=0.1)  # RK4, one period
true_final = math.cos(6.283)

# Hand-rolled Euler at the same step for contrast:
y = [1.0, 0.0]
for _ in range(63):
    d = harmonic(0.0, y)
    y = [yj + dj * 0.1 for yj, dj in zip(y, d)]
print(f"RK4 error:   {ys[-1][0] - true_final:+.2e}")
print(f"Euler error: {y[0] - true_final:+.2e}")
```

```
RK4 error:   -4.33e-07
Euler error: +3.68e-01
```

**The lesson.** Same function evaluations per step, but RK4 is roughly **a million times** more accurate here. Euler's error falls as $O(h)$; RK4's as $O(h^4)$. Halving the step helps Euler tenfold but RK4 *sixteenfold*. Higher order isn't free (more arithmetic per step), but for smooth problems it is almost always the better trade — the reason production solvers default to RK45, an adaptive RK4(5) pair, which CDS also provides in [`diffeq/solvers.py`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/diffeq/solvers.py).

---

## Stop 6 — Optimization: following the gradient downhill

Minimize the bowl $f(x,y) = (x-3)^2 + (y+1)^2$, whose minimum sits at $(3, -1)$. Gradient descent samples the slope via finite differences and steps against it:

```python
from cds.optimization import gradient_descent

result = gradient_descent(lambda c: (c[0] - 3) ** 2 + (c[1] + 1) ** 2, [0.0, 0.0])
print(result.x, result.value, result.converged)  # → [3.0, -1.0]  ~0.0  True
```

```
[2.99999999, -0.99999999]  2.41e-17  True
```

**The lesson.** Gradient descent is the engine behind training nearly every modern ML model — and here you can read its entire loop in [`optimization/minimize.py`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/optimization/minimize.py), alongside `newton_method` (uses curvature, faster near the minimum), `adam` (adaptive per-parameter step sizes), and `line_search`. CDS's own educational NLP module (`cds.nlp`) trains a tiny GPT using exactly this machinery.

---

## Stop 7 — Quantum: superposition in two lines

A Hadamard gate turns the basis state $|0\rangle$ into the equal superposition $\frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$. Measuring then gives each outcome with ~50% probability:

```python
from cds.quantum import QuantumCircuit, hadamard, simulate

circuit = QuantumCircuit().add(hadamard())
print(circuit.run().probabilities())  # → (0.5, 0.5)
print(simulate(circuit, shots=10000))  # → {0: 5060, 1: 4940}
```

```
(0.5, 0.5)
{0: 5060, 1: 4940}
```

**The lesson.** Superposition and interference are the substrate of quantum advantage — and unlike most libraries, CDS implements the gates, state vectors, and measurement in plain Python you can audit ([`quantum/`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/quantum/)). From here the module extends to multi-qubit circuits, Bell/GHZ entangled states, and the algorithms that exploit them.

---

## Stop 8 — 2-D Integration: tensor products and the curse of dimension

A 1-D rule extends to a rectangle by the *tensor product*: integrate in $x$ with a rule, then integrate *that result* in $y$. The catch is the work multiplies too — an $n \times n$ grid costs $n^2$ evaluations, and in $d$ dimensions a full grid needs $n^d$. So just as in Stop 1, *high-order* rules pay off: a few well-placed points beat many evenly-spaced ones.

```python
import math
from cds.numerical_integration import simpson_2d, gaussian_quadrature_2d


# ∬_{[0,1]²} x² y² dx dy = (1/3)·(1/3) = 1/9 ≈ 0.111111 — exact for both rules.
def poly(x, y):
    return x * x * y * y


true_poly = 1.0 / 9.0


# Area of the unit disk ∬_{[-1,1]²} 1_{x²+y²≤1} dx dy should be π ≈ 3.14159.
def disk(x, y):
    return 1.0 if x * x + y * y <= 1.0 else 0.0


print(f"Simpson 2-D  poly err = {simpson_2d(poly, 0, 1, 0, 1, 2, 2) - true_poly:+.2e}")
print(f"Gauss-3 2-D poly err = {gaussian_quadrature_2d(poly, 0, 1, 0, 1, n=3) - true_poly:+.2e}")
print(f"Simpson 2-D  disk    = {simpson_2d(disk, -1, 1, -1, 1, 100, 100):.5f}  (π = 3.14159)")
```

```
Simpson 2-D  poly err = +0.00e+00
Gauss-3 2-D poly err = -6.94e-16
Simpson 2-D  disk    = 3.14222  (π = 3.14159)
```

**The lesson.** Both rules hit the polynomial $x^2 y^2$ exactly — Simpson because it is exact through cubics in each axis, 3-point Gauss because it is exact through degree 5 in each axis (the $-10^{-16}$ is just round-off). The real lesson is what this scales to: the *disk* integral is a discontinuous indicator, and a $100 \times 100 = 10{,}000$-point grid still only gets $\pi$ to three decimals because the rule wastes evaluations straddling the edge. As dimensions grow, the tensor-product grid explodes as $n^d$ — the precise pain Monte Carlo (Stop 2) sidesteps. See [`numerical_integration/quadrature.py`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/numerical_integration/quadrature.py) for both 2-D rules.

---

## Stop 9 — Time-series: structure in the order

Strip away the time index and a series is just numbers. Put it back, and *order* starts to matter: yesterday's value predicts today's. The **autocorrelation function** (ACF) measures exactly how much, at every lag, and the Ljung–Box test asks whether *any* of those correlations are real signal rather than noise.

```python
import random
from cds.stats import autocorrelation_function, ljung_box

random.seed(0)
N = 240
signal = [random.gauss(0.0, 1.0) for _ in range(N)]  # white-noise base...
for k in range(4, N):  # ...plus a lag-4 echo:
    signal[k] += 0.6 * signal[k - 4]  # each value repeats 4 steps back

acf = autocorrelation_function(signal, max_lag=16)
print(f"r[1]  = {acf[1]:+.3f}   (no lag-1 structure)")
print(f"r[4]  = {acf[4]:+.3f}   (lag-4 echo peaks here)")
print(f"r[8]  = {acf[8]:+.3f}   (echo of the echo)")
lb = ljung_box(signal, lags=12)
print(f"Ljung–Box p-value = {lb.p_value:.2e}  → autocorrelation present? {lb.has_autocorrelation}")
```

```
r[1]  = -0.109   (no lag-1 structure)
r[4]  = +0.504   (lag-4 echo peaks here)
r[8]  = +0.222   (echo of the echo)
Ljung–Box p-value = 8.06e-15  → autocorrelation present? True
```

**The lesson.** The ACF is flat near zero *except* at lag 4 — exactly where we injected the echo — with a smaller secondary bump at lag 8 (each echo repeats its own predecessor). The *order* of the data is information a plain mean discards. The Ljung–Box test aggregates the first dozen autocorrelations into one statistic and reports overwhelming evidence they are not all zero (the p-value is essentially nil). This is the first step of any Box–Jenkins workflow: detect structure, then model it (ARIMA, exponential smoothing). CDS implements the ACF, the partial ACF (via Durbin–Levinson), KPSS-style stationarity testing, and seasonal decomposition in [`stats/time_series.py`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/stats/time_series.py).

---

## Stop 10 — Filter Design: maximally flat, from scratch

A filter's job is to pass some frequencies and stop others. The **Butterworth** family does this with the flattest possible passband — no ripple — which is why it's the default "honest" filter. Designing one means placing poles on a circle (analog prototype), warping them to the digital domain (bilinear transform), and reading off difference-equation coefficients. CDS derives all of that from first principles.

```python
import math
from cds.signals import butter_lowpass, apply_filter

# A slow 5-cycle sine plus faster 200-cycle interference, sampled at N points.
# Normalised freq = (cycles) / (N/2): k=5 -> 0.010, k=200 -> 0.391.
N = 1024
clean = [math.sin(2 * math.pi * 5 * k / N) for k in range(N)]
high = [0.6 * math.sin(2 * math.pi * 200 * k / N) for k in range(N)]
noisy = [a + b for a, b in zip(clean, high)]

coef = butter_lowpass(order=4, cutoff=0.15)  # keep freqs below 15% of Nyquist
filtered = apply_filter(noisy, coef)

# Cosine similarity in the steady-state half (phase-robust): 1.0 = perfect recovery.
half = N // 2
dot = sum(clean[i] * filtered[i] for i in range(half, N))
norm = math.sqrt(sum(c * c for c in clean[half:]) * sum(f * f for f in filtered[half:]))
print(f"recovery (cos sim vs clean) = {dot / norm:.3f}   (1.000 = perfect)")
# Attenuation of the interference alone: pass it through and compare amplitudes.
attenuated = apply_filter([math.sin(2 * math.pi * 200 * k / N) for k in range(N)], coef)
a_in = max(
    abs(x)
    for x in attenuated[:1] and [math.sin(2 * math.pi * 200 * k / N) for k in range(N)][half:]
)
a_out = max(abs(x) for x in attenuated[half:])
print(
    f"interference amplitude: {a_in:.3f} -> {a_out:.4f}  ({a_in / max(a_out, 1e-12):.0f}x smaller)"
)
```

```
recovery (cos sim vs clean) = 0.986   (1.000 = perfect)
interference amplitude: 1.000 -> 0.0135  (74x smaller)
```

**The lesson.** The slow 5-cycle signal passes through almost untouched (cosine similarity 0.986), while the 200-cycle interference is crushed ~74× (−37 dB). That selectivity comes from a 4th-order polynomial whose poles sit on a circle — a design you can re-derive from first principles in [`signals/filters.py`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/signals/filters.py). The same module adds high-pass, band-pass (cascade), band-stop (parallel), and a `moving_median` denoiser that shrugs off impulsive outliers the way no linear filter can.

---

## Where to go next

- **Per-module tutorials** — each stop above has a full tutorial in [Tutorials](tutorials/quick_start.md) with deeper examples.
- **API reference** — every public function, fully typed: [API](api.md).
- **Case study** — see these tools composed on real data: [Measuring the Hubble constant](CASE_STUDY_HUBBLE.md).
- **Read the source** — the whole package is designed to be read. Start anywhere in `src/cds/`; nothing is hidden behind compiled extensions.

> Every algorithm on this tour exists because someone, somewhere, found a smarter way to compute something that mattered. CDS is an invitation to understand *how* — not just *that* — it works.
