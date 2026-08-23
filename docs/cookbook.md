# CDS Cookbook

A problem-oriented recipe book. Each recipe solves one concrete task with
copy-pasteable code that runs against the real CDS API — no toy stubs. The
recipes are grouped by domain, and every snippet is pure-Python: no NumPy,
no pandas, no extra installs unless the recipe says so.

> Looking for a guided walkthrough instead? See
> [Tour of Numerical Methods](tour_of_numerical_methods.md). For the public
> surface of any module, see the [API Reference](api.md).

---

## Table of contents

- [Core data models](#core-data-models)
- [Math utilities (linear algebra & calculus)](#math-utilities)
- [Probability](#probability)
- [Scientific formulas & constants](#scientific-formulas)
- [Graph algorithms](#graph-algorithms)
- [Signal processing](#signal-processing)
- [Statistics & hypothesis testing](#statistics)
- [Time-series analysis](#time-series)
- [Monte Carlo](#monte-carlo)
- [Numerical integration (quadrature)](#numerical-integration)
- [Differential equations](#differential-equations)
- [Optimization](#optimization)
- [Symbolic modeling](#symbolic-modeling)
- [Machine learning](#machine-learning)
- [NLP — tokenization & attention](#nlp)
- [Data analysis & visualization](#data-analysis)
- [Matplotlib plotting (optional)](#matplotlib-plotting)
- [Hypothesis generation](#hypothesis-generation)
- [Knowledge organization](#knowledge-organization)

---

## Core data models

CDS carries structured research artifacts in `Domain`, `Hypothesis`, and
`HypothesisStatus`. These travel through every higher-level module.

### Encode a falsifiable Hypothesis

A `Hypothesis` needs an `id`, a `statement`, a `domain`, and a
`research_question`. Testable consequences go in `predictions`; `confidence`
is clamped to `[0.0, 1.0]`.

```python
from cds.core.models import Domain, Hypothesis, HypothesisStatus

h = Hypothesis(
    id="H-001",
    statement="Increasing reactor temperature from 410K to 415K raises yield by >5%.",
    domain=Domain.CHEMISTRY,
    research_question="How does temperature affect yield?",
    assumptions=["reaction rate is Arrhenius in this range", "no side reactions"],
    predictions=["yield_delta > 0.05"],
    confidence=0.7,
    status=HypothesisStatus.TESTABLE,
)
print(h.predictions)  # the testable consequences
print(h.to_markdown())  # a structured write-up
```

### Serialize a Hypothesis to JSON

```python
import json

print(json.dumps(h.to_dict(), indent=2))  # enums -> strings, ready to store
```

---

## Math utilities

### Solve a linear system A·x = b

```python
from cds.math_utils import solve_linear

# 2x +  y = 5
#  x + 3y = 10
A = [[2.0, 1.0], [1.0, 3.0]]
b = [5.0, 10.0]
print(solve_linear(A, b))  # [1.0, 3.0]
```

### Largest eigenvalue via power iteration

```python
from cds.math_utils import power_iteration

M = [[4.0, 1.0], [1.0, 3.0]]
eigval, eigvec = power_iteration(M, max_iter=200)
print(eigval)  # ~4.618 (the dominant eigenvalue)
```

### Numerical derivative and integral

```python
from cds.math_utils import derivative, integral

# d/dx of x^3 at x=2 is 12
print(derivative(lambda x: x**3, 2.0))  # ~12.0
# integral of x^2 from 0..3 is 9
print(integral(lambda x: x**2, 0.0, 3.0))  # ~9.0
```

### Orthonormalize a basis (Gram-Schmidt)

```python
from cds.math_utils import gram_schmidt

basis = [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]
print(gram_schmidt(basis))  # two orthonormal vectors
```

---

## Probability

### Evaluate and compare distributions

```python
from cds.probability import gaussian_pdf, exponential_pdf

# Probability density of a Normal(0,1) at z=1.0
print(gaussian_pdf(1.0, mu=0.0, sigma=1.0))  # ~0.2420
# Exponential(lambda=2) density at t=0.5  (``lam`` is the rate parameter)
print(exponential_pdf(0.5, lam=2.0))  # ~0.7358
```

### PMFs for count data

```python
from cds.probability import binomial_pmf, poisson_pmf

# 3 successes out of 10 fair-coin flips
print(binomial_pmf(3, n=10, p=0.5))  # ~0.1172
# A call center with mean 4 calls/hour seeing exactly 6
print(poisson_pmf(6, lam=4.0))  # ~0.1042
```

### Uniform sampling for a quick simulation

```python
from cds.probability import uniform_sample

# 5 draws between 0 and 10
print(uniform_sample(0.0, 10.0, n=5))
```

---

## Scientific formulas

### Look up a physical constant and chain a formula

```python
from cds.scientific import get_constant, photon_energy, de_broglie_wavelength

h = get_constant("h")  # Planck constant
# Energy of a 500nm photon: E = h*f, f = c/lambda
c = get_constant("c")
E = photon_energy(h * c / 500e-9)
print(f"{E:.3e} J")
print(de_broglie_wavelength(9.109e-31, 1e6))  # electron at 1e6 m/s
```

### Gravity, escape velocity, and black-hole radius

```python
from cds.scientific import gravitational_force, escape_velocity, schwarzschild_radius

F = gravitational_force(5.972e24, 7.348e22, 3.844e8)  # Earth–Moon pull
v = escape_velocity(5.972e24, 6.371e6)  # Earth escape v ~11186 m/s
rs = schwarzschild_radius(5.972e24)  # Earth as a BH ~8.87mm
print(F, v, rs)
```

---

## Graph algorithms

CDS graphs use **integer vertices** (a dense array model, optimized for
algorithms). Build one with `Graph(n_vertices=..., directed=...)`.

### Shortest path on a weighted graph

```python
from cds.graph import Graph, dijkstra

# 5 nodes (0..4). Dijkstra returns a distance map and a predecessor map.
g = Graph(n_vertices=5, directed=False)
for u, v, w in [(0, 1, 4), (0, 2, 2), (1, 2, 1), (1, 3, 5), (2, 4, 10), (3, 4, 2)]:
    g.add_edge(u, v, weight=w)

dists, prev = dijkstra(g, 0)
print(dists[4])  # cost to reach node 4 from node 0
```

### Minimum spanning tree and cycle detection

```python
from cds.graph import Graph, kruskal_mst, has_cycle

g = Graph(n_vertices=4, directed=False)
for u, v, w in [(0, 1, 1), (1, 2, 2), (0, 2, 3), (2, 3, 5)]:
    g.add_edge(u, v, weight=w)

mst_edges, total = kruskal_mst(g)
print(mst_edges)  # the cheapest tree connecting all vertices
print(total)  # MST total weight
print(has_cycle(g))  # True (the 0-2 edge closes a triangle)
```

### Topological order of a dependency graph

```python
from cds.graph import Graph, topological_sort

# 0=compile, 1=test, 2=package, 3=deploy
dag = Graph(n_vertices=4, directed=True)
dag.add_edge(0, 1, weight=1)  # compile -> test
dag.add_edge(1, 2, weight=1)  # test -> package
dag.add_edge(2, 3, weight=1)  # package -> deploy
print(topological_sort(dag))  # [0, 1, 2, 3] — build order, sources first
```

---

## Signal processing

### Pull a frequency out of a noisy signal

```python
import math
from cds.signals import power_spectrum

# A 50Hz sine sampled at 1000Hz for 0.2s, plus a slower drift.
n = 200
signal = [math.sin(2 * math.pi * 50 * i / 1000) + 0.3 * i / n for i in range(n)]
freqs = power_spectrum(signal)
peak = max(range(len(freqs)), key=lambda k: freqs[k])
print(peak)  # bin near 50Hz dominates
```

### Fast FFT (radix-2) on a power-of-two window

```python
import math
from cds.signals import fft_radix2

x = [math.sin(2 * math.pi * 3 * k / 64) for k in range(64)]  # exactly 3 cycles
spec = fft_radix2(x)
print(abs(spec[3]))  # the 3rd bin carries almost all the energy
```

### Design and apply a Butterworth low-pass

`butter_lowpass` takes the filter `order` and a normalized `cutoff`
(in `(0, 1)` relative to Nyquist); `apply_filter` runs it over a signal.

```python
from cds.signals import butter_lowpass, apply_filter

coeffs = butter_lowpass(order=4, cutoff=0.2)
smoothed = apply_filter([1, 2, 5, 3, 8, 6, 9, 4], coeffs)
print([round(v, 3) for v in smoothed])  # high-frequency jitter attenuated
```

---

## Statistics

### Descriptive stats and correlation

```python
from cds.stats import mean, stdev, correlation

a = [12.5, 14.3, 11.8, 15.1, 13.7]
b = [3.1, 4.0, 2.9, 4.4, 3.8]
print(mean(a), stdev(a))  # 13.48, ~1.31
print(correlation(a, b))  # strongly positive
```

### Fit a line and read off the slope

```python
from cds.stats import linear_regression

x = [1, 2, 3, 4, 5]
y = [2.1, 3.9, 6.2, 7.8, 10.1]
reg = linear_regression(x, y)
print(f"y = {reg.slope:.3f}x + {reg.intercept:.3f}  (r²={reg.r_squared:.3f})")
```

### One-sample and two-sample t-tests

```python
from cds.stats import one_sample_ttest, two_sample_ttest

sample = [4.8, 5.1, 4.9, 5.0, 5.2]
r1 = one_sample_ttest(sample, popmean=5.0)
print(r1.p_value)  # is the mean different from 5.0?

group_a = [22, 25, 19, 24, 23]
group_b = [28, 30, 27, 31, 29]
r2 = two_sample_ttest(group_a, group_b)
print(r2.p_value)  # do the two groups differ?
```

### Chi-square goodness-of-fit and effect size

```python
from cds.stats import chi_square_gof, cohens_d

# Is a die fair? observed counts vs uniform expected
observed = [8, 9, 12, 7, 11, 13]
gof = chi_square_gof(observed, expected=[10] * 6)
print(gof.p_value)  # large => consistent with fair

print(cohens_d([1, 2, 3, 4], [4, 5, 6, 7]))  # standardized mean gap
```

---

## Time-series

### Smooth, difference, and decompose

```python
import math
from cds.stats import moving_average, difference, seasonal_decompose

series = [10 + 0.5 * i + 3 * math.sin(i / 2) for i in range(48)]
print(moving_average(series, window=4)[:4])  # local smoothing
stationary = difference(series)  # remove the linear trend
trend, seasonal, residual = seasonal_decompose(series, period=12)
```

### Test for stationarity (KPSS) and autocorrelation (Ljung-Box)

```python
from cds.stats import kpss_statistic, ljung_box, autocorrelation_function

kpss = kpss_statistic(series)
print(kpss.statistic, kpss.p_value)  # low p => reject stationarity

lb = ljung_box(series, lags=10)
print(lb.p_value)  # are values autocorrelated?

acfs = autocorrelation_function(series, max_lag=12)
print([round(a, 3) for a in acfs])
```

### Forecast with exponential smoothing

```python
from cds.stats import exponential_smoothing

forecast = exponential_smoothing(series, alpha=0.3)
print([round(f, 2) for f in forecast[-5:]])  # smoothed recent level
```

---

## Monte Carlo

### Estimate π by dart-throwing

`estimate_pi` returns an `MCResult` with `.estimate`, `.samples`, and
`.std_error` (the Monte Carlo standard error).

```python
from cds.montecarlo import estimate_pi

result = estimate_pi(n_samples=50_000)
print(result.estimate, result.std_error)  # ~3.14, with an error bar
```

### Integrate a 1-D function by random sampling

`mc_integrate` is single-variable: pass `f`, the bounds `[a, b]`, and a
sample count.

```python
from cds.montecarlo import mc_integrate

# Integral of sin(x) from 0..pi is 2.0.
r = mc_integrate(math.sin, 0.0, math.pi, n_samples=200_000)
print(r.estimate, r.std_error)  # ~2.0, with a Monte Carlo error estimate
```

### Simulate a 2-D random walk

```python
from cds.montecarlo import random_walk_2d

path = random_walk_2d(steps=1000)  # list of (x, y) positions
print(len(path))  # 1001 points (start + 1000 steps)
```

---

## Numerical integration

The 1-D rules (`trapezoid`, `simpson`, `gaussian_quadrature`) return a plain
`float`. The higher-accuracy rules (`romberg`, `adaptive_simpson`) return a
`QuadratureResult` carrying `.value` and an `.error_estimate`.

### Trapezoid vs Simpson vs Gauss-Legendre on a smooth curve

```python
import math
from cds.numerical_integration import trapezoid, simpson, gaussian_quadrature

f = lambda x: math.exp(-(x**2))  # integral 0..1 ~0.746824
print(trapezoid(f, 0.0, 1.0, n=100))  # float
print(simpson(f, 0.0, 1.0, n=100))  # float
print(gaussian_quadrature(f, 0.0, 1.0, n=5))  # float, high accuracy w/ few pts
```

### Adaptive Simpson where the integrand is spiky

```python
from cds.numerical_integration import adaptive_simpson

# A narrow Gaussian peak: adaptive refinement concentrates effort near it.
f = lambda x: math.exp(-((x - 0.5) ** 2) / 0.001)
r = adaptive_simpson(f, 0.0, 1.0, tol=1e-8)  # QuadratureResult
print(r.value, r.error_estimate)
```

### Integrate over a 2-D rectangle

The 2-D rules also return a plain `float`.

```python
from cds.numerical_integration import simpson_2d, gaussian_quadrature_2d

# Volume of a Gaussian bump over the unit square.
g = lambda x, y: math.exp(-(x**2 + y**2))
print(simpson_2d(g, 0.0, 1.0, 0.0, 1.0, nx=20, ny=20))
print(gaussian_quadrature_2d(g, 0.0, 1.0, 0.0, 1.0, n=4))
```

---

## Differential equations

CDS ODE solvers take the signature `f(t, y) -> y'`, plus `t0`, `y0`,
`t_end`, and a step size `dt`.

### Solve an ODE with RK4

```python
import math
from cds.diffeq import rk4

# dy/dt = -y, y(0)=1  =>  y(t) = e^-t, so y(2) = e^-2 ≈ 0.1353
decay = lambda t, y: -y
sol = rk4(decay, t0=0.0, y0=1.0, t_end=2.0, dt=0.01)
print(sol.y[-1])  # ~0.1353
print(len(sol.t))  # number of time steps
```

### Adaptive stepping with RK45

```python
from cds.diffeq import rk45

# Logistic growth — adaptive RK45 takes bigger steps where it can.
logistic = lambda t, y: 0.5 * y * (1 - y / 10)
sol = rk45(logistic, t0=0.0, y0=0.5, t_end=20.0, rtol=1e-6)
print(sol.y[-1])  # settles near the carrying capacity 10
```

### A coupled system (predator–prey)

`solve_system` returns `(times, states)` — a list of time points and a list
of state vectors.

```python
from cds.diffeq import solve_system


# Lotka–Volterra: dx/dt = ax - bxy, dy/dt = dxy - cy
def lotka(t, state):
    x, y = state
    return [1.1 * x - 0.4 * x * y, 0.1 * x * y - 0.4 * y]


ts, ys = solve_system(lotka, t0=0.0, y0=[10.0, 5.0], t_end=40.0, dt=0.05)
print(len(ys))  # two trajectories oscillating out of phase
```

---

## Optimization

### Minimize a scalar function

```python
from cds.optimization import gradient_descent

# Minimum of (x-3)^2 is at x=3
r = gradient_descent(lambda x: (x - 3) ** 2, x0=0.0, lr=0.1)
print(r.x, r.value, r.converged)  # 3.0, 0.0, True
```

### Minimize a multivariate quadratic

For the vector overload, the objective takes a **single list** argument
(the parameter vector), not separate coordinates.

```python
from cds.optimization import gradient_descent

# Rosenbrock-ish bowl: f(x,y) = (x-1)^2 + 100*(y-x^2)^2, written as f(v)
f = lambda v: (v[0] - 1) ** 2 + 100 * (v[1] - v[0] ** 2) ** 2
r = gradient_descent(f, x0=[0.0, 0.0], lr=0.001, max_iter=50000)
print(r.x)  # approaches [1.0, 1.0]
```

### Find a root with Newton's method

```python
import math
from cds.optimization import newton_method

# Root of cos(x) near x=1 is pi/2
root = newton_method(math.cos, x0=1.0)
print(root.x, math.pi / 2)  # ~1.5708
```

### Adam on a noisy objective

```python
from cds.optimization import adam

r = adam(lambda x: (x - 2) ** 2, x0=5.0, lr=0.1, max_iter=1000)
print(r.x)  # ~2.0; Adam's momentum handles the flat gradient well
```

---

## Symbolic modeling

### Build, differentiate, and export an expression

```python
from cds.modeling import Variable, Sin, Exp

x = Variable("x")
f = Sin(x) * Exp(x)  # sin(x)·e^x
df = f.diff("x")  # symbolic derivative
print(f.to_str(), df.to_str())
print(f.to_latex())  # LaTeX for papers / docs
print(df.evaluate({"x": 0.0}))  # 1.0 (= e^0 (sin0 + cos0))
```

### Group equations into a model and take a Jacobian

```python
from cds.modeling import MathModel, Variable, Constant

t, a = Variable("t"), Variable("a")
m = MathModel(name="kinematics", parameters={"a": 9.81}, variables=["t", "v0", "x0"])
v0, x0 = Variable("v0"), Variable("x0")
m.add_equation("velocity", v0 + a * t)
m.add_equation("position", x0 + v0 * t + Constant(0.5) * a * t * t)
print(m.evaluate({"t": 2.0, "v0": 0.0, "x0": 0.0}))
print(m.jacobian("t"))  # symbolic partials w.r.t. time
```

### Solve and fit a model

```python
from cds.modeling import Variable, Constant, MathModel, solve_equation, fit_parameters

x = Variable("x")
root = solve_equation(x**2 - Constant(2.0), variable="x", x0=1.0)
print(root.x)  # sqrt(2) ~1.4142

# Recover a velocity v from position observations p = v*t.
fit_model = MathModel(name="linear", parameters={"v": 0.0}, variables=["t"])
fit_model.add_equation("position", Variable("v") * Variable("t"))
observed = [({"t": float(i)}, 3.0 * i) for i in range(5)]
fit = fit_parameters(
    fit_model, observed=observed, parameter_names=["v"], x0=[0.1], target_label="position"
)
print(fit.parameters["v"])  # ~3.0
```

---

## Machine learning

### Train a small MLP to fit a curve

```python
from cds.ml import MLP, Layer

net = MLP(
    [
        Layer(input_size=1, output_size=8, activation="relu"),
        Layer(input_size=8, output_size=1, activation="identity"),
    ]
)
X = [[i / 10] for i in range(-20, 21)]  # inputs in [-2, 2]
y = [[xi[0] ** 2] for xi in X]  # learn y = x^2
info = net.train(X, y, epochs=500, lr=0.01)
print(info["final_loss"], info["converged"])
print(net.predict([1.5]))  # close to 2.25
```

### Inspect what each layer learned

```python
for i, layer in enumerate(net.layers):
    print(f"layer {i}: {len(layer.weights)} neurons, activation={layer.activation}")
```

---

## NLP

### Train a BPE tokenizer and encode text

`train_bpe` takes a single corpus string and returns a ready `BPETokenizer`
(equipped with `encode` / `decode`).

```python
from cds.nlp import train_bpe

corpus = "the quick brown fox the lazy dog a quick fox jumps"
tok = train_bpe(corpus, vocab_size=80, min_frequency=1)
ids = tok.encode("the quick fox")
print(ids)  # subword token ids
print(tok.decode(ids))  # round-trip back to text
```

### Compute scaled dot-product attention

`scaled_dot_product_attention` returns the attended values (the weighted sum
of `V`); derive the attention weights yourself when you need them.

```python
from cds.nlp import scaled_dot_product_attention

# Three "tokens" with 4-dim embeddings.
Q = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
K = Q
V = [[1.0], [2.0], [3.0]]
out = scaled_dot_product_attention(Q, K, V)
print(out)  # 3x1 attended output
```

### Run a mini autograd computation

`cds.nlp.Tensor` is a scalar autograd node: build one per scalar value and
chain it with `*`, `+`, `relu`, ... `backward()` takes **no argument** — it
seeds the output gradient to `1.0` and walks the graph back to the leaves.

```python
from cds.nlp import Tensor, relu

a = Tensor(2.0, requires_grad=True)
b = Tensor(3.0, requires_grad=True)
c = relu(a * b)  # 6.0, with a and b as parents
c.backward()  # seeds 1.0, propagates grads back
print(c.data, a.grad, b.grad)  # 6.0  3.0  2.0
```

---

## Data analysis

### Load CSV data and normalize a column

A `DataTable` exposes columns via `column(name)` and `column_as_float(name)`.

```python
from cds.data_analysis import load_csv, z_score

table = load_csv("measurements.csv")  # DataTable
col = table.column_as_float("temp")
print(z_score(col)[:3])  # standardized values, mean 0
```

### Moving average and visualization

```python
from cds.data_analysis import moving_average, plot_line

series = [1, 3, 2, 5, 4, 6, 5, 8, 7, 9]
smoothed = moving_average(series, window=3)
print(plot_line(series))  # ASCII line chart in the terminal
```

### Round-trip data through pandas (optional extra)

`to_dataframe` takes a `DataSet`, which wraps a list of row dicts.

```python
# pip install "cds[pandas]"
from cds.data_analysis import DataSet, to_dataframe, from_dataframe

ds = DataSet(data=[{"temp": 1.0}, {"temp": 2.0}, {"temp": 3.0}])
df = to_dataframe(ds)  # DataSet -> pandas.DataFrame
print(df.describe())
ds2 = from_dataframe(df)  # and back, losslessly
```

---

## Matplotlib plotting

Optional extra — core CDS stays zero-dependency.

```bash
pip install "scientific-computing-system[plot]"
```

### Plot a series, histogram, and ACF

```python
import math
from cds.plot import plot_series, plot_histogram, plot_acf, save_figure

y = [math.sin(2 * math.pi * i / 40) for i in range(100)]
save_figure(plot_series(y, title="Sine"), "series.png")
save_figure(plot_histogram(y, bins=20), "hist.png")
save_figure(plot_acf(y, max_lag=20), "acf.png")
```

### Regression, multi-series, heatmap

```python
from cds.plot import plot_regression, plot_multi_series, plot_heatmap, save_figure

x = [1, 2, 3, 4, 5]
y = [2.1, 3.9, 6.0, 8.2, 9.8]
save_figure(plot_regression(x, y), "fit.png")
save_figure(plot_multi_series({"y": y, "2x": [2 * v for v in x]}), "multi.png")
save_figure(plot_heatmap([[1, 2], [3, 4], [5, 6]]), "heat.png")
```

### Waveform + power spectrum with `cds.signals`

```python
import math
from cds.plot import plot_waveform, plot_power_spectrum, save_figure

n, fs = 128, 128.0
signal = [math.sin(2 * math.pi * 8 * t / fs) for t in range(n)]
save_figure(plot_waveform(signal, sample_rate=fs), "wave.png")
save_figure(plot_power_spectrum(signal, sample_rate=fs), "power.png")
```

CLI (with `[plot]` installed):

```bash
cds plot 1,5,3,8,2 --file out.png
cds plot 1,2,2,3,5 --kind hist --file hist.png
cds plot 1,0,1,0,1,0,1,0,1,0 --kind acf --file acf.png
```

See also `examples/plot_demo.py`.

---

## Hypothesis generation

### Generate structured hypotheses offline

```python
from cds.hypothesis import generate_hypotheses

hypos = generate_hypotheses(
    "Why is the measured Hubble constant lower in some methods?",
    domain="cosmology",
    n=3,
)
for h in hypos:
    print(h.statement, "=>", h.predictions)
```

### Evaluate a hypothesis against observed data

`HypothesisEvaluator.evaluate(hypothesis, data)` takes a `Hypothesis` and an
`EvaluationData` dict keyed by the dispatch you want to run — here
`"chi_square_gof"`, whose value is a `ChiSquareGofPayload` (`{"observed": ...,
"expected": ...}`).

```python
from cds.core.models import Domain, Hypothesis, HypothesisStatus
from cds.hypothesis import HypothesisEvaluator, ChiSquareGofPayload

h = Hypothesis(
    id="H-die",
    statement="a six-sided die is fair",
    domain=Domain.MATHEMATICS,
    research_question="is the die fair?",
    status=HypothesisStatus.TESTABLE,
)
payload: ChiSquareGofPayload = {
    "observed": [8, 12, 10, 11, 9, 10],
    "expected": [10, 10, 10, 10, 10, 10],
}
result = HypothesisEvaluator().evaluate(h, {"chi_square_gof": payload})
print(result.p_value, result.decision)  # large p => consistent with fair
```

---

## Knowledge organization

### Build a concept graph and traverse it

A `KnowledgeGraph` is created with a `name`. `add_concept(name,
description=...)` creates a node; `add_relation(source, target, kind)` links
two named concepts.

```python
from cds.knowledge import KnowledgeGraph

kg = KnowledgeGraph(name="mechanics")
kg.add_concept("momentum", description="p = m·v")
kg.add_concept("force", description="F = dp/dt")
kg.add_relation("force", "momentum", "depends-on")
print(kg.neighbors("force"))  # what force relates to
```

### Attach research notes and search across everything

A `Notebook` is created with a `name`. Add notes with `add_note(note_id,
title, body, ...)`, then `search` ranks hits from both the graph and the
notebook by relevance.

```python
from cds.knowledge import Notebook, search

nb = Notebook(name="mechanics")
nb.add_note(
    note_id="n1",
    title="Newton II",
    body="F equals the rate of change of momentum.",
    linked_concepts=["force", "momentum"],
)
hits = search(kg, nb, query="rate of change")
print(len(hits), "matches")  # ranked by relevance
```

---

Every recipe above uses only the documented public API. If a recipe doesn't
behave as shown, the API reference for that module (`api.md`) is the source
of truth for signatures and defaults.
