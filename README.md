<p align="center">
  <img src="assets/logo.svg" alt="scientific-computing-system" width="640">
</p>

<h1 align="center">scientific-computing-system</h1>

<p align="center"><b>An open-source computational science platform — quantum, stats, signals, optimization &amp; hypothesis generation, in one pure-Python package.</b></p>

<p align="center">
  <a href="https://pypi.org/project/scientific-computing-system/"><img src="https://img.shields.io/pypi/v/scientific-computing-system.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/scientific-computing-system/"><img src="https://img.shields.io/pypi/dm/scientific-computing-system.svg" alt="PyPI downloads"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-green.svg" alt="Python 3.10+"></a>
  <a href="https://codecov.io/gh/Furox-Art/scientific-computing-system"><img src="https://codecov.io/gh/Furox-Art/scientific-computing-system/branch/main/graph/badge.svg" alt="codecov"></a>
  <a href="https://github.com/Furox-Art/scientific-computing-system/actions/workflows/tests.yml"><img src="https://github.com/Furox-Art/scientific-computing-system/actions/workflows/tests.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://furox-art.github.io/scientific-computing-system/"><img src="https://img.shields.io/badge/docs-mkdocs-teal.svg" alt="Docs"></a>
  <a href="https://github.com/Furox-Art/scientific-computing-system/releases"><img src="https://img.shields.io/github/v/release/Furox-Art/scientific-computing-system.svg" alt="GitHub release"></a>
  <a href="https://github.com/Furox-Art/scientific-computing-system"><img src="https://img.shields.io/github/stars/Furox-Art/scientific-computing-system.svg?style=social" alt="GitHub stars"></a>
</p>

<p align="center">
  <a href="https://mybinder.org/v2/gh/Furox-Art/scientific-computing-system/main?urlpath=lab/tree/examples/tour_of_numerical_methods.ipynb"><img src="https://img.shields.io/badge/try%20it-on%20Binder-orange.svg?logo=jupyter" alt="Launch on Binder"></a>
</p>

<p align="center">
  <a href="https://furox-art.github.io/scientific-computing-system/">Documentation</a> &nbsp;·&nbsp;
  <a href="https://furox-art.github.io/scientific-computing-system/tour_of_numerical_methods/">Tour of Numerical Methods</a> &nbsp;·&nbsp;
  <a href="https://furox-art.github.io/scientific-computing-system/cookbook/">Cookbook</a> &nbsp;·&nbsp;
  <a href="https://github.com/Furox-Art/scientific-computing-system/releases">Releases</a> &nbsp;·&nbsp;
  <a href="docs/tutorials/">Tutorials</a> &nbsp;·&nbsp;
  <a href="#quick-start">Quick Start</a> &nbsp;·&nbsp;
  <a href="docs/CASE_STUDY_HUBBLE.md">Case Studies</a>
</p>

---

## Learning scientific Python?

CDS was built for you. Every algorithm you'd normally call as a black box —
numpy's FFT, scipy's integrators, sklearn's clustering — lives here as
readable pure Python you can open, run, and modify line by line.

**Start here:**

1. `pip install scientific-computing-system` — no NumPy, no compilers,
   nothing to configure
2. `cds modules` — see everything inside
3. Run any demo: `python examples/quantum_demo.py`
4. Follow the tutorials — zero-install via
   [Binder](https://mybinder.org/v2/gh/Furox-Art/scientific-computing-system/main?urlpath=lab/tree/examples/tour_of_numerical_methods.ipynb)

> **One package, no heavy dependencies.** CDS brings together quantum circuit simulation, statistical analysis, signal processing, optimization, probability, ODE/numerical solvers, symbolic modeling, knowledge graphs, educational NLP, and **structured hypothesis generation** — all in readable pure Python. **No NumPy. No SciPy. No compiled extensions.** Just `pip install` and you can read every line of source.

**Why CDS exists** — for teaching, prototyping, scientific exploration, and edge deployments where a single, dependency-light, fully-readable Python package beats juggling six libraries. Every algorithm is implemented from scratch so you can *learn how it works*, not just call it.

> **If CDS saves you time, a star helps others find it — and keeps the project maintained.** Thank you!

---
**Latest Update (v1.6.0):** scikit-learn-flavored deepening — **PCA**, **k-NN / k-means / decision tree / logistic & linear regression**, **StandardScaler + train/test split**, chi-square/Student-t/gamma/beta distributions with seeded samplers, **implicit stiff ODE solvers** (backward Euler, Crank–Nicolson), **Nelder–Mead** and **simulated annealing**, plus Mann–Whitney U and Wilcoxon tests. Install: `pip install -U "scientific-computing-system[plot]"`.
---

## Contents

- [Why CDS?](#why-cds)
- [Citing CDS](#citing-cds)
- [Modules](#modules)
- [Quick Start](#quick-start)
- [Intelligence over Brute Force](#intelligence-over-brute-force)
- [ASCII Visualization & Tools](#ascii-visualization--tools)
- [Interactive Dashboard](#interactive-dashboard)
- [Scientific Case Studies](#scientific-case-studies)
- [Examples](#examples)
- [Architecture](#architecture)
- [Vision](#vision)
- [Recent Improvements](#recent-improvements)
- [Contributing](#contributing)
- [Automation and Maintenance Workflows](#automation-and-maintenance-workflows)
- [Security](#security)
- [License](#license)
- [Contact](#contact)

---

## Why CDS?

Most scientific Python stacks are **black boxes**: they work, but you can't see
*how* an algorithm works without leaving Python for C, Fortran, or CUDA. CDS is
the opposite trade-off. Every algorithm is written in readable pure Python, so
the source *is* the documentation. You get a single package that spans quantum
simulation, statistics, signal processing, optimization, numerical methods,
symbolic modeling, and structured hypothesis generation, and you can step
through any of it line by line.

**The pitch in numbers:** 21 modules · **2,000+ tests** on Linux/Windows/macOS
(Python 3.10–3.13) · **100% coverage** (statement *and* branch) enforced as a
CI gate · `mypy --strict` clean across source and tests · **zero runtime
dependencies**.

And because every domain lives under one namespace, crossing them costs you
nothing:

```python
from cds.quantum import bell_state  # quantum simulation
from cds.signals import fft_radix2  # signal processing
from cds.stats import linear_regression  # statistics

reg = bell_state(0)
spec = fft_radix2([complex(i) for i in range(8)])
fit = linear_regression([1, 2, 3], [2.1, 3.9, 6.2])
# …no NumPy, no SciPy, no compiler — this *is* the whole install.
```

That positioning makes CDS a good fit when one of these matters more than raw
throughput:

- **You want to understand an algorithm**, not just call it. The FFT, the LU
  pivot, the RK45 step controller, and the BPE merge loop are all a click away
  from the function you called.
- **You need a dependency-light runtime**. No NumPy, no SciPy, no BLAS, no
  compiled extensions. `pip install` and you're done; it runs the same on a
  locked-down server or a teaching laptop.
- **You're crossing domains in one project**. A research workflow that touches
  stats, signals, and hypothesis generation normally means three or four
  libraries with three or four sets of conventions. CDS keeps them under one
  namespace with a shared data model.
- **You want a falsifiable hypothesis, not a chatbot answer.** `cds.hypothesis`
  emits structured objects (assumptions, predictions, confidence) that you can
  then feed straight back into the statistical tests in the same package.

If your priority is heavy numerical performance on arrays of >10⁷ elements,
CDS is the wrong tool — use NumPy/SciPy. The [comparison table](#cds-vs-other-libraries)
below spells out exactly where each fits.

<details>
<summary><b>The reliability floor</b></summary>

These aren't the pitch, but they remove the usual reasons to hesitate:

- The full test suite runs on every push across Linux, Windows, and macOS,
  Python 3.10–3.13. See the [CI badge](https://github.com/Furox-Art/scientific-computing-system/actions/workflows/tests.yml)
  for the live test count.
- **100% code coverage (statement + branch)** is enforced as a gate — CI fails
  if either drops. See the [codecov badge](https://codecov.io/gh/Furox-Art/scientific-computing-system).
- `mypy --strict` passes clean across `src/` and `tests/`.
- An interactive CLI with ASCII visualization is included, no plotting deps.

</details>

### CDS vs other libraries

| Need | CDS | NumPy/SciPy | SymPy | PennyLane |
|---|---|---|---|---|
| Pure-Python (no compile, no binary) | yes | no (C/Fortran) | yes | no (needs Qiskit/Cirq) |
| Quantum simulation | yes, single/multi-qubit | no | minimal | yes, full SDK |
| Hypothesis generation (structured) | yes | no | no | no |
| Educational NLP (BPE, embeddings) | yes, from-scratch | no | no | no |
| Single-package umbrella (math+physics+stats+ML+signals+NLP) | yes | no, split across 6+ | partial | no, focused |
| Production-ready CI/CD (multi-OS matrix, signed releases) | yes | n/a | partial | yes |
| Educational / readable source | yes | no, large surface | yes | no |
| Edge runtime (no BLAS) | yes | no | partial | no |
| Heavy numerical performance (>10⁷ ops) | no, use NumPy instead | yes | no | yes (GPU) |

**When to use CDS:** teaching, prototyping, scientific exploration, edge deployments, custom algorithm development.
**When to reach for NumPy/SciPy/PennyLane:** production HPC, GPU-accelerated quantum, distributed compute.

## Citing CDS

If CDS is useful in your research or publications, please cite it using the information in `CITATION.cff` at the repository root. This helps give proper credit and track adoption in scientific work.

## Modules

| Module | Description |
|--------|-------------|
| `cds.core` | Shared data model — `Domain`, `Hypothesis`, `HypothesisStatus` types used across modules |
| `cds.interpolate` | Interpolation — 1-D linear & nearest-neighbour evaluators, bilinear interpolation on regular 2-D grids |
| `cds.quantum` | Single & multi-qubit simulation — Hadamard, Pauli, CNOT, SWAP, Toffoli, Bell/GHZ states, entanglement detection |
| `cds.optimization` | Gradient descent, Newton's method, Adam optimizer, golden section search, **Nelder–Mead simplex**, **simulated annealing** (box bounds, seeded), **projected gradient descent** & **quadratic-penalty constrained solves** |
| `cds.ml` | Pure Python Neural Networks — MLP, dense layers, Adam-based training — plus **k-NN classifier/regressor**, **k-means clustering** (k-means++), **CART decision tree** (Gini), **logistic regression**, **vector OLS `LinearRegression`**, **PCA** (Jacobi eigen-solver), **StandardScaler** & seeded **train/test split**, **Random Forest**, **gradient-boosted trees**, **Gaussian Naive Bayes**, **VotingClassifier** (hard/soft), classification metrics (accuracy, P/R/F1, ROC-AUC) & **k-fold cross-validation** |
| `cds.signals` | DFT, radix-2 FFT/IFFT (O(N log N)), 2D FFT/IFFT, convolution, power spectrum, **Butterworth IIR filter design** (low/high/band), moving-median denoiser, **STFT & spectrogram** (Hann/Hamming windows), **Haar wavelet DWT/IDWT** (multi-level) |
| `cds.probability` | Gaussian, uniform, exponential, binomial, Poisson distributions — plus **chi-square / Student-t (pdf·cdf·ppf)**, **gamma / beta densities with seeded samplers** (Marsaglia–Tsang), **hypergeometric & negative-binomial pmf/cdf** |
| `cds.stats` | Descriptive stats, Pearson correlation, linear regression, t-test, chi-square, ANOVA, effect-size measures (Cohen's d, η², Cramér's V), Bonferroni correction, **time-series analysis** (ACF/PACF, KPSS, Ljung-Box, exponential smoothing, seasonal decomposition), **nonparametric tests** (Mann–Whitney U, Wilcoxon signed-rank with tie corrections), **bootstrap confidence intervals** (percentile method), **power analysis & sample-size solver** (t / proportion tests) |
| `cds.math_utils` | Numerical calculus, O(N³) LU / QR / Cholesky, eigenvalue (power iteration), Gram-Schmidt, matrix inverse, **SVD via Jacobi rotations** (+ rank & condition number) |
| `cds.data_analysis` | Mini-Pandas `DataSet` for filtering/grouping, CSV loading, ASCII visualization, optional pandas interop (`to_dataframe` / `from_dataframe` via `cds[pandas]`) |
| `cds.scientific` | Physical constants, formulas (KE, gravity, gas law, Schwarzschild, de Broglie, escape velocity) |
| `cds.graph` | BFS, DFS, Dijkstra shortest path, Kruskal MST, topological sort, cycle detection |
| `cds.modeling` | Symbolic algebra — expressions, symbolic differentiation, simplification, LaTeX export, `MathModel` equation systems, root-finding & parameter fitting |
| `cds.knowledge` | Knowledge organization — concept graph with typed relations, research notes notebook, ranked structured retrieval (JSON persistence), **TF-IDF ranked search** across concepts & notes |
| `cds.montecarlo` | Monte Carlo integration, π estimation, Buffon's needle, random walks (1D/2D), **Metropolis-Hastings MCMC sampler** |
| `cds.diffeq` | Euler method, RK4, midpoint method, ODE system solver — plus **implicit stiff solvers**: backward Euler & Crank–Nicolson (scalar + systems, analytic or finite-difference Jacobian), **symplectic integrators** (semi-implicit Euler, velocity Verlet) |
| `cds.numerical_integration` | Deterministic quadrature — trapezoid, Simpson 1/3 & 3/8, Romberg, Gauss-Legendre, adaptive Simpson, **2-D tensor-product quadrature** (Simpson + Gauss-Legendre) |
| `cds.nlp` | Educational NLP from scratch — BPE tokenizer, sinusoidal embeddings, multi-head attention, Transformer block, scalar autograd (SGD/Adam), MiniGPT demo |
| `cds.hypothesis` | Structured hypothesis generation with prompt templates for custom research workflows, plus the **correlation-mining engine**: `mine_correlations` turns numeric datasets into falsifiable, statistically tested Hypothesis objects |
| `cds.pde` | 1-D finite-difference PDE solvers — explicit FTCS heat equation (Dirichlet/Neumann, von Neumann stability guard) and wave equation (CFL-guarded) |
| `cds.plot` | Optional matplotlib charts — series/multi-series, scatter, regression, histograms, waveforms, power spectra, ACF/PACF, seasonal decompose, heatmaps, loss curves (`pip install scientific-computing-system[plot]`) |

## Quick Start

The fastest way in is from PyPI — no clone, no download needed:

```bash
pip install scientific-computing-system
cds --help
cds hypothesis "What causes the Hubble tension?"
```

### From source

**macOS / Linux**

```bash
git clone https://github.com/Furox-Art/scientific-computing-system.git
cd scientific-computing-system
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

pytest            # run the test suite
cds --help        # CLI
cds constants     # physical constants
cds calc ke       # kinetic-energy calculator
cds modules       # list all modules
cds hypothesis "What causes the Hubble tension?"
```

**Windows (PowerShell)**

```powershell
git clone https://github.com/Furox-Art/scientific-computing-system.git
cd scientific-computing-system
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

pytest
cds --help
cds hypothesis "What causes the Hubble tension?"
```

> **On Windows you don't need to download anything manually.**
> Just `git clone` (or copy the repo folder) and the commands above set everything up.
> No `.zip` to extract — clone gives you the live, up-to-date source that you can `git pull` anytime.
> If you prefer not to use Git, you can also click **Code → Download ZIP** on GitHub,
> but **cloning is recommended** so updates are a single `git pull`.

> **Execution policy:** if `Activate.ps1` is blocked, run
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use `cmd.exe`
> with `.venv\Scripts\activate.bat` instead.

## Intelligence over Brute Force

CDS is built on the principle that algorithmic improvements matter more than raw loop speed. Pure Python cannot match C-extensions for tight loops, so CDS closes the gap where it can through better algorithms:

- **Quantum Simulation:** Instead of multiplying state matrices for every shot, CDS uses **O(1) probabilistic sampling** with explicit state collapse, which is significantly faster than the naive shot-by-shot approach.
- **Linear Algebra:** Uses **O(N³)** Partial Pivoting LU Decomposition in place of the naive O(N!) determinant expansion.
- **Signal Processing:** Zero-padded **O(N log N)** FFT and FFT-based convolution via the Convolution Theorem.
- **Neural Networks:** Adam optimizers with momentum state persistence.

See the full [Intelligence & Performance Benchmark Report](docs/benchmarks.md) for detailed figures.

## ASCII Visualization & Tools

You don't need heavy plotting libraries to inspect your data. CDS includes a built-in terminal visualization engine:

```bash
# Plot a sine wave or data series directly in your terminal
cds plot "1, 5, 3, 8, 4, 9" --title "My Data"
```
*Outputs scale-aware ASCII line plots and bar charts.*

## Interactive Dashboard

CDS includes an **Interactive Web Dashboard** for scientific exploration. Launch it from your terminal:

```bash
pip install "scientific-computing-system[dashboard]"
cds dashboard
```

*The dashboard includes a Hypothesis Engine, Quantum Circuit Simulator, Neural Network training visualizer, and Statistical testing lab.*

## Scientific Case Studies

Explore how CDS is used to solve real-world research problems:

1.  [**Hubble Tension Analysis**](docs/CASE_STUDY_HUBBLE.md): Generating and testing hypotheses for the expansion rate of the universe.
2.  [**Quantum-ML Integration**](docs/CASE_STUDY_QUANTUM_ML.md): Using quantum circuit measurements as features for classical Neural Network training.

## Examples

CDS ships **27 runnable demo scripts** in [`examples/`](examples/) — one per module.
Each is a self-contained `.py` you can run directly:

```bash
python examples/quantum_demo.py       # quantum circuits & entanglement
python examples/signals_demo.py       # FFT, convolution, power spectrum
python examples/ml_and_viz_demo.py   # neural network training + plots
python examples/montecarlo_demo.py    # π estimation & integration
python examples/hypothesis_demo.py    # structured hypothesis generation
```

See `docs/research-workflows.md` for guidance on embedding CDS in research pipelines.

### Hypothesis Generation (cognitive discovery)

```bash
# Basic demo
python examples/hypothesis_demo.py

# With stats / experiment sketch example
python examples/hypothesis_with_stats_demo.py

# Custom generator implementation (using the HypothesisGenerator Protocol)
python examples/hypothesis_custom_generator.py

# Or via CLI
cds hypothesis "What causes the Hubble tension?"
```

### Quantum Circuit (single qubit)
```python
from cds.quantum import QuantumCircuit, hadamard, pauli_x, simulate

circuit = QuantumCircuit().add(hadamard()).add(pauli_x())
result = circuit.run()
print(result.probabilities())

counts = simulate(circuit, shots=1000)
print(counts)  # {0: ~500, 1: ~500}
```

### Multi-Qubit & Entanglement
```python
from cds.quantum import (
    QuantumRegister,
    h_gate,
    cnot,
    bell_state,
    ghz_state,
    is_entangled,
)

# Bell state (|00⟩ + |11⟩) / √2
reg = bell_state(0)
print(is_entangled(reg))  # True
print(reg.measure_shots(shots=1000))  # {'00': ~500, '11': ~500}

# 4-qubit GHZ state
ghz = ghz_state(4)
counts = ghz.measure_shots(shots=1000)
print(counts)  # {'0000': ~500, '1111': ~500}
```

### Optimization
```python
from cds.optimization import gradient_descent, newton_method

# Find minimum of (x-3)²
result = gradient_descent(lambda x: (x - 3) ** 2, x0=10.0, lr=0.1)
print(f"x = {result.x:.6f}")  # ~3.0

# Find √2 using Newton's method
result = newton_method(lambda x: x**2 - 2, x0=1.5)
print(f"√2 = {result.x:.10f}")  # 1.4142135624
```

### Signal Processing
```python
from cds.signals import dft, fft_radix2, convolve, low_pass_filter

# FFT of a signal
signal = [complex(i) for i in range(8)]
spectrum = fft_radix2(signal)

# Convolution
result = convolve([1.0, 2.0, 3.0], [0.5, 0.5])
print(result)  # [0.5, 1.5, 2.5, 1.5]
```

### Probability Distributions
```python
from cds.probability import gaussian_pdf, binomial_pmf, poisson_pmf

# Gaussian PDF at x=0
print(gaussian_pdf(0.0, mu=0, sigma=1))  # 0.3989...

# Binomial: P(3 heads in 5 fair flips)
print(binomial_pmf(3, 5, 0.5))  # 0.3125

# Poisson: P(k=2, λ=3)
print(poisson_pmf(2, 3.0))  # 0.2240...
```

### Statistics
```python
from cds.stats import mean, stdev, correlation, linear_regression

data = [12.5, 14.3, 11.8, 15.1, 13.7]
print(f"mean={mean(data):.2f}, std={stdev(data):.2f}")

x = [1, 2, 3, 4, 5]
y = [2.1, 3.9, 6.2, 7.8, 10.1]
reg = linear_regression(x, y)
print(f"y = {reg.slope:.2f}x + {reg.intercept:.2f}, R²={reg.r_squared:.3f}")
```

### Machine Learning
```python
from cds.ml import Layer, MLP

# Simple XOR-like Neural Network
net = MLP([Layer(2, 4, activation="relu"), Layer(4, 1, activation="sigmoid")])
X, y = [[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [0]]

# Train with built-in Adam optimizer
history = net.train(X, y, epochs=50, lr=0.1)
print(f"Final loss: {history['final_loss']:.4f}")
```

### Data Analysis & Visualization
```python
from cds.data_analysis import DataSet, plot_bar

# Mini-Pandas DataSet for filtering and grouping
data = [{"name": "A", "score": 88}, {"name": "B", "score": 92}]
ds = DataSet(data)
filtered = ds.filter(lambda row: row["score"] > 90)
print(filtered.column("name"))  # ['B']

# Terminal Visualization
scores = {row["name"]: row["score"] for row in ds.to_list()}
print(plot_bar(scores, title="Scores"))
```

### Scientific Computing
```python
from cds.scientific import kinetic_energy, escape_velocity, get_constant

print(get_constant("c"))  # speed of light
print(kinetic_energy(10, 5))  # 125.0 J
print(escape_velocity(5.972e24, 6.371e6))  # ~11186 m/s
```

### Graph Theory
```python
from cds.graph import Graph, dijkstra, kruskal_mst, bfs

g = Graph(n_vertices=4, directed=False)
g.add_edge(0, 1, 1.0)
g.add_edge(1, 2, 2.0)
g.add_edge(2, 3, 3.0)
g.add_edge(0, 3, 10.0)

dist, prev = dijkstra(g, 0)
print(dist)  # {0: 0.0, 1: 1.0, 2: 3.0, 3: 6.0}

edges, total = kruskal_mst(g)
print(f"MST weight: {total}")  # 6.0
```

### Mathematical Modeling
```python
from cds.modeling import Variable, Sin, Exp, solve_equation

x = Variable("x")
expr = Sin(x) * Exp(x)  # sin(x) * e^x

# Symbolic derivative (chain + product rules)
print(expr.diff("x").to_str())

# Solve x^2 - 2 = 0  =>  x = sqrt(2)
root = solve_equation(Variable("x") ** 2 - 2, variable="x", x0=1.0)
print(root.x)  # ~1.4142
print(root.converged)  # True
```

### Knowledge Organization
```python
from cds.knowledge import KnowledgeGraph, Notebook, search

kg = KnowledgeGraph(name="Cosmology")
kg.link_concepts("Dark Energy", "Hubble Constant", kind="affects")
kg.link_concepts("Hubble Constant", "CMB", kind="constrains")

# Shortest path across the (undirected) graph
print(kg.find_path("Dark Energy", "CMB"))
# ['Dark Energy', 'Hubble Constant', 'CMB']

nb = Notebook(name="Lab Book")
nb.add_note(
    "n1",
    "Hubble Tension",
    "Local vs CMB H0 disagree.",
    tags=["experiment"],
    linked_concepts=["Hubble Constant"],
)

# Ranked retrieval across both concepts and notes
for hit in search(kg, nb, query="hubble"):
    print(hit.concept_name or hit.note_id, hit.score)
```

### Monte Carlo Simulation
```python
import math
from cds.montecarlo import estimate_pi, mc_integrate

if __name__ == "__main__":
    # Unit-circle method
    result = estimate_pi(n_samples=100_000, seed=42)
    print(f"PI approximation: {result.estimate:.4f}")

    # Integration
    area = mc_integrate(math.sin, 0, math.pi, n_samples=100_000)
    print(f"Integral of sin(x): {area.estimate:.4f}")
```

### Differential Equations
```python
from cds.diffeq import rk4, solve_system
import math

# dy/dt = -y, y(0)=1  =>  y(t) = e^(-t)
sol = rk4(lambda t, y: -y, t0=0, y0=1.0, t_end=2.0)
print(f"y(2) = {sol.y[-1]:.6f}")  # ~0.135335 (e^-2)


# Harmonic oscillator: x'' = -x
def harmonic(t, y):
    return [y[1], -y[0]]


t_vals, y_vals = solve_system(harmonic, 0, [1.0, 0.0], math.pi)
print(f"x(π) = {y_vals[-1][0]:.4f}")  # ~-1.0
```

### Numerical Integration
```python
import math
from cds.numerical_integration import simpson, gaussian_quadrature, romberg

# ∫_0^π sin(x) dx = 2
print(simpson(math.sin, 0, math.pi, n=100))  # ~2.0, O(h⁴)

# Gauss-Legendre: exact for polynomials up to degree 2n-1
print(gaussian_quadrature(lambda x: x**7, 0, 1, n=4))  # 0.125 (exact)

# Romberg reaches full machine precision on smooth integrands
result = romberg(math.exp, 0, 1, tol=1e-12)
print(f"∫e^x = {result.value:.10f}")  # ~1.7182818285
```

## Architecture

```
src/cds/
├── quantum/        # Quantum circuit simulation (single & multi-qubit)
├── optimization/   # Gradient descent, Newton, Adam, line search
├── ml/             # Neural Networks (MLP, Layers, Adam training)
├── signals/        # DFT, FFT, convolution, filtering
├── probability/    # Probability distributions & sampling
├── stats/          # Statistical analysis & regression
├── math_utils/     # Calculus, linear algebra, eigenvalues, Gram-Schmidt
├── data_analysis/  # Mini-Pandas DataSet, CSV loading, ASCII viz
├── scientific/     # Physical constants & formulas
├── graph/          # Graph algorithms (Dijkstra, BFS, DFS, Kruskal MST)
├── modeling/       # Symbolic math (expressions, MathModel, solvers)
├── knowledge/      # Knowledge graph, concepts, notes, structured retrieval
├── montecarlo/     # Monte Carlo methods (π, integration, random walks)
├── diffeq/         # ODE solvers (Euler, RK4, midpoint)
├── numerical_integration/  # Deterministic quadrature (trapezoid, Simpson, Romberg, Gauss-Legendre)
├── nlp/            # Educational NLP (BPE, embeddings, attention, autograd, MiniGPT)
├── interpolate/    # 1-D linear/nearest + 2-D bilinear interpolation
├── pde/            # 1-D finite-difference PDE solvers (heat FTCS, wave CFL)
├── hypothesis/     # Hypothesis generation + correlation-mining engine
├── core/           # Shared models, config
└── cli/           # Command-line interface package

examples/           # Runnable demo scripts
tests/              # full test suite (see CI badge for the live count)
docs/               # MkDocs documentation, tutorials, benchmarks
.github/workflows/  # Automation for PRs (labels + checklist), releases, and dependency updates
```

## Vision

The long-term goal of CDS is a lightweight, dependency-free platform for scientific exploration: numerical foundations (quantum simulation, FFT, linear algebra, statistics, ODE solvers) in the same package as higher-level tools for hypothesis generation and research workflows, all readable end-to-end.

A distinctive part is the `cds.hypothesis` module, which generates structured, falsifiable hypotheses with explicit assumptions and predictions. The `cds hypothesis` CLI command and `examples/hypothesis_demo.py` make this side immediately usable.

Development is incremental: each release adds a module or hardens an existing one, with the test suite, type checker, and coverage gate kept green throughout. The README "Recent improvements" section and [`CHANGELOG.md`](CHANGELOG.md) track what landed when.

Run `cds modules` after installation to explore the current modules.

## Recent improvements

**v1.2.0 (2026-06-25)** — horizontal expansion + hardening:

- **Time-series analysis** (`cds.stats`) — ACF/PACF, KPSS & Ljung-Box tests, exponential smoothing, seasonal decomposition, differencing, moving average.
- **Signal-filter design** (`cds.signals`) — Butterworth IIR low/high/band-pass coefficient design, direct-form `apply_filter`, edge-preserving `moving_median` denoiser.
- **2-D quadrature** (`cds.numerical_integration`) — tensor-product Simpson and Gauss-Legendre rules over rectangular domains.
- **`cds[pandas]` optional extra** — `to_dataframe` / `from_dataframe` round-trip for `DataSet`, guarded so the core stays zero-dependency.
- **Docs overhaul** — new Cookbook (~48 verified recipes), Architecture guide (`docs/ARCHITECTURE.md`), expanded Tour of Numerical Methods.
- **Refactors & stability** — `cds.modeling.expression` split into `_base`/`_nodes`; `cds.stats` distribution functions extracted to `_distributions`; numerical-stability fixes; no public API removed or renamed.

Earlier v1.1.x releases focused on reliability, type-safety, and a hardened release pipeline:

- **100% blended coverage gate** (v1.1.5) — CI now fails unless both statement and branch coverage reach 100%; property-based invariant tests and shared fixtures were added.
- **Python 3.13 support** (v1.1.5) — the full suite is green on 3.10–3.13.
- **Automated release pipeline** (v1.1.6) — pushing a `v*` tag now builds, publishes to PyPI via a scoped API token, cuts a GitHub Release, and attests build provenance (sigstore). `release.yml` is the sole publish authority.
- **PEP 639 SPDX license metadata** (v1.1.7) — the license declaration is now a valid SPDX expression recognized on both PyPI and GitHub.
- **ODE backward integration bug fix** (v1.1.8) — the fixed-step and adaptive ODE solvers (`euler`/`rk4`/`midpoint`/`rk45`/`solve_system`) silently returned only the initial value when `t_end < t0`; integration direction is now derived from `sign(t_end - t0)` so backward integration actually works. Forward behavior is unchanged. Includes 7 new regression tests and corrected deep-verification scripts.
- **Tutorials & architecture docs** (v1.1.5) — guided walkthroughs for optimization, signals, ML, statistics, and an architecture section in `CONTRIBUTING.md`.

See [`CHANGELOG.md`](CHANGELOG.md) for the full release history.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and guidelines.

Looking for:
- Researchers with domain expertise
- People interested in pure-Python scientific computing
- Contributors for new modules (ML basics, PDE solvers, etc.)
- People who want to help make scientific tools easier to maintain and use


## Automation and Maintenance Workflows

A few GitHub Actions handle repetitive aspects of keeping the project running:

- Dependabot for regular updates to dependencies and GitHub Actions
- Automatic labeling and review checklists for pull requests
- An automated release pipeline: pushing a version tag builds, publishes to PyPI via a scoped API token, cuts a GitHub Release with artifacts, and attests build provenance (sigstore)

These help ensure that time spent on the project goes more toward developing new modules, improving hypothesis tools, and supporting research use cases rather than manual upkeep.

See `.github/workflows/` for the current setup.

## License

MIT — see [LICENSE](LICENSE).

## Contact

- Maintainer: [@Furox88](https://github.com/Furox88)
- Issues & Discussions: [GitHub](https://github.com/Furox-Art/scientific-computing-system/issues)


## Security

Found a vulnerability? **Please do not open a public issue.** Report it privately:

- GitHub private advisory: [Report a vulnerability](https://github.com/Furox-Art/scientific-computing-system/security/advisories/new)
- Or email the maintainer directly.

Acknowledgement target: **48 hours** · Fix SLA: **7 days**. Full threat model,
supported versions, and out-of-scope items are in [SECURITY.md](SECURITY.md).

> **Why all the automation?** CDS is maintained solo. The workflows above (PR labeling, checklists, releases, dependency rotation) exist so routine housekeeping takes minutes, leaving the bulk of maintainer time for the actual science — improving the hypothesis tools, adding modules, and writing better examples.

---

## Support the project

CDS is built and maintained solo, for free. If it helped your research, teaching, or prototyping:

- **Star the repo** — [github.com/Furox-Art/scientific-computing-system](https://github.com/Furox-Art/scientific-computing-system) — it costs nothing and is the single biggest signal that helps others discover CDS.
- **Share it** — a post on X, Reddit, or with a colleague who'd find it useful.
- **Cite it** — see [CITATION.cff](CITATION.cff) if CDS appears in your work.
- **Contribute** — new modules, docs, examples, and issue triage are all welcome. See [Contributing](#contributing).

Thank you for using CDS.
