<p align="center">
  <img src="assets/logo.svg" alt="scientific-computing-system" width="640">
</p>

<h1 align="center">scientific-computing-system</h1>

<p align="center"><b>A pure-Python scientific computing library: numerical linear algebra, ODE/PDE solvers, statistics, uncertainty quantification, Monte Carlo, signal processing and quantum simulation — with <b>zero runtime dependencies</b>.</b></p>

<p align="center">
  <a href="https://pypi.org/project/scientific-computing-system/"><img src="https://img.shields.io/pypi/v/scientific-computing-system.svg" alt="PyPI version"></a>
  <a href="https://www.npmjs.com/package/scientific-computing-system"><img src="https://img.shields.io/npm/v/scientific-computing-system.svg" alt="npm version"></a>
  <a href="https://pypi.org/project/scientific-computing-system/"><img src="https://img.shields.io/pypi/dm/scientific-computing-system.svg" alt="PyPI downloads"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-green.svg" alt="Python 3.10+"></a>
  <a href="https://codecov.io/gh/Furox-Art/scientific-computing-system"><img src="https://codecov.io/gh/Furox-Art/scientific-computing-system/branch/main/graph/badge.svg" alt="codecov"></a>
  <a href="https://github.com/Furox-Art/scientific-computing-system/actions/workflows/tests.yml"><img src="https://github.com/Furox-Art/scientific-computing-system/actions/workflows/tests.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://furox-art.github.io/scientific-computing-system/"><img src="https://img.shields.io/badge/docs-mkdocs-teal.svg" alt="Docs"></a>
  <a href="https://github.com/Furox-Art/scientific-computing-system/releases"><img src="https://img.shields.io/github/v/release/Furox-Art/scientific-computing-system.svg" alt="GitHub release"></a>
</p>

<p align="center">
  <b>34 domain modules</b> · <b>zero runtime dependencies</b> · <b>2,900+ tests</b> · <b>100% branch coverage</b> · <b>Python 3.10–3.13</b> · <b>MIT</b>
</p>

## Why this library

NumPy and SciPy are excellent, and this library is **not** a competitor to them.
It exists for the cases where they are the wrong tool:

- **You need to read the algorithm.** Every routine is plain Python you can open
  and step through. When a solver returns something surprising, you can find out
  why — not file a bug against a binary wheel.
- **You cannot take the dependency.** `pip install scientific-computing-system`
  pulls in nothing. No BLAS/MKL mismatch, no broken ARM wheel, no resolver
  conflict with the rest of your environment. The core is enforced by
  `tests/test_zero_dependencies.py`.
- **You are teaching or auditing.** A lecture on RK4, SVD or the central limit
  theorem is shorter when the implementation fits on one screen.

**The honest tradeoff:** it is slower. The docs
[measure this honestly](https://furox-art.github.io/scientific-computing-system/benchmarks/)
— the benchmark page reports CDS roughly 1000× slower than NumPy on dense matrix
multiplication, and shows where it still wins (FFT complexity, O(1) quantum
sampling). For production number crunching, reach for NumPy. For prototype,
teaching, audit, or a dependency-constrained environment, use this.

## Install

```bash
pip install scientific-computing-system
```

Requires Python 3.10+. The install has no transitive dependencies. Optional
extras are opt-in and never needed for the core:

```bash
pip install "scientific-computing-system[pandas,plot]"   # pandas interop, matplotlib plots
pip install "scientific-computing-system[scientific,io]"  # NumPy/SciPy adapters, HDF5/NetCDF
```

## Quickstart

This is real output from `examples/quickstart_demo.py`, which runs on the
standard library alone. Every computation is seeded or compared against a closed-form
answer, so it is reproducible byte-for-byte:

```python
import math

from cds import diffeq, math_utils, montecarlo, stats, units
from cds.uncertainty import propagate_linear

# --- Linear algebra: SVD, and solve a small system ---
a = [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]]
svd = math_utils.svd(a)
print([round(s, 6) for s in svd.singular_values])  # [4.732051, 3.0, 1.267949]
print(math_utils.solve_linear([[2.0, 1.0], [4.0, 3.0]], [5.0, 11.0]))  # [2.0, 1.0]

# --- ODE: RK4 on y' = y, whose exact solution is exp(t) ---
sol = diffeq.rk4(lambda t, y: y, 0.0, 1.0, 1.0, dt=0.01)
print(round(sol.y[-1], 9), round(math.exp(1.0), 9))  # 2.718281828 2.718281828

# --- Statistics: least squares + a hypothesis test ---
fit = stats.linear_regression([1.0, 2.0, 3.0, 4.0, 5.0], [2.1, 3.95, 6.1, 7.85, 10.2])
print(fit.slope, round(fit.r_squared, 6))  # 2.01 0.998122

# --- Monte Carlo: E[X^2] over Uniform(0,1) is exactly 1/3 ---
mc = montecarlo.mc_expectation(lambda x: x**2, n_samples=200_000, seed=20260930)
print(round(mc.estimate, 6))  # 0.333985

# --- Uncertainty: propagate standard uncertainties (GUM-style budget) ---
r = propagate_linear(lambda x, y: math.hypot(x, y), [3.0, 4.0], standard_uncertainties=[0.3, 0.4])
print(r.value, round(r.standard_uncertainty, 6))  # 5.0 0.367151

# --- Dimensional analysis: catch unit mistakes at the type level ---
print(units.dimensions_compatible(units.NEWTON * units.METER, units.JOULE))  # True
```

Actual output:

```text
CDS 2.1.1 -- quickstart (seed=20260930)

[1] SVD of a symmetric 3x3 matrix
    singular values  : [4.732051, 3.0, 1.267949]
    matrix rank      : 3
    residual ||A - USVt||_F: 1.021e-15
    determinant       : 18.000000
    solve_linear([[2,1],[4,3]], [5,11]) -> [2.0, 1.0]  (exact [2, 1])
    linear-system residual: 0.000e+00

[2] RK4 on y' = y, y(0) = 1, exact solution exp(t)
    steps taken         : 100
    y(1) numeric        : 2.718281828234
    y(1) exact          : 2.718281828459
    absolute error      : 2.246e-10

[3] Ordinary least squares on y = 1.9x + 0.2
    slope     : 2.010000
    intercept : 0.010000
    R^2       : 0.998122
    one-sample t-test vs mu=6.0 -> statistic 0.028117, p = 0.978916

[4] Monte Carlo: E[X^2] for X ~ Uniform(0,1), exact value 1/3
    samples   : 200000
    estimate  : 0.333985  (exact 0.333333)
    std error : 0.000667

[5] Uncertainty propagation for f(a,b) = sqrt(a^2 + b^2)
    output value            : 5.000000  (exact 5.000000)
    combined std uncertainty: 0.367151
    method                  : linearized
    MC cross-check std      : 0.366898
    95% interval            : [4.296696, 5.736151]

[6] Dimensional analysis
    N*m symbol              : N*m
    compatible with joule?  : True
    compatible with newton? : False

All six checks ran on the Python standard library alone.
Next: `cds modules` lists every scientific module with its capabilities.
```

Run it yourself:

```bash
python examples/quickstart_demo.py
```

Note on the two-module discrepancy: each step checks its answer against a
closed-form or analytically known value (exact ODE solution, exact integral,
RSS uncertainty), so the residuals above are real correctness evidence, not
just "it ran without raising".

## Import name

The distribution is `scientific-computing-system`; the **import name is `cds`**:

```bash
pip install scientific-computing-system
```

```python
import cds  # correct
from cds.math_utils import svd  # correct
```

## Command line

A stdlib `argparse` CLI ships with the package — no extra install:

```bash
cds --help        # every subcommand
cds modules       # the 25 scientific modules and their capabilities
cds constants     # physical constants
cds calc gravity  # quick physics calculation
cds stats 1,2,3,4 # descriptive statistics
cds integrate sin 0 3.14159
```

`cds modules` prints the full inventory:

```text
| Module                    | Key Capabilities                                             |
+---------------------------+--------------------------------------------------------------+
| cds.quantum               | Single/multi-qubit circuits, Bell/GHZ states, entanglement   |
| cds.signals               | DFT/FFT, convolution, filtering, spectral utilities          |
| cds.math_utils            | Linear algebra, decompositions, calculus utilities           |
| cds.optimization          | Gradient/Newton/Adam, Nelder-Mead, annealing, search         |
| cds.stats                 | Inference, regression, tests, time-series statistics         |
| cds.probability           | Distributions, CDFs, quantiles, and sampling                 |
| cds.montecarlo            | Monte Carlo integration, simulation, and random walks        |
| cds.diffeq                | Explicit/adaptive/stiff ODE solvers and PDE utilities        |
| cds.modeling              | Symbolic models, equation solving, and parameter fitting     |
| cds.ml                    | Classical ML estimators, preprocessing, validation, PCA      |
| cds.data_analysis         | Tabular analysis, normalization, visualization helpers       |
| cds.data_io               | Memory-bounded streaming plus optional HDF5/NetCDF adapters  |
| cds.units                 | SI units, conversions, and dimensional analysis              |
| cds.uncertainty           | Analytic/correlated Monte Carlo uncertainty propagation      |
| cds.sensitivity           | Dependency-free local parameter sensitivity analysis         |
| cds.validation            | Scientific checks, cross-method verification, final audit    |
| cds.workflow              | Approval-gated scientific workflow orchestration             |
| cds.provenance            | Run manifests, hashes, tool versions, decisions, checkpoints |
| cds.tools                 | Lazy scientific backends plus SciPy/SymPy/Z3 adapters        |
| cds.knowledge             | Knowledge graph, concept mapping, notes, retrieval           |
| cds.hypothesis            | Structured scientific hypothesis generation                  |
| cds.scientific            | Physical constants and common scientific formulas            |
| cds.numerical_integration | Adaptive and fixed numerical quadrature                      |
| cds.graph                 | Graph traversal, shortest paths, MST, topological sort       |
| cds.nlp                   | Educational tokenizer, embeddings, attention, MiniGPT        |
| cds.plot                  | Optional matplotlib scientific plots                         |
+---------------------------+--------------------------------------------------------------+
```

## Scientific domains covered

| Domain | Modules |
|--------|---------|
| **Numerical linear algebra** | `math_utils` — SVD, LU, QR, Cholesky, rank, condition number, matrix inverse |
| **ODE / PDE solving** | `diffeq` (Euler, midpoint, RK4, RK45, implicit, velocity Verlet, symplectic), `pde` (heat & wave, stability- and CFL-guarded) |
| **Numerical integration** | `numerical_integration` — trapezoid, Simpson, Romberg, Gauss-Legendre, 2-D tensor rules |
| **Interpolation** | `interpolate` — `interp1d`, bilinear `interp2d` |
| **Optimization** | `optimization` — gradient descent, Newton, Adam, Nelder-Mead, simulated annealing, projected gradient, quadratic penalty |
| **Statistics** | `stats` — regression, t-tests, ANOVA, chi-square, nonparametric, multiple-testing correction, bootstrap CIs, time-series (ACF/PACF, KPSS, Ljung-Box, decomposition) |
| **Effect sizes** | `stats` — Cohen's *d*, Cramér's V, eta-squared |
| **Power analysis** | `stats.power` — `power_t_test`, `required_n_per_group` |
| **Probability** | `probability` — pmf/pdf/cdf/ppf for binomial, Poisson, Gaussian, uniform, exponential, gamma, chi-square, *t*, beta, geometric, hypergeometric, negative-binomial |
| **Bayesian inference** | `bayes` — conjugate posteriors |
| **Monte Carlo** | `montecarlo` — integration, expectation, Buffon, random walks, Metropolis-Hastings MCMC |
| **Uncertainty quantification** | `uncertainty` — linearized and Monte Carlo propagation, correlated inputs |
| **Sensitivity & identifiability** | `sensitivity` — local and global parameter sensitivity |
| **Causal estimation** | `causal` |
| **Signal processing** | `signals`, `wavelets` — DFT/FFT, STFT, convolution, Butterworth IIR filters, Haar DWT |
| **Quantum simulation** | `quantum` — state-vector simulation, gates, Bell/GHZ states, entanglement, multi-qubit registers |
| **Machine learning** | `ml` — PCA, KMeans, kNN, decision trees, random forest, gradient boosting, Naive Bayes, logistic/linear regression, MLP, voting ensembles, cross-validation, metrics |
| **Symbolic mathematics** | `modeling` — expression trees, differentiation, LaTeX export, `MathModel` equation systems, root finding, parameter fitting |
| **Information theory** | `infotheory` |
| **Information & graph algorithms** | `graph` — BFS, DFS, Dijkstra, Kruskal MST, topological sort |
| **Physics & chemistry** | `scientific` (constants, formulas), `genetics`, `fractals` |
| **Scientific data & I/O** | `data_analysis` (DataSet/DataTable, optional pandas bridge), `data_io` (memory-bounded streaming, optional HDF5/NetCDF) |
| **Units & dimensional analysis** | `units` — SI quantities, conversions, `dimensions_compatible` |
| **Validation & drift** | `validation` — adequacy checks, cross-method verification, distribution drift |
| **Scientific workflows** | `workflow` — approval-gated orchestration, selection, gates |
| **Reproducibility & provenance** | `provenance` — run manifests, hashes, tool versions, checkpoints |
| **Hypothesis generation** | `hypothesis` — structured hypotheses from a research question, plus statistical evaluation |
| **Educational NLP** | `nlp` — BPE tokenizer, embeddings, attention, autograd tensor engine, MiniGPT |
| **Plotting** | `plot` — optional matplotlib charts via the `[plot]` extra |

## Documentation

| Resource | Link |
|----------|------|
| **Documentation home** | [furox-art.github.io/scientific-computing-system](https://furox-art.github.io/scientific-computing-system/) |
| **Getting started** | [Install & first steps](https://furox-art.github.io/scientific-computing-system/getting-started/) |
| **Tutorials** | [32 guided walkthroughs with worked output](https://furox-art.github.io/scientific-computing-system/tutorials/quick_start/) |
| **Cookbook** | [Problem-oriented recipes — pick a task, copy the snippet](https://furox-art.github.io/scientific-computing-system/cookbook/) |
| **API reference** | [Every public function, generated from docstrings](https://furox-art.github.io/scientific-computing-system/api/) |
| **Case studies** | [Hubble tension](https://furox-art.github.io/scientific-computing-system/CASE_STUDY_HUBBLE/) · [Quantum + ML](https://furox-art.github.io/scientific-computing-system/CASE_STUDY_QUANTUM_ML/) |
| **Benchmarks** | [Measured performance and complexity scaling](https://furox-art.github.io/scientific-computing-system/benchmarks/) |
| **Why pure Python** | [The honest tradeoff](https://furox-art.github.io/scientific-computing-system/why-pure-python/) |
| **Architecture** | [Module dependency graph and data flow](https://furox-art.github.io/scientific-computing-system/ARCHITECTURE/) |
| **Runnable examples** | [`examples/` — 33 runnable demo scripts](https://github.com/Furox-Art/scientific-computing-system/tree/main/examples) |

## Adoption path

**Trying it (2 minutes)**

```bash
pip install scientific-computing-system
python -c "from cds import math_utils; print(math_utils.svd([[4.,1.,0.],[1.,3.,1.],[0.,1.,2.]]).singular_values)"
```

**Evaluating it against your own numbers (15 minutes)**

Run the quickstart, then check a result you already trust. Each of the six
sections above compares against a closed-form or analytically known answer.
If your own quantity agrees to the printed residual, the implementation is
behaving; if it does not, you have a reproducible counterexample and a bug report.

**Depending on it (production)**

The core has zero runtime dependencies and PEP 561 type hints
(`py.typed`). Pin the version (`pip install "scientific-computing-system==2.1.1"`).
Use optional extras only for the interop you need. Check
[CHANGELOG.md](CHANGELOG.md) for breaking changes — the project follows
[Semantic Versioning](https://semver.org/).

## Troubleshooting

| Symptom | Cause and fix |
|---------|---------------|
| `ModuleNotFoundError: No module named 'scs'` | The import name is `cds`, not `scs`. Use `from cds.math_utils import svd`. |
| `ModuleNotFoundError: No module named 'cds'` | Not installed in the active environment. `pip install scientific-computing-system`, then confirm with `python -m pip show scientific-computing-system`. |
| `command not found: cds` | The CLI entry point is not on `PATH`. Activate your venv, or run `python -m cds` instead. |
| `ImportError` from `cds.plot`, `cds.data_io` or `cds.tools` | Optional backend missing. Install the matching extra: `pip install "scientific-computing-system[plot,io,scientific]"`. |
| `pandas` not found in a recipe | The pandas bridge is opt-in: `pip install "scientific-computing-system[pandas]"`. |
| `estimate_pi` raises about `__main__` on Windows/macOS | It parallelizes with `ProcessPoolExecutor`, which on spawn-based platforms requires the `if __name__ == "__main__":` guard. Put the call inside that guard, or use the single-process `mc_expectation` / `mc_integrate`. |
| Results differ from the docs | Pin the version — docs track the released version. `pip install "scientific-computing-system==2.1.1"`. |
| A solver returns `converged=False` | Loosen `tol` or raise `max_iter`. Many optimizers return an `OptResult` dataclass with `.converged`, `.iterations` and `.value` rather than raising. |
| Want to confirm the install is clean | `python -m pytest tests/test_zero_dependencies.py` after `pip install -e .` verifies the zero-dependency claim. |

## Project links

| | |
|---|---|
| **Source** | [github.com/Furox-Art/scientific-computing-system](https://github.com/Furox-Art/scientific-computing-system) |
| **PyPI** | [pypi.org/project/scientific-computing-system](https://pypi.org/project/scientific-computing-system/) |
| **npm** | [npmjs.com/package/scientific-computing-system](https://www.npmjs.com/package/scientific-computing-system) |
| **Changelog** | [CHANGELOG.md](CHANGELOG.md) · [Releases](https://github.com/Furox-Art/scientific-computing-system/releases) |
| **Security policy** | [SECURITY.md](SECURITY.md) — please report vulnerabilities privately, not as public issues |
| **Contributing** | [CONTRIBUTING.md](CONTRIBUTING.md) |
| **Citation** | [CITATION.cff](CITATION.cff) |
| **Discussions & issues** | [Open an issue](https://github.com/Furox-Art/scientific-computing-system/issues) |

## Related project

For production workloads that need real speed, there is a companion package on
the same NumPy/SciPy stack:
[scientific-computing-system-2.0](https://github.com/Furox-Art/scientific-computing-system-2.0).
This repository stays pure-Python and dependency-free by design.

## License

[MIT](LICENSE).
