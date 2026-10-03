# Scientific Computing System (CDS)

<div class="cds-hero" markdown>
<p class="cds-hero__title">Computational science, from scratch.</p>
<p class="cds-hero__tagline">
CDS is an open-source platform for research, simulation and discovery: 34 subpackages
spanning quantum simulation, FFT, linear algebra, statistics, ODE/PDE, symbolic math,
machine learning, uncertainty quantification and NLP, implemented in readable pure
Python with <strong>zero runtime dependencies</strong>.
</p>
<div class="cds-hero__actions" markdown>
[Get started](getting-started.md){ .md-button .md-button--primary }
[Take the tour](tour_of_numerical_methods.md){ .md-button }
[API reference](api.md){ .md-button }
</div>
</div>

```bash
pip install scientific-computing-system
cds modules          # see what's inside
```

## Start here

<div class="cds-cards" markdown>

<div class="cds-card" markdown>
### [Getting Started](getting-started.md)
Install, run your first simulation, and learn the CLI in about five minutes.
</div>

<div class="cds-card" markdown>
### [Quick Start Tutorial](tutorials/quick_start.md)
A guided first session: constants, statistics, a quantum circuit, an ODE.
</div>

<div class="cds-card" markdown>
### [Tour of Numerical Methods](tour_of_numerical_methods.md)
How the solvers actually work: quadrature, RK45, LU, FFT, with worked output.
</div>

<div class="cds-card" markdown>
### [Cookbook](cookbook.md)
Problem-oriented recipes: pick the task, copy the snippet.
</div>

<div class="cds-card" markdown>
### [API Reference](api.md)
All 34 public subpackages, generated from the source docstrings.
</div>

<div class="cds-card" markdown>
### [Architecture](ARCHITECTURE.md)
Module dependency graph and data flow, for contributors and auditors.
</div>

<div class="cds-card" markdown>
### [Why Pure Python?](why-pure-python.md)
The trade-off stated plainly, and when to reach for the NumPy-based 2.0 project instead.
</div>

</div>

!!! tip "New to scientific Python?"

    CDS is designed as a first stop: every algorithm is readable pure Python, so
    you learn *how* things work, not just how to call them. Follow the tutorials
    in order: [Quick Start](tutorials/quick_start.md),
    [Statistics](tutorials/stats_demo.md),
    [Machine Learning](tutorials/ml_demo.md), then branch out.

## What people use CDS for

CDS is designed for searchable, practical scientific-Python tasks such as
**numerical methods**, **ODE and PDE solving**, **Monte Carlo simulation**,
**statistics and hypothesis testing**, **uncertainty quantification**,
**sensitivity analysis**, **dimensional analysis**, **signal processing**,
**linear algebra**, **symbolic mathematics**, and **reproducible computational
research**.

## Key Features

- **Pure Python:** Every module is implemented from scratch using the Python standard library. No heavy dependencies like NumPy or SciPy required.
- **Quantum Simulation:** Full state-vector simulation for single and multi-qubit circuits with entanglement and O(1) sampling.
- **Advanced Mathematics:** O(N³) Partial Pivoting LU decomposition, SVD, QR, Cholesky, and adaptive ODE solvers (RK45).
- **Scientific Assurance:** Uncertainty propagation, local & variance-based global sensitivity, identifiability analysis, validation checks, cross-method verification, assumption-gated causal estimation, and SI dimensional analysis.
- **Reproducible Orchestration:** Approval-gated workflow execution that fails closed, plus run manifests, canonical hashing, and checkpoints.
- **Hypothesis Engine:** Built-in tools for generating and statistically validating scientific hypotheses, complemented by effect-size measures (Cohen's d, Cramér's V) that quantify the magnitude of an effect alongside its significance.
- **High Reliability:** Comprehensive test suite with 100% blended code coverage (statement + branch) on the reference CI cell. See the CI and codecov badges in the [README](https://github.com/Furox-Art/scientific-computing-system) for live status.
- **Interactive Tools:** `cds` CLI (pure stdlib) and an optional Streamlit dashboard.

## Overview of Modules

`src/cds/` contains 34 subpackages plus `causal.py` and `sensitivity.py` — 36
public feature modules. This table summarises them and the
[API Reference](api.md) documents each one. (`cds modules` prints a curated
26-entry subset, not the full catalog.)

| Module | Description |
|--------|-------------|
| `cds.core` | Shared data models (`Domain`, `Hypothesis`, `HypothesisStatus`) |
| `cds.hypothesis` | Structured hypothesis generation and evaluation |
| `cds.knowledge` | Concept graphs, research notes, ranked structured retrieval |
| `cds.stats` | Descriptive stats, regression, hypothesis testing, effect sizes, time series |
| `cds.probability` | Probability distributions, quantiles, seeded sampling |
| `cds.bayes` | Bayesian conjugate updates, credible intervals, Bayes factors |
| `cds.math_utils` | Calculus + linear algebra (SVD, QR, Cholesky, LU, eigen) |
| `cds.interpolate` | `interp1d` / `interp2d` over sampled domains |
| `cds.numerical_integration` | Trapezoid, Simpson, Romberg, Gauss-Legendre, adaptive, 2-D |
| `cds.montecarlo` | π estimation, Monte-Carlo integration, random walks |
| `cds.diffeq` | Euler, RK4, adaptive RK45, implicit stiff, symplectic |
| `cds.pde` | Finite-difference heat & wave equation solvers (1-D) |
| `cds.optimization` | Gradient descent, Newton, Adam, Nelder-Mead, annealing, constrained |
| `cds.ml` | MLP, k-NN, k-means, CART, random forest, boosting, PCA (educational) |
| `cds.signals` | DFT/FFT, convolution, Butterworth design, STFT, spectral tools |
| `cds.wavelets` | Haar DWT/IDWT, multi-level decomposition, denoising |
| `cds.quantum` | Single & multi-qubit state-vector simulation |
| `cds.scientific` | Physical constants & classical physics formulas |
| `cds.genetics` | GC content, k-mers, reverse complement, Needleman-Wunsch |
| `cds.fractals` | Mandelbrot, Julia, Barnsley fern, Sierpinski triangle |
| `cds.infotheory` | Entropy, cross-entropy, KL / JS divergence, mutual information |
| `cds.modeling` | Symbolic expressions, differentiation, LaTeX, `MathModel`, fitting |
| `cds.graph` | BFS, DFS, Dijkstra, Kruskal MST, topological sort, cycles |
| `cds.uncertainty` | Analytic + correlated Monte-Carlo uncertainty propagation |
| `cds.sensitivity` | Local & variance-based global sensitivity, identifiability |
| `cds.validation` | Validation checks, data adequacy, drift / OOD reports |
| `cds.causal` | Assumption-gated causal effect estimators |
| `cds.units` | SI quantities, conversions, dimensional compatibility |
| `cds.workflow` | Approval-gated scientific workflow orchestration |
| `cds.provenance` | Run manifests, canonical SHA-256, decisions, checkpoints |
| `cds.data_analysis` | CSV/tabular loading, normalisation, ASCII visualisation |
| `cds.data_io` | Streaming I/O, online moments, optional HDF5 / NetCDF |
| `cds.tools` | Lazy discovery + adapters for NumPy/SciPy/sklearn/SymPy/Z3 |
| `cds.plot` | Optional matplotlib charts via the `[plot]` extra |
| `cds.nlp` | Educational NLP from scratch (BPE, embeddings, attention, MiniGPT) |
| `cds.cli` | The `cds` console script (pure-stdlib `argparse`) |

## Quick Navigation

- [Getting Started](getting-started.md)
- [API Reference](api.md)
- [Cookbook](cookbook.md): problem-oriented recipes for every module
- [Tour of Numerical Methods](tour_of_numerical_methods.md): guided walkthrough
- [Why Pure Python?](why-pure-python.md): the trade-off, stated plainly
- [Architecture](ARCHITECTURE.md): module dependency graph & data flow
- [Case Studies](CASE_STUDY_HUBBLE.md)
- [Benchmarks](benchmarks.md)
- [Research Workflows](research-workflows.md)
- [Related projects](related.md)

---
*CDS is stable and actively developed. Contributions are welcome — see
[CONTRIBUTING.md](https://github.com/Furox-Art/scientific-computing-system/blob/main/CONTRIBUTING.md).*
