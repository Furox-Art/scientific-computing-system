# Scientific Computing System (CDS)

<div class="cds-hero" markdown>
<p class="cds-hero__title">Computational science, from scratch.</p>
<p class="cds-hero__tagline">
CDS is an open-source platform for research, simulation and discovery:
<strong>36 domain modules</strong> spanning quantum simulation, FFT, linear algebra,
statistics, ODE/PDE solving, uncertainty quantification, Monte Carlo, symbolic
math, machine learning and NLP, implemented in readable pure Python with
<strong>zero runtime dependencies</strong>.
</p>
<div class="cds-hero__actions" markdown>
[Get started](getting-started.md){ .md-button .md-button--primary }
[Take the tour](tour_of_numerical_methods.md){ .md-button }
[API reference](api.md){ .md-button }
</div>
</div>

```bash
pip install scientific-computing-system   # zero runtime dependencies
cds modules                               # inventory of every module
```

The import name is `cds` (not `scs`). Run the verified five-minute transcript:

```bash
python examples/quickstart_demo.py
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
Every public function and class, generated from the source docstrings.
</div>

<div class="cds-card" markdown>
### [Architecture](ARCHITECTURE.md)
Module dependency graph and data flow, for contributors and auditors.
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
**sensitivity analysis**, **signal processing**, **linear algebra**,
**symbolic mathematics**, and **reproducible computational research**.

## Key Features

- **Pure Python:** Every module is implemented from scratch using the Python standard library. No heavy dependencies like NumPy or SciPy required.
- **Quantum Simulation:** Full state-vector simulation for single and multi-qubit circuits with entanglement and O(1) sampling.
- **Advanced Mathematics:** O(N³) Partial Pivoting LU decomposition, vectorized optimizers, and adaptive ODE solvers (RK45).
- **Hypothesis Engine:** Built-in tools for generating and statistically validating scientific hypotheses, complemented by effect-size measures (Cohen's d, Cramér's V) that quantify the magnitude of an effect alongside its significance.
- **High Reliability:** Comprehensive test suite with 100% code coverage (statement + branch) on the reference CI cell. See the CI and codecov badges in the [README](https://github.com/Furox-Art/scientific-computing-system) for the live test count and coverage.
- **Interactive Tools:** Beautiful CLI and a Streamlit-based web dashboard.

## Overview of Modules

| Module | Description |
|--------|-------------|
| `cds.core` | Shared data models (`Domain`, `Hypothesis`, `HypothesisStatus`) |
| `cds.quantum` | Single & multi-qubit quantum circuit simulation |
| `cds.optimization` | Gradient-based and numerical optimizers |
| `cds.ml` | Pure Python Neural Networks (MLP, Adam-based training) |
| `cds.signals` | Fast signal processing (DFT, FFT/IFFT, convolution) + Butterworth IIR filter design & moving-median denoiser |
| `cds.probability` | Probability distributions & sampling |
| `cds.stats` | Descriptive stats, regression, hypothesis testing, effect-size measures (Cohen's d, Cramér's V) & time-series analysis (ACF/PACF, KPSS, Ljung-Box, decomposition) |
| `cds.math_utils` | Numerical calculus, linear algebra, eigenvalues |
| `cds.data_analysis` | Structured data management, visualization & optional pandas interop (`cds[pandas]`) |
| `cds.scientific` | Physical constants & scientific formulas |
| `cds.graph` | Graph algorithms (BFS, DFS, Dijkstra, Kruskal MST) |
| `cds.modeling` | Symbolic algebra: expressions, differentiation, simplification, LaTeX export, `MathModel` equation systems, root-finding & parameter fitting |
| `cds.knowledge` | Knowledge organization: concept graph with typed relations, research notes notebook, ranked structured retrieval (JSON persistence) |
| `cds.montecarlo` | Monte Carlo integration, π estimation, random walks |
| `cds.diffeq` | ODE solvers (Euler, RK4, midpoint) |
| `cds.numerical_integration` | Deterministic quadrature (trapezoid, Simpson, Romberg) + 2-D tensor-product rules (Simpson, Gauss-Legendre) |
| `cds.nlp` | Educational NLP from scratch (BPE, embeddings, attention, autograd, MiniGPT) |
| `cds.hypothesis` | Structured scientific hypothesis generation and statistical evaluation |
| `cds.plot` | Optional matplotlib charts (series, scatter, regression, spectra, ACF, seasonal, heatmaps, …) via `cds[plot]` |
| `cds.uncertainty` | Measurement-uncertainty representation and propagation (linearised & Monte Carlo) |
| `cds.sensitivity` | Local/global sensitivity, variance-based screening and identifiability analysis |
| `cds.causal` | Conservative, assumption-gated causal effect estimators |
| `cds.units` | SI units, conversions and dimensional-analysis compatibility checks |
| `cds.validation` | Scientific validation, cross-method checking, distribution drift and final audit |
| `cds.workflow` | Approval-gated scientific workflow orchestration |
| `cds.provenance` | Reproducibility manifests, hashes, tool versions and local checkpoints |
| `cds.data_io` | Memory-bounded streaming file I/O plus lazy HDF5/NetCDF backends via `cds[io]` |
| `cds.tools` | Capability discovery and adapters for optional NumPy/SciPy/SymPy/Z3 backends |
| `cds.bayes` | Bayesian conjugate analysis (pure Python) |
| `cds.infotheory` | Information-theory measures (entropy, divergence, dependence) |
| `cds.interpolate` | 1-D interpolation (`interp1d`) and bilinear `interp2d` on regular grids |
| `cds.pde` | Finite-difference PDE solvers on uniform 1-D grids (heat & wave, CFL/stability guarded) |
| `cds.wavelets` | Haar wavelet transform (`dwt`/`idwt`), orthonormal with perfect reconstruction |
| `cds.fractals` | Fractal geometry (escape-time and IFS fractals) |
| `cds.genetics` | DNA sequence analysis |
| `cds.cli` | Stdlib `argparse` command-line interface (`cds --help`) |

## Quick Navigation

- [Getting Started](getting-started.md)
- [API Reference](api.md)
- [Cookbook](cookbook.md): problem-oriented recipes for every module
- [Tour of Numerical Methods](tour_of_numerical_methods.md): guided walkthrough
- [Architecture](ARCHITECTURE.md): module dependency graph & data flow
- [Case Studies](CASE_STUDY_HUBBLE.md)
- [Benchmarks](benchmarks.md)

---
*CDS is stable and actively developed. Contributions are welcome!*
