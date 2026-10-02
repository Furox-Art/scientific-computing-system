# API Reference

Every public CDS subpackage, rendered from the module's own docstrings by
[mkdocstrings](https://mkdocstrings.github.io/). The distribution is
`scientific-computing-system`; the import name is **`cds`**.

`src/cds/` currently contains **34 subpackages**. Every one is listed below.
`cds modules` prints the same catalog from the installed package, so it stays
authoritative if this page ever drifts.

Optional integrations (`cds.plot`, `cds.data_analysis.pandas_io`) need an extra;
the zero-dependency core does not.

---

## Foundations

### Core Data Models

Shared `Domain`, `Hypothesis`, and `HypothesisStatus` types — the foundation the
hypothesis engine builds on.

::: cds.core

### Hypothesis Generation

Structured, falsifiable hypothesis generation from a research question, with an
offline generator, prompt templates, a `Protocol` for custom generators, and a
statistical evaluator.

::: cds.hypothesis

### Knowledge Organization

Concept graphs joined by typed, directed relations, a notebook of research notes
linked to concept names, and ranked structured retrieval across both. JSON
persistence via the stdlib.

::: cds.knowledge

---

## Analysis

### Statistics

Descriptive statistics, regression, frequentist hypothesis tests (t-test,
chi-square, ANOVA) with companion effect-size measures (Cohen's d, eta-squared,
Cramer's V), nonparametric rank tests, multiple-testing correction, bootstrap,
power analysis, and time-series work (ACF/PACF, KPSS, Ljung-Box, seasonal
decomposition).

::: cds.stats

### Probability

Continuous PDFs and discrete PMFs (Gaussian, uniform, exponential, binomial,
Poisson, chi-square, Student-t), quantile functions including Marsaglia-Tsang
gamma/beta samplers, and reproducible seeded sampling.

::: cds.probability

### Bayesian Conjugate Analysis

Beta-binomial, gamma-Poisson and normal-normal updates, credible intervals, and
the binomial Bayes factor `BF10`.

::: cds.bayes

### Mathematical Utilities

Calculus (derivative, integral, gradient) and a compact linear-algebra toolkit:
SVD, QR, Cholesky, partial-pivoting LU, matrix inverse, rank, condition number,
eigenvalues via power iteration.

::: cds.math_utils

### Interpolation

Factory functions returning plain callables: `interp1d` on strictly increasing
samples and `interp2d` on regular rectangular grids. No extrapolation, no hidden
state, deterministic.

::: cds.interpolate

### Optimization

Gradient descent, Newton's method, Adam, golden-section line search, Nelder-Mead,
simulated annealing, projected gradient descent, and quadratic-penalty constrained
optimization.

::: cds.optimization

### Machine Learning

Educational, from-scratch estimators: MLP with Adam training, k-NN (classifier and
regressor), k-means, Gaussian naive Bayes, CART decision trees, random forest,
gradient boosting, voting ensembles, linear/logistic regression, PCA,
`StandardScaler`, and train/test splitting. Not intended for production use.

::: cds.ml

---

## Compute

### Numerical Integration

Deterministic quadrature: trapezoid, Simpson 1/3 and 3/8, Romberg,
Gauss-Legendre, adaptive Simpson, and 2-D tensor-product rules over rectangular
domains.

::: cds.numerical_integration

### Monte Carlo Methods

π estimation, generic Monte-Carlo integration and expectation, hit-or-miss
sampling, 1-D/2-D random walks, and Buffon's needle.

::: cds.montecarlo

### Differential Equations

IVP solvers: Euler, midpoint, RK4, adaptive RK45, implicit backward Euler,
Crank-Nicolson, and symplectic integrators (symplectic Euler, velocity Verlet,
trapezoid method). Scalar and system-of-ODEs variants.

::: cds.diffeq

### PDE Solvers

Finite-difference solvers on uniform 1-D grids for the two classic model
equations: the heat equation (`u_t = alpha * u_xx`, explicit time marching) and
the wave equation (`u_tt = c^2 * u_xx`).

::: cds.pde

### Signal Processing

A from-scratch Fourier toolkit (DFT/FFT/IFFT, 2-D FFT, convolution, power
spectrum, frequency-domain low-pass) plus a classical digital-filter *design*
suite: Butterworth low/high/band-pass/band-stop via the analog-prototype +
bilinear-transform recipe, applied in the time domain through a direct-form II
difference equation, STFT, and a robust moving-median denoiser.

::: cds.signals

### Wavelets

Haar discrete wavelet transform, inverse DWT, multi-level decomposition, and
wavelet denoising.

::: cds.wavelets

### Quantum Computing

Single- and multi-qubit state-vector simulation with O(1) shot sampling; Bell and
GHZ states, entanglement detection, and the standard gate set.

::: cds.quantum

---

## Scientific Domains

### Scientific Constants & Formulas

Curated physical constants and classical physics formulas: mechanics, waves,
relativity, and thermodynamics.

::: cds.scientific

### Genetics

DNA sequence analysis: GC content, k-mers, reverse complement, and
Needleman-Wunsch global alignment.

::: cds.genetics

### Fractals

Escape-time Mandelbrot and Julia sets, the Barnsley fern, and the Sierpinski
triangle.

::: cds.fractals

### Information Theory

Shannon entropy, cross-entropy, KL and Jensen-Shannon divergence, and mutual
information.

::: cds.infotheory

### Symbolic Modeling

An expression tree (`+`, `-`, `*`, `/`, `**`, `sin`, `cos`, `exp`, `log`, `sqrt`)
with symbolic differentiation, simplification, and LaTeX export; named `MathModel`
systems of equations with Jacobians; and numeric solvers (root finding, parameter
fitting) built on `cds.optimization`.

::: cds.modeling

### Graph Theory

BFS, DFS, Dijkstra shortest paths, Kruskal MST, topological sort, and cycle
detection.

::: cds.graph

---

## Scientific Assurance

### Uncertainty Propagation

`UncertainValue` plus analytic linear propagation and correlated Monte-Carlo
propagation.

::: cds.uncertainty

### Sensitivity & Identifiability

Local parameter sensitivity, variance-based global screening, and local
identifiability analysis — dependency-free.

::: cds.sensitivity

### Validation

Scientific validation checks, cross-method verification, data-adequacy
assessment, and drift / out-of-distribution reports.

::: cds.validation

### Causal Estimation

Conservative, assumption-gated effect estimators. This module does **not** infer
causal structure from observational correlation: callers must declare causal
assumptions explicitly and supply adjustment covariates themselves. Unmet
assumptions are reported rather than silently ignored.

::: cds.causal

### Units & Dimensional Analysis

Pure-Python SI quantities, unit conversion, and dimension-compatibility checks.

::: cds.units

---

## Orchestration & Reproducibility

### Workflow Orchestration

Approval-gated scientific workflow orchestration: analysis plans and requests,
gate policies and decisions, execution traces, and independent validation. The
orchestrator is fail-closed — blocked methods, missing tools, denied approvals,
failed validation, or unresolved method suitability all block an unqualified
conclusion.

::: cds.workflow

### Provenance

Reproducibility manifests, canonical SHA-256 hashing, git-SHA detection, decision
records, and local checkpoints.

::: cds.provenance

### Data Analysis

CSV loading, normalisation, z-scoring, moving average, and ASCII visualisation.

::: cds.data_analysis

### Pandas Interoperability (optional)

A lossless bridge between CDS's `DataTable`/`DataSet` and a pandas `DataFrame`:
`to_dataframe` / `from_dataframe`. Install with
`pip install "scientific-computing-system[pandas]"`; the core stays
zero-dependency.

::: cds.data_analysis.pandas_io

### Data I/O

Memory-bounded streaming file I/O, online first/second moments, and lazy optional
HDF5 and NetCDF backends. Install the backends with
`pip install "scientific-computing-system[io]"`.

::: cds.data_io

### Scientific Tool Adapters

Capability discovery and normalised adapters for optional NumPy, SciPy,
statsmodels, scikit-learn, SymPy, and Z3 backends. Dependencies are resolved
lazily, so the core stays importable without them.

::: cds.tools

### Plotting (optional)

Matplotlib helpers for series, multi-series, scatter, linear regression,
histograms, waveforms, spectra, ACF/PACF, seasonal decomposition, heatmaps, loss
curves, and optimizer trajectories. Install with
`pip install "scientific-computing-system[plot]"`. Matplotlib is imported lazily
on the first plot call.

::: cds.plot

---

## Educational NLP

From-scratch transformer primitives: BPE tokeniser, sinusoidal embeddings,
scaled dot-product and multi-head attention, a minimal autograd `Tensor` engine,
and MiniGPT. Educational only.

::: cds.nlp

---

## Command-Line Interface

The `cds` console script. Pure-stdlib `argparse`; no `typer`/`rich` dependency,
which is what keeps the core zero-dependency. Run `cds --help` for the live
command list; the commands are:

| Command | Purpose |
|---|---|
| `cds --version` | Print the installed version |
| `cds info` | Version, module status, and health summary |
| `cds modules` | List every scientific module available in the installation |
| `cds constants` | Table of available physical constants |
| `cds calc <formula>` | Quick physics calculation (`ke`, `gravity`, `wave`, `gas`) |
| `cds stats <numbers>` | Descriptive statistics for a comma-separated list |
| `cds sample <dist>` | Draw samples from a probability distribution |
| `cds integrate <fn>` | Numerically integrate a built-in function over `[a, b]` |
| `cds hypothesis <question>` | Generate scientific hypotheses for a question |
| `cds prompt` | Print a ready-to-use prompt for a custom generator |
| `cds benchmark` | Run the built-in benchmarks |
| `cds plot <numbers>` | ASCII plot, or PNG with `--file` when `[plot]` is installed |
| `cds dashboard` | Launch the Streamlit dashboard (needs `[dashboard]`) |

`cds modules` is the authoritative live catalog — it is generated from the
installed package, so it cannot drift from the code the way this table could.