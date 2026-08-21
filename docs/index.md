# Cognitive Discovery System (CDS)

Welcome to the official documentation for the **Cognitive Discovery System (CDS)**.

CDS is an open-source computational science platform designed for research, simulation, and discovery. It provides a lightweight, dependency-free environment for scientific exploration, featuring 19 modules covering everything from quantum simulation to symbolic math, knowledge organization, educational NLP primitives, and automated hypothesis generation.

<video controls loop muted autoplay playsinline poster="assets/cds_promo_poster.png" style="max-width:100%;border-radius:12px;border:1px solid rgba(148,163,184,.25)">
  <source src="assets/cds_promo.mp4" type="video/mp4">
  <source src="assets/cds_promo.webm" type="video/webm">
</video>

## Key Features

- **Pure Python:** Every module is implemented from scratch using the Python standard library. No heavy dependencies like NumPy or SciPy required.
- **Quantum Simulation:** Full state-vector simulation for single and multi-qubit circuits with entanglement and O(1) sampling.
- **Advanced Mathematics:** O(N³) Partial Pivoting LU decomposition, vectorized optimizers, and adaptive ODE solvers (RK45).
- **Hypothesis Engine:** Built-in tools for generating and statistically validating scientific hypotheses, complemented by effect-size measures (Cohen's d, Cramér's V) that quantify the magnitude of an effect alongside its significance.
- **High Reliability:** Comprehensive test suite with 100% code coverage (statement + branch) on the reference CI cell. See the CI and codecov badges in the [README](https://github.com/Furox88/cognitive-discovery-system) for the live test count and coverage.
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
| `cds.modeling` | Symbolic algebra — expressions, differentiation, simplification, LaTeX export, `MathModel` equation systems, root-finding & parameter fitting |
| `cds.knowledge` | Knowledge organization — concept graph with typed relations, research notes notebook, ranked structured retrieval (JSON persistence) |
| `cds.montecarlo` | Monte Carlo integration, π estimation, random walks |
| `cds.diffeq` | ODE solvers (Euler, RK4, midpoint) |
| `cds.numerical_integration` | Deterministic quadrature (trapezoid, Simpson, Romberg) + 2-D tensor-product rules (Simpson, Gauss-Legendre) |
| `cds.nlp` | Educational NLP from scratch (BPE, embeddings, attention, autograd, MiniGPT) |
| `cds.hypothesis` | Cognitive discovery and structured hypothesis generation |
| `cds.plot` | Optional matplotlib charts (series, scatter, regression, spectra, ACF, seasonal, heatmaps, …) via `cds[plot]` |

## Quick Navigation

- [Getting Started](getting-started.md)
- [API Reference](api.md)
- [Cookbook](cookbook.md) — problem-oriented recipes for every module
- [Tour of Numerical Methods](tour_of_numerical_methods.md) — guided walkthrough
- [Architecture](ARCHITECTURE.md) — module dependency graph & data flow
- [Case Studies](CASE_STUDY_HUBBLE.md)
- [Benchmarks](benchmarks.md)

---
*CDS v1.5.0 is stable and actively developed. Contributions are welcome!*
