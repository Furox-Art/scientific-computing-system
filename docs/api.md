# API Reference

Auto-generated reference for every public CDS module. Each entry below is rendered from the module's own docstrings by mkdocstrings.

## Core Data Models

The shared `Domain`, `Hypothesis`, and `HypothesisStatus` types used throughout CDS, the foundation the hypothesis engine builds on.

::: cds.core

## Hypothesis Generation

The cognitive-discovery centrepiece: structured hypothesis generation from a research question, plus a statistical evaluator.

::: cds.hypothesis

## Statistics

Descriptive statistics, regression, and frequentist hypothesis tests (t-test, chi-square, ANOVA) with companion effect-size measures (Cohen's d, eta-squared, Cramer's V).

::: cds.stats

## Probability

Continuous PDFs (Gaussian, uniform, exponential) and discrete PMFs (binomial, Poisson) with reproducible sampling.

::: cds.probability

## Mathematical Utilities

Calculus (derivative, integral, gradient) and a compact linear-algebra toolkit (PLU, QR, Cholesky, eigenvalues via power iteration).

::: cds.math_utils

## Numerical Integration

Deterministic quadrature rules: trapezoid, Simpson 1/3 and 3/8, Gauss–Legendre, Romberg, adaptive Simpson, and 2-D tensor-product rules (composite Simpson 2-D and tensor-product Gauss–Legendre) over rectangular domains.

::: cds.numerical_integration

## Differential Equations

Initial-value-problem solvers: Euler, midpoint, RK4, adaptive RK45, and a system-of-ODEs integrator.

::: cds.diffeq

## Monte Carlo Methods

Stochastic integration: π estimation, generic Monte-Carlo integration, 1D/2D random walks, and Buffon's needle.

::: cds.montecarlo

## Optimization

Gradient descent, Newton's method, Adam, and golden-section line search.

::: cds.optimization

## Machine Learning

Pure-Python neural networks: an MLP with Adam-based training.

::: cds.ml

## Signal Processing

A from-scratch Fourier toolkit (DFT/FFT/IFFT, convolution, power spectrum,
frequency-domain low-pass) plus a classical digital-filter *design* suite:
Butterworth low/high/band-pass/band-stop via the analog-prototype + bilinear-
transform recipe, applied in the time domain through a direct-form II
difference equation, and a robust moving-median denoiser.

::: cds.signals

## Quantum Computing

Single- and multi-qubit state-vector simulation with O(1) sampling.

::: cds.quantum

## Scientific Computing

Curated physical constants and classical physics formulas (mechanics, waves, relativity, thermo).

::: cds.scientific

## Graph Theory

BFS, DFS, Dijkstra shortest paths, Kruskal MST, topological sort, cycle detection.

::: cds.graph

## Mathematical Modeling

Symbolic algebra for equation development: an expression tree (`+`, `-`, `*`, `/`, `**`, `sin`, `cos`, `exp`, `log`, `sqrt`) with symbolic differentiation, simplification, LaTeX export, named `MathModel` systems of equations, and numeric solvers (root finding and parameter fitting) built on `cds.optimization`.

::: cds.modeling

## Knowledge Organization

A knowledge graph of named concepts connected by typed, directed relations (`is-a`, `depends-on`, …) with undirected traversal (shortest path, transitive closure, cycle detection), a notebook of research notes linked to concept names, and ranked structured retrieval across both. Persistence is JSON via the stdlib.

::: cds.knowledge

## Data Analysis

CSV loading, normalisation, smoothing, and ASCII visualisation.

::: cds.data_analysis

### Pandas Interoperability (optional)

A lossless bridge between CDS's [`DataSet`](#data-analysis) and a pandas
`DataFrame`: `to_dataframe` / `from_dataframe`. Install with
`pip install ".[pandas]"`; the core package stays zero-dependency.

::: cds.data_analysis.pandas_io

## Plotting (optional)

Matplotlib helpers for series, multi-series, scatter, linear regression,
histograms, waveforms, spectra / power spectra, ACF/PACF, seasonal
decomposition, heatmaps, loss curves, and optimizer trajectories. Install with
`pip install ".[plot]"`; the core package stays zero-dependency. Matplotlib is
imported lazily on first plot call.

::: cds.plot

## Educational NLP

From-scratch transformer primitives: BPE tokeniser, sinusoidal embeddings, attention, autograd, MiniGPT.

::: cds.nlp
