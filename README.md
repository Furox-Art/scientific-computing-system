<p align="center">
  <img src="assets/logo.svg" alt="scientific-computing-system" width="640">
</p>

<h1 align="center">scientific-computing-system</h1>

<p align="center"><b>A pure-Python computational science platform for numerical methods, modeling, validation, uncertainty, scientific workflows, dimensional analysis, and reproducible research.</b></p>

<p align="center">
  <a href="https://pypi.org/project/scientific-computing-system/"><img src="https://img.shields.io/pypi/v/scientific-computing-system.svg" alt="PyPI version"></a>
  <a href="https://www.npmjs.com/package/scientific-computing-system"><img src="https://img.shields.io/npm/v/scientific-computing-system.svg" alt="npm version"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-green.svg" alt="Python 3.10+"></a>
  <a href="https://codecov.io/gh/Furox-Art/scientific-computing-system"><img src="https://codecov.io/gh/Furox-Art/scientific-computing-system/branch/main/graph/badge.svg" alt="codecov"></a>
  <a href="https://github.com/Furox-Art/scientific-computing-system/actions/workflows/tests.yml"><img src="https://github.com/Furox-Art/scientific-computing-system/actions/workflows/tests.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://furox-art.github.io/scientific-computing-system/"><img src="https://img.shields.io/badge/docs-mkdocs-teal.svg" alt="Docs"></a>
  <a href="https://github.com/Furox-Art/scientific-computing-system/releases"><img src="https://img.shields.io/github/v/release/Furox-Art/scientific-computing-system.svg" alt="GitHub release"></a>
</p>

<p align="center">
  <b>Docs</b> ·
  <a href="https://furox-art.github.io/scientific-computing-system/getting-started/">Getting Started</a> ·
  <a href="https://furox-art.github.io/scientific-computing-system/api/">API Reference</a> ·
  <a href="https://furox-art.github.io/scientific-computing-system/cookbook/">Cookbook</a> ·
  <a href="https://furox-art.github.io/scientific-computing-system/benchmarks/">Benchmarks</a> ·
  <b>Project</b> ·
  <a href="https://github.com/Furox-Art/scientific-computing-system/blob/main/CHANGELOG.md">Changelog</a> ·
  <a href="https://github.com/Furox-Art/scientific-computing-system/blob/main/CONTRIBUTING.md">Contributing</a> ·
  <a href="https://github.com/Furox-Art/scientific-computing-system/blob/main/SECURITY.md">Security</a> ·
  <a href="https://github.com/Furox-Art/scientific-computing-system/blob/main/CODE_OF_CONDUCT.md">Code of Conduct</a> ·
  <a href="https://github.com/Furox-Art/scientific-computing-system/blob/main/CITATION.cff">Citation</a> ·
  <a href="https://github.com/Furox-Art/scientific-computing-system/blob/main/paper.md">Paper (JOSS)</a> ·
  <a href="https://github.com/Furox-Art/scientific-computing-system/discussions">Discussions</a> ·
  <a href="https://github.com/Furox-Art/scientific-computing-system/issues">Issues</a>
</p>

I wrote this because NumPy and SciPy are incredible, but they're also 20 years old and carry two decades of design decisions that don't always make sense anymore.

Download counts: [PyPI stats](https://pypi.org/project/scientific-computing-system/)

This is a from-scratch rethinking of what scientific computing in Python could look like if we started today. No C extensions, no Fortran legacy, no dependency hell. Just Python, type hints, and algorithms that are actually readable.

The import name is **`cds`** (the distribution is `scientific-computing-system`). The zero-dependency core needs nothing but the standard library. The sibling NumPy build of this work is [`scientific-computing-system-2.0`](https://github.com/Furox-Art/scientific-computing-system-2.0); related projects are listed in [docs/related.md](docs/related.md).

## Install

| Channel | Command | What you get |
|---|---|---|
| **PyPI** (the library) | `pip install scientific-computing-system` | The actual Python package. Zero runtime dependencies. This is what you want. |
| **npm** (a launcher shim) | `npm i -g scientific-computing-system` | A thin Node wrapper that runs `python -m cds`. It ships no Python and needs the Python distribution installed so that `python -m cds` resolves. |

Both channels are published at the same version. They are **not** equally
verifiable:

- **PyPI** — carries a **PEP 740 provenance attestation** (Sigstore-signed,
  binding the file digest to the publishing workflow). Verified on 2.2.3, and
  on every release since attestations were enabled.
- **npm** — carries a **Sigstore attestation as of 2.2.3**, minted by npm's
  OIDC trusted publisher (the registry serves both an npm publish attestation
  and a SLSA provenance v1 bundle for that tarball). 1.0.0 through 2.2.2
  predate the publisher registration and have none, so pin 2.2.3 or newer. npm
  also exposes `dist.integrity` (sha512), which is a content pin, not proof of
  origin, and `dist.signatures`, which is npm's registry transport signature,
  not build provenance.

Prefer PyPI if you need an attestation for a version older than 2.2.3.
[SECURITY.md](SECURITY.md#distribution-channels-and-provenance) has the
per-channel table, the published digests to pin, the commands to verify them,
and what to register to close the npm gap.

## What's inside

`src/cds/` holds **34 subpackages plus `causal.py` and `sensitivity.py`**. That is
35 public feature modules — the `cds` CLI entry point is the 36th subpackage —
exporting **505 names** (342 functions, 163 classes), all pure standard library.
`tests/test_readme_surface.py` derives these numbers from the package, so they
cannot drift silently.

| Area | Modules | What you get |
|---|---|---|
| Core models | `cds.core` | Shared `Domain`, `Hypothesis`, `HypothesisStatus` types |
| Linear algebra & calculus | `cds.math_utils`, `cds.interpolate` | SVD, QR, Cholesky, LU, power iteration, derivatives, integrals |
| ODE / PDE | `cds.diffeq`, `cds.pde` | Euler, RK4, adaptive RK45, implicit stiff methods, symplectic integrators; 1-D heat & wave equations |
| Quadrature | `cds.numerical_integration` | Trapezoid, Simpson 1/3 & 3/8, Romberg, Gauss-Legendre, adaptive Simpson, 2-D tensor rules |
| Optimization | `cds.optimization` | Gradient descent, Newton, Adam, Nelder-Mead, annealing, constrained search |
| Statistics | `cds.stats`, `cds.probability`, `cds.bayes` | Descriptive stats, regression, t/chi-square/ANOVA, effect sizes, nonparametric ranks, time series, multiple testing, Bayesian conjugate updates |
| Monte Carlo | `cds.montecarlo` | π estimation, Monte-Carlo integration, random walks |
| Machine learning | `cds.ml` | MLP, k-NN, k-means, CART, random forest, boosting, PCA (educational, not production) |
| Signal processing | `cds.signals`, `cds.wavelets` | DFT/FFT/IFFT, convolution, Butterworth design, STFT, Haar DWT |
| Quantum | `cds.quantum` | Single & multi-qubit state-vector circuits, Bell/GHZ states, entanglement |
| Graphs | `cds.graph` | BFS, DFS, Dijkstra shortest paths, Kruskal MST, topological sort, cycle detection |
| Scientific domains | `cds.scientific`, `cds.genetics`, `cds.fractals`, `cds.infotheory` | Physical constants & formulas, DNA analysis & alignment, fractal sets, entropy/divergence/mutual information |
| Symbolic modeling | `cds.modeling` | Expression trees, symbolic differentiation, LaTeX export, `MathModel` systems, root finding, fitting |
| Uncertainty & sensitivity | `cds.uncertainty`, `cds.sensitivity` | Analytic + correlated Monte-Carlo propagation, local & variance-based global sensitivity, identifiability |
| Validation & causality | `cds.validation`, `cds.causal` | Cross-method checks, drift/OOD reports, assumption-gated causal estimators |
| Units | `cds.units` | SI quantities, conversions, dimensional compatibility checks |
| Workflow & provenance | `cds.workflow`, `cds.provenance` | Approval-gated orchestration, run manifests, hashes, checkpoints |
| Data | `cds.data_analysis`, `cds.data_io` | CSV/tabular analysis, streaming I/O, optional HDF5/NetCDF |
| Knowledge & hypothesis | `cds.knowledge`, `cds.hypothesis` | Concept graphs, research notes, structured hypothesis generation & evaluation |
| NLP (educational) | `cds.nlp` | BPE, embeddings, attention, autograd, MiniGPT |
| Plotting | `cds.plot` | Optional matplotlib helpers (`[plot]` extra) |
| Backend adapters | `cds.tools` | Lazy discovery for NumPy/SciPy/statsmodels/scikit-learn/SymPy/Z3 |

The authoritative per-module list is the [API reference](https://furox-art.github.io/scientific-computing-system/api/).

!!! note "`cds modules` is a curated subset"
    `cds modules` prints 26 entries. It is **not** the full catalog: it omits
    `bayes`, `causal`, `core`, `fractals`, `genetics`, `infotheory`, `interpolate`,
    `pde`, and `wavelets`. Use the [API reference](https://furox-art.github.io/scientific-computing-system/api/)
    or `src/cds/` for the complete set.

### Runnable examples and the dashboard

- **[`examples/`](examples/)** — 35 runnable scripts and notebooks (33 `.py`, 2
  `.ipynb`), each self-contained and dependency-free unless noted. Every tutorial
  page in the docs has a matching example.
- **[`dashboard/app.py`](dashboard/app.py)** — a Streamlit dashboard:
  `cds dashboard`, or `pip install "scientific-computing-system[dashboard]"`.

## Quick Start

```bash
pip install scientific-computing-system
```

```python
from cds.math_utils import svd
from cds.diffeq import solve_system
from cds.stats import linear_regression

# 1. SVD — result is an SVDResult, not a tuple
result = svd([[1.0, 2.0], [3.0, 4.0]])
print("singular_values =", [round(v, 6) for v in result.singular_values])


# 2. Damped harmonic oscillator: y'' = -y - 0.1 y'
def rhs(t, y):
    return [y[1], -y[0] - 0.1 * y[1]]


t, y = solve_system(rhs, 0.0, [1.0, 0.0], 50.0, dt=0.01)
print("final t =", round(t[-1], 2), "| final y =", [round(v, 8) for v in y[-1]])

# 3. Least-squares line fit
fit = linear_regression([1.0, 2.0, 3.0, 4.0, 5.0], [2.1, 3.9, 6.2, 7.8, 10.1])
print(f"slope = {fit.slope:.4f} | intercept = {fit.intercept:.4f} | r^2 = {fit.r_squared:.4f}")
```

Real output (executed verbatim; pasted from an actual run):

```text
singular_values = [5.464986, 0.365966]
final t = 50.0 | final y = [0.07638443, 0.0264785]
slope = 1.9900 | intercept = 0.0500 | r^2 = 0.9973
```

## CLI

The package installs a `cds` command with 13 subcommands:

| Command | Purpose |
|---|---|
| `cds --help` | Full command list |
| `cds modules` | Curated module catalog (see the caveat above) |
| `cds info` | Version, module status, health summary |
| `cds constants` | Table of physical constants |
| `cds calc <formula>` | Quick physics calculation (`ke`, `gravity`, `wave`, `gas`) |
| `cds stats <numbers>` | Descriptive statistics for a comma-separated list |
| `cds sample <dist>` | Draw samples from a probability distribution |
| `cds integrate <fn>` | Integrate a built-in function over `[a, b]` |
| `cds hypothesis <q>` | Generate scientific hypotheses for a question |
| `cds prompt` | Prompt text for a custom hypothesis generator |
| `cds benchmark` | Run the built-in benchmarks |
| `cds plot <numbers>` | ASCII plot, or PNG with `--file` when `[plot]` is installed |
| `cds dashboard` | Launch the Streamlit dashboard (needs `[dashboard]`) |

```console
$ cds --version
System version 2.2.3
$ cds constants        # physical constants table
```

## Optional extras

The core is zero-dependency. Everything below is opt-in:

```bash
pip install "scientific-computing-system[scientific]"  # NumPy, SciPy, statsmodels, scikit-learn, SymPy, Z3
pip install "scientific-computing-system[io]"          # HDF5 + NetCDF
pip install "scientific-computing-system[plot]"        # matplotlib
pip install "scientific-computing-system[pandas]"      # DataFrame bridge: cds.data_analysis.pandas_io
pip install "scientific-computing-system[dashboard]"   # Streamlit dashboard
```

## The catch

Pure Python is slower than NumPy on dense numerics: the committed benchmark
artifact records 100×100 matrix multiplication at 0.0696 s against NumPy's
0.000060 s, about **1155× slower**. Those numbers are from one CI run whose
platform, CPU, and library versions were not recorded, so treat them as
order-of-magnitude only — [docs/benchmarks.md](docs/benchmarks.md) states exactly
what that artifact does and does not support, and
[docs/why-pure-python.md](docs/why-pure-python.md) covers when to reach for
NumPy or the 2.0 build instead.

## Contributing

```bash
pip install -e ".[dev]"
pytest            # CI also enforces 100% blended coverage (statement + branch)
mkdocs serve      # docs at http://127.0.0.1:8000/
```

Issues and PRs welcome; see [CONTRIBUTING.md](CONTRIBUTING.md) and
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

MIT — see [LICENSE](LICENSE).

Cite via [`CITATION.cff`](CITATION.cff). [`codemeta.json`](codemeta.json) carries
the same metadata in codemeta form, and [`paper.md`](paper.md) /
[`paper.bib`](paper.bib) are the JOSS manuscript.
