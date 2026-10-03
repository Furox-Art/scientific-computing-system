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
  <a href="https://github.com/Furox-Art/scientific-computing-system/discussions">Discussions</a> ·
  <a href="https://github.com/Furox-Art/scientific-computing-system/issues">Issues</a>
</p>

Download counts: [PyPI stats](https://pypi.org/project/scientific-computing-system/)

I wrote this because NumPy and SciPy are incredible, but they're also 20 years old and carry two decades of design decisions that don't always make sense anymore.

This is a from-scratch rethinking of what scientific computing in Python could look like if we started today. No C extensions, no Fortran legacy, no dependency hell. Just Python, type hints, and algorithms that are actually readable.

This repository is the front door. The NumPy build is [scientific-computing-system-2.0](https://github.com/Furox-Art/scientific-computing-system-2.0), not a second project. Beside it: [axiomize](https://github.com/Furox-Art/axiomize) for units and Model IR, [quantum-reasoning-skill](https://github.com/Furox-Art/quantum-reasoning-skill) for the reasoning skill, and [plan-auditor](https://github.com/Furox-Art/plan-auditor) for a fail-closed agent check.

The import name is **`cds`** (the distribution is `scientific-computing-system`). The zero-dependency core needs nothing but the standard library.

## Install: two channels, two different things

| Channel | Command | What you get |
|---|---|---|
| **PyPI** (the library) | `pip install scientific-computing-system` | The actual Python package. Zero runtime dependencies. This is what you want. |
| **npm** (a launcher shim) | `npm i -g scientific-computing-system` | A thin Node wrapper that forwards to the `cds` CLI. **Ships no Python code** — it needs the PyPI package installed too. |

Both are published at the same version, and the npm `scs` launcher only works
once `pip install scientific-computing-system` has put `cds` on your PATH. If
you want the library, use PyPI; npm exists so Node-based tooling can invoke the
same CLI.

The two registries are not equally verifiable. See
[Supply-chain provenance](#supply-chain-provenance) before choosing a channel
for anything security-sensitive.

## What's inside

34 subpackages under `src/cds/`. Run `cds modules` for the live list.

| Area | Modules | What you get |
|---|---|---|
| Linear algebra & calculus | `cds.math_utils`, `cds.interpolate` | SVD, QR, Cholesky, LU, power iteration, derivatives, integrals |
| ODE / PDE | `cds.diffeq`, `cds.pde` | Euler, RK4, adaptive RK45, implicit stiff methods, symplectic integrators; 1-D heat & wave equations |
| Quadrature | `cds.numerical_integration` | Trapezoid, Simpson 1/3 & 3/8, Romberg, Gauss-Legendre, adaptive Simpson, 2-D tensor rules |
| Optimization | `cds.optimization` | Gradient descent, Newton, Adam, Nelder-Mead, simulated annealing, constrained search |
| Statistics | `cds.stats`, `cds.probability` | Descriptive stats, regression, t/chi-square/ANOVA, effect sizes, nonparametric ranks, time series (ACF/PACF, KPSS, Ljung-Box, decomposition), multiple testing |
| Bayesian | `cds.bayes` | Conjugate updates, credible intervals, Bayes factors |
| Monte Carlo | `cds.montecarlo` | π estimation, Monte-Carlo integration, random walks |
| Machine learning | `cds.ml` | MLP, k-NN, k-means, CART trees, random forest, boosting, PCA, scalers (educational, not production) |
| Signal processing | `cds.signals`, `cds.wavelets` | DFT/FFT/IFFT, convolution, Butterworth design, STFT, Haar DWT |
| Quantum | `cds.quantum` | Single & multi-qubit state-vector circuits, Bell/GHZ states, entanglement |
| Scientific domains | `cds.scientific`, `cds.genetics`, `cds.fractals`, `cds.infotheory` | Physical constants & formulas, DNA analysis, fractals, entropy/divergence |
| Symbolic modeling | `cds.modeling` | Expression trees, symbolic differentiation, LaTeX export, `MathModel` systems, root finding, fitting |
| Uncertainty & sensitivity | `cds.uncertainty`, `cds.sensitivity` | Analytic + correlated Monte-Carlo propagation, local & variance-based global sensitivity, identifiability |
| Validation & causality | `cds.validation`, `cds.causal` | Cross-method checks, drift/OOD reports, assumption-gated causal estimators |
| Units & dimensional analysis | `cds.units` | SI quantities, conversions, dimension compatibility checks |
| Workflow & provenance | `cds.workflow`, `cds.provenance` | Approval-gated orchestration, run manifests, hashes, checkpoints |
| Data | `cds.data_analysis`, `cds.data_io` | CSV/tabular analysis, streaming I/O, optional HDF5/NetCDF |
| Knowledge & hypothesis | `cds.knowledge`, `cds.hypothesis` | Concept graphs, research notes, structured hypothesis generation & evaluation |
| NLP (educational) | `cds.nlp` | BPE, embeddings, attention, autograd, MiniGPT |
| Plotting | `cds.plot` | Optional matplotlib helpers (`[plot]` extra) |
| Backend adapters | `cds.tools` | Lazy discovery for NumPy/SciPy/statsmodels/scikit-learn/SymPy/Z3 |

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

A CLI ships with the package:

```console
$ cds --version
System version 2.2.0
$ cds modules          # live module catalog
$ cds constants        # physical constants table
```

Full docs: [furox-art.github.io/scientific-computing-system](https://furox-art.github.io/scientific-computing-system/).

## Optional extras

The core is zero-dependency. Everything below is opt-in:

```bash
pip install "scientific-computing-system[scientific]"  # NumPy, SciPy, statsmodels, scikit-learn, SymPy, Z3
pip install "scientific-computing-system[io]"          # HDF5 + NetCDF
pip install "scientific-computing-system[plot]"        # matplotlib
pip install "scientific-computing-system[dashboard]"   # Streamlit dashboard
```

## Common use cases

- Learn and inspect **numerical methods in pure Python** without compiled extensions.
- Prototype **scientific computing** workflows with transparent implementations.
- Explore **ODE/PDE solvers**, numerical integration, optimization, Monte Carlo, signal processing, and linear algebra.
- Run **statistics, uncertainty quantification, sensitivity analysis, dimensional analysis, and reproducible research** workflows.
- Teach or audit algorithms where readable source code matters more than raw NumPy/SciPy performance.

## The catch

Pure Python is slower than NumPy on dense numerics. On this machine's benchmark run, 100×100 matrix multiplication took ~0.05 s versus ~0.0001 s for NumPy — roughly **580× slower**. That is the price of readable source. See [docs/benchmarks.md](https://github.com/Furox-Art/scientific-computing-system/blob/main/docs/benchmarks.md) for the full measurements, their provenance, and the hardware they came from.

The trade-off buys you something real: you can read every algorithm, follow every step, and change anything without compiling C.

I use it for prototyping, for teaching, and for cases where I need to know exactly what the computer is doing. For production number crunching, I still reach for NumPy — or the companion project [`scientific-computing-system-2.0`](https://github.com/Furox-Art/scientific-computing-system-2.0), which takes the opposite trade-off on NumPy/SciPy.

## Supply-chain provenance

The two registries are **not** equally verifiable, and the difference matters if
you are installing this in anything security-sensitive.

**PyPI 2.2.0 carries a real attestation.** The wheel and sdist were published
through Trusted Publishing (OIDC) from the release workflow, and PyPI serves a
PEP 740 provenance bundle for both files. You can check it yourself against the
SHA-256 digests PyPI shows for each artifact.

**npm 2.2.0 carries no attestation.** That release went out in *token mode*: the
workflow publishes with `--provenance` on the OIDC path, but a long-lived
registry token cannot mint a Sigstore identity, so the token fallback
deliberately omits the flag and prints a notice saying so. Consequently:

- `npm view scientific-computing-system` shows a legacy registry signature
  (`dist.signatures`) — that is npm's own transport signing, **not** a build
  provenance attestation.
- There is no attestation bundle for it; the npm attestations endpoint returns
  404 for this package@version.
- The publish workflow supports OIDC trusted publishing with `--provenance`, and
  uses it by default. It is not in effect for the currently published npm
  2.2.0, because that dispatch ran with `use_token_fallback=true`.

If you need attested provenance today, use the PyPI channel. The npm channel is
a convenience shim, not a supply-chain-trust path. Full policy, threat model,
and reporting instructions are in [SECURITY.md](SECURITY.md).

## License

MIT — see [LICENSE](LICENSE).

If you use this in published work, please cite it via [`CITATION.cff`](CITATION.cff).
