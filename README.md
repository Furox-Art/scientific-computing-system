<p align="center">
  <img src="assets/logo.svg" alt="scientific-computing-system" width="640">
</p>

<h1 align="center">scientific-computing-system</h1>

<p align="center"><b>A pure-Python computational science platform for numerical methods, modeling, validation, uncertainty, scientific workflows, dimensional analysis, and reproducible research.</b></p>

<p align="center">
  <a href="https://pypi.org/project/scientific-computing-system/"><img src="https://img.shields.io/pypi/v/scientific-computing-system.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/scientific-computing-system/"><img src="https://img.shields.io/pypi/dm/scientific-computing-system.svg" alt="PyPI downloads"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-green.svg" alt="Python 3.10+"></a>
  <a href="https://codecov.io/gh/Furox-Art/scientific-computing-system"><img src="https://codecov.io/gh/Furox-Art/scientific-computing-system/branch/main/graph/badge.svg" alt="codecov"></a>
  <a href="https://github.com/Furox-Art/scientific-computing-system/actions/workflows/tests.yml"><img src="https://github.com/Furox-Art/scientific-computing-system/actions/workflows/tests.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://furox-art.github.io/scientific-computing-system/"><img src="https://img.shields.io/badge/docs-mkdocs-teal.svg" alt="Docs"></a>
  <a href="https://github.com/Furox-Art/scientific-computing-system/releases"><img src="https://img.shields.io/github/v/release/Furox-Art/scientific-computing-system.svg" alt="GitHub release"></a>
</p>

<p align="center">
  <a href="https://furox-art.github.io/scientific-computing-system/">Documentation</a> ·
  <a href="https://github.com/Furox-Art/scientific-computing-system/releases">Releases</a> ·
  <a href="docs/tutorials/">Tutorials</a> ·
  <a href="examples/">Examples</a> ·
  <a href="SECURITY.md">Security</a>
</p>

---

## Current system

**Current package line: 2.0.0.**

CDS keeps a **zero-runtime-dependency pure-Python core** while extending the original readable, from-scratch numerical platform into a broader scientific workflow system. The current codebase combines computational methods with explicit validation, uncertainty propagation, sensitivity analysis, dimensional analysis, provenance, approval-gated workflow orchestration, scalable local data I/O, and optional adapters to established scientific libraries.

The core is intended to remain inspectable: algorithms can be read, tested, modified, and used without requiring NumPy, SciPy, BLAS, compiled extensions, or a separate runtime stack. Optional integrations are loaded only when requested.

```bash
pip install scientific-computing-system
cds info
cds modules
```

## What 2.0 adds

The 2.0 codebase is not only a collection of numerical algorithms. It now has explicit layers for scientific assurance and reproducibility:

- **Scientific validation** — structured checks, cross-method verification, and final-audit support.
- **Uncertainty** — analytic and correlated Monte Carlo uncertainty propagation.
- **Sensitivity analysis** — dependency-free local parameter sensitivity analysis.
- **Units and dimensional analysis** — SI units, conversions, and dimension-aware checks.
- **Workflow orchestration** — approval-gated scientific workflows rather than silent expensive or consequential execution.
- **Provenance** — run manifests, hashes, tool versions, decisions, and checkpoints for reproducibility.
- **Research-data I/O** — memory-bounded streaming plus optional HDF5 and NetCDF backends.
- **Scientific tool adapters** — lazy capability discovery and normalized optional adapters for NumPy/SciPy, statsmodels, scikit-learn, SymPy, and Z3.
- **Modeling and fitting** — symbolic models, equation solving, numerical fitting, diagnostics, and validation paths.
- **Structured hypothesis generation** — falsifiable scientific hypotheses that can be connected to the rest of the computational stack.

## Architecture

The current CLI groups the system into five layers:

```text
compute       quantum / signals / math / ODE-PDE / integration
analysis      stats / probability / ML / modeling / sensitivity
assurance     validation / uncertainty / units / provenance
orchestration workflow / optional scientific tools
data          data_analysis / streaming I/O / knowledge
```

This separation is deliberate: numerical execution, scientific interpretation, verification, orchestration, and data handling are related but not treated as the same concern.

## Module map

| Area | Modules | Main capabilities |
|---|---|---|
| Quantum | `cds.quantum` | Single- and multi-qubit circuits, Bell/GHZ states, measurement and entanglement utilities |
| Signals | `cds.signals`, `cds.wavelets` | DFT/FFT, convolution, filters, spectral tools, STFT and wavelet operations |
| Mathematics | `cds.math_utils`, `cds.numerical_integration` | Linear algebra, decompositions, calculus, quadrature and numerical utilities |
| Dynamics | `cds.diffeq` | Explicit, adaptive, stiff and symplectic ODE methods plus PDE-related utilities |
| Optimization | `cds.optimization` | Gradient/Newton/Adam methods, Nelder-Mead, annealing and constrained search |
| Statistics | `cds.stats`, `cds.probability`, `cds.bayes` | Inference, regression, statistical tests, distributions, sampling and Bayesian utilities |
| Monte Carlo | `cds.montecarlo` | Monte Carlo integration, simulation, random walks and MCMC utilities |
| Modeling | `cds.modeling` | Symbolic models, equation solving, parameter fitting and model-oriented workflows |
| Machine learning | `cds.ml` | Classical estimators, preprocessing, validation, PCA and readable ML implementations |
| Scientific domains | `cds.scientific`, `cds.genetics`, `cds.fractals`, `cds.infotheory` | Physical constants/formulas, genetics helpers, fractals and information-theory utilities |
| Data | `cds.data_analysis`, `cds.data_io` | Tabular analysis, normalization, visualization helpers, streaming and optional HDF5/NetCDF I/O |
| Units | `cds.units` | SI quantities, conversions and dimensional analysis |
| Uncertainty | `cds.uncertainty` | Analytic and correlated Monte Carlo uncertainty propagation |
| Sensitivity | `cds.sensitivity` | Local parameter sensitivity analysis without mandatory external dependencies |
| Validation | `cds.validation` | Scientific checks, cross-method verification and final audit |
| Workflow | `cds.workflow` | Approval-gated scientific workflow orchestration |
| Provenance | `cds.provenance` | Run manifests, hashes, tool versions, decisions and checkpoints |
| Tools | `cds.tools` | Lazy optional scientific backends and normalized SciPy/SymPy/Z3-style adapters |
| Knowledge | `cds.knowledge` | Knowledge graphs, concept mapping, notes and retrieval |
| Hypotheses | `cds.hypothesis` | Structured, falsifiable scientific hypothesis generation |
| NLP | `cds.nlp` | Educational tokenizer, embeddings, attention, autograd and MiniGPT components |
| Graphs | `cds.graph` | Traversal, shortest paths, spanning trees and topological operations |
| Plotting | `cds.plot` | Optional matplotlib-based scientific plots |

Run the installed package for the authoritative live module list:

```bash
cds modules
```

## Installation profiles

### Pure-Python core

```bash
pip install scientific-computing-system
```

The base package has **zero runtime dependencies**.

### Scientific reference backends

```bash
pip install "scientific-computing-system[scientific]"
```

This enables optional NumPy, SciPy, statsmodels, scikit-learn, SymPy, and Z3-backed tooling used through the capability/adaptor layer.

### Research data I/O

```bash
pip install "scientific-computing-system[io]"
```

Adds optional HDF5 and NetCDF support while keeping core streaming/CSV functionality dependency-free.

### Plotting

```bash
pip install "scientific-computing-system[plot]"
```

### Dashboard

```bash
pip install "scientific-computing-system[dashboard]"
cds dashboard
```

### Everything

```bash
pip install "scientific-computing-system[all]"
```

## Quick examples

### Numerical computing

```python
from cds.stats import linear_regression
from cds.signals import fft_radix2
from cds.quantum import bell_state, is_entangled

fit = linear_regression([1, 2, 3], [2.1, 3.9, 6.2])
spectrum = fft_radix2([complex(i) for i in range(8)])
state = bell_state(0)

print(fit.slope)
print(spectrum)
print(is_entangled(state))
```

### Mathematical modeling

```python
from cds.modeling import Variable, solve_equation

x = Variable("x")
result = solve_equation(x**2 - 2, variable="x", x0=1.0)
print(result.x)
```

### Scientific constants

```python
from cds.scientific import get_constant, kinetic_energy

print(get_constant("c"))
print(kinetic_energy(10, 5))
```

## Scientific assurance

CDS 2.0 separates a computed result from the evidence supporting that result. Depending on the workflow, the package can combine:

1. explicit inputs and assumptions;
2. unit/dimensional checks;
3. numerical or statistical computation;
4. uncertainty propagation;
5. parameter sensitivity analysis;
6. independent or cross-method validation;
7. provenance capture;
8. a final workflow/audit decision.

The goal is not to claim that every result is automatically correct. The goal is to make the path from input to result **inspectable, reproducible, and falsifiable**.

## Optional scientific backends

The core package does not silently require external numerical libraries. Optional backends exist for cases where an independent reference implementation or specialized solver is useful.

The current optional scientific extra includes:

- NumPy
- SciPy
- statsmodels
- scikit-learn
- SymPy
- Z3

Availability is discovered lazily through `cds.tools`; the dependency-free core remains usable when those packages are absent.

## Why a pure-Python core?

CDS makes a different trade-off from high-performance scientific libraries. Its priority is readability, portability, inspectability, and cross-domain composition.

Use CDS when you want to:

- inspect an algorithm rather than treat it as a compiled black box;
- teach or learn numerical/scientific methods from readable source;
- prototype across multiple scientific domains under one package;
- build reproducible local workflows with explicit validation and provenance;
- run a lightweight scientific stack where binary dependencies are undesirable;
- compare a pure-Python implementation with optional established scientific backends.

For very large array workloads, production HPC, GPU-heavy computation, or specialized high-performance solvers, use the appropriate NumPy/SciPy/JAX/PyTorch/domain-specific stack directly or through an optional integration where appropriate.

## Development and validation

The repository uses automated testing, strict typing and CI across supported Python versions/platforms. The live CI badges and workflow files are the authoritative source for current test counts and status; the README intentionally avoids hard-coding test totals that quickly become stale.

```bash
git clone https://github.com/Furox-Art/scientific-computing-system.git
cd scientific-computing-system
python -m venv .venv
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
```

## Documentation and examples

- [`examples/`](examples/) — runnable scientific examples
- [`docs/`](docs/) — documentation, tutorials, benchmarks and research workflows
- [`CHANGELOG.md`](CHANGELOG.md) — historical release record
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — contribution workflow
- [`SECURITY.md`](SECURITY.md) — security policy and trust boundaries
- [`CITATION.cff`](CITATION.cff) — citation metadata

## Relationship to scientific-computing-system-2.0

This repository is the **pure-Python, zero-runtime-dependency CDS line**. The separate [`scientific-computing-system-2.0`](https://github.com/Furox-Art/scientific-computing-system-2.0) project deliberately makes the opposite performance trade-off: it builds on NumPy/SciPy/pandas/matplotlib and adds compiled/GPU acceleration and a broader production-oriented scientific Python stack.

## License

MIT — see [`LICENSE`](LICENSE).

## Contact

Maintainer: [@Furox-Art](https://github.com/Furox-Art)

For bugs and feature requests, use the repository issue tracker. Security vulnerabilities should be reported through the process in [`SECURITY.md`](SECURITY.md), not as public issues.
