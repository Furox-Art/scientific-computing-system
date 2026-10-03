# Getting Started

## Installation

Most people just want the package from PyPI:

```bash
pip install scientific-computing-system
```

That gives you the `cds` import and the `cds` command, with **zero runtime
dependencies** — the core is pure standard library. Optional extras are listed
in the [README](https://github.com/Furox-Art/scientific-computing-system#optional-extras).

Install from source when you intend to contribute:

```bash
git clone https://github.com/Furox-Art/scientific-computing-system.git
cd scientific-computing-system
python -m venv .venv
source .venv/bin/activate   # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Requires Python 3.10 or newer.

### The npm channel (Node launcher only)

There is also an npm package of the same name. It is a **thin Node wrapper**, not
the library: it contains no Python and simply forwards to the `cds` CLI.

```bash
pip install scientific-computing-system   # required — provides `cds`
npm i -g scientific-computing-system      # optional — provides the `scs` shim
```

Without the PyPI install the `scs` command exits non-zero and tells you so.
For anything security-sensitive prefer the PyPI channel: the two registries carry
different provenance guarantees, detailed in
[SECURITY.md](https://github.com/Furox-Art/scientific-computing-system/blob/main/SECURITY.md#distribution-channels-and-provenance).

!!! note "Two different things named `scs`"
    The Python **import** name is `cds`, never `scs` — `from scs... import`
    fails, and an unrelated PyPI distribution also uses that name. Separately,
    `scs` **is** the name of the npm launcher's command, so `scs --help` works
    once both channels are installed.

## Quick Usage

### CLI

```bash
# Show available commands
cds --help

# List physical constants
cds constants

# Interactive physics calculator — prompts for each variable, so it needs a TTY
cds calc ke        # kinetic energy: asks for mass and velocity
cds calc gravity   # asks for both masses and the separation
cds calc wave
cds calc gas

# Generate hypotheses
cds hypothesis "what causes the Hubble tension?" --domain cosmology

# Live module catalog — the authoritative list of what is installed
cds modules
```

### Python API

```python
# Quantum simulation
from cds.quantum import bell_state, is_entangled, ghz_state

reg = bell_state(0)
print(is_entangled(reg))  # True

# Optimization
from cds.optimization import gradient_descent

result = gradient_descent(lambda x: (x - 3) ** 2, x0=10.0, lr=0.1)
print(result.x)  # ~3.0

# Signal processing
from cds.signals import fft_radix2

spectrum = fft_radix2([complex(i) for i in range(8)])

# Probability
from cds.probability import gaussian_pdf, binomial_pmf

print(gaussian_pdf(0.0))  # 0.3989...
print(binomial_pmf(3, 5, 0.5))  # 0.3125

# Statistics
from cds.stats import mean, linear_regression

print(mean([1, 2, 3, 4, 5]))  # 3.0

# Effect-size measures (companion to the significance tests above)
from cds.stats import cohens_d, cramers_v

print(
    cohens_d([20.0, 22.0, 19.0], [28.0, 31.0, 26.0])
)  # -3.843076, signed: negative = first group lower
print(
    cramers_v([[10.0, 20.0], [30.0, 40.0]])
)  # 0.089087, association strength for a contingency table

# Scientific computing
from cds.scientific import kinetic_energy, get_constant

print(get_constant("c"))  # 299792458.0
print(kinetic_energy(10, 5))  # 125.0

# Graph theory
from cds.graph import Graph, dijkstra

g = Graph(n_vertices=3, directed=False)
g.add_edge(0, 1, 1.0)
g.add_edge(1, 2, 2.0)
dist, _ = dijkstra(g, 0)
print(dist)  # {0: 0.0, 1: 1.0, 2: 3.0}

# Monte Carlo
from cds.montecarlo import estimate_pi

result = estimate_pi(n_samples=50_000, seed=42)
print(f"π ≈ {result.estimate:.4f}")

# Differential equations
from cds.diffeq import rk4
import math

sol = rk4(lambda t, y: -y, 0, 1.0, 1.0)
print(f"e^-1 ≈ {sol.y[-1]:.6f}")  # 0.367879

# Linear algebra
from cds.math_utils import solve_linear, power_iteration

x = solve_linear([[2, 1], [4, 3]], [5, 11])
print(x)  # [2.0, 1.0]
```

## Running Tests

```bash
pytest           # run the full suite (see the CI badge for current status)
pytest -v        # verbose output
pytest -x        # stop on first failure
```

## Running Examples

Every script in `examples/` is standalone and runnable. This is the full list:

```bash
# Core models, data & knowledge
python examples/core_demo.py
python examples/data_analysis_demo.py
python examples/graph_demo.py
python examples/knowledge_demo.py
python examples/modeling_demo.py

# Math & numerics
python examples/math_utils_demo.py
python examples/linalg_demo.py
python examples/numerical_integration_demo.py
python examples/diffeq_demo.py
python examples/pde_demo.py
python examples/montecarlo_demo.py
python examples/probability_demo.py
python examples/optimization_demo.py
python examples/interpolate_demo.py
python examples/scientific_demo.py

# Signals & ML
python examples/signals_demo.py
python examples/fft2_demo.py
python examples/ml_and_viz_demo.py
python examples/ml_advanced_demo.py
python examples/pca_demo.py
python examples/ensemble_showcase.py

# NLP (educational)
python examples/nlp_bpe_demo.py
python examples/nlp_attention_demo.py
python examples/nlp_mini_gpt_demo.py
python examples/nlp_viz_demo.py

# Quantum
python examples/quantum_demo.py

# Statistics & hypothesis engine
python examples/stats_demo.py
python examples/hypothesis_demo.py
python examples/hypothesis_tests_demo.py
python examples/hypothesis_with_stats_demo.py
python examples/hypothesis_custom_generator.py
```

Notebooks live alongside the scripts:
`examples/tour_of_numerical_methods.ipynb` and
`examples/plotting_notebook.ipynb`. `plot_demo.py` and `plotting_notebook.ipynb`
need the `[plot]` extra.

See [Research Workflows](research-workflows.md) for guidance on using CDS inside
larger research scripts and discovery pipelines.

## Project Structure

`src/cds/` holds **34 subpackages** plus `sensitivity.py` and `causal.py`. Run
`cds modules` for the live catalog.

```
src/cds/
├── cli/                 # The `cds` console script (argparse, zero-dependency)
├── core/                # Shared models: Domain, Hypothesis, HypothesisStatus
├── math_utils/          # Calculus + linear algebra (SVD, QR, Cholesky, LU, eigen)
├── interpolate/         # interp1d / interp2d
├── probability/         # Distributions, quantiles, seeded sampling
├── bayes/               # Conjugate updates, credible intervals, Bayes factors
├── stats/               # Descriptive stats, regression, tests, time series
├── signals/             # DFT, FFT, convolution, Butterworth filters, STFT
├── wavelets/            # Haar DWT/IDWT, denoising
├── montecarlo/          # π estimation, MC integration, random walks
├── numerical_integration/ # Quadrature (trapezoid, Simpson, Romberg) + 2-D rules
├── diffeq/              # ODE solvers (Euler, RK4, RK45, implicit, symplectic)
├── pde/                 # 1-D heat & wave equation solvers
├── optimization/        # Gradient descent, Newton, Adam, Nelder-Mead, annealing
├── quantum/             # Quantum circuit simulation (single & multi-qubit)
├── ml/                  # Estimators (MLP, k-NN, k-means, CART, forest, boosting)
├── nlp/                 # BPE tokenizer, attention, autograd, MiniGPT
├── modeling/            # Symbolic algebra, expression trees, MathModel
├── scientific/          # Physical constants & formulas
├── genetics/            # GC content, k-mers, alignment
├── fractals/            # Mandelbrot, Julia, Barnsley, Sierpinski
├── infotheory/          # Entropy, KL/JS divergence, mutual information
├── graph/               # Graph algorithms (Dijkstra, BFS, DFS, Kruskal)
├── uncertainty/         # Analytic + Monte-Carlo propagation
├── sensitivity.py       # Local/global sensitivity, identifiability
├── validation/          # Validation checks, drift/OOD reports
├── causal.py            # Assumption-gated causal estimators
├── units/               # SI quantities, dimensional analysis
├── workflow/            # Approval-gated orchestration
├── provenance/          # Run manifests, hashing, checkpoints
├── data_io/             # Streaming I/O, optional HDF5/NetCDF
├── tools/               # Lazy adapters for NumPy/SciPy/SymPy/Z3
├── data_analysis/       # DataSet/DataTable + optional pandas interop ([pandas])
├── hypothesis/          # Hypothesis generation & evaluation
├── knowledge/           # Concept graph, notebook, structured retrieval
└── plot/                # Optional matplotlib helpers ([plot])

examples/                # Runnable demo scripts and notebooks
tests/                   # Test suite (see the CI badge for current status)
docs/                    # Documentation, API reference, benchmarks
```
