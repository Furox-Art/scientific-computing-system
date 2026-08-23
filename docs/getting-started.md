# Getting Started

## Installation

```bash
git clone https://github.com/Furox-Art/scientific-computing-system.git
cd scientific-computing-system
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Quick Usage

### CLI

```bash
# Show available commands
cds --help

# List physical constants
cds constants

# Interactive physics calculator
cds calc ke    # kinetic energy
cds calc gravity
cds calc wave
cds calc gas

# Generate hypotheses
cds hypothesize "what causes the Hubble tension?" --domain cosmology
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

print(cohens_d([20.0, 22.0, 19.0], [28.0, 31.0, 26.0]))  # standardized mean difference
print(cramers_v([[10.0, 20.0], [30.0, 40.0]]))  # association strength for a contingency table

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
pytest           # run all 1441 tests (see CI)
pytest -v        # verbose output
pytest -x        # stop on first failure
```

## Running Examples

```bash
# Core models & data
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
python examples/montecarlo_demo.py
python examples/probability_demo.py
python examples/optimization_demo.py
python examples/scientific_demo.py

# Signals & ML
python examples/signals_demo.py
python examples/fft2_demo.py
python examples/ml_and_viz_demo.py

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

See `docs/research-workflows.md` for guidance on using CDS inside larger research scripts and discovery pipelines.

## Project Structure

```
src/cds/
├── quantum/             # Quantum circuit simulation (single & multi-qubit)
├── optimization/        # Gradient descent, Newton, Adam, line search
├── ml/                  # Pure Python neural networks (MLP, Adam training)
├── signals/             # DFT, FFT, convolution, Butterworth IIR filters
├── probability/         # Probability distributions & sampling
├── stats/               # Descriptive stats, regression, hypothesis tests, time-series
├── math_utils/          # Calculus, linear algebra, eigenvalues
├── data_analysis/       # DataSet/DataTable + optional pandas interop (cds[pandas])
├── scientific/          # Physical constants & formulas
├── graph/               # Graph algorithms (Dijkstra, BFS, DFS, Kruskal)
├── montecarlo/          # Monte Carlo methods (π, integration, random walks)
├── diffeq/              # ODE solvers (Euler, RK4, midpoint)
├── numerical_integration/ # Quadrature (trapezoid, Simpson, Romberg) + 2-D rules
├── modeling/            # Symbolic algebra, expression trees, MathModel
├── knowledge/           # Concept graph, notebook, structured retrieval
├── nlp/                 # BPE tokenizer, attention, autograd, MiniGPT
├── hypothesis/          # Hypothesis generation
├── core/                # Shared models, config
└── cli/                 # Command-line interface (argparse, zero-dependency)

examples/                # Runnable demo scripts
tests/                   # 1441 tests (see CI)
docs/                    # Documentation, API reference, benchmarks
```
