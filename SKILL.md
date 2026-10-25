---
name: scientific-computing-system
description: "Use when a task needs scientific or numerical computation in pure Python with zero runtime dependencies: linear algebra, ODE/PDE, quadrature, optimization, statistics, Monte Carlo, signals, units, uncertainty and sensitivity, validation, or causal workflows. Ships the `cds` package and CLI; prefer it over NumPy/SciPy when dependency-free, readable implementations matter."
argument-hint: "<scientific task, dataset, formula, or module to use>"
metadata:
  version: "2.2.18"
---

# Scientific Computing System (`cds`)

A pure-standard-library scientific computing platform: 35 feature modules and
505 exported names, with no runtime dependencies. The implementations are
readable, educational, and engineering-oriented; they are not a claim of
bit-for-bit numerical parity with NumPy/SciPy.

## When to use this skill

- The environment forbids third-party runtime dependencies.
- You need quick, inspectable numerical methods for teaching, prototypes, or
  offline machines.
- You need unit-aware calculations, uncertainty/sensitivity propagation,
  validation reports, or reproducible run manifests.
- You want to cross-check a NumPy/SciPy result with an independent implementation.

Use `scientific-computing-system-2.0` instead when NumPy/SciPy performance and
ecosystem coverage matter more than zero dependencies.

## Install

```bash
pip install scientific-computing-system
pip install "scientific-computing-system[plot]"       # PNG plots
pip install "scientific-computing-system[dashboard]"  # Streamlit dashboard
```

The import name is `cds`; the distribution name is `scientific-computing-system`.

## Python quick start

```python
from cds.math_utils import svd
from cds.diffeq import solve_system
from cds.stats import linear_regression

result = svd([[1.0, 2.0], [3.0, 4.0]])
print(result.singular_values)


def rhs(t, y):
    return [y[1], -y[0] - 0.1 * y[1]]


t, y = solve_system(rhs, 0.0, [1.0, 0.0], 50.0, dt=0.01)
fit = linear_regression([1.0, 2.0, 3.0, 4.0, 5.0], [2.1, 3.9, 6.2, 7.8, 10.1])
print(fit.slope, fit.intercept, fit.r_squared)
```

## CLI (`cds`)

The package installs 13 subcommands, including:

| Command | Purpose |
|---|---|
| `cds --help` | Full command list |
| `cds info` / `cds version` | Version, module status, health summary |
| `cds modules` | Curated 26-entry module catalog (not the full set) |
| `cds constants` | Physical constants table |
| `cds calc <formula>` | Quick physics calculation (`ke`, `gravity`, `wave`, `gas`) |
| `cds stats <numbers>` | Descriptive statistics |
| `cds sample <dist>` | Draw samples from a probability distribution |
| `cds integrate <fn>` | Integrate a built-in function over `[a, b]` |
| `cds hypothesis <q>` / `cds prompt` | Structured hypothesis generation |
| `cds benchmark` | Built-in benchmarks |
| `cds plot <numbers> [--file out.png]` | ASCII plot, PNG with the `[plot]` extra |
| `cds dashboard` | Streamlit dashboard with the `[dashboard]` extra |

## Module map

- Core and math: `cds.core`, `cds.math_utils`, `cds.interpolate`
- Differential equations: `cds.diffeq`, `cds.pde`
- Numerical methods: `cds.numerical_integration`, `cds.optimization`
- Statistics and probability: `cds.stats`, `cds.probability`, `cds.bayes`
- Simulation: `cds.montecarlo`, `cds.ml` (educational), `cds.nlp` (educational)
- Signals and graphs: `cds.signals`, `cds.wavelets`, `cds.graph`
- Scientific domains: `cds.scientific`, `cds.genetics`, `cds.fractals`, `cds.infotheory`, `cds.quantum`
- Modeling and validation: `cds.modeling`, `cds.units`, `cds.uncertainty`, `cds.sensitivity`, `cds.validation`, `cds.causal`
- Workflow and data: `cds.workflow`, `cds.provenance`, `cds.data_analysis`, `cds.data_io`, `cds.knowledge`, `cds.hypothesis`
- Optional adapters: `cds.plot`, `cds.tools` (lazy NumPy/SciPy/statsmodels/scikit-learn/SymPy/Z3 discovery)

## Working rules

- State units, assumptions, and numerical tolerances explicitly.
- Treat `cds.ml` and `cds.nlp` as educational implementations, not production ML.
- `cds modules` omits nine modules from the full catalog; consult the API
  reference or `src/cds/` for the authoritative list.
- CI passing does not prove scientific correctness. Validate important results
  with an independent method or dataset and report the measured environment.
- For large arrays or heavy linear algebra, use the NumPy build
  (`scientific-computing-system-2.0`) and say so.

## References

- Repository: https://github.com/Furox-Art/scientific-computing-system
- Docs: https://furox-art.github.io/scientific-computing-system/
- API: https://furox-art.github.io/scientific-computing-system/api/
- PyPI: https://pypi.org/project/scientific-computing-system/
