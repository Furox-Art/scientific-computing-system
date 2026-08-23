# Architecture

This document describes how the Scientific Computing System (CDS) is
organized: the module layers, what depends on what, and how data flows
from a user call to a result. CDS is deliberately a **pure-Python,
zero-heavy-dependency** library — every algorithm lives in `src/cds/` as
code you can read and step through. The architecture reflects that goal:
layers are thin, the dependency graph points downward toward stable
primitives, and no module reaches back up into a higher-level one.

---

## 1. Module map

CDS is split into **18 top-level subpackages** under `src/cds/`, each
owning one scientific domain. Every subpackage exposes its public API
through an `__init__.py` with an explicit `__all__`.

| Module | Responsibility |
| --- | --- |
| [`core`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/core) | Shared data models (`DataPoint`, `Dataset`) and numeric guards. |
| [`math_utils`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/math_utils) | Linear algebra, special functions, combinatorics, number theory. |
| [`probability`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/probability) | Discrete/continuous distributions and sampling, plus chi-square/Student-t quantiles and gamma/beta (Marsaglia–Tsang) samplers. |
| [`scientific`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/scientific) | Physical constants and closed-form scientific formulas. |
| [`graph`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/graph) | Graph algorithms — BFS/DFS, Dijkstra, Kruskal MST, etc. |
| [`signals`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/signals) | DFT/FFT, convolution, power spectra, Butterworth filter design. |
| [`stats`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/stats) | Descriptive stats, hypothesis tests, regression, time-series, nonparametric rank tests. |
| [`montecarlo`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/montecarlo) | Stochastic estimation and integration (e.g. π by dart-throwing). |
| [`numerical_integration`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/numerical_integration) | Deterministic quadrature — Newton-Cotes, Romberg, Gauss-Legendre, 2-D. |
| [`diffeq`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/diffeq) | ODE solvers — Euler, RK4, RK45, plus implicit stiff methods (backward Euler, Crank–Nicolson). |
| [`optimization`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/optimization) | Gradient descent, Newton, Adam, line search, Nelder–Mead, simulated annealing. |
| [`quantum`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/quantum) | Single- and multi-qubit circuit/state-vector simulation. |
| [`ml`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/ml) | From-scratch neural networks plus k-NN, k-means, CART trees, logistic/linear regression, PCA, scaling and splitting. |
| [`nlp`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/nlp) | BPE tokenizer, embeddings, attention, a mini-GPT, autograd. |
| [`modeling`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/modeling) | Symbolic expressions, equation systems, solvers. |
| [`data_analysis`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/data_analysis) | Data loading, analysis, optional pandas interop, visualization. |
| [`hypothesis`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/hypothesis) | Structured research-hypothesis generation and evaluation. |
| [`knowledge`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/knowledge) | Concept graphs and structured research-note retrieval. |

---

## 2. The dependency graph

Dependencies **only point downward**: low-level numerical primitives at
the bottom, domain toolkits in the middle, application-level composition
at the top. No module imports something above it, which keeps the graph
acyclic and lets each layer be understood (and tested) in isolation.

```mermaid
graph TD
    %% Application / composition layer
    knowledge["knowledge<br/>(concept graphs)"]
    hypothesis["hypothesis<br/>(research ideas)"]
    data_analysis["data_analysis<br/>(load + viz)"]
    modeling["modeling<br/>(symbolic math)"]

    %% Domain toolkits layer
    ml["ml<br/>(neural nets)"]
    nlp["nlp<br/>(mini-GPT)"]
    quantum["quantum<br/>(circuits)"]
    diffeq["diffeq<br/>(ODEs)"]
    optimization["optimization<br/>(minimize)"]
    numerical_integration["numerical_integration<br/>(quadrature)"]
    stats["stats<br/>(tests, time-series)"]
    signals["signals<br/>(FFT, filters)"]
    montecarlo["montecarlo<br/>(sampling)"]
    graph["graph<br/>(algorithms)"]
    probability["probability<br/>(distributions)"]
    scientific["scientific<br/>(constants)"]

    %% Primitives layer
    math_utils["math_utils<br/>(linalg)"]
    core["core<br/>(models, guards)"]

    %% Application -> toolkits
    hypothesis --> stats
    data_analysis --> stats

    %% Domain toolkits -> toolkits / primitives
    ml --> optimization
    nlp --> math_utils
    modeling --> optimization
    optimization --> math_utils

    %% Everything rests on core
    math_utils --> core
    stats --> core
    diffeq --> core
    optimization --> core
    quantum --> core
    ml --> core
    nlp --> core
    modeling --> core
    numerical_integration --> core
    hypothesis --> core

    classDef app fill:#dbeafe,stroke:#1d4ed8,color:#1e3a8a;
    classDef domain fill:#dcfce7,stroke:#15803d,color:#14532d;
    classDef prim fill:#fef3c7,stroke:#b45309,color:#78350f;
    class knowledge,hypothesis,data_analysis,modeling app;
    class ml,nlp,quantum,diffeq,optimization,numerical_integration,stats,signals,montecarlo,graph,probability,scientific domain;
    class math_utils,core prim;
```

The three colors denote the layers:

- **Blue (application)** — composes lower toolkits into a workflow
  (turning `stats` into hypothesis generation, or into a data pipeline).
  Note these modules are at the *top* of the graph even though some have
  no incoming edges: their role is composition, not primitives.
- **Green (domain toolkits)** — the scientific algorithms. Each is
  self-contained within its field and depends only on primitives or a
  sibling toolkit.
- **Amber (primitives)** — `core` (shared models and numeric guards) and
  `math_utils` (linear algebra and special functions). These change the
  least and are depended on the most.

> **Leaf modules** with *no* internal dependencies — `core`, `probability`,
> `scientific`, `graph`, `signals`, `montecarlo`, `knowledge` — form the
> foundation that everything else is built on. They can be imported with
> zero cross-package coupling.
>
> The graph is generated from actual `import` statements (docstring
> references such as `:mod:` roles are excluded), so it always reflects
> the real coupling in the code.

---

## 3. Layered design rules

The graph above is enforced by convention and encodes three rules:

1. **Downward-only imports.** A module may import from the layer below,
   never from one above. `optimization` may use `math_utils`, but
   `math_utils` may never reach into `optimization`. This is what keeps
   the system acyclic and each layer independently testable.
2. **One concern per package.** Each subpackage owns a single scientific
   domain. Statistics lives in `stats`, signal processing in `signals`,
   quadrature in `numerical_integration` — never mixed. Cross-domain
   needs are expressed as *dependencies*, not by sprawling a module.
3. **Explicit public surface.** Every `__init__.py` declares `__all__`,
   so the public API is documented in code and verified by `mypy`. The
   underscore-prefixed files (`_distributions.py`, `_nodes.py`, `_base.py`,
   `_numeric.py`) are implementation detail and are not re-exported.

---

## 4. Data flow

A CDS computation moves through three stages: **input → algorithm →
result**, where each stage is a plain Python object with no framework
involvement.

```
        ┌─────────────────┐
input → │  parse / load   │  lists, floats, core.DataPoint/Dataset
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │   algorithm      │  pure functions in a domain toolkit
        │   (the lesson)   │  (e.g. fft_radix2, gradient_descent, simpson_2d)
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
result ← │  typed result    │  plain floats, lists, or a frozen @dataclass
        │  (dataclass)     │  (e.g. QuadratureResult, TestResult, StationarityResult)
        └─────────────────┘
```

Concretely, a call flows like this:

1. **Input** arrives as native Python types — a `list[float]` signal, a
   2-D `list[list[float]]` matrix, a callable `f(x)`, or a
   `core.models.Dataset`. No DataFrame or array library is required.
2. **Algorithm** runs in a domain toolkit. It is a pure function: same
   input always yields the same output, no hidden state, no I/O. When a
   routine needs a lower-level primitive (e.g. `kpss_statistic` needs a
   mean, `gradient_descent` needs a matrix solve) it imports it from the
   layer below — that is the only cross-module coupling.
3. **Result** is either a native value (the integrated area, the
   optimized vector) or an immutable `@dataclass` that bundles the
   headline answer with diagnostics (`TestResult.p_value`,
   `QuadratureResult.error_estimate`). Frozen dataclasses make results
   hashable, comparable, and safe to thread through higher-level
   workflows like `hypothesis`.

Because every stage is a plain object, **the whole pipeline composes**:
the output of `linear_regression` feeds `one_sample_ttest`, which feeds
`hypothesis.generate_hypotheses`, with no adapters or glue code.

---

## 5. The optional-dependency boundary

CDS keeps its zero-dependency guarantee by isolating anything that needs
a third-party library behind an **optional adapter**. Today that is the
pandas interop in [`data_analysis/pandas_io.py`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/data_analysis/pandas_io.py):

- The core algorithms operate on plain `list` and `dict`.
- `to_dataframe` / `from_dataframe` are the *only* places pandas appears,
  imported lazily inside the function.
- Installing via the `cds[pandas]` extra pulls in pandas; without it,
  the rest of the library is unaffected.

This is the pattern for any future interop (NumPy arrays, plotting
backends): keep the algorithm pure-Python, and put the bridge behind a
clearly-marked, lazily-imported adapter. Nothing in the dependency graph
of section 2 ever crosses the optional-dependency line.

---

## 6. Where things live

```
src/cds/
├── __init__.py            # top-level convenience re-exports
├── cli/                    # command-line entry point (cds command)
├── _version.py
├── core/                  # ← primitives layer
├── math_utils/
├── probability/
├── scientific/
├── graph/
├── signals/               # ← domain toolkits layer
├── stats/
├── montecarlo/
├── numerical_integration/
├── diffeq/
├── optimization/
├── quantum/
├── ml/
├── nlp/
├── modeling/
├── data_analysis/         # ← application / composition layer
├── hypothesis/
└── knowledge/
```

Tests mirror this layout one-to-one under `tests/` (e.g.
`src/cds/stats/time_series.py` ↔ `tests/test_stats_time_series.py`), so
finding the coverage for any module is a matter of matching the name.

---

## Where to go next

- **[Cookbook](cookbook.md)** — copy-pasteable recipes for every module.
- **[Tour of Numerical Methods](tour_of_numerical_methods.md)** — a guided
  end-to-end walkthrough.
- **[API Reference](api.md)** — the authoritative signatures and defaults.
