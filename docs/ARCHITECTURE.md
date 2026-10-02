# Architecture

This document describes how the Scientific Computing System (CDS) is
organized: the module layers, what depends on what, and how data flows
from a user call to a result. CDS is deliberately a **pure-Python,
zero-heavy-dependency** library, every algorithm lives in `src/cds/` as
code you can read and step through. The architecture reflects that goal:
layers are thin, the dependency graph points downward toward stable
primitives, and no module reaches back up into a higher-level one.

---

## 1. Module map

`src/cds/` contains **34 subpackages** plus two single-file modules
(`causal.py`, `sensitivity.py`). Each subpackage owns one scientific domain
and exposes its public API through an `__init__.py` with an explicit
`__all__`. Run `cds modules` for the same catalog from an installed package.

| Module | Responsibility |
| --- | --- |
| [`core`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/core) | Shared data models (`Domain`, `Hypothesis`, `HypothesisStatus`) and numeric guards. |
| [`math_utils`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/math_utils) | Calculus, linear algebra (SVD, QR, Cholesky, LU, eigen), special functions. |
| [`interpolate`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/interpolate) | `interp1d` / `interp2d` factory functions over sampled domains. |
| [`probability`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/probability) | Discrete/continuous distributions and sampling, plus chi-square/Student-t quantiles and gamma/beta (Marsaglia–Tsang) samplers. |
| [`bayes`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/bayes) | Conjugate updates, credible intervals, and the binomial Bayes factor. |
| [`scientific`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/scientific) | Physical constants and closed-form scientific formulas. |
| [`graph`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/graph) | Graph algorithms: BFS/DFS, Dijkstra, Kruskal MST, etc. |
| [`signals`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/signals) | DFT/FFT, convolution, power spectra, STFT, Butterworth filter design. |
| [`wavelets`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/wavelets) | Haar DWT/IDWT, multi-level decomposition, denoising. |
| [`stats`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/stats) | Descriptive stats, hypothesis tests, regression, effect sizes, time-series, nonparametric rank tests, multiple testing. |
| [`montecarlo`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/montecarlo) | Stochastic estimation and integration (e.g. π by dart-throwing), random walks. |
| [`numerical_integration`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/numerical_integration) | Deterministic quadrature: Newton-Cotes, Romberg, Gauss-Legendre, 2-D. |
| [`diffeq`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/diffeq) | ODE solvers: Euler, RK4, adaptive RK45, implicit stiff methods (backward Euler, Crank–Nicolson), symplectic integrators. |
| [`pde`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/pde) | Finite-difference heat and wave equation solvers on uniform 1-D grids. |
| [`optimization`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/optimization) | Gradient descent, Newton, Adam, line search, Nelder–Mead, simulated annealing, constrained search. |
| [`quantum`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/quantum) | Single- and multi-qubit circuit/state-vector simulation. |
| [`ml`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/ml) | From-scratch neural networks plus k-NN, k-means, CART trees, random forest, boosting, regression, PCA, scaling and splitting. |
| [`nlp`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/nlp) | BPE tokenizer, embeddings, attention, a mini-GPT, autograd. |
| [`modeling`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/modeling) | Symbolic expressions, equation systems, solvers. |
| [`genetics`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/genetics) | GC content, k-mers, reverse complement, Needleman-Wunsch alignment. |
| [`fractals`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/fractals) | Escape-time Mandelbrot/Julia and IFS fractals. |
| [`infotheory`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/infotheory) | Shannon entropy, cross-entropy, KL/JS divergence, mutual information. |
| [`uncertainty`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/uncertainty) | `UncertainValue` with analytic and correlated Monte-Carlo propagation. |
| [`sensitivity`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/sensitivity.py) | Local and variance-based global sensitivity, identifiability analysis. |
| [`validation`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/validation) | Validation checks, cross-method verification, data adequacy, drift/OOD reports. |
| [`causal`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/causal.py) | Assumption-gated causal effect estimators (module, not subpackage). |
| [`units`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/units) | SI quantities, conversions, dimension-compatibility checks. |
| [`workflow`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/workflow) | Approval-gated orchestration, gate decisions, execution traces. |
| [`provenance`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/provenance) | Run manifests, canonical hashing, decision records, checkpoints. |
| [`data_analysis`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/data_analysis) | Data loading, analysis, optional pandas interop, ASCII visualization. |
| [`data_io`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/data_io) | Streaming I/O, online moments, lazy HDF5/NetCDF backends. |
| [`tools`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/tools) | Capability discovery and adapters for optional scientific backends. |
| [`hypothesis`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/hypothesis) | Structured research-hypothesis generation and evaluation. |
| [`knowledge`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/knowledge) | Concept graphs and structured research-note retrieval. |
| [`plot`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/plot) | Optional matplotlib plotting helpers (`[plot]` extra). |
| [`cli`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/cli) | The `cds` console script (pure-stdlib `argparse`). |

---

## 2. The dependency graph

Dependencies **only point downward**: low-level numerical primitives at
the bottom, domain toolkits in the middle, application-level composition
at the top. No module imports something above it, which keeps the graph
acyclic and lets each layer be understood (and tested) in isolation.

Only cross-subpackage edges that actually appear in `import` statements are
drawn. 18 subpackages have no outgoing internal dependency at all, so they are
shown as standalone leaves rather than being forced into a layer.

```mermaid
graph TD
    %% Application / composition
    cli["cli<br/>(console script)"]
    workflow["workflow<br/>(orchestration)"]
    hypothesis["hypothesis<br/>(research ideas)"]
    data_analysis["data_analysis<br/>(load + viz)"]
    modeling["modeling<br/>(symbolic math)"]

    %% Domain toolkits
    ml["ml<br/>(estimators)"]
    nlp["nlp<br/>(mini-GPT)"]
    quantum["quantum<br/>(circuits)"]
    diffeq["diffeq<br/>(ODEs)"]
    bayes["bayes<br/>(conjugate)"]
    optimization["optimization<br/>(minimize)"]
    numerical_integration["numerical_integration<br/>(quadrature)"]
    stats["stats<br/>(tests, time-series)"]
    signals["signals<br/>(FFT, filters)"]
    probability["probability<br/>(distributions)"]
    plot["plot<br/>(matplotlib)"]

    %% Assurance layer
    provenance["provenance<br/>(manifests)"]
    validation["validation<br/>(checks)"]
    tools["tools<br/>(adapters)"]

    %% Primitives
    math_utils["math_utils<br/>(linalg)"]
    core["core<br/>(models, guards)"]

    %% Standalone leaves (no internal deps)
    knowledge["knowledge<br/>(concept graphs)"]
    montecarlo["montecarlo<br/>(sampling)"]
    graph["graph<br/>(algorithms)"]
    scientific["scientific<br/>(constants)"]
    wavelets["wavelets<br/>(DWT)"]
    genetics["genetics<br/>(DNA)"]
    fractals["fractals"]
    infotheory["infotheory"]
    interpolate["interpolate"]
    uncertainty["uncertainty<br/>(propagation)"]
    sensitivity["sensitivity<br/>(identifiability)"]
    causal["causal<br/>(estimators)"]
    units["units<br/>(dimensional)"]
    pde["pde<br/>(heat, wave)"]
    data_io["data_io<br/>(streaming)"]

    %% cli -> toolkits
    cli --> stats
    cli --> probability
    cli --> scientific
    cli --> numerical_integration
    cli --> hypothesis
    cli --> data_analysis
    cli --> core
    cli --> plot

    %% orchestration -> assurance
    workflow --> validation
    workflow --> provenance
    workflow --> tools

    %% composition -> toolkits
    hypothesis --> stats
    hypothesis --> core
    data_analysis --> stats

    %% toolkits -> toolkits / primitives
    ml --> optimization
    ml --> math_utils
    ml --> core
    nlp --> math_utils
    nlp --> core
    modeling --> optimization
    modeling --> core
    optimization --> math_utils
    optimization --> core
    diffeq --> math_utils
    diffeq --> core
    probability --> math_utils
    stats --> math_utils
    stats --> core
    numerical_integration --> core
    quantum --> core
    bayes --> math_utils
    plot --> signals
    plot --> stats

    %% primitives
    math_utils --> core

    classDef app fill:#dbeafe,stroke:#1d4ed8,color:#1e3a8a;
    classDef domain fill:#dcfce7,stroke:#15803d,color:#14532d;
    classDef assurance fill:#fce7f3,stroke:#be185d,color:#831843;
    classDef prim fill:#fef3c7,stroke:#b45309,color:#78350f;
    classDef leaf fill:#f3f4f6,stroke:#6b7280,color:#374151;
    class cli,workflow,hypothesis,data_analysis,modeling app;
    class ml,nlp,quantum,diffeq,bayes,optimization,numerical_integration,stats,signals,probability,plot domain;
    class provenance,validation,tools assurance;
    class math_utils,core prim;
    class knowledge,montecarlo,graph,scientific,wavelets,genetics,fractals,infotheory,interpolate,uncertainty,sensitivity,causal,units,pde,data_io leaf;
```

The five colours denote the layers:

- **Blue (application)**: composes lower toolkits into a workflow
  (turning `stats` into hypothesis generation, or into a data pipeline),
  plus the `cli` entry point. Some of these sit at the *top* of the graph
  even where nothing imports them: their role is composition, not
  primitives.
- **Green (domain toolkits)**: the scientific algorithms. Each is
  self-contained within its field and depends only on primitives or a
  sibling toolkit.
- **Pink (assurance)**: the evidence layer — `validation` (cross-method
  checks), `provenance` (run manifests and hashes), and `tools`
  (lazy backend adapters). `workflow` consumes all three and is
  deliberately fail-closed when they cannot vouch for a result.
- **Amber (primitives)**: `core` (shared models and numeric guards) and
  `math_utils` (linear algebra and calculus). These change the least and
  are depended on the most.
- **Grey (standalone leaves)**: 15 subpackages with **no** internal
  dependency in either direction — `knowledge`, `montecarlo`, `graph`,
  `scientific`, `wavelets`, `genetics`, `fractals`, `infotheory`,
  `interpolate`, `uncertainty`, `sensitivity`, `causal`, `units`, `pde`,
  and `data_io`. They are pure-Python and self-contained, importable with
  zero cross-package coupling, and therefore independently testable.

> The graph is derived from actual `import` statements (docstring
> references such as `:mod:` roles are excluded), so it reflects the real
> coupling in the code rather than an intended layering.
>
> **The "downward-only" rule is a convention, not an enforced invariant.**
> The graph is acyclic today, and section 3 states the rules contributors
> are expected to hold to, but no static check currently rejects a
> back-edge. Read the diagram as descriptive, not as a guarantee.

---

## 3. Layered design rules

The graph above is enforced by convention and encodes three rules:

1. **Downward-only imports.** A module may import from the layer below,
   never from one above. `optimization` may use `math_utils`, but
   `math_utils` may never reach into `optimization`. This is what keeps
   the system acyclic and each layer independently testable.
2. **One concern per package.** Each subpackage owns a single scientific
   domain. Statistics lives in `stats`, signal processing in `signals`,
   quadrature in `numerical_integration`, never mixed. Cross-domain
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
 input → │  parse / load   │  lists, floats, core.Domain/Hypothesis
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

1. **Input** arrives as native Python types: a `list[float]` signal, a
   2-D `list[list[float]]` matrix, a callable `f(x)`, or a
   `core.Domain` / `core.Hypothesis` value. No DataFrame or array library
   is required.
2. **Algorithm** runs in a domain toolkit. It is a pure function: same
   input always yields the same output, no hidden state, no I/O. When a
   routine needs a lower-level primitive (e.g. `kpss_statistic` needs a
   mean, `gradient_descent` needs a matrix solve) it imports it from the
   layer below, that is the only cross-module coupling.
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

CDS keeps its zero-dependency guarantee by isolating anything that needs a
third-party library behind an **optional adapter**. There are four such
boundaries today, all resolved lazily at call time:

| Boundary | Extra | How it is isolated |
| --- | --- | --- |
| [`data_analysis/pandas_io.py`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/data_analysis/pandas_io.py) | `[pandas]` | `to_dataframe` / `from_dataframe` are the only places pandas appears; imported inside the function. |
| [`plot/`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/plot) | `[plot]` | matplotlib is imported lazily on the first plot call, so `import cds.plot` itself is cheap and safe. |
| [`data_io/`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/data_io) | `[io]` | HDF5 (h5py) and NetCDF (netCDF4) backends load only when requested; core streaming/CSV I/O stays stdlib-only. |
| [`tools/`](https://github.com/Furox-Art/scientific-computing-system/blob/main/src/cds/tools) | `[scientific]` | NumPy, SciPy, statsmodels, scikit-learn, SymPy and Z3 are discovered through a capability registry; none is imported by the core. |

The shared pattern:

- Core algorithms operate on plain `list` and `dict`.
- Third-party imports live **inside** the adapter function, never at module
  top level, so a missing dependency degrades one call rather than breaking
  `import cds`.
- Installing the extra is opt-in; without it the rest of the library is
  unaffected.

This is the pattern for any future interop: keep the algorithm pure-Python,
and put the bridge behind a clearly-marked, lazily-imported adapter.

!!! warning "Not a sandbox"
    These adapters do not make third-party libraries safe for adversarial
    input. `sympy_verify_identity()` passes caller-provided symbolic strings
    to SymPy's parser, and HDF5/NetCDF files carry their own security
    guidance. See [SECURITY.md](https://github.com/Furox-Art/scientific-computing-system/blob/main/SECURITY.md)
    for the full threat model.

---

## 6. Where things live

```
src/cds/
├── __init__.py            # top-level re-exports, __version__
├── _version.py            # static version source
├── __main__.py            # `python -m cds`
├── cli/                   # ← the `cds` console script (argparse)
│
├── core/                  # ← primitives layer
├── math_utils/
│
├── probability/           # ← domain toolkits layer
├── bayes/
├── interpolate/
├── stats/
├── signals/
├── wavelets/
├── montecarlo/
├── numerical_integration/
├── diffeq/
├── pde/
├── optimization/
├── quantum/
├── ml/
├── nlp/
├── modeling/
├── scientific/
├── genetics/
├── fractals/
├── infotheory/
├── graph/
├── plot/                  # optional matplotlib helpers
│
├── uncertainty/           # ← assurance layer
├── sensitivity.py         #   single-file module
├── validation/
├── causal.py              #   single-file module
├── units/
├── workflow/
├── provenance/
├── tools/
├── data_io/
│
├── data_analysis/         # ← application / composition layer
├── hypothesis/
└── knowledge/
```

That is 34 subpackages plus `sensitivity.py` and `causal.py`. The
`cds modules` command prints the live catalog if this tree ever drifts.

Tests mirror this layout one-to-one under `tests/` (e.g.
`src/cds/stats/time_series.py` ↔ `tests/test_stats_time_series.py`), so
finding the coverage for any module is a matter of matching the name.

---

## Where to go next

- **[Cookbook](cookbook.md)**: copy-pasteable recipes for every module.
- **[Tour of Numerical Methods](tour_of_numerical_methods.md)**: a guided
  end-to-end walkthrough.
- **[API Reference](api.md)**: the authoritative signatures and defaults.
