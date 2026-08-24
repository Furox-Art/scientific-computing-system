# Changelog

All notable changes to **scientific-computing-system** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Random Forest classifier** (`cds.ml`) — bagged CART trees with
  bootstrap resampling and per-split feature randomization (`sqrt`/`log2`
  / explicit count), fully deterministic under a seed.
- **Metrics** (`cds.ml`) — `accuracy`, `confusion_matrix`,
  `precision_recall_f1`, `macro_prf`, regression `mse`/`mae`/`r2_score`,
  and tie-aware rank-based `roc_auc`.
- **Cross-validation** (`cds.ml.model_selection`) — `k_fold_indices` and
  `cross_val_score` over the shared estimator interface.
- **Interpolation module** (`cds.interpolate`) — `interp1d` (linear /
  nearest) and `interp2d` (bilinear on regular grids).
- **SVD** (`cds.math_utils`) — one-sided Jacobi decomposition with
  `svd`, `rank` and `condition_number`; full U/Vt factors.
- **STFT / spectrogram** (`cds.signals`) — Hann/Hamming windows, framed
  zero-padded transforms reusing the radix-2 FFT, `STFTResult`.
- **Gradient Boosting classifier** (`cds.ml`) — logistic-loss boosted
  regression trees with Newton leaf values; binary labels.
- **Bootstrap confidence intervals** (`cds.stats`) — percentile-method
  `bootstrap_ci` and two-sample `bootstrap_diff_ci`.
- **Metropolis-Hastings sampler** (`cds.montecarlo`) — 1-D random-walk
  MCMC with burn-in/thinning and acceptance-rate reporting.
- **Symplectic integrators** (`cds.diffeq`) — semi-implicit Euler and
  velocity Verlet with forward/backward integration.
- **Constrained optimization** (`cds.optimization`) — projected gradient
  descent for box bounds and quadratic-penalty constrained solves.
- **Hypergeometric & negative-binomial distributions**
  (`cds.probability`) — pmf + cdf for both.
- **TF-IDF ranked retrieval** (`cds.knowledge`) — `rank_tfidf` across
  concepts and notes with smoothed idf.
- **Haar wavelet transform** (`cds.signals`) — `dwt` / `idwt` /
  `dwt_multi_level`, orthonormal with perfect reconstruction.
- **PDE solvers** (`cds.pde`) — explicit FTCS heat equation (Dirichlet /
  Neumann, stability-guarded) and wave equation (CFL-guarded).
- **Gaussian Naive Bayes** (`cds.ml`) — log-space scoring with variance
  smoothing.
- **Correlation-mining engine** (`cds.hypothesis`) — `mine_correlations`
  turns numeric datasets into statistically tested, falsifiable
  `Hypothesis` objects.
- **Power analysis** (`cds.stats`) — `power_t_test`,
  `power_proportion_test` and `required_n_per_group` sample-size solver.
- **VotingClassifier** (`cds.ml`) — hard (majority vote) and soft
  (averaged probabilities) ensembles over the shared estimator interface.
- Four new runnable demos under `examples/` — interpolation, PDE,
  ensemble showcase and the end-to-end discovery pipeline.

### Changed

- `DecisionTreeClassifier` now accepts `max_features` and `seed` for
  random-forest-style split randomization (default behavior unchanged).

## [v1.6.1] - 2026-08-21

A **patch** release completing the JSON round trip for core data models.

### Added

- **`cds.core`**: `Hypothesis.from_dict()` — the inverse of `to_dict()`.
  CLI/JSON output can now be re-imported: enum strings are coerced back to
  `Domain` / `HypothesisStatus`, ISO-8601 timestamps are parsed, and missing
  optional keys fall back to constructor defaults. Invalid `status` values
  raise `ValueError`.

### Changed

- Internal layering cleanup — shared special-function kernels moved to their
  single home (`cds.math_utils.special`) so lower-level packages never depend
  on higher-level ones; a suite-level import-direction guard enforces this.
- `cds.cli` is now a package (`_style` / `_handlers` / `_parser`) with
  identical behaviour; refreshed module descriptions in `cds modules`.

## [v1.6.0] - 2026-08-21

A **minor** release deepening the ML, probability, ODE, and optimization
toolkits toward scikit-learn / SciPy teaching parity — still zero runtime
dependencies, still 100% blended coverage.

### Added

- **`cds.ml`**: `KNeighborsClassifier` / `KNeighborsRegressor` (brute-force,
  deterministic tie-breaking), `KMeans` (k-means++ seeding, seeded),
  `DecisionTreeClassifier` (CART + Gini), `LogisticRegression` (stable-BCE
  gradient descent, optional L2), `LinearRegression` (closed-form vector OLS
  via normal equations), `PCA` (cyclic Jacobi eigen-solver, transform /
  inverse_transform), `StandardScaler`, `train_test_split` (seeded)
- **`cds.probability`**: chi-square and Student-t distributions
  (`chi2_pdf/cdf/ppf`, `t_pdf/cdf/ppf`), `gamma_pdf` + `sample_gamma`
  (Marsaglia–Tsang with shape<1 boost), `beta_pdf` + `sample_beta`
  (exact gamma-ratio identity)
- **`cds.diffeq`**: implicit stiff solvers — `backward_euler`,
  `trapezoid_method` (Crank–Nicolson) and their `_system` variants; unified
  θ-method core with Newton iteration over analytic or central-difference
  Jacobians via the shared PLU solver
- **`cds.optimization`**: `nelder_mead` simplex (degenerate-simplex and
  symmetric false-convergence guards), `simulated_annealing` (Metropolis rule,
  box bounds, seeded)
- **`cds.stats`**: nonparametric tests — `mann_whitney_u` and
  `wilcoxon_signed_rank` with tie/zero corrections (normal approximation)

### Fixed

- Test-suite pyplot figures are now closed per test, silencing matplotlib's
  "more than 20 figures" memory warning in long sessions

### Changed

- README "Why CDS?" pitch strengthened with proof numbers and a cross-domain
  snippet; stale counts/versions synced across metadata (19 modules)

## [v1.5.0] - 2026-07-20

A **minor** release expanding algorithms across graph theory, Monte Carlo,
linear algebra, and CLI — still zero runtime dependencies.

### Added

- **`cds.graph`**: `degree`, `connected_components`, `bellman_ford`,
  `floyd_warshall`, `prim_mst`
- **`cds.montecarlo`**: `mc_expectation`, `hit_or_miss` (2-D area estimation)
- **`cds.math_utils`**: `vector_norm`, `frobenius_norm`, `matrix_trace`
- **CLI**: `cds integrate <sin|cos|exp|x2|unit> --a --b -n --method`

## [v1.4.0] - 2026-07-20

A **minor** release expanding core scientific APIs and CLI ergonomics.
Zero runtime dependencies preserved.

### Added

- **`cds.probability`**: CDFs (`gaussian_cdf`, `uniform_cdf`, `exponential_cdf`),
  `geometric_pmf`, samplers (`gaussian_sample` Box–Muller, `exponential_sample`,
  `poisson_sample` Knuth)
- **`cds.stats`**: `spearman_correlation` (midranks), `percentile`, `z_scores`
- **`cds.scientific`**: constants `epsilon_0`, `g`, `mu_0`; formulas
  `coulomb_force`, `centripetal_acceleration`, `pendulum_period`, `doppler_frequency`
- **CLI**: `cds stats <nums>`, `cds sample <dist> -n N --seed …`

### Changed

- `cds info` no longer hardcodes a stale test count; module list includes modeling/knowledge/nlp

## [v1.3.2] - 2026-07-20

A **patch** release expanding the optional ``cds[plot]`` surface.

### Added

- **`cds.plot` charts**: `plot_multi_series`, `plot_scatter`, `plot_regression`
  (OLS via `cds.stats`), `plot_power_spectrum` (via `cds.signals`),
  `plot_seasonal_decompose` (4-panel), `plot_heatmap`, `plot_loss`,
  `save_figure(..., close=)` helper; histogram gains ``density=``
- **CLI**: `cds plot … --kind series|hist|acf --file out.png`

### Changed

- Docs / Cookbook / demo updated for the expanded plot API

## [v1.3.1] - 2026-07-20

A **patch** release. Fixes the console entry point so `cds` works after a
plain PyPI install.

### Fixed

- **`[project.scripts]`**: entry point was `cds.cli:app` but the CLI exports
  `main` (argparse). Installing from PyPI then raised
  `ImportError: cannot import name 'app' from 'cds.cli'`. Now points to
  `cds.cli:main`. `python -m cds` was already correct.

## [v1.3.0] - 2026-07-20

A **minor** release focused on **visualization and ergonomics**. Adds the
optional ``cds[plot]`` matplotlib extra, a CLI PNG export path, a companion
notebook, and documentation. The zero-dependency core is unchanged: matplotlib
is only imported when a plot function runs (or when ``cds plot --file`` is
used).

### Added

- **`cds.plot` optional package** (`src/cds/plot/`):
  - `plot_series`, `plot_histogram`, `plot_waveform`, `plot_spectrum`
  - `plot_acf` / PACF stems via `cds.stats` time-series helpers
  - `plot_optimization_path` for 2-D optimizer trajectories
  - Lazy matplotlib backend with install-hint `ImportError`
- **Extra**: `pip install scientific-computing-system[plot]` (also in `[all]`)
- **CLI**: `cds plot 1,2,3 --file out.png` saves a PNG when matplotlib is
  installed; without `--file` the existing ASCII terminal plot remains
- **Examples**: `examples/plot_demo.py`, `examples/plotting_notebook.ipynb`
- **Docs**: Cookbook matplotlib section, API reference for `cds.plot`, README /
  module tables, ROADMAP v1.3.0 track

### Changed

- CI main matrix installs `.[pandas,plot]` so optional bridges stay at 100%
  coverage; minimal-deps job omits `cds/plot/*` like `pandas_io`
- PyPI description mentions optional matplotlib plotting

## [v1.2.3] - 2026-06-29

A **patch** release. Fixes the release pipeline that silently broke every
publish since v1.1.8.

### Fixed

- **Release workflow (`release.yml`)**: removed an invalid
  `administration: write` entry from the `sync-about` job's `permissions:`
  block. `administration` is not a valid `permissions:` key, so GitHub
  rejected the entire workflow file as a "workflow file issue" before any
  job could start, failing in 0s and leaving v1.1.9 onward without a PyPI
  upload or GitHub Release. The About-sync step is now best-effort
  (`continue-on-error: true`) since `gh repo edit` needs an admin REST
  scope the default `GITHUB_TOKEN` can't grant; the repo description is
  synced by hand on release.

## [v1.2.2] - 2026-06-29

A **patch** release. No code changes — documentation-only refresh.

### Changed

- **README: removed pictographic emoji** from the top navigation row, the
  callout blocks, the "Support the project" section heading, and the trailing
  sign-off. Em dashes are preserved. Brings the README in line with the
  emoji-free house style already applied to the docs and live PR bodies.
- **`.gitignore`: ignore local promo-GIF QA scratch** (audit/compare/measure
  scripts under `scripts/_*.py` and extracted frames under `assets/_qa_*/`),
  which are one-off diagnostic tooling and were never meant to ship.

## [v1.2.1] - 2026-06-28

A **patch** release. No code changes — metadata-only refresh to improve
PyPI discoverability.

### Changed

- **Strengthened PyPI `description`**: the short one-liner now names all 18
  domain modules (quantum circuits, ML, NLP, signal processing, optimization,
  statistics & time-series, ODE solvers, Monte Carlo, symbolic math, knowledge
  graphs, hypothesis generation) and states the zero-runtime-dependency
  guarantee. Improves search-result relevance on PyPI.
- **Expanded `keywords`** from 12 to 20 entries, aligning them with the GitHub
  repository topics (added `machine-learning`, `nlp`, `numerical-integration`,
  `knowledge-graph`, `symbolic-math`, `data-analysis`, `physics`,
  `hypothesis-generation`, `numerical-methods`; dropped the redundant `fft`
  in favour of the broader `signal-processing`).

## [v1.2.0] - 2026-06-25

A **minor** release. The theme is **horizontal expansion + hardening**: three
new domain features (time-series analysis, signal-filter design, 2-D
quadrature), an optional pandas interop extra, and a documentation overhaul
(Cookbook, Architecture guide, expanded Tour). Plus numerical-stability fixes
and a no-behavior-change refactor of the two largest source files.

No existing public API is removed or renamed; the `cds[pandas]` adapter is
gated behind an optional extra so the zero-dependency core is untouched.

### Added

- **`cds.stats` — time-series analysis** (`src/cds/stats/time_series.py`):
  - `moving_average`, `exponential_smoothing`, `difference`, `seasonal_decompose`
  - `autocorrelation_function` (sample ACF), `partial_autocorrelation_function` (PACF)
  - `kpss_statistic` (stationarity test) and `ljung_box` (autocorrelation test),
    both returning result objects with `.statistic` / `.p_value`
  - Exported from the top-level `cds.stats` namespace.
- **`cds.signals` — filter design** (`src/cds/signals/filters.py`):
  - `butter_lowpass(order, cutoff)` — Butterworth IIR low-pass coefficient design
  - `apply_filter(signal, coeffs)` — direct-form IIR application
  - `moving_median(signal, window)` — order-statistic denoiser (edge-preserving)
- **`cds.numerical_integration` — 2-D quadrature** (`src/cds/numerical_integration/`):
  - `simpson_2d(f, ax, bx, ay, by, nx, ny)` — 2-D tensor-product Simpson rule
  - `gaussian_quadrature_2d(f, ax, bx, ay, by, n)` — 2-D tensor-product Gauss-Legendre
- **`cds[pandas]` optional extra** (`src/cds/data_analysis/pandas_io.py`):
  - `to_dataframe(data_set)` / `from_dataframe(df)` round-trip a `DataSet` to/from
    a `pandas.DataFrame`, guarded so importing `cds` never requires pandas.
  - Declared as an optional dependency in `pyproject.toml` (`pandas = {version = ...}`).
- **Documentation**:
  - **`docs/cookbook.md`** — a new problem-oriented recipe book (~48 recipes, one
    per concrete task), verified end-to-end against the real API by
    `_verify_cookbook.py`.
  - **`docs/ARCHITECTURE.md`** — layered module dependency graph and data flow.
  - **Tour of Numerical Methods** expanded with 2-D quadrature, time-series, and
    filter-design stops.
  - `docs/index.md` Quick Navigation now links the Cookbook, Tour, and Architecture.

### Changed

- **`cds.modeling` refactor** — `expression.py` (708 lines) split into
  `_base.py` (AST base classes, visitors) and `_nodes.py` (operator/leaf node
  types). Pure reorganization; no public symbols moved or renamed.
- **`cds.stats` refactor** — distribution functions (normal/t/chi²/F CDFs and
  inverses) extracted from `hypothesis_tests.py` into the private
  `_distributions.py`, with `__all__` declared so the re-exports pass
  `mypy --strict`.

### Fixed

- **`cds.math_utils.linalg` — numerical stability of pivoting.** Pivoting now
  uses a scale-relative threshold (`max(|row|) * epsilon`) instead of a fixed
  absolute epsilon, and rejects exact-zero pivots at sub-normal matrix scales.
  Prevents spurious singular-matrix errors and mis-pivoting on
  small-magnitude matrices. No change to well-scaled inputs.
- **`cds.stats`** — `__all__` declared so the `_distributions` re-exports are
  recognized as intentional public surface under `mypy --strict`.
- **Tests** — stabilized a flaky mean-homogeneity property test by anchoring on
  an absolute tolerance instead of a relative one.

## [v1.1.9] - 2026-06-24

### Minor — effect-size measures for the statistics & hypothesis stack

A backward-compatible feature release. Adds standardized **effect-size**
measures to `cds.stats` so users can report the *magnitude* of an effect
alongside its significance — closing the long-standing gap where a
hypothesis test answered *"is there an effect?"* but not *"how large?"*.
No existing API or behavior changes; all prior tests remain green.

### Added

- **`cds.stats` — effect-size measures** (`src/cds/stats/hypothesis_tests.py`):
  - `cohens_d(group_a, group_b)` — Cohen's *d* standardized mean difference
    between two samples (pooled-SD denominator).
  - `eta_squared_from_f(f, df1, df2)` — η² proportion of variance explained
    by group membership, derived from a one-way ANOVA *F* statistic.
  - `cramers_v(contingency_table)` — Cramér's *V* association strength for
    an *r* × *c* contingency table (χ²-based, normalized to `[0, 1]`).
  All three functions are exported from the top-level `cds.stats` namespace and
  documented in the tutorial (`docs/tutorials/hypothesis_tests_demo.md`).
- **Tutorial section** — *"Effect sizes"* walkthrough in
  `hypothesis_tests_demo.md` showing how `cohens_d` / `cramers_v` pair with
  the existing `t_test` / `chi_square_independence` tests.
- **Getting-started snippets** (EN + TR) and `docs/index.md` now surface the
  effect-size functions in the Python-API quick-start and module table.

### Documentation

- README module table, `docs/index.md` Key Features & module description,
  and the EN/TR getting-started guides now mention effect-size support.

## [v1.1.8] - 2026-06-21

### Patch — ODE backward integration bug fix

A user-facing **bug-fix release**. Forward integration is byte-identical
to v1.1.7; only the previously-broken `t_end < t0` case changes behavior.
Includes corrected deep-verification scripts and a docs sync — no new
modules, no API additions.

### Fixed

- **`diffeq.solvers` — backward integration silently returned only the
  initial value.** The fixed-step (`euler`, `rk4`, `midpoint`) and
  adaptive (`rk45`) ODE solvers, plus `solve_system`, used a forward-only
  loop guard `while t < t_end - LOOP_EPSILON`. When `t_end < t0` the
  guard was immediately `False`, so the solvers did nothing and returned
  only `y0`. The integration direction is now derived from
  `copysign(1.0, t_end - t0)` and applied as
  `direction * min(dt, abs(t_end - t0))`; `dt` stays an unambiguous
  positive magnitude and the loop guard becomes
  `(t_end - t) * direction > LOOP_EPSILON`. A zero-length span
  (`t_end == t0`) still skips the loop because `copysign(1, 0) == +1`.
  For `rk45`, step size is tracked as an always-positive `h_mag` so the
  shrink/grow/snap logic keeps its forward semantics in either direction.
  7 new `TestBackwardIntegration` regression tests cover all five
  solvers.

- **`hypothesis.generator` — confidence overflow past `n >= 12`.** The
  generated-confidence formula `0.45 + i*0.05` overflowed the
  `Hypothesis.confidence` `le=1.0` constraint for 12+ hypotheses,
  raising `ValidationError`. Now clamped to `min(0.9, 0.4 + i*0.05)`.

### Changed (verification tooling)

- **`scripts/verify_d1_numerics.py`, `verify_d2_edge_cases.py`,
  `verify_d4_hypothesis.py`** — stale reference values/thresholds that
  produced spurious failures against correct code: D1.3 Romberg
  monotone-decrease check only asserted above `1e-13`; D1.4 RK45
  singularity now uses `t_end=2.0` (past the pole) so `RuntimeError` is
  genuinely triggered; D1.7/D1.2 odd-degree Gauss-Legendre exact
  integral corrected to `0`; Simpson `O(h^4)` thresholds loosened to
  match the method's order; D4.2 confidence expectation updated to the
  clamped formula. All three scripts now report **0 failures**.

### Documentation

- README "recent improvements" → v1.1.8 ODE fix entry; ROADMAP marks
  hatch-vcs / git-cliff as tried-and-abandoned with the real rollback
  reasons; SECURITY supported-versions refresh; dashboard UX
  touch-ups; nav/cross-ref drift. **No code or behavior changes.**

## [v1.1.7] - 2026-06-20

### Patch — PEP 639 SPDX license metadata

A maintenance release with **no API or behavior changes**. Standardizes
the license declaration so it is recognized as a valid SPDX identifier
both on PyPI and in package metadata, and removes drift-prone
hardcoded numbers from the README.

### Changed

- **`pyproject.toml`** — adopted [PEP 639](https://peps.python.org/pep-0639/):
  replaced the legacy `license = {text = "MIT"}` table form with the
  SPDX `license = "MIT"` expression plus `license-files = ["LICENSE"]`.
  The wheel/sdist METADATA now emit `License-Expression: MIT` and
  `License-File: LICENSE`, which PyPI and GitHub recognize as a valid
  SPDX id (the old table form was not parsed, so the license read as
  unrecognized). Removed the now-redundant `License :: OSI Approved ::
  MIT License` classifier (deprecated under PEP 639; would trigger a
  metadata warning).

### Documentation

- **`README.md`** — removed hardcoded, drift-prone numbers (version
  tag, test count, module count) in favor of the existing PyPI / CI /
  codecov badges, so the docs no longer go stale between releases.

## [v1.1.6] - 2026-06-20

### Patch — release pipeline fix (OIDC → scoped API token)

A maintenance release with **no API or behavior changes**. Fixes the
broken automated release pipeline: the previous Trusted Publishing (OIDC)
path required a PyPI publisher configuration that was never set up, so
the v1.1.5 tag push failed at the upload step with an `invalid-publisher`
error. The pipeline now uses a scoped **PyPI API token** (stored as the
`PYPI_API_TOKEN` repo secret), which is the sole publish authority.

This is the first release actually published by the automated pipeline —
v1.1.4 and v1.1.5 were uploaded manually before the pipeline existed.

### Changed

- **`release.yml`** — removed the `id-token: write` permission and the
  Trusted Publishing path; publishes via `pypa/gh-action-pypi-publish`
  with `password: ${{ secrets.PYPI_API_TOKEN }}`. The workflow now also
  creates the GitHub Release (previously done by the local script).
- **`scripts/publish.py`** — no longer uploads to PyPI or creates the
  GitHub release. It now builds, verified the built version against the
  tag, runs tests, and pushes the tag (which triggers CI). Removed the
  `--skip-release` flag and the local `~/.pypi-token` handling.
- **`pyproject.toml`** — dropped `twine` from the `[all]` and `[dev]`
  extras (no longer needed locally).
- **Version bump `1.1.5` → `1.1.6`** in `pyproject.toml`,
  `src/cds/_version.py`, and `CITATION.cff`.

### Documentation

- **`README.md`** and **`docs/maintenance.md`** — updated the release
  pipeline description to reflect the scoped-API-token flow and
  `release.yml` as the sole publish authority.

## [v1.1.5] - 2026-06-20

### Patch — quality, type-safety, test depth, and a 100% coverage gate

A patch release: no API changes, no behavior changes. A coordinated sweep
of refactoring (type-safety, code deduplication), deeper testing
(property-based invariants, shared fixtures), expanded documentation
(tutorials, architecture guide), and a hardening of CI to enforce full
blended coverage. The result is the cleanest, most thoroughly verified
release to date — **17 modules, 1230 tests, and 100% code coverage
(statement + branch)**.

### Added

- **Python 3.13 support** — added 3.13 to the CI test matrix and the
  `pyproject.toml` classifiers; the full suite is now green on 3.10–3.13.
- **Shared test fixtures** — `tests/conftest.py` plus `tests/test_core.py`
  centralize reusable fixtures and core invariants, reducing duplication
  across the suite.
- **Property-based invariant suite (B5)** — a Hypothesis-driven test set
  that exercises numerical invariants (associativity, identity,
  round-trip) over generated inputs, catching edge cases that
  example-based tests miss.
- **Tutorials for optimization, signals, ML, and statistics (B3)** — new
  guided walkthroughs under `docs/tutorials/` covering the four most-used
  subsystems.
- **Contributing architecture section (B4)** — `CONTRIBUTING.md` now
  documents the module layout and the data flow between subsystems, with
  stale cross-references fixed.
- **`examples/modeling_demo.py`** — a new end-to-end example demonstrating
  the modeling workflow referenced by the tutorials.

### Changed

- **Closed remaining `Any` type escapes** — the last untyped boundaries in
  the core and hypothesis modules now carry concrete types; `mypy --strict`
  stays green.
- **Generic `OptResult`** — the optimization result type is now generic
  over its scalar type, and the narrowing casts that worked around the old
  concrete signature have been removed.
- **Deduplicated scalar central-difference** — the scalar central-
  difference implementation that existed in both `calculus` and
  `optimization` is now shared from a single source.
- **Version bump `1.1.4` → `1.1.5`** in `pyproject.toml`,
  `src/cds/_version.py`, and `CITATION.cff`.

### CI

- **100% blended coverage gate** — CI now fails unless both statement and
  branch coverage reach 100%, locking in the coverage level as a
  non-regression boundary.
- **Regenerated benchmark report** — the benchmark report and
  `results.json` were regenerated against the current codebase.

### Documentation

- **Synced stale test count and coverage** — corrected the documented test
  count (1165 → 1230) and coverage figure (~99.6% → 100%) to match CI.
- **Roadmap items shipped** — roadmap entries #2 (modeling) and #3
  (knowledge organization) are marked released; the corresponding issues
  are closed.

## [v1.1.4] - 2026-06-19

### Patch — documentation, CI, and housekeeping cleanup

A patch release: no API changes, no behavior changes. Removes cosmetic
distractions (emojis, sprint references) from issue templates and docs,
fixes CI workflow schemas, and deletes ~3 500 lines of no-op autograd
shim code that was never exercised at runtime.

### Changed

- **Issue templates** — removed decorative emojis from
  `bug_report.md`, `config.yml`, and `changelog.yml` for a cleaner,
  more professional appearance.
- **`.github/labeler.yml`** — migrated to the `actions/labeler` v5
  configuration schema (`triage` → `labels` key).
- **`.github/workflows/changelog.yml`** — fixed a missing trailing
  newline flagged by lint checks.
- **Documentation** — stripped sprint-tracking references and
  ephemeral planning files (`docs/superpowers/plans/`,
  `docs/superpowers/specs/`); unified inline-code formatting across
  tutorials and case-study pages.
- **Removed no-op autograd shims** — deleted `src/cds/nlp/autograd/`
  shim functions (`_grad.py` stubs, pass-through wrappers in
  `ops.py` and `tensor.py`) that were placeholders and never
  participated in any real gradient computation. ~3 500 lines removed.
- **Examples** — aligned `ruff` formatting in
  `examples/data_analysis_demo.py` and
  `examples/ml_and_viz_demo.py`.
- **Version bump `1.1.3` → `1.1.4`** in `pyproject.toml`,
  `src/cds/_version.py`, and `CITATION.cff`.

## [v1.1.3] - 2026-06-19

### Patch — type-safety cleanup

A patch release: no API changes, no behavior changes. Resolves the final
block of `mypy --strict` errors so the type checker reports **0 errors
across 136 files**. No public surface changed; all 1165 tests pass and
the suite is unchanged.

### Fixed

- **43 `mypy --strict` errors resolved** across `examples/`, `dashboard/`,
  and `scripts/`. The fixes fall into five categories, none of which
  alter runtime behavior:
  - **`int` → `float` argument types** — numeric literals passed to APIs
    typed as `float` (e.g. integration tolerances, attention scaling)
    now use explicit `float` literals so the call matches the signature.
  - **`Domain` enum usage** — iteration and membership checks against the
    `Domain` enum now go through `Domain.__members__.values()` / explicit
    member references, matching the pattern established in v1.1.2 for
    CodeQL compatibility.
  - **Attention tensor rank** — reshape/transpose calls in
    `examples/nlp_attention_demo.py` and `examples/nlp_mini_gpt_demo.py`
    annotated to reflect the actual tensor rank rather than an
    over-narrow `Any`/scalar guess.
  - **Invariant list widening** — a few demo helper functions that
    return heterogeneous sequences now declare `list[object]` (or the
    specific union) instead of an impossible precise element type.
  - **`Protocol` signature alignment** — a duck-typed callback protocol
    in `scripts/publish.py` had a signature that diverged from its
    concrete implementations; the protocol and call sites now agree.

### Changed

- **Version bump `1.1.2` → `1.1.3`** in `pyproject.toml` and
  `src/cds/_version.py`.

## [v1.1.2] - 2026-06-19

### Patch — security hardening & CodeQL closure

A patch release: no API changes, no behavior changes. Closes the remaining
CodeQL code-scanning alerts with a fix (not a workaround),
tightens `main` branch protection, and improves the discoverability of the
vulnerability-reporting flow. Brings the repo to **0 open** code-scanning
alerts (22/22 closed: 20 fixed by code, 2 dismissed as documented false
positives).

### Security

- **CodeQL `py/non-iterable-in-for-loop` — fix.** Enum iteration
  in `examples/core_demo.py` and `tests/test_hypothesis.py` now uses
  `Domain.__members__.values()` instead of `for x in Domain:`. This is the
  Enum API's standard member-collection view: functionally identical (same
  members, same insertion order — all 1164 tests pass) but the return type
  is an explicitly iterable mapping view that CodeQL's type resolver can
  follow. Previous attempts (idiomatic iteration, direct import from the
  defining module) did not satisfy the analyzer; this does, with no
  camouflage (`list()` wrapping) and no behavior change.
- **`main` branch protection strengthened.** Added `required_status_checks`
  for `CI` and `CodeQL` — a PR can no longer merge into `main` while those
  checks are red. `enforce_admins` is left disabled so the maintainer
  retains a direct-push path; everything else (linear history, conversation
  resolution, no force-pushes, no deletions) is unchanged.
- **Security-reporting guidance.** The README now links to `SECURITY.md`
  and the private-advisory flow; the bug-report issue template warns against
  opening public issues for vulnerabilities and redirects to private
  advisories.

### Changed

- **`CITATION.cff` version refresh.** Two stale `1.1.0` references (root and
  `preferred-citation` blocks) bumped to `1.1.2`. This was missed in v1.1.1.

## [v1.1.1] - 2026-06-19

### Patch — supply-chain & CI hardening

A patch release: no API changes, no behavior changes. Strengthens release
integrity and CI correctness so downstream consumers and contributors get
verifiable, low-churn workflows.

### Security

- **CodeQL static analysis** (`codeql.yml`) — runs `security-and-quality`
  query pack on every push/PR into `main` plus a weekly schedule; findings
  surface as code-scanning alerts under the Security tab. Complements
  `attest.yml` (which signs release artifacts with build provenance) and
  `dependabot.yml` (dependency CVEs).
- **README security badge** — live CodeQL status added next to the CI badge.

### Changed

- **`changelog.yml` made idempotent** — all dispatches now target a single
  fixed branch (`chore/changelog-regen`); `peter-evans/create-pull-request`
  updates the existing open PR instead of opening a duplicate per tag.
  Previously up to 6 stale changelog PRs accumulated; now only one is ever
  open. Also bumped the action `v6 → v7` and added a `concurrency` group to
  serialize simultaneous dispatches.
- **`benchmarks.yml` respects branch protection** — switched from a direct
  bot push (rejected on protected `main`) to opening a PR via
  `peter-evans/create-pull-request@v7`, so regenerated artifacts merge
  through the normal review flow without breaking protection rules.

### Removed

- 6 stale auto-generated changelog PRs (#10, #19, #20, #21, #22, #23) and
  their per-tag branches — all superseded by the hand-curated CHANGELOG on
  `main`.

## [v1.1.0] - 2026-06-19

### Minor — two new modules (symbolic math + knowledge organization)

A backward-compatible feature release. Adds two new public subpackages
(`cds.modeling`, `cds.knowledge`) without changing any existing API. The
platform now spans **17 modules**, **1164 tests**, and **99.59%** coverage.

### Added

- **`cds.modeling`** — symbolic algebra for equation development:
  - An expression tree (`+`, `-`, `*`, `/`, `**`, unary `-`, variables,
    numbers) built by overloading Python operators on a `Variable`/`Constant`
    AST.
  - Symbolic differentiation (`diff`), algebraic simplification
    (`simplify`), substitution (`subs`), evaluation (`evaluate`), and
    LaTeX export (`to_latex`).
  - A `MathModel` for equation systems with `solve_equation` (Newton-Raphson
    root finding) and `fit_parameters` (least-squares parameter fitting to
    observations).
  - Runnable demo (`examples/modeling_demo.py`) + tutorial.
- **`cds.knowledge`** — a knowledge-organization layer in three pure-Python,
  dependency-free files:
  - `graph.py`: `Concept`, `Relation` (typed directed edges), and a
    `KnowledgeGraph` with undirected traversal (shortest path via BFS,
    transitive closure, cycle detection), neighbors, and JSON persistence.
  - `notes.py`: `Note` + `Notebook` research notebook with tag and
    concept lookups, and JSON persistence.
  - `retrieval.py`: `search()` producing ranked `SearchResult` hits across
    both concepts and notes (matched field + normalized score).
  - Runnable demo (`examples/knowledge_demo.py`) + tutorial.
- Both modules are wired into the package `__all__`, the CLI `modules`
  listing, and the API reference (`docs/api.md`).

### Maintenance

- Docs: resync README, `docs/index.md`, getting-started (EN + TR), and
  CITATION.cff to v1.1.0 — module count 16→17, test count 883→1164,
  coverage 99.48%→99.59%. Keep `pyproject.toml` + `src/cds/_version.py`
  lockstepped at 1.1.0.
- Chore(repo): ignore the `_demo_*.json` runtime artifacts the
  knowledge demo writes next to its script, so a demo run no longer
  dirties the working tree.

## [v1.0.4] - 2026-06-18

### Security & Tooling — CI hardening

A follow-up patch to v1.0.3. No behavior changes to the published package;
fully backward compatible. Adds continuous dependency-vulnerability scanning
to CI, tightens the type-check configuration, and resynchronizes docs that
had drifted after v1.0.0.

### Maintenance

- Chore(ci): add a `pip-audit` job to the CI workflow (`tests.yml`). Runs
  once on the reference cell (Linux + Python 3.12) against the PyPI Advisory
  database. Currently `continue-on-error: true` — the dev/docs/test toolchain
  (jupyter, mkdocs, build, twine, pillow, lxml, ...) carries transitive deps
  with open CVEs unrelated to the published runtime (typer/pydantic/rich,
  which audit clean). Surfacing them every run gives visibility without
  red-flashing the matrix on a fresh advisory.
- Chore(types): remove the global `ignore_missing_imports = true` from
  `[tool.mypy]`. CDS only depends on PEP 561 typed runtime libs (typer,
  pydantic, rich), all of which resolve cleanly under strict mypy (98 files,
  0 errors). Dropping the escape hatch means a future un-stubbed dependency
  surfaces a real error instead of being silently typed as `Any`.
- Chore(repo): delete a stray Windows `nul` artifact left in the working tree.
- Docs: resync README + `docs/` after v1.0.0 — test count 845→878, version
  strings updated, `CITATION.cff` bumped to 1.0.4. Keep `pyproject.toml` +
  `src/cds/_version.py` lockstepped at 1.0.4.

## [v1.0.3] - 2026-06-18

### Hygiene — test-suite type correctness

A follow-up patch to v1.0.2. No behavior changes to the published package;
fully backward compatible. Brings `tests/` to the same mypy baseline as
`src/` so the full type-check (`mypy src/ tests/`) is green, and resolves
post-release lint/test/publish findings caught after v1.0.2 shipped.

### Testing

- Test(types): clear all remaining mypy errors across `tests/` (39 files,
  0 errors). Fixes list-variance issues (`Parameter <: Tensor`), redundant
  `type: ignore` comments, and intentional error-path `**` operands.
- Test(benchmarks): isolate benchmark test artifacts — `run_all()` /
  `_write_json()` accept an `output_dir` parameter so `pytest` no longer
  clobbers the committed `benchmarks/results.json`.

### Maintenance

- Chore(ci): resolve post-1.0.2 lint/test/publish findings (commit 5e429ec).
- Docs: keep `pyproject.toml` + `src/cds/_version.py` lockstepped at 1.0.3.

## [v1.0.2] - 2026-06-18

### Hygiene — test isolation for benchmark artifacts

A follow-up patch to v1.0.1. No behavior changes to the published package;
fully backward compatible. Resolves a repo-hygiene issue where running the
benchmark test suite silently clobbered the committed
`benchmarks/results.json` artifact (and `docs/benchmarks.md`) on every
`pytest` invocation.

### Bug Fixes

- Fix(tests): `test_run_all_generates_report` no longer writes into the
  working tree. `run_all()` and `_write_json()` now accept an optional
  `output_dir` parameter; the test passes its `tmp_path`, so the committed
  `benchmarks/results.json` and `docs/benchmarks.md` are never modified by a
  test run. Verified: object hash of `results.json` is byte-identical before
  and after `pytest`.

### Testing

- Test(benchmarks): added a regression assertion that `results.json` lands
  in `tmp_path` (not `benchmarks/`) after `run_all(output_dir=...)`.

### Features

- Feat(benchmarks): `run_all(output_dir=None)` and
  `_write_json(record, output_dir=None)` — new optional parameter for
  redirecting artifact output. Defaults preserve existing CLI behaviour.

## [v1.0.1] - 2026-06-18

### Maintenance — typed, documented, hardened

A follow-up to the v1.0 stable release focused on packaging quality and
numerical correctness guarantees. No behavior changes; fully backward
compatible.

### Features

- Feat(packaging): declare PEP 561 type information — ship `src/cds/py.typed`
  marker and force-include it in the wheel so downstream users get full
  static type-checking support out of the box.
- Feat(docs): add "type checked" badge to the README.

### Testing

- Test(invariants): add `tests/test_numerical_invariants.py` — 32
  property-based numerical invariants across 8 modules (linalg, signals,
  stats, quadrature, quantum, diffeq, monte carlo, probability). Fixed
  seed, fully reproducible, zero new dev dependencies.

### Documentation

- Docs(api): add docstrings to all 51 previously-undocumented public
  functions identified via AST scan (3 empty-docstring gaps closed).

### Bug Fixes

- Chore: bump version to 1.0.1.

## [v1.0.0] - 2026-06-18

### 1.0 — Stable release

Cognitive Discovery Platform reaches its first stable release. All five
subsystems of the v1.0 completion spec are implemented, gated, and
verified: 99.16% statement coverage (845 tests passing), zero mypy
errors, clean ruff, and a strict mkdocs build.

### Features

- Feat(nlp): NLP visualisation module — attention heatmap, PCA projection, training curve ([7956597](7956597))
- Feat(benchmarks): emit results.json with timestamp + git SHA provenance, add CI workflow ([4e18500](4e18500))

### Testing

- Test(coverage): close NLP edge-case gaps — tensor/ops/optim/bpe/layers/model/attention; reach 99.16% statement coverage ([6f0a487](6f0a487))
- Style(tests): ruff format cleanup — collapse over-split listcomp, expand train() signature, remove duplicate inline imports ([36f4ee4](36f4ee4))

### Documentation

- Docs(examples): add 9 module demos + matching tutorials ([9f00fc5](9f00fc5))
- Docs(api): restructure mkdocs nav, add api.md lead-ins, fix stale test counts ([4cad8be](4cad8be))
- Docs(nlp): fix griffe warnings — split combined params in docstrings ([3aedff4](3aedff4))
- Docs(benchmarks): regenerate results.json with current SHA provenance ([69c5a88](69c5a88))

### Bug Fixes

- Chore: bump version to 1.0.0 + Production/Stable classifier; sync _version.py

### Other

- Promoted from `Development Status :: 4 - Beta` to `Development Status :: 5 - Production/Stable`.

## [v0.10.0b1] - 2026-06-17


### Features


- Feat(nlp): autograd engine + MiniGPT-from-scratch ([5e0d308](5e0d308))


### Documentation


- Docs(changelog): regenerate for v0.9.0b8 (#17) ([2b0a7d3](2b0a7d3))


## [v0.9.0b8] - 2026-06-17


### Features


- Feat(nlp): add multi-head attention + transformer block ([4f38cae](4f38cae))


### Documentation


- Docs(changelog): regenerate for v0.9.0b7 (#16) ([44ec93c](44ec93c))


## [v0.9.0b7] - 2026-06-17


### Bug Fixes


- Fix: catch two more stale surfaces (CITATION.cff, mkdocs api.md) ([b59b6e3](b59b6e3))


### Documentation


- Docs(changelog): regenerate for v0.9.0b6 (#15) ([e1244ce](e1244ce))


## [v0.9.0b6] - 2026-06-17


### Bug Fixes


- Fix: sync test count 572 to 655 across all surfaces ([952ffe0](952ffe0))


### Documentation


- Docs(changelog): regenerate for v0.9.0b5 (#14) ([f7a037b](f7a037b))


## [v0.9.0b5] - 2026-06-17


### Features


- Feat(nlp): add cds.nlp module — BPE tokenizer + sinusoidal embeddings ([045b9d1](045b9d1))


### Other


- Build: drop hatch-vcs + setuptools_scm, switch to static versioning ([6e04d31](6e04d31))


### Documentation


- Docs(changelog): regenerate for v0.9.0b4 (#13) ([da52ee0](da52ee0))


## [v0.9.0b4] - 2026-06-17


### Bug Fixes


- Fix: drop stale 'auditor' comment + sync 570 to 572 across docs and cli ([0dcaba6](0dcaba6))


### Documentation


- Docs(changelog): regenerate for v0.9.0b4 (#11) ([b1eda78](b1eda78))


## [v0.9.0b3] - 2026-06-17


### Bug Fixes


- Fix(docs): include getting-started.tr.md in mkdocs nav; fix broken CONTRIBUTING link ([85316fe](85316fe))

- Fix: bump CITATION.cff top-level version to 0.9.0b2 ([80cc3b4](80cc3b4))

- Fix: bump CITATION.cff top-level version to 0.9.0b2 (was 0.8.5, missed during release) ([adee53c](adee53c))

- Fix: keep cast() in test_legendre (Python 3.10/3.11 mypy needs it, 3.12 ignores) ([bcd20fc](bcd20fc))

- Fix: resolve 34 mypy errors (matrix/fft type unions, plot float types, missing confidence arg, cli.Path) ([eea36a3](eea36a3))

- Fix: type local _legendre import as Callable[[int, float], tuple[float, float]] (cross-Python mypy fix) ([486bad5](486bad5))

- Fix: use # type: ignore[no-any-return] instead of cast() for cross-Python compat ([bd3aaa8](bd3aaa8))


### Other


- Build: add setuptools_scm local-scheme config (no-local-version, fallback) ([82d07c9](82d07c9))

- Build: exclude hatch-vcs generated _version.py from ruff ([76760d3](76760d3))

- Build: publish.py ignores hatch-vcs regenerated _version.py in dirty check ([bea5f71](bea5f71))

- Build: setuptools_scm version_scheme=no-guess-dev; fix wheel version regex ([8ccb292](8ccb292))

- Release: v0.9.0b1 - Beta milestone, CI multi-OS, hatch-vcs, git-cliff, attestation ([3e3928f](3e3928f))

- Release: v0.9.0b2 - same code, re-tag to trigger attest workflow (checkout fix) ([c54e900](c54e900))

- Release: v0.9.0b3 - branch protection, threat model, i18n, signed commits guide ([31983fe](31983fe))


### Documentation


- Docs(changelog): regenerate for v0.9.0b1 ([44f7823](44f7823))

- Docs(changelog): regenerate for v0.9.0b2 ([404532c](404532c))

- Docs(changelog): regenerate for v0.9.0b3 (#8) ([3a8ef95](3a8ef95))

- Docs(changelog): regenerate for v0.9.0b3 (#9) ([c4fcb9b](c4fcb9b))

- Docs: add threat model, signed commits guide, Why CDS comparison, Turkish i18n; update ROADMAP ([0ba07cf](0ba07cf))

- Docs: update SECURITY.md supported versions (0.6.x → 0.8.x) ([b42cf5f](b42cf5f))


### Styling


- Style: pre-commit auto-fix (trailing whitespace, import order, ruff format) ([9adbf5a](9adbf5a))

- Style: pre-commit auto-fix _version.py (add blank line before __version__) ([277b535](277b535))


### Miscellaneous Tasks


- Ci: changelog PR action needs explicit base branch (checked out on tag) ([b96d322](b96d322))

- Ci: changelog action checks out main (not tag) to avoid detached HEAD issues ([af3cd12](af3cd12))

- Ci: changelog workflow needs write contents + pull-requests perms ([a319a9f](a319a9f))

- Ci: changelog workflow uses PR instead of direct push (branch protection compat) ([3f593cb](3f593cb))

- Ci: fix attest + changelog workflows (checkout step, detached HEAD push) ([d2c9d6e](d2c9d6e))

- Ci: quote step name with colon (YAML parse error) ([0731ca8](0731ca8))

- Ci: rewrite tests.yml with simple bash conditional (no GitHub conditional parse issues) ([2cc8483](2cc8483))

- Ci: split test step into 2 by matrix (gate vs report-only) - fix missing fi ([9e68ff0](9e68ff0))


## [v0.8.5] - 2026-06-16


### Other


- Release: v0.8.5 - cds[all] extras, auto-generated API ref, docs deploy ([12b03b0](12b03b0))


## [v0.8.4] - 2026-06-16


### Other


- Release: v0.8.4 - pydantic upgrade ([58ee8d9](58ee8d9))


## [v0.8.3] - 2026-06-16


### Other


- Release: v0.8.3 - example guard consistency ([5017680](5017680))


## [v0.8.2] - 2026-06-16


### Other


- Release: v0.8.2 - docstrings, knowledge/ removal, CHANGELOG backfill ([f22150c](f22150c))

- Scripts: replace publish.ps1 with pure-Python publish.py ([c7c08e2](c7c08e2))


### Miscellaneous Tasks


- Ci: fix pre-commit and coverage gate for stable CI ([ca66c9f](ca66c9f))

- Ci: remove Docs workflow; relax coverage gate; re-enable legacy Pages ([cdcf5d0](cdcf5d0))


## [v0.8.1] - 2026-06-16


### Other


- Release: v0.8.1 ([4dc49e2](4dc49e2))


### Documentation


- Docs(roadmap): link open enhancement issues #2 and #3 under Longer Term ([ee6103b](ee6103b))

- Docs: mark v0.8.0 as released; move 3 pending perf items to v0.8.1 ([80e6afe](80e6afe))


### Testing


- Test: hit 100% statement coverage; keep branch coverage off ([5c606c4](5c606c4))


### Miscellaneous Tasks


- Ci(pypi): fix pypi-publish workflow ([8335cb6](8335cb6))

- Ci(scripts): add publish.ps1 for local PyPI releases ([57e81a4](57e81a4))

- Ci: harden repo to 10/10 - pre-commit, lockfiles, coverage gate ([7c3b18f](7c3b18f))

- Ci: remove pypi-publish workflow ([043b7f5](043b7f5))


## [v0.8.0] - 2026-06-16


### Other


- Squashed: collapse 108 commits into a single cohesive history ([d4e8aee](d4e8aee))

- Release: v0.8.0 - Performance & Benchmarks ([83211ef](83211ef))


<!-- generated by git-cliff -->
