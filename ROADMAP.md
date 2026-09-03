# Roadmap

Planned direction for CDS. All work maintains the **zero-dependency, pure Python** philosophy.

## v0.7.0 — Coverage & Polish (Released)

- [x] ~~Push test coverage to **97%+** across every module~~: achieved **100%**
- [x] ~~Add `typing.Protocol`-based extension points for custom hypothesis generators~~: shipped as `HypothesisGenerator` Protocol
- [x] ~~Document the PyPI package name (`cognitive-discovery-platform`) vs repo name (`scientific-computing-system`) distinction in CONTRIBUTING.md~~: superseded: the package was renamed to `scientific-computing-system`, so PyPI and repo names now match (see §"Package Name")
- [x] ~~Replace remaining generic error messages with actionable guidance~~: updated 11 messages in linalg, signals, and stats modules

## v0.8.x — Performance & Benchmarks (Released 2026-06-16)

- [x] ~~Publish automated benchmark suite (FFT speed, Monte Carlo convergence, LU decomposition vs naive)~~: see `benchmarks/run_benchmarks.py` & `docs/benchmarks.md`
- [x] ~~Add `cds[all]` meta-extras~~ (single-install convenience)
- [x] ~~Bump pydantic 2.12 → 2.13~~ (security patch)

## v0.9.x — Beta Milestone (Released 2026-06-17)

- [x] ~~Classifier `3 - Alpha` → `4 - Beta`~~ (maturity signal)
- [x] ~~Multi-OS CI matrix~~ (Linux + Windows + macOS × Python 3.10/3.11/3.12)
- [x] ~~Branch coverage enabled~~ (`branch = true` in coverage config)
- [x] ~~Mypy `--strict` on both `src/` and `tests/`~~ (0 issues, 73 files)
- [~] ~~Hatch-vcs dynamic versioning~~: **tried and abandoned**: hatch-vcs 0.5.0 silently ignored the `version-scheme = "release"` override, so static versioning (`version` in `pyproject.toml` mirrored in `src/cds/_version.py`) was shipped instead. See `pyproject.toml` release checklist.
- [~] ~~Git-cliff auto-CHANGELOG on tag push~~: **rolled back to manual**: the tag-push trigger rewrote the entire `CHANGELOG.md` from commit history and discarded hand-curated narrative entries (see PR #24). `changelog.yml` is now `workflow_dispatch` only; `CHANGELOG.md` is hand-curated.
- [x] ~~Sigstore release attestation~~ (`actions/attest-build-provenance@v2`)
- [x] ~~Branch protection on main~~ (1 PR reviewer, linear history, no force push, no deletes)
- [x] ~~Threat model in SECURITY.md~~ (in-scope vs out-of-scope, user best practices)
- [x] ~~Türkçe başlangıç rehberi~~ (`docs/getting-started.tr.md`)
- [x] ~~Signed commits guide in CONTRIBUTING.md~~ (SSH or GPG)

## v1.0.0 — Stability RELEASE (Released 2026-06-18)

- [x] ~~Freeze public API: backward-compatible guarantees for all `cds.*` exports~~
- [x] ~~Full API reference documentation on GitHub Pages (auto-generated via mkdocstrings)~~
- [x] ~~Type stubs (`.pyi`) for IDE autocompletion~~ (skipped: pure-Python with full type hints doesn't need them per project policy)
- [x] ~~Security audit~~ (dependency pinning, signed releases: `requirements.lock` pinned, sigstore attestation live)
- [x] ~~**Mark as stable**: remove alpha/beta labels, bump to 1.0.0~~ (Development Status `5 - Production/Stable`)
- [ ] Optional: enable `required_signatures` in branch protection (after maintainer configures GPG/SSH signing key)
- [ ] Optional: Enable Dependabot security-only updates (currently weekly, can switch to security-only mode)
- [x] ~~Automated benchmark regeneration on releases + weekly schedule~~ (`.github/workflows/benchmarks.yml`): CI artifact `benchmarks/results.json` for regression tracking

## v1.0.1 – v1.0.4 — Post-Stable Hardening (Released 2026-06-18)

Backward-compatible patch train after the stable cut. No API or behavior changes.

- [x] ~~**v1.0.1**: PEP 561 type marker~~ (`src/cds/py.typed` shipped in the wheel) + 32 property-based numerical invariants (`tests/test_numerical_invariants.py`) + docstrings for all 51 previously-undocumented public functions (AST-scanned, 3 gaps closed)
- [x] ~~**v1.0.2**: benchmark artifact isolation~~: `run_all(output_dir=...)` so `pytest` no longer clobbers the committed `benchmarks/results.json` / `docs/benchmarks.md` (byte-identical verified)
- [x] ~~**v1.0.3**: `tests/` brought to the same mypy baseline as `src/`~~ (`mypy src/ tests/` green across 39 test files) + benchmark test artifact isolation in `tmp_path`
- [x] ~~**v1.0.4**: CI pip-audit job + drop the global `ignore_missing_imports = true`~~ from `[tool.mypy]` so future un-stubbed dependencies surface real errors instead of silently typing as `Any`

## v1.1.0 — Modeling & Knowledge Modules (Released 2026-06-19)

A backward-compatible **minor** release. Adds two new public subpackages; the platform now spans 17 modules. Tracked under issues [#2](https://github.com/Furox-Art/scientific-computing-system/issues/2) and [#3](https://github.com/Furox-Art/scientific-computing-system/issues/3).

- [x] ~~**`cds.modeling`**: symbolic algebra~~: expression-tree AST (`+ - * / **`, unary `-`, `Variable`/`Constant`), `diff`, `simplify`, `subs`, `evaluate`, `to_latex`; `MathModel` with `solve_equation` (Newton-Raphson) and `fit_parameters` (least squares). See `examples/modeling_demo.py`.
- [x] ~~**`cds.knowledge`**: knowledge organization layer~~: `KnowledgeGraph` (`Concept`/`Relation`, BFS shortest path, transitive closure, cycle detection, JSON persistence), `Notebook` (`Note` with tag/concept lookups), `retrieval.search()` ranked across concepts+notes. See `examples/knowledge_demo.py`.
- [x] ~~Both modules wired into `__all__`, CLI `modules` listing, and `docs/api.md`~~

## v1.1.1 – v1.1.7 — Polish & Release Pipeline (Released 2026-06-19 → 2026-06-20)

Backward-compatible patches. Highlights:

- [x] ~~**v1.1.5**: type-safety sweep + 100% coverage gate~~: blended statement+branch coverage at 100% across 1230 tests, property-based invariant tests, shared fixtures, expanded docs (tutorials + architecture guide)
- [x] ~~**v1.1.6**: release pipeline fix~~: switched from Trusted Publishing (OIDC, never configured) to a scoped `PYPI_API_TOKEN`; first release actually published by the automated pipeline
- [x] ~~**v1.1.7**: PEP 639 SPDX license metadata~~: `license = "MIT"` expression replaces the legacy table form so PyPI/GitHub recognize the license; removed drift-prone hardcoded numbers from README

## v1.1.7 – v1.1.8 — Bug-fix patches (Released 2026-06-20 → 2026-06-21)

- [x] ~~**v1.1.7**: PEP 639 SPDX license metadata~~ (`license = "MIT"` expression; license-files declared; classifier drift removed)
- [x] ~~**v1.1.8**: ODE backward integration bug fix~~ (see CHANGELOG for full details) + hypothesis confidence overflow clamp

## v1.1.9 — Effect-size measures for stats & hypothesis (Released 2026-06-24)

A backward-compatible **minor** release. Adds standardized effect-size
measures to `cds.stats` so significance tests can be paired with a
quantitative magnitude estimate.

- [x] ~~`cds.stats.cohens_d(group_a, group_b)`~~: Cohen's *d* standardized mean difference (pooled-SD denominator)
- [x] ~~`cds.stats.eta_squared_from_f(f, df1, df2)`~~: η² proportion of variance explained by group membership (derived from one-way ANOVA *F*)
- [x] ~~`cds.stats.cramers_v(contingency_table)`~~: Cramér's *V* association strength for contingency tables (χ²-based, ∈ `[0, 1]`)
- [x] ~~Tutorial section in `docs/tutorials/hypothesis_tests_demo.md`~~: "Effect sizes" walkthrough pairing the measures with the existing `t_test` / `chi_square_independence`
- [x] ~~Docs sync~~: README module table, `docs/index.md` Key Features + stats description, EN/TR getting-started Python-API snippets

## v1.2.0 — Multi-module expansion (Released 2026-06-25)

A backward-compatible **minor** release. Theme: horizontal expansion of the
domain toolkits plus a documentation overhaul, with zero-dependency guarantees
preserved (pandas is an opt-in extra). The platform now spans **18 modules**.
See `CHANGELOG.md` for full detail.

- [x] ~~**`cds.stats` time-series analysis**~~: ACF/PACF, KPSS stationarity test,
      Ljung-Box autocorrelation test, moving average / exponential smoothing /
      differencing / seasonal decomposition
- [x] ~~**`cds.signals` filter design**~~: `butter_lowpass` IIR design +
      `apply_filter` + `moving_median` denoiser
- [x] ~~**`cds.numerical_integration` 2-D quadrature**~~: tensor-product Simpson
      and Gauss-Legendre rules
- [x] ~~**`cds[pandas]` optional extra**~~: `to_dataframe` / `from_dataframe`
      interop bridge, gated behind the extra so the core stays dependency-free
- [x] ~~**Refactors**~~: `modeling/expression.py` split into `_base.py` +
      `_nodes.py`; stats distribution functions extracted into `_distributions.py`
- [x] ~~**Numerical-stability fixes**~~: scale-relative pivoting in `linalg`,
      exact-zero pivot rejection at sub-normal scales
- [x] ~~**Documentation**~~: new Cookbook (`docs/cookbook.md`), Architecture
      guide (`docs/ARCHITECTURE.md`), expanded Tour of Numerical Methods,
      README/`index.md` cross-links

## v1.3.0 — Visualization & Ergonomics (Released 2026-07-20)

- [x] ~~**`cds[plot]` optional extra**~~ (+ v1.3.2 chart expansion)
- [x] ~~**CLI PNG export**~~: `cds plot … --file / --kind`
- [x] ~~**Companion notebook**~~: `examples/plotting_notebook.ipynb`

## v1.4.0 — Core APIs & CLI (Released 2026-07-20)

- [x] ~~Probability CDFs & samplers, Spearman/percentile/z-scores, physics formulas, CLI stats/sample~~

## v1.5.0 — Algorithms wave (Released 2026-07-20)

- [x] ~~**Graph**~~: Bellman–Ford, Floyd–Warshall, Prim MST, components, degree
- [x] ~~**Monte Carlo**~~: `mc_expectation`, `hit_or_miss`
- [x] ~~**Linear algebra**~~: vector / Frobenius norms, matrix trace
- [x] ~~**CLI**~~: `cds integrate`

### Still open

- [ ] **More Jupyter notebook templates**: additional Cookbook / Tour notebooks
- [ ] **Issue #62**: lift the `numpy<2.5` dev-pin once `mypy`'s default
      `python_version` advances to 3.13+ (tracking issue; dev-only, no runtime
      impact).

## Longer Term

Open ideas, not version-committed. Contributions welcome.

- Community-contributed domain modules (bioinformatics, finance)
- **CDS Script Templates**: domain-specific scientific workflows (quantum chemistry, signal processing demos)

### Completed tracks (graduated out of this section)

- **Educational NLP track**: shipped across v0.9.0b5 → v1.0.0 (now `cds.nlp`): BPE tokenizer + embeddings, multi-head self-attention, pure-Python autograd with optional `cds[fast-jit]` Numba backend, tiny GPT-from-scratch, attention/embedding visualisations. Scope explicitly excludes production-scale training (PyTorch / JAX / MLX territory).
- **Mathematical Modeling Framework** ([#2](https://github.com/Furox-Art/scientific-computing-system/issues/2)): shipped in v1.1.0 as `cds.modeling`.
- **Knowledge Organization System** ([#3](https://github.com/Furox-Art/scientific-computing-system/issues/3)): shipped in v1.1.0 as `cds.knowledge`.
- **Optional pandas interop**: shipped in v1.2.0 as the `cds[pandas]` extra (`to_dataframe` / `from_dataframe`).
- **"Tour of Numerical Methods" guide**: shipped in v1.0.0 and expanded through v1.2.0 (`docs/tour_of_numerical_methods.md`).

---

Contributions and ideas are welcome, especially for cross-module demos, new algorithms, and the hypothesis generation system.

**Status legend:**
- `[x]` Released
- `[~]` In progress
- `[ ]` Planned
