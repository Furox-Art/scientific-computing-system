# CDS Performance Report

!!! info "How to read this page"
    The **Raw generator output** section below is produced by
    `benchmarks/run_benchmarks.py` and written verbatim by
    `benchmarks/robust_runner.py`. Do not hand-edit those tables — edit the
    script and re-run it. Everything else on this page is written by hand and
    explains what the numbers mean and which claims they support.

## Provenance

The numbers below come from a single run of one script. This is everything the
committed artifact `benchmarks/results.json` records about that run:

| Field | Recorded value |
|---|---|
| Script | `benchmarks/run_benchmarks.py`, via `benchmarks/robust_runner.py` |
| Command | `python benchmarks/robust_runner.py` |
| Commit | `7213425` (short SHA, as written by the generator) |
| Timestamp | 2026-06-25T16:35:53+00:00 |
| Package version | **not recorded** — this artifact predates `package_version` in the JSON schema |
| Platform / CPU / Python | **not recorded** — see the warning below |
| NumPy version | **not recorded** |
| Timing method | best-of-N via `timeit`; the warmup/median/MAD/p95 metadata is not present in this artifact |

**These numbers are stale and under-documented.** They are over three months old
and predate the schema that records environment metadata, so the machine, CPU,
Python build, and NumPy version behind them are unknown and cannot be recovered
from the repository.

!!! warning "Treat the absolute timings below as unverified"
    The scaling and convergence figures are still informative — they compare two
    CDS implementations on one machine, or measure error against a closed form.
    The absolute seconds and the NumPy ratio cannot be reproduced or attributed,
    because the environment is not recorded.

The `Benchmarks` workflow is supposed to regenerate this page weekly and on `v*`
tags. It can only open its follow-up pull request when the `BOT_APP_ID` secret is
configured; when it is not, the job still succeeds and the committed report
silently ages. **Check the commit SHA and timestamp above before citing any
number from this page.**

## What these numbers do and do not support

**Supported by the measurement, independent of the machine:**

- The asymptotic scaling checks. The determinant ratio and the FFT-vs-DFT
  comparison pit two CDS implementations against each other on the same run, so
  the algorithmic-complexity conclusion holds regardless of hardware.
- The quadrature convergence figures. Those are errors against a known closed
  form (`e - 1`), not timings, so they are deterministic and reproduce exactly.

**Not supported — read with care:**

- **"Naive Brute Force (Est.)" and the speedup derived from it are an
  extrapolation, not a measurement.** The script times a single `circuit.run()`
  call and multiplies by the 100,000-shot count; it never executes 100,000
  circuits. Treat the resulting speedup as an order-of-magnitude illustration
  of why O(1) sampling is the right design, not as a benchmark result.
- **"CPU Cores Saturated" is a core count, not a saturation measurement.** It
  reports `multiprocessing.cpu_count()`. Nothing here verifies that every core
  stayed busy or that the workload actually scaled.
- **The generated section headers describe algorithms, not outcomes.** Strings
  like "Algorithmic Intelligence", "Approaching C-Speed" and "Hardware
  Saturation" come from the generator, not from the data. The first table's
  honest reading is the opposite of its own header: on that run CDS was
  **1154.8× slower** than NumPy on dense matrix multiplication.
- **Two different numbers for the same division.** The quantum table reports
  "53.2x Faster" while the conclusion line below it prints "53.3 times faster".
  Both come from the same ratio at different rounding points; neither is a
  measurement of 100,000 full circuit runs.
- **Single-run dispersion is not shown.** Best-of-N hides variance, and this
  artifact carries no MAD or p95.

## Raw generator output

<!-- BEGIN GENERATED: benchmarks/run_benchmarks.py via benchmarks/robust_runner.py -->
<!-- The tables below are written verbatim by the benchmark runner. Edit the -->
<!-- script, not this block. -->

> **Last measured:** commit `7213425` at 2026-06-25 16:35 UTC. Regenerated
> automatically by the `benchmarks` GitHub Actions workflow (weekly + on release
> tags). Raw data: `benchmarks/results.json`. Note: the generator prints the
> commit SHA where a measurement value belongs, so "Last measured: `7213425`"
> means "measured at commit 7213425", not a count.

This report measures both raw speed and algorithmic scaling. Pure Python is slower than C-extensions for dense numerics, so rather than only racing NumPy, the comparisons below also check that the implemented algorithms scale with their theoretical complexity (e.g. O(N log N) FFT, O(N^3) PLU determinant) and converge to machine precision where expected.

#### Linear algebra vs. NumPy

| Metric | Value |
|--------|-------|
| CDS Matrix Mul (100x100) | 0.0696s |
| CDS LU Decomp (100x100) | 0.0247s |
| NumPy Matrix Mul (Baseline) | 0.000060s |
| Speed Status | CDS is 1154.8x slower than NumPy (pure Python vs C) |

#### Determinant scaling

| Metric | Value |
|--------|-------|
| Determinant @ N=50 | 0.004818s |
| Determinant @ N=100 | 0.029788s |
| Ratio (doubling N) | 6.2x |
| Expected for O(N^3) | 8.0x |
| Complexity | O(N^3) PLU |

#### Monte Carlo

| Metric | Value |
|--------|-------|
| Parallel Pi (100k samples) | 0.0388s |
| CPU Cores Saturated | 4 |
| Estimate error vs π | 0.00791 |

#### Quantum sampling (estimated)

| Metric | Value |
|--------|-------|
| Intelligent O(1) Sampling | 0.0152s |
| Naive Brute Force (Est.) | 0.81s |
| Intelligence Speedup | 53.2x Faster |

#### Signal processing: FFT vs. DFT

| Metric | Value |
|--------|-------|
| Signal length | 1024 samples |
| CDS FFT (radix-2, O(N log N)) | 0.002745s |
| Naive DFT (O(N^2)) | 0.351954s |
| Algorithmic speedup | 128x |

#### Numerical integration (convergence errors)

| Metric | Value |
|--------|-------|
| Integral | ∫_0^1 e^x dx = e - 1 |
| Trapezoid n=1000 | 1.43e-07 |
| Simpson n=100 | 9.55e-11 |
| Gauss-Legendre n=8 | 6.66e-16 |
| Romberg (auto tol) | 8.88e-16 |
| Adaptive Simpson | 6.66e-16 |

#### Visual comparison: quantum sampling

```text
Naive Brute Force: ######################################## (0.81s)
CDS O(1) Sampling: # (0.0152s)

Conclusion: CDS is 53.3 times faster via O(1) probabilistic sampling vs running the circuit shot-by-shot.
```

<!-- END GENERATED -->

## Regenerating

```bash
pip install "scientific-computing-system[all]"
python benchmarks/robust_runner.py
```

This rewrites `docs/benchmarks.md` and `benchmarks/results.json` in place. The
current `robust_runner.py` records the package version, full commit SHA, Python
build, platform, CPU, warmup count, seed policy, and per-workload median / MAD /
p95 in the JSON — so a fresh run gives a fully attributable report. The numbers
in the generated section above predate that.

In CI, the `Benchmarks` workflow runs this on `v*` tags and weekly, then opens a
pull request with both artifacts when `BOT_APP_ID` / `BOT_APP_PRIVATE_KEY` are
configured.

## Related

- [Why Pure Python?](why-pure-python.md) — the trade-off in prose, and when to
  reach for the NumPy-based 2.0 project instead
- [ML Reference Values](ml_reference.md) — correctness checks against
  scikit-learn, which are reproducible and separate from these timings
