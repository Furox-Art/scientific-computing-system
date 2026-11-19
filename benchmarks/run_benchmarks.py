"""
Performance benchmark suite for CDS.

Measures execution time for core scientific modules and optionally compares with
NumPy. Run directly to regenerate ``docs/benchmarks.md`` and the
machine-readable ``benchmarks/results.json`` (timestamped + git-SHA-stamped for
regression tracking):

    python benchmarks/run_benchmarks.py

In CI, the ``Benchmarks`` workflow (``.github/workflows/benchmarks.yml``)
runs this automatically on ``v*`` tag pushes and weekly (Monday 03:00 UTC),
committing both artifacts back to ``main``.

Each ``bench_*`` function returns an ordered dict of ``{metric: value}`` so the
report stays deterministic.
"""

import json
import math
import subprocess
import time
import timeit
from collections import OrderedDict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

# Try to import industry standards for comparison (Optional)
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def _git_sha() -> str:
    """Return the current commit SHA, or ``"unknown"`` if git is unavailable.

    CI runs inside a checkout; local runs may not have git on PATH. Failure
    must never break a benchmark run, so every error path returns a sentinel.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip() or "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        # git missing or repo not available — fall back to "unknown".
        pass
    return "unknown"


def _coerce_metric(value: str) -> float | str:
    """Best-effort parse of a metric string into a number.

    Benchmark values are formatted as ``"0.0423s"``, ``"727.3x"``, ``"22"``
    (CPU cores), ``"8.0x"``, etc. We strip a trailing unit and try ``float``;
    anything that doesn't parse stays a string (e.g. ``"O(N^3) PLU"``).
    """
    s = value.strip()
    stripped = s.rstrip()
    if stripped and stripped[-1].isalpha():
        stripped = stripped[:-1].strip()
    try:
        return float(stripped)
    except ValueError:
        return s


def _build_json_record(
    results: dict[str, "OrderedDict[str, str]"],
) -> dict[str, object]:
    """Assemble the machine-readable artifact: provenance + metric tree."""
    numeric: dict[str, dict[str, float | str]] = {}
    for category, metrics in results.items():
        numeric[category] = {k: _coerce_metric(v) for k, v in metrics.items()}
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "metrics": numeric,
    }


def _write_json(record: dict[str, object], json_dir: Path | None = None) -> Path:
    """Write the JSON artifact and return its path.

    Args:
        record: the assembled metric tree from :func:`_build_json_record`.
        json_dir: directory to write ``results.json`` into. Defaults to the
            directory containing this script (``benchmarks/``), so the
            committed artifact lands at ``benchmarks/results.json`` regardless
            of where the script is invoked from. Tests pass a ``tmp_path`` so
            they don't clobber the committed artifact.

    The canonical location is ``benchmarks/results.json`` — the workflow's
    diff check and ``add-paths``, and the report's "Raw data" line, all
    reference that path. Previously ``run_all`` passed ``output_dir=root``
    (the repo root) here, which wrote to ``./results.json`` and silently
    diverged from the tracked file. ``json_dir`` defaults to the script's own
    directory so the two can never drift again.
    """
    out_dir = json_dir if json_dir is not None else Path(__file__).resolve().parent
    out = out_dir / "results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, sort_keys=False)
        f.write("\n")
    return out


def _bench(func: Callable[[], object], number: int, repeat: int = 1) -> float:
    """Return the mean wall-clock seconds per call of ``func`` over ``number`` runs."""
    best = math.inf
    for _ in range(repeat):
        elapsed = timeit.timeit(func, number=number)
        best = min(best, elapsed / number)
    return best


# --- Linear Algebra ---
def bench_linalg() -> OrderedDict[str, str]:
    from cds.math_utils.linalg import lu_decomposition, mat_mul

    size = 100
    # Create a non-singular matrix
    A = [[0.0] * size for _ in range(size)]
    for i in range(size):
        A[i][i] = 1.0
        if i > 0:
            A[i][i - 1] = 0.5

    B = [[2.0] * size for _ in range(size)]

    t_mul_cds = _bench(lambda: mat_mul(A, B), number=5)
    t_lu_cds = _bench(lambda: lu_decomposition(A), number=5)

    results: OrderedDict[str, str] = OrderedDict()
    results["CDS Matrix Mul (100x100)"] = f"{t_mul_cds:.4f}s"
    results["CDS LU Decomp (100x100)"] = f"{t_lu_cds:.4f}s"

    if HAS_NUMPY:
        A_np = np.array(A)
        B_np = np.array(B)
        t_mul_np = _bench(lambda: np.dot(A_np, B_np), number=100)
        ratio = t_mul_cds / t_mul_np if t_mul_np > 0 else math.inf
        results["NumPy Matrix Mul (Baseline)"] = f"{t_mul_np:.6f}s"
        results["Speed Status"] = f"CDS is {ratio:.1f}x slower than NumPy (pure Python vs C)"

    return results


def bench_linalg_intelligence() -> OrderedDict[str, str]:
    """Determinant via PLU (O(N^3)) — confirm the cubic scaling law.

    A naive Laplace-expansion determinant is O(N!) and unrunnable past ~N=13, so
    instead of a misleading apples-to-oranges race this benchmark verifies that
    CDS's determinant *actually* scales as O(N^3): doubling N should take ~8x
    longer. That is the real evidence the implementation is not brute force.
    """
    from cds.math_utils.linalg import determinant

    def _spd_matrix(n: int) -> list[list[float]]:
        # Diagonally dominant => well-conditioned, clean PLU.
        A = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                A[i][j] = 1.0 / (1 + i + j)
            A[i][i] = n + 1.0
        return A

    t_n50 = _bench(lambda: determinant(_spd_matrix(50)), number=5)
    t_n100 = _bench(lambda: determinant(_spd_matrix(100)), number=5)
    # Expected ratio for true O(N^3): (100/50)^3 = 8.0x
    observed = t_n100 / t_n50 if t_n50 > 0 else math.inf

    return OrderedDict(
        [
            ("Determinant @ N=50", f"{t_n50:.6f}s"),
            ("Determinant @ N=100", f"{t_n100:.6f}s"),
            ("Ratio (doubling N)", f"{observed:.1f}x"),
            ("Expected for O(N^3)", "8.0x"),
            ("Complexity", "O(N^3) PLU"),
        ]
    )


# --- Monte Carlo ---
def bench_montecarlo() -> OrderedDict[str, str]:
    from cds.montecarlo import estimate_pi

    samples = 100_000

    start = time.perf_counter()
    result = estimate_pi(samples, seed=42)
    elapsed = time.perf_counter() - start

    return OrderedDict(
        [
            ("Pi (100k samples)", f"{elapsed:.4f}s"),
            ("Sampling mode", "single in-process random.Random stream"),
            ("Estimate error vs π", f"{abs(result.estimate - math.pi):.5f}"),
        ]
    )


# --- Quantum: O(1) sampling vs an extrapolated brute-force cost ---
def bench_quantum() -> OrderedDict[str, str]:
    from cds.quantum.circuit import QuantumCircuit, hadamard
    from cds.quantum.simulator import simulate

    c = QuantumCircuit()
    for _ in range(10):
        c.add(hadamard())

    shots = 100_000

    # O(1) sampling: one simulate() call draws all shots from the state vector.
    start = time.perf_counter()
    simulate(c, shots=shots)
    t_intelligent = time.perf_counter() - start

    # Extrapolated naive cost: time a single circuit run and multiply by the
    # shot count. The 100,000 shot-by-shot runs are NEVER executed, so this is
    # an estimate, not a measurement -- the metric labels say so explicitly.
    t_single_run = timeit.timeit(lambda: c.run(), number=100) / 100
    t_naive_estimate = t_single_run * shots

    # Derive the speedup from the *displayed* (rounded) timings so the table
    # and the visual conclusion can never quote two different numbers for the
    # same division (the old report printed 53.2x in one and 53.3x in other).
    t_int_display = round(t_intelligent, 4)
    t_naive_display = round(t_naive_estimate, 2)
    speedup = t_naive_display / t_int_display if t_int_display > 0 else 1.0

    return OrderedDict(
        [
            ("O(1) Sampling (measured)", f"{t_int_display:.4f}s"),
            ("Naive Brute Force (extrapolated)", f"{t_naive_display:.2f}s"),
            ("Speedup vs extrapolation", f"{speedup:.1f}x"),
        ]
    )


# --- Signal Processing: FFT vs naive DFT ---
def bench_signals() -> OrderedDict[str, str]:
    from cds.signals import dft, fft_radix2

    n = 1024  # power of two for radix-2
    signal = [complex(i % 17) for i in range(n)]

    t_fft = _bench(lambda: fft_radix2(signal), number=10)
    t_dft = _bench(lambda: dft(signal), number=1)

    speedup = t_dft / t_fft if t_fft > 0 else math.inf
    return OrderedDict(
        [
            ("Signal length", f"{n} samples"),
            ("CDS FFT (radix-2, O(N log N))", f"{t_fft:.6f}s"),
            ("Naive DFT (O(N^2))", f"{t_dft:.6f}s"),
            ("Algorithmic speedup", f"{speedup:.0f}x"),
        ]
    )


# --- Numerical Integration: convergence ---
def bench_numerical_integration() -> OrderedDict[str, str]:
    """Accuracy of deterministic quadrature on ∫_0^1 e^x dx = e - 1."""
    from cds.numerical_integration import (
        adaptive_simpson,
        gaussian_quadrature,
        romberg,
        simpson,
        trapezoid,
    )

    exact = math.e - 1
    f = math.exp

    def _err(val: float) -> str:
        return f"{abs(val - exact):.2e}"

    return OrderedDict(
        [
            ("Integral", "∫_0^1 e^x dx = e - 1"),
            ("Trapezoid n=1000", _err(trapezoid(f, 0, 1, 1000))),
            ("Simpson n=100", _err(simpson(f, 0, 1, 100))),
            ("Gauss-Legendre n=8", _err(gaussian_quadrature(f, 0, 1, 8))),
            ("Romberg (auto tol)", _err(romberg(f, 0, 1).value)),
            ("Adaptive Simpson", _err(adaptive_simpson(f, 0, 1).value)),
        ]
    )


def _bar(value: float, max_value: float, width: int = 40) -> str:
    """Render a scaled ASCII bar of length proportional to ``value / max_value``."""
    if max_value <= 0:
        return ""
    n = max(1, round(width * value / max_value))
    return "#" * n


# Category names describe what is measured, never an outcome the data may
# contradict. The old headers ("Approaching C-Speed", "Hardware Saturation",
# "Algorithmic Intelligence") advertised the opposite of the committed
# numbers: the linear-algebra table shows CDS ~1155x SLOWER than NumPy, the
# Monte Carlo row was a bare cpu_count(), and the quantum "speedup" is an
# extrapolation, not a measurement.
QUANTUM_CATEGORY = "Quantum (O(1) Sampling vs Extrapolated Brute Force)"

BEGIN_GENERATED = (
    "<!-- BEGIN GENERATED: benchmarks/run_benchmarks.py via benchmarks/robust_runner.py -->"
)
END_GENERATED = "<!-- END GENERATED -->"


def _render_generated_block(results: dict[str, "OrderedDict[str, str]"], sha: str, ts: str) -> str:
    """Render the verbatim block written between the GENERATED markers.

    The block is a pure function of the metric tree plus the run's SHA and
    timestamp, so ``tests/test_benchmarks.py`` can assert that the committed
    ``docs/benchmarks.md`` block and ``benchmarks/results.json`` agree row for
    row. Hand-editing the tables breaks that gate; edit this renderer instead.
    """
    lines = [
        "<!-- The tables below are written verbatim by the benchmark runner. Edit the -->",
        "<!-- script, not this block. -->",
        "",
        f"> **Last measured:** commit `{sha}` at {ts}. Regenerated",
        "> automatically by the `benchmarks` GitHub Actions workflow (weekly + on release",
        "> tags). Raw data: `benchmarks/results.json`. Note: the generator prints the",
        f'> commit SHA where a measurement value belongs, so "Last measured: `{sha}`"',
        f'> means "measured at commit {sha}", not a count.',
        "",
        "This report measures both raw speed and algorithmic scaling. Pure Python is "
        "slower than C-extensions for dense numerics, so rather than only racing NumPy, "
        "the comparisons below also check that the implemented algorithms scale with "
        "their theoretical complexity (e.g. O(N log N) FFT, O(N^3) PLU determinant) and "
        "converge to machine precision where expected.",
        "",
    ]
    for category, metrics in results.items():
        lines.append(f"#### {category}")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for key, value in metrics.items():
            lines.append(f"| {key} | {value} |")
        lines.append("")

    # Visual comparison scaled by the *displayed* quantum timings. The speedup
    # is recomputed from the same rounded strings the table shows, so the two
    # can never disagree (the old report printed 53.2x vs 53.3x).
    q_metrics = results[QUANTUM_CATEGORY]
    t_intelligent = float(q_metrics["O(1) Sampling (measured)"].rstrip("s"))
    t_naive = float(q_metrics["Naive Brute Force (extrapolated)"].rstrip("s"))
    speedup_val = t_naive / t_intelligent if t_intelligent > 0 else 1.0

    lines.append("#### Visual comparison: quantum sampling (extrapolated)")
    lines.append("")
    lines.append("```text")
    lines.append(f"Naive Brute Force (extrapolated): {_bar(t_naive, t_naive)} ({t_naive:.2f}s)")
    lines.append(f"CDS O(1) Sampling: {_bar(t_intelligent, t_naive)} ({t_intelligent:.4f}s)")
    lines.append("")
    lines.append(
        f"Conclusion (extrapolated, not measured): O(1) sampling is ~{speedup_val:.1f}x "
        "faster than the estimated shot-by-shot cost."
    )
    lines.append("```")
    return "\n".join(lines) + "\n"


def run_all(output_dir: Path | None = None, json_dir: Path | None = None) -> None:
    """Run every benchmark and emit the markdown + JSON artifacts.

    Args:
        output_dir: root directory under which ``docs/benchmarks.md`` is
            written. Defaults to the current working directory, preserving CLI
            behaviour. Tests pass a ``tmp_path`` so they don't clobber the
            committed markdown.
        json_dir: directory to write ``results.json`` into. Defaults to the
            directory containing this script (``benchmarks/``), so the
            committed artifact always lands at ``benchmarks/results.json``
            regardless of ``output_dir``. Tests pass a ``tmp_path`` to keep
            the committed JSON untouched.

    The markdown page is curated: everything outside the GENERATED markers is
    hand-written provenance and analysis. When the markers are present, only
    the block between them is replaced, so a regeneration can never destroy
    the hand-written half of the page (the old writer overwrote the whole
    file). Without markers -- a fresh tree or a test's ``tmp_path`` -- the
    full report is written. The two outputs stay decoupled: the markdown goes
    to ``<output_dir>/docs/benchmarks.md`` while ``results.json`` lands at the
    canonical ``benchmarks/results.json`` the workflow diffs and commits.
    """
    root = Path.cwd() if output_dir is None else output_dir
    print("Running Benchmarks...")
    results: dict[str, OrderedDict[str, str]] = {}
    results["Linear Algebra (Pure Python vs NumPy)"] = bench_linalg()
    results["Linear Algebra (Determinant Scaling)"] = bench_linalg_intelligence()
    results["Monte Carlo (Single-Stream Pi Sampling)"] = bench_montecarlo()
    results[QUANTUM_CATEGORY] = bench_quantum()
    results["Signal Processing (FFT vs DFT)"] = bench_signals()
    results["Numerical Integration (Convergence)"] = bench_numerical_integration()

    sha = _git_sha()
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    block = _render_generated_block(results, sha, ts)

    docs_dir = root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    docs_path = docs_dir / "benchmarks.md"
    full_report = "# CDS Performance Report\n\n" + block
    existing = docs_path.read_text(encoding="utf-8") if docs_path.exists() else ""
    begin = existing.find(BEGIN_GENERATED)
    end = existing.find(END_GENERATED)
    if begin != -1 and end > begin:
        head = existing[: begin + len(BEGIN_GENERATED)]
        tail = existing[end:]
        docs_path.write_text(head + "\n" + block + "\n" + tail, encoding="utf-8")
    else:
        docs_path.write_text(full_report, encoding="utf-8")

    # Machine-readable artifact for regression tracking (spec §E).
    record = _build_json_record(results)
    json_path = _write_json(record, json_dir=json_dir)

    print(f"Benchmarks completed. Report saved to {docs_path}")
    print(f"JSON artifact saved to {json_path}")


if __name__ == "__main__":
    run_all()
