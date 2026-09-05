"""Reproducible benchmark runner with robust timing summaries.

This wrapper preserves the benchmark workloads in ``run_benchmarks.py`` while
replacing its best-of timing helper with warmup + repeated sampling and enriching
the JSON artifact with enough environment metadata to compare runs responsibly.
"""

from __future__ import annotations

import math
import multiprocessing
import os
import platform
import statistics
import subprocess
import timeit
from collections import OrderedDict
from collections.abc import Callable

import run_benchmarks as legacy

from cds import __version__

_TIMING_SAMPLES: list[dict[str, object]] = []
_original_build_json = legacy._build_json_record


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def robust_bench(func: Callable[[], object], number: int, repeat: int = 1) -> float:
    """Warm up once, collect repeated per-call timings, and return the median."""
    if number <= 0 or repeat <= 0:
        raise ValueError("number and repeat must be positive")
    func()
    sample_count = max(5, repeat)
    samples = [timeit.timeit(func, number=number) / number for _ in range(sample_count)]
    median = statistics.median(samples)
    mad = statistics.median(abs(value - median) for value in samples)
    _TIMING_SAMPLES.append(
        {
            "index": len(_TIMING_SAMPLES),
            "number_per_sample": number,
            "sample_count": sample_count,
            "median_seconds": median,
            "mad_seconds": mad,
            "p95_seconds": _percentile(samples, 0.95),
            "min_seconds": min(samples),
            "max_seconds": max(samples),
            "samples_seconds": samples,
        }
    )
    return median


def _full_git_sha() -> str:
    value = os.environ.get("GITHUB_SHA", "").strip()
    if value:
        return value
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else "unknown"


def build_robust_json(
    results: dict[str, OrderedDict[str, str]],
) -> dict[str, object]:
    """Extend legacy benchmark JSON with reproducibility and timing metadata."""
    record = _original_build_json(results)
    record["schema_version"] = 2
    record["package_version"] = __version__
    record["git_sha_full"] = _full_git_sha()
    record["environment"] = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": multiprocessing.cpu_count(),
    }
    record["reproducibility"] = {
        "timing_clock": "timeit.default_timer",
        "warmup_calls": 1,
        "minimum_timing_samples": 5,
        "reported_central_tendency": "median",
        "dispersion": "median_absolute_deviation",
        "tail_statistic": "p95",
        "known_seeds": {"monte_carlo_pi": 42},
    }
    record["timing_distributions"] = list(_TIMING_SAMPLES)
    return record


def main() -> None:
    _TIMING_SAMPLES.clear()
    legacy._bench = robust_bench
    legacy._build_json_record = build_robust_json
    legacy.run_all()


if __name__ == "__main__":
    main()
