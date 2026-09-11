"""Monte Carlo simulation methods.

References:
    - Metropolis, N. & Ulam, S. (1949). The Monte Carlo Method.
    - Buffon, G.L. (1777). Essai d'arithmétique morale.
    - Robert, C.P. & Casella, G. Monte Carlo Statistical Methods (2nd ed.)
"""

from __future__ import annotations

import math
import multiprocessing
import random
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass


@dataclass
class MCResult:
    """Result of a Monte Carlo estimation."""

    estimate: float
    samples: int
    std_error: float


def _validate_integer(name: str, value: int, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _validate_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _evaluate_finite(function: Callable[[float], float], x: float) -> float:
    value = float(function(x))
    if not math.isfinite(value):
        raise ValueError("Monte Carlo integrand must return finite values")
    return value


def _finite_width(lower: float, upper: float, *, name: str) -> float:
    width = upper - lower
    if not math.isfinite(width):
        raise ArithmeticError(f"{name} width became non-finite")
    return width


def _pi_worker(samples_seed: tuple[int, int | None]) -> int:
    """Worker function for parallel pi estimation."""
    samples, seed = samples_seed
    rng = random.Random(seed)
    inside = 0
    for _ in range(samples):
        x = rng.random()
        y = rng.random()
        if x * x + y * y <= 1.0:
            inside += 1
    return inside


def estimate_pi(n_samples: int = 100_000, seed: int | None = None) -> MCResult:
    """Estimate π using the unit-circle method (parallelized).

    Throw random points into the unit square [0,1]×[0,1].
    Fraction inside the quarter-circle ≈ π/4.
    """
    _validate_integer("n_samples", n_samples, minimum=1)

    cores = min(multiprocessing.cpu_count(), n_samples)
    chunk_size = n_samples // cores
    chunks = [chunk_size] * cores
    chunks[-1] += n_samples - sum(chunks)

    if seed is None:
        import os
        import sys

        seed = int.from_bytes(os.urandom(4), sys.byteorder)

    seeds = [seed + i for i in range(cores)]
    tasks = list(zip(chunks, seeds))

    inside = 0
    with ProcessPoolExecutor(max_workers=cores) as executor:
        for result in executor.map(_pi_worker, tasks):
            inside += result

    p = inside / n_samples
    estimate = 4.0 * p
    se = 4.0 * math.sqrt(p * (1 - p) / n_samples) if n_samples > 1 else 0.0
    return MCResult(estimate=estimate, samples=n_samples, std_error=se)


def mc_integrate(
    f: Callable[[float], float],
    a: float,
    b: float,
    n_samples: int = 100_000,
    seed: int | None = None,
) -> MCResult:
    """Monte Carlo integration of ``f`` over the finite interval [a, b].

    Reversed bounds are supported and preserve the usual signed-integral
    convention. Equal bounds return an exact zero-width estimate after finite
    function evaluations.
    """
    _validate_integer("n_samples", n_samples, minimum=1)
    _validate_finite("a", a)
    _validate_finite("b", b)
    width = _finite_width(a, b, name="integration interval")

    rng = random.Random(seed)
    total = 0.0
    total_sq = 0.0
    for _ in range(n_samples):
        x = a + rng.random() * width
        val = _evaluate_finite(f, x)
        total += val
        total_sq += val * val
        if not math.isfinite(total) or not math.isfinite(total_sq):
            raise ArithmeticError("Monte Carlo accumulation became non-finite")

    mean_val = total / n_samples
    estimate = mean_val * width
    var = total_sq / n_samples - mean_val**2 if n_samples > 1 else 0.0
    if not math.isfinite(estimate) or not math.isfinite(var):
        raise ArithmeticError("Monte Carlo integral became non-finite")
    se = abs(width) * math.sqrt(max(0.0, var) / n_samples)
    if not math.isfinite(se):
        raise ArithmeticError("Monte Carlo standard error became non-finite")
    return MCResult(estimate=estimate, samples=n_samples, std_error=se)


def random_walk_1d(
    steps: int,
    step_size: float = 1.0,
    seed: int | None = None,
) -> list[float]:
    """1D symmetric random walk with a finite non-negative step magnitude."""
    _validate_integer("steps", steps, minimum=0)
    _validate_finite("step_size", step_size)
    if step_size < 0:
        raise ValueError("step_size must be non-negative")

    rng = random.Random(seed)
    positions = [0.0]
    pos = 0.0
    for _ in range(steps):
        pos += step_size if rng.random() < 0.5 else -step_size
        if not math.isfinite(pos):
            raise ArithmeticError("random walk position became non-finite")
        positions.append(pos)
    return positions


def random_walk_2d(
    steps: int,
    step_size: float = 1.0,
    seed: int | None = None,
) -> list[tuple[float, float]]:
    """2D random walk on a plane with a finite non-negative step magnitude."""
    _validate_integer("steps", steps, minimum=0)
    _validate_finite("step_size", step_size)
    if step_size < 0:
        raise ValueError("step_size must be non-negative")

    rng = random.Random(seed)
    positions: list[tuple[float, float]] = [(0.0, 0.0)]
    x, y = 0.0, 0.0
    for _ in range(steps):
        angle = rng.uniform(0, 2 * math.pi)
        x += step_size * math.cos(angle)
        y += step_size * math.sin(angle)
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ArithmeticError("random walk position became non-finite")
        positions.append((x, y))
    return positions


def buffon_needle(
    needle_length: float = 1.0,
    line_spacing: float = 2.0,
    n_throws: int = 100_000,
    seed: int | None = None,
) -> MCResult:
    """Buffon's needle experiment for estimating π.

    ``needle_length`` and ``line_spacing`` must be finite and positive, with
    ``needle_length <= line_spacing`` for the standard formula used here.
    """
    _validate_integer("n_throws", n_throws, minimum=1)
    _validate_finite("needle_length", needle_length)
    _validate_finite("line_spacing", line_spacing)
    if needle_length <= 0 or line_spacing <= 0:
        raise ValueError("needle_length and line_spacing must be positive")
    if needle_length > line_spacing:
        raise ValueError("needle_length must not exceed line_spacing")

    rng = random.Random(seed)
    crossings = 0
    for _ in range(n_throws):
        center = rng.uniform(0, line_spacing / 2)
        angle = rng.uniform(0, math.pi)
        tip = (needle_length / 2) * math.sin(angle)
        if tip >= center:
            crossings += 1

    if crossings == 0:
        raise ArithmeticError(
            "Buffon estimator is undefined with zero crossings; increase n_throws"
        )

    p = crossings / n_throws
    estimate = (2 * needle_length) / (line_spacing * p)
    se_p = math.sqrt(p * (1 - p) / n_throws)
    se = (2 * needle_length * se_p) / (line_spacing * p * p)
    if not math.isfinite(estimate) or not math.isfinite(se):
        raise ArithmeticError("Buffon estimator became non-finite")
    return MCResult(estimate=estimate, samples=n_throws, std_error=se)


def mc_expectation(
    f: Callable[[float], float],
    n_samples: int = 10_000,
    a: float = 0.0,
    b: float = 1.0,
    seed: int | None = None,
) -> MCResult:
    """Estimate ``E[f(X)]`` for ``X ~ Uniform(a, b)`` by plain MC."""
    _validate_integer("n_samples", n_samples, minimum=1)
    _validate_finite("a", a)
    _validate_finite("b", b)
    if a >= b:
        raise ValueError("a must be less than b")
    width = _finite_width(a, b, name="expectation interval")

    rng = random.Random(seed)
    total = 0.0
    total_sq = 0.0
    for _ in range(n_samples):
        x = a + rng.random() * width
        val = _evaluate_finite(f, x)
        total += val
        total_sq += val * val
        if not math.isfinite(total) or not math.isfinite(total_sq):
            raise ArithmeticError("Monte Carlo accumulation became non-finite")
    mean_val = total / n_samples
    var = total_sq / n_samples - mean_val**2 if n_samples > 1 else 0.0
    if not math.isfinite(mean_val) or not math.isfinite(var):
        raise ArithmeticError("Monte Carlo expectation became non-finite")
    se = math.sqrt(max(0.0, var) / n_samples)
    if not math.isfinite(se):
        raise ArithmeticError("Monte Carlo standard error became non-finite")
    return MCResult(estimate=mean_val, samples=n_samples, std_error=se)


def hit_or_miss(
    predicate: Callable[[float, float], bool],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    n_samples: int = 50_000,
    seed: int | None = None,
) -> MCResult:
    """Estimate the area of a 2-D region defined by ``predicate(x, y)``."""
    _validate_integer("n_samples", n_samples, minimum=1)
    x0, x1 = x_range
    y0, y1 = y_range
    for name, value in (("x0", x0), ("x1", x1), ("y0", y0), ("y1", y1)):
        _validate_finite(name, value)
    if x0 >= x1 or y0 >= y1:
        raise ValueError("ranges must be non-empty (lo < hi)")
    x_width = _finite_width(x0, x1, name="x-range")
    y_width = _finite_width(y0, y1, name="y-range")
    box = x_width * y_width
    if not math.isfinite(box):
        raise ArithmeticError("bounding-box area became non-finite")

    rng = random.Random(seed)
    hits = 0
    for _ in range(n_samples):
        x = x0 + rng.random() * x_width
        y = y0 + rng.random() * y_width
        if predicate(x, y):
            hits += 1
    p = hits / n_samples
    estimate = box * p
    se = box * math.sqrt(p * (1 - p) / n_samples) if n_samples > 1 else 0.0
    if not math.isfinite(estimate) or not math.isfinite(se):
        raise ArithmeticError("hit-or-miss estimate became non-finite")
    return MCResult(estimate=estimate, samples=n_samples, std_error=se)
