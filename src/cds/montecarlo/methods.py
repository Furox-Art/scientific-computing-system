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


def _validate_positive_count(value: int, name: str) -> None:
    """Require a genuine positive integer for a Monte Carlo sample count."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_walk(steps: int, step_size: float) -> None:
    """Validate random-walk dimensions while allowing a zero-length walk."""
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise TypeError("steps must be an integer")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if not math.isfinite(step_size) or step_size < 0:
        raise ValueError("step_size must be finite and non-negative")


def _validate_finite_interval(a: float, b: float, *, ordered: bool) -> None:
    """Validate endpoints for uniform Monte Carlo sampling."""
    if not (math.isfinite(a) and math.isfinite(b)):
        raise ValueError("interval bounds must be finite")
    if ordered and a >= b:
        raise ValueError("a must be less than b")


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

    Args:
        n_samples: positive number of random points.
        seed: optional random seed.
    """
    _validate_positive_count(n_samples, "n_samples")

    cores = min(multiprocessing.cpu_count(), n_samples)
    chunk_size = n_samples // cores
    chunks = [chunk_size] * cores
    chunks[-1] += n_samples - sum(chunks)  # add remainder to last chunk

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

    The estimator is ``E[f(X)] * (b-a)`` for a uniform draw between the two
    endpoints. Reversed bounds are supported and produce a signed integral;
    the reported standard error remains non-negative.
    """
    _validate_positive_count(n_samples, "n_samples")
    _validate_finite_interval(a, b, ordered=False)

    rng = random.Random(seed)
    total = 0.0
    total_sq = 0.0
    width = b - a
    for _ in range(n_samples):
        x = a + rng.random() * width
        val = f(x)
        total += val
        total_sq += val * val

    mean_val = total / n_samples
    estimate = mean_val * width
    var = (total_sq / n_samples - mean_val**2) if n_samples > 1 else 0.0
    se = abs(width) * math.sqrt(max(0.0, var) / n_samples)
    return MCResult(estimate=estimate, samples=n_samples, std_error=se)


def random_walk_1d(
    steps: int,
    step_size: float = 1.0,
    seed: int | None = None,
) -> list[float]:
    """1D symmetric random walk.

    At each step, move +step_size or -step_size with equal probability. A
    zero-step walk is valid and contains only the origin.
    """
    _validate_walk(steps, step_size)
    rng = random.Random(seed)
    positions = [0.0]
    pos = 0.0
    for _ in range(steps):
        pos += step_size if rng.random() < 0.5 else -step_size
        positions.append(pos)
    return positions


def random_walk_2d(
    steps: int,
    step_size: float = 1.0,
    seed: int | None = None,
) -> list[tuple[float, float]]:
    """2D random walk on a plane.

    At each step, move in a random direction (uniform angle). A zero-step walk
    is valid and contains only the origin.
    """
    _validate_walk(steps, step_size)
    rng = random.Random(seed)
    positions: list[tuple[float, float]] = [(0.0, 0.0)]
    x, y = 0.0, 0.0
    for _ in range(steps):
        angle = rng.uniform(0, 2 * math.pi)
        x += step_size * math.cos(angle)
        y += step_size * math.sin(angle)
        positions.append((x, y))
    return positions


def buffon_needle(
    needle_length: float = 1.0,
    line_spacing: float = 2.0,
    n_throws: int = 100_000,
    seed: int | None = None,
) -> MCResult:
    """Buffon's needle experiment for estimating π.

    The classical short-needle formula requires finite positive dimensions and
    ``needle_length <= line_spacing``. ``n_throws`` must be positive.
    """
    _validate_positive_count(n_throws, "n_throws")
    if not math.isfinite(needle_length) or needle_length <= 0:
        raise ValueError("needle_length must be finite and positive")
    if not math.isfinite(line_spacing) or line_spacing <= 0:
        raise ValueError("line_spacing must be finite and positive")
    if needle_length > line_spacing:
        raise ValueError("needle must be shorter than or equal to line spacing")
    rng = random.Random(seed)

    crossings = 0
    for _ in range(n_throws):
        center = rng.uniform(0, line_spacing / 2)
        angle = rng.uniform(0, math.pi)
        tip = (needle_length / 2) * math.sin(angle)
        if tip >= center:
            crossings += 1

    if crossings == 0:
        return MCResult(estimate=0.0, samples=n_throws, std_error=0.0)

    p = crossings / n_throws
    estimate = (2 * needle_length) / (line_spacing * p)
    se_p = math.sqrt(p * (1 - p) / n_throws)
    se = (2 * needle_length * se_p) / (line_spacing * p * p) if p > 0 else 0.0
    return MCResult(estimate=estimate, samples=n_throws, std_error=se)


def mc_expectation(
    f: Callable[[float], float],
    n_samples: int = 10_000,
    a: float = 0.0,
    b: float = 1.0,
    seed: int | None = None,
) -> MCResult:
    """Estimate ``E[f(X)]`` for ``X ~ Uniform(a, b)`` by plain MC.

    Differs from :func:`mc_integrate` by **not** multiplying by ``(b-a)`` —
    this is the expectation, not the integral.
    """
    _validate_positive_count(n_samples, "n_samples")
    _validate_finite_interval(a, b, ordered=True)
    rng = random.Random(seed)
    width = b - a
    total = 0.0
    total_sq = 0.0
    for _ in range(n_samples):
        x = a + rng.random() * width
        val = f(x)
        total += val
        total_sq += val * val
    mean_val = total / n_samples
    var = (total_sq / n_samples - mean_val**2) if n_samples > 1 else 0.0
    se = math.sqrt(max(0.0, var) / n_samples)
    return MCResult(estimate=mean_val, samples=n_samples, std_error=se)


def hit_or_miss(
    predicate: Callable[[float, float], bool],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    n_samples: int = 50_000,
    seed: int | None = None,
) -> MCResult:
    """Estimate the area of a 2-D region defined by ``predicate(x, y)``.

    Samples uniformly in the finite bounding box ``x_range × y_range`` and
    returns ``area_box * fraction_true``.
    """
    _validate_positive_count(n_samples, "n_samples")
    x0, x1 = x_range
    y0, y1 = y_range
    if not all(math.isfinite(value) for value in (x0, x1, y0, y1)):
        raise ValueError("ranges must contain finite bounds")
    if x0 >= x1 or y0 >= y1:
        raise ValueError("ranges must be non-empty (lo < hi)")
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_samples):
        x = x0 + rng.random() * (x1 - x0)
        y = y0 + rng.random() * (y1 - y0)
        if predicate(x, y):
            hits += 1
    p = hits / n_samples
    box = (x1 - x0) * (y1 - y0)
    estimate = box * p
    se = box * math.sqrt(p * (1 - p) / n_samples) if n_samples > 1 else 0.0
    return MCResult(estimate=estimate, samples=n_samples, std_error=se)
