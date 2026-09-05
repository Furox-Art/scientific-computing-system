"""Pure-Python uncertainty propagation for scientific calculations."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class UncertainValue:
    """A scalar measurement represented by value ± standard uncertainty."""

    value: float
    standard_uncertainty: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise ValueError("value must be finite")
        if not math.isfinite(self.standard_uncertainty) or self.standard_uncertainty < 0:
            raise ValueError("standard_uncertainty must be finite and non-negative")


@dataclass(frozen=True)
class PropagationResult:
    """First-order propagated uncertainty and local sensitivity coefficients."""

    value: float
    standard_uncertainty: float
    variance: float
    sensitivities: tuple[float, ...]
    method: str = "linearized"


@dataclass(frozen=True)
class MonteCarloResult:
    """Exact Monte Carlo summary retaining all outputs for exact sample quantiles."""

    mean: float
    standard_deviation: float
    lower: float
    upper: float
    samples: int
    seed: int | None
    confidence: float
    method: str = "monte-carlo"


@dataclass(frozen=True)
class StreamingMonteCarloResult:
    """Bounded-memory Monte Carlo summary with reservoir-estimated quantiles."""

    mean: float
    standard_deviation: float
    lower: float
    upper: float
    samples: int
    seed: int | None
    confidence: float
    reservoir_size: int
    quantiles_exact: bool
    method: str = "monte-carlo-streaming-reservoir"


def _validate_means(means: Sequence[float]) -> tuple[float, ...]:
    if not means:
        raise ValueError("means must not be empty")
    values = tuple(float(value) for value in means)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("means must contain only finite values")
    return values


def _covariance_matrix(
    size: int,
    standard_uncertainties: Sequence[float] | None,
    covariance: Sequence[Sequence[float]] | None,
) -> tuple[tuple[float, ...], ...]:
    if covariance is not None and standard_uncertainties is not None:
        raise ValueError("provide either standard_uncertainties or covariance, not both")

    if covariance is None:
        if standard_uncertainties is None:
            uncertainties = (0.0,) * size
        else:
            if len(standard_uncertainties) != size:
                raise ValueError("standard_uncertainties must match the number of means")
            uncertainties = tuple(float(value) for value in standard_uncertainties)
            if any(not math.isfinite(value) or value < 0 for value in uncertainties):
                raise ValueError("standard_uncertainties must be finite and non-negative")
        return tuple(
            tuple(uncertainties[row] ** 2 if row == column else 0.0 for column in range(size))
            for row in range(size)
        )

    if len(covariance) != size or any(len(row) != size for row in covariance):
        raise ValueError("covariance must be a square matrix matching the number of means")
    matrix = tuple(tuple(float(value) for value in row) for row in covariance)
    if any(not math.isfinite(value) for row in matrix for value in row):
        raise ValueError("covariance must contain only finite values")
    for row in range(size):
        if matrix[row][row] < 0:
            raise ValueError("covariance diagonal entries must be non-negative")
        for column in range(row + 1, size):
            if not math.isclose(
                matrix[row][column], matrix[column][row], rel_tol=1e-12, abs_tol=1e-15
            ):
                raise ValueError("covariance must be symmetric")
    return matrix


def _evaluate(function: Callable[..., float], values: Sequence[float]) -> float:
    result = float(function(*values))
    if not math.isfinite(result):
        raise ValueError("function returned a non-finite result")
    return result


def _sensitivities(
    function: Callable[..., float],
    means: tuple[float, ...],
    relative_step: float,
) -> tuple[float, ...]:
    if relative_step <= 0 or not math.isfinite(relative_step):
        raise ValueError("relative_step must be finite and positive")
    sensitivities: list[float] = []
    for index, value in enumerate(means):
        step = relative_step * max(1.0, abs(value))
        left = list(means)
        right = list(means)
        left[index] -= step
        right[index] += step
        derivative = (_evaluate(function, right) - _evaluate(function, left)) / (2.0 * step)
        sensitivities.append(derivative)
    return tuple(sensitivities)


def _quadratic_form(vector: Sequence[float], matrix: Sequence[Sequence[float]]) -> float:
    return sum(
        vector[row] * matrix[row][column] * vector[column]
        for row in range(len(vector))
        for column in range(len(vector))
    )


def propagate_linear(
    function: Callable[..., float],
    means: Sequence[float],
    *,
    standard_uncertainties: Sequence[float] | None = None,
    covariance: Sequence[Sequence[float]] | None = None,
    relative_step: float = 1e-6,
) -> PropagationResult:
    """Propagate covariance through ``function`` using first-order sensitivities."""
    normalized_means = _validate_means(means)
    matrix = _covariance_matrix(len(normalized_means), standard_uncertainties, covariance)
    value = _evaluate(function, normalized_means)
    sensitivities = _sensitivities(function, normalized_means, relative_step)
    variance = _quadratic_form(sensitivities, matrix)
    scale = max(1.0, max((abs(entry) for row in matrix for entry in row), default=0.0))
    if variance < -1e-12 * scale:
        raise ValueError("covariance produced a negative propagated variance")
    variance = max(0.0, variance)
    return PropagationResult(
        value=value,
        standard_uncertainty=math.sqrt(variance),
        variance=variance,
        sensitivities=sensitivities,
    )


def _cholesky(matrix: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    size = len(matrix)
    lower = [[0.0] * size for _ in range(size)]
    for row in range(size):
        for column in range(row + 1):
            subtotal = sum(lower[row][k] * lower[column][k] for k in range(column))
            if row == column:
                diagonal = matrix[row][row] - subtotal
                if diagonal < -1e-12:
                    raise ValueError("covariance must be positive semidefinite")
                lower[row][column] = math.sqrt(max(0.0, diagonal))
            elif lower[column][column] > 0:
                lower[row][column] = (matrix[row][column] - subtotal) / lower[column][column]
            elif not math.isclose(matrix[row][column] - subtotal, 0.0, abs_tol=1e-12):
                raise ValueError("covariance must be positive semidefinite")
    return tuple(tuple(row) for row in lower)


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return sorted_values[lower_index]
    fraction = position - lower_index
    return sorted_values[lower_index] * (1.0 - fraction) + sorted_values[upper_index] * fraction


def _draw_correlated(
    generator: random.Random,
    means: tuple[float, ...],
    lower: Sequence[Sequence[float]],
) -> list[float]:
    standard_normals = [generator.normalvariate(0.0, 1.0) for _ in means]
    return [
        means[row]
        + sum(lower[row][column] * standard_normals[column] for column in range(row + 1))
        for row in range(len(means))
    ]


def _mc_setup(
    means: Sequence[float],
    standard_uncertainties: Sequence[float] | None,
    covariance: Sequence[Sequence[float]] | None,
    samples: int,
    confidence: float,
) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
    normalized_means = _validate_means(means)
    if samples < 2:
        raise ValueError("samples must be at least 2")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    matrix = _covariance_matrix(len(normalized_means), standard_uncertainties, covariance)
    return normalized_means, _cholesky(matrix)


def propagate_monte_carlo(
    function: Callable[..., float],
    means: Sequence[float],
    *,
    standard_uncertainties: Sequence[float] | None = None,
    covariance: Sequence[Sequence[float]] | None = None,
    samples: int = 10_000,
    seed: int | None = 0,
    confidence: float = 0.95,
) -> MonteCarloResult:
    """Propagate Gaussian uncertainty with exact empirical sample quantiles.

    This compatibility API retains every output and is therefore ``O(samples)``
    in memory.  Use :func:`propagate_monte_carlo_streaming` for large runs.
    """
    normalized_means, lower = _mc_setup(
        means, standard_uncertainties, covariance, samples, confidence
    )
    generator = random.Random(seed)
    outputs = [
        _evaluate(function, _draw_correlated(generator, normalized_means, lower))
        for _ in range(samples)
    ]
    mean = sum(outputs) / samples
    variance = sum((value - mean) ** 2 for value in outputs) / (samples - 1)
    ordered = sorted(outputs)
    tail = (1.0 - confidence) / 2.0
    return MonteCarloResult(
        mean=mean,
        standard_deviation=math.sqrt(variance),
        lower=_quantile(ordered, tail),
        upper=_quantile(ordered, 1.0 - tail),
        samples=samples,
        seed=seed,
        confidence=confidence,
    )


def propagate_monte_carlo_streaming(
    function: Callable[..., float],
    means: Sequence[float],
    *,
    standard_uncertainties: Sequence[float] | None = None,
    covariance: Sequence[Sequence[float]] | None = None,
    samples: int = 100_000,
    seed: int | None = 0,
    confidence: float = 0.95,
    reservoir_size: int = 4096,
) -> StreamingMonteCarloResult:
    """Propagate uncertainty in bounded memory using online moments.

    Mean and sample standard deviation are exact for the simulated stream via
    Welford's online algorithm.  Confidence bounds use uniform reservoir
    sampling and are therefore approximate when ``reservoir_size < samples``;
    the result exposes ``quantiles_exact`` so this approximation is never
    silent.  Memory is ``O(min(samples, reservoir_size))``.
    """
    normalized_means, lower = _mc_setup(
        means, standard_uncertainties, covariance, samples, confidence
    )
    if reservoir_size < 2:
        raise ValueError("reservoir_size must be at least 2")
    retained = min(samples, reservoir_size)
    generator = random.Random(seed)
    reservoir_rng = random.Random(None if seed is None else seed ^ 0x5DEECE66D)
    reservoir: list[float] = []
    mean = 0.0
    m2 = 0.0

    for count in range(1, samples + 1):
        output = _evaluate(function, _draw_correlated(generator, normalized_means, lower))
        delta = output - mean
        mean += delta / count
        m2 += delta * (output - mean)
        if len(reservoir) < retained:
            reservoir.append(output)
        else:
            replacement = reservoir_rng.randrange(count)
            if replacement < retained:
                reservoir[replacement] = output

    ordered = sorted(reservoir)
    tail = (1.0 - confidence) / 2.0
    return StreamingMonteCarloResult(
        mean=mean,
        standard_deviation=math.sqrt(m2 / (samples - 1)),
        lower=_quantile(ordered, tail),
        upper=_quantile(ordered, 1.0 - tail),
        samples=samples,
        seed=seed,
        confidence=confidence,
        reservoir_size=retained,
        quantiles_exact=retained == samples,
    )
