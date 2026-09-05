"""Local and global sensitivity analysis with no runtime dependencies."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

ScalarModel = Callable[[Sequence[float]], float]


@dataclass(frozen=True)
class ParameterSensitivity:
    """Sensitivity of one model parameter around a declared baseline."""

    index: int
    baseline: float
    step: float
    derivative: float
    normalized: float | None


@dataclass(frozen=True)
class SensitivityReport:
    """Central finite-difference sensitivity report for one scalar output."""

    output: float
    parameters: tuple[ParameterSensitivity, ...]

    def most_influential(self) -> ParameterSensitivity | None:
        if not self.parameters:
            return None

        def score(item: ParameterSensitivity) -> float:
            return abs(item.normalized) if item.normalized is not None else abs(item.derivative)

        return max(self.parameters, key=score)


@dataclass(frozen=True)
class MorrisParameter:
    """Morris elementary-effect summary for one normalized input."""

    index: int
    mean: float
    mean_absolute: float
    standard_deviation: float
    effects: tuple[float, ...]


@dataclass(frozen=True)
class MorrisReport:
    """Global Morris screening report."""

    parameters: tuple[MorrisParameter, ...]
    trajectories: int
    levels: int
    seed: int | None

    def most_influential(self) -> MorrisParameter | None:
        if not self.parameters:
            return None
        return max(self.parameters, key=lambda item: item.mean_absolute)


@dataclass(frozen=True)
class SobolParameter:
    """Monte Carlo first-order and total-order Sobol indices."""

    index: int
    first_order: float
    total_order: float


@dataclass(frozen=True)
class SobolReport:
    """Variance-based Sobol/Jansen sensitivity report."""

    parameters: tuple[SobolParameter, ...]
    samples: int
    variance: float
    seed: int | None


@dataclass(frozen=True)
class InteractionEffect:
    """Local mixed second derivative for a parameter pair."""

    first: int
    second: int
    derivative: float
    normalized: float | None


@dataclass(frozen=True)
class InteractionReport:
    """Pairwise local parameter-interaction report."""

    output: float
    effects: tuple[InteractionEffect, ...]

    def strongest(self) -> InteractionEffect | None:
        if not self.effects:
            return None
        return max(
            self.effects,
            key=lambda item: abs(item.normalized)
            if item.normalized is not None
            else abs(item.derivative),
        )


def _evaluate(model: ScalarModel, parameters: Sequence[float]) -> float:
    result = float(model(parameters))
    if not math.isfinite(result):
        raise ValueError("model output must be finite")
    return result


def _validate_bounds(bounds: Sequence[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    if not bounds:
        raise ValueError("bounds must contain at least one parameter range")
    normalized = tuple((float(lower), float(upper)) for lower, upper in bounds)
    if any(
        not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper
        for lower, upper in normalized
    ):
        raise ValueError("each bound must be finite and satisfy lower < upper")
    return normalized


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _sample_stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = _mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def local_sensitivity(
    model: ScalarModel,
    parameters: Sequence[float],
    *,
    relative_step: float = 1e-6,
    absolute_step: float = 1e-8,
) -> SensitivityReport:
    """Estimate local parameter derivatives with symmetric perturbations."""
    if relative_step <= 0 or absolute_step <= 0:
        raise ValueError("relative_step and absolute_step must be positive")

    baseline = tuple(float(value) for value in parameters)
    if any(not math.isfinite(value) for value in baseline):
        raise ValueError("parameters must contain only finite values")
    output = _evaluate(model, baseline)

    results: list[ParameterSensitivity] = []
    for index, parameter in enumerate(baseline):
        step = max(abs(parameter) * relative_step, absolute_step)
        lower = list(baseline)
        upper = list(baseline)
        lower[index] -= step
        upper[index] += step
        lower_output = _evaluate(model, lower)
        upper_output = _evaluate(model, upper)
        derivative = (upper_output - lower_output) / (2.0 * step)
        normalized = None if output == 0.0 else derivative * parameter / output
        results.append(
            ParameterSensitivity(
                index=index,
                baseline=parameter,
                step=step,
                derivative=derivative,
                normalized=normalized,
            )
        )
    return SensitivityReport(output=output, parameters=tuple(results))


def morris_screening(
    model: ScalarModel,
    bounds: Sequence[tuple[float, float]],
    *,
    trajectories: int = 20,
    levels: int = 6,
    seed: int | None = 0,
) -> MorrisReport:
    """Run Morris elementary-effects screening across the full parameter space.

    Inputs are sampled on a regular normalized grid.  Each trajectory perturbs
    every input exactly once by the standard Morris grid step in a randomized
    order and direction.  Elementary effects are reported per unit normalized
    input, making magnitudes comparable across differently scaled bounds.
    """
    normalized_bounds = _validate_bounds(bounds)
    if trajectories < 2:
        raise ValueError("trajectories must be at least 2")
    if levels < 4 or levels % 2 != 0:
        raise ValueError("levels must be an even integer of at least 4")

    dimensions = len(normalized_bounds)
    delta = levels / (2.0 * (levels - 1.0))
    grid_step = 1.0 / (levels - 1.0)
    allowed_base = [index * grid_step for index in range(levels) if index * grid_step <= 1.0 - delta]
    generator = random.Random(seed)
    effects: list[list[float]] = [[] for _ in range(dimensions)]

    def physical(normalized: Sequence[float]) -> list[float]:
        return [
            lower + value * (upper - lower)
            for value, (lower, upper) in zip(normalized, normalized_bounds)
        ]

    for _ in range(trajectories):
        point = [generator.choice(allowed_base) for _ in range(dimensions)]
        directions = [1 if generator.random() < 0.5 else -1 for _ in range(dimensions)]
        for index, direction in enumerate(directions):
            if direction < 0:
                point[index] += delta
        order = list(range(dimensions))
        generator.shuffle(order)
        current_value = _evaluate(model, physical(point))
        for index in order:
            next_point = list(point)
            next_point[index] += directions[index] * delta
            next_value = _evaluate(model, physical(next_point))
            effects[index].append((next_value - current_value) / (directions[index] * delta))
            point = next_point
            current_value = next_value

    summaries = tuple(
        MorrisParameter(
            index=index,
            mean=_mean(values),
            mean_absolute=_mean([abs(value) for value in values]),
            standard_deviation=_sample_stdev(values),
            effects=tuple(values),
        )
        for index, values in enumerate(effects)
    )
    return MorrisReport(
        parameters=summaries,
        trajectories=trajectories,
        levels=levels,
        seed=seed,
    )


def sobol_indices(
    model: ScalarModel,
    bounds: Sequence[tuple[float, float]],
    *,
    samples: int = 2048,
    seed: int | None = 0,
) -> SobolReport:
    """Estimate first- and total-order Sobol indices with Saltelli/Jansen estimators."""
    normalized_bounds = _validate_bounds(bounds)
    if samples < 2:
        raise ValueError("samples must be at least 2")
    dimensions = len(normalized_bounds)
    generator = random.Random(seed)

    def draw() -> list[float]:
        return [generator.uniform(lower, upper) for lower, upper in normalized_bounds]

    matrix_a = [draw() for _ in range(samples)]
    matrix_b = [draw() for _ in range(samples)]
    values_a = [_evaluate(model, row) for row in matrix_a]
    values_b = [_evaluate(model, row) for row in matrix_b]
    combined = values_a + values_b
    center = _mean(combined)
    variance = sum((value - center) ** 2 for value in combined) / (len(combined) - 1)
    if variance <= 0.0:
        raise ValueError("Sobol indices are undefined for zero-variance model output")

    parameters: list[SobolParameter] = []
    for index in range(dimensions):
        hybrid_values: list[float] = []
        for row_a, row_b in zip(matrix_a, matrix_b):
            hybrid = list(row_a)
            hybrid[index] = row_b[index]
            hybrid_values.append(_evaluate(model, hybrid))

        first = sum(
            value_b * (hybrid - value_a)
            for value_a, value_b, hybrid in zip(values_a, values_b, hybrid_values)
        ) / (samples * variance)
        total = sum(
            (value_a - hybrid) ** 2 for value_a, hybrid in zip(values_a, hybrid_values)
        ) / (2.0 * samples * variance)
        parameters.append(SobolParameter(index=index, first_order=first, total_order=total))

    return SobolReport(
        parameters=tuple(parameters),
        samples=samples,
        variance=variance,
        seed=seed,
    )


def pairwise_interactions(
    model: ScalarModel,
    parameters: Sequence[float],
    *,
    relative_step: float = 1e-4,
    absolute_step: float = 1e-7,
) -> InteractionReport:
    """Estimate local pairwise interactions using mixed central differences."""
    if relative_step <= 0 or absolute_step <= 0:
        raise ValueError("relative_step and absolute_step must be positive")
    baseline = tuple(float(value) for value in parameters)
    if any(not math.isfinite(value) for value in baseline):
        raise ValueError("parameters must contain only finite values")
    output = _evaluate(model, baseline)
    steps = [max(abs(value) * relative_step, absolute_step) for value in baseline]
    effects: list[InteractionEffect] = []

    for first in range(len(baseline)):
        for second in range(first + 1, len(baseline)):
            h_first = steps[first]
            h_second = steps[second]
            evaluations: list[float] = []
            for first_sign, second_sign in ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)):
                point = list(baseline)
                point[first] += first_sign * h_first
                point[second] += second_sign * h_second
                evaluations.append(_evaluate(model, point))
            derivative = (evaluations[0] - evaluations[1] - evaluations[2] + evaluations[3]) / (
                4.0 * h_first * h_second
            )
            normalized = (
                None
                if output == 0.0
                else derivative * baseline[first] * baseline[second] / output
            )
            effects.append(
                InteractionEffect(
                    first=first,
                    second=second,
                    derivative=derivative,
                    normalized=normalized,
                )
            )

    return InteractionReport(output=output, effects=tuple(effects))
