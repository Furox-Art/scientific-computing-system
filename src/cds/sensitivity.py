"""Local and global sensitivity analysis with no external runtime dependencies."""

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
    """Central finite-difference sensitivity report for one scalar model output."""

    output: float
    parameters: tuple[ParameterSensitivity, ...]

    def most_influential(self) -> ParameterSensitivity | None:
        """Return the parameter with the largest local influence magnitude."""
        if not self.parameters:
            return None

        def score(item: ParameterSensitivity) -> float:
            if item.normalized is not None:
                return abs(item.normalized)
            return abs(item.derivative)

        return max(self.parameters, key=score)


@dataclass(frozen=True)
class GlobalParameterSensitivity:
    """Elementary-effect summary across a parameter's declared range."""

    index: int
    mean_effect: float
    mean_absolute_effect: float
    effect_std: float
    monotonicity: float


@dataclass(frozen=True)
class GlobalSensitivityReport:
    """Randomized range-wide elementary-effects sensitivity summary."""

    trajectories: int
    levels: int
    evaluations: int
    parameters: tuple[GlobalParameterSensitivity, ...]

    def most_influential(self) -> GlobalParameterSensitivity | None:
        if not self.parameters:
            return None
        return max(self.parameters, key=lambda item: item.mean_absolute_effect)


def local_sensitivity(
    model: ScalarModel,
    parameters: Sequence[float],
    *,
    relative_step: float = 1e-6,
    absolute_step: float = 1e-8,
) -> SensitivityReport:
    """Estimate local parameter derivatives with symmetric perturbations.

    The perturbation for parameter ``p`` is
    ``max(abs(p) * relative_step, absolute_step)``. Normalized sensitivity is
    ``(d output / d p) * p / output`` when the baseline output is non-zero.
    """
    if (
        not math.isfinite(relative_step)
        or relative_step <= 0
        or not math.isfinite(absolute_step)
        or absolute_step <= 0
    ):
        raise ValueError("relative_step and absolute_step must be positive and finite")

    baseline = tuple(float(value) for value in parameters)
    if any(not math.isfinite(value) for value in baseline):
        raise ValueError("parameters must contain only finite values")

    output = float(model(baseline))
    if not math.isfinite(output):
        raise ValueError("model baseline output must be finite")

    results: list[ParameterSensitivity] = []
    for index, parameter in enumerate(baseline):
        step = max(abs(parameter) * relative_step, absolute_step)
        lower = list(baseline)
        upper = list(baseline)
        lower[index] -= step
        upper[index] += step
        lower_output = float(model(lower))
        upper_output = float(model(upper))
        if not (math.isfinite(lower_output) and math.isfinite(upper_output)):
            raise ValueError(f"model perturbation output for parameter {index} must be finite")
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


def global_sensitivity(
    model: ScalarModel,
    bounds: Sequence[tuple[float, float]],
    *,
    trajectories: int = 32,
    levels: int = 8,
    seed: int = 0,
) -> GlobalSensitivityReport:
    """Screen range-wide influence with randomized elementary effects.

    For each randomized baseline, every parameter is perturbed by one grid step
    within its declared range and the signed elementary effect is recorded.
    The returned mean absolute effect measures global influence, while effect
    spread reveals nonlinearity/interactions and ``monotonicity`` approaches 1
    when effects retain a consistent sign across the domain.
    """
    if not bounds:
        raise ValueError("bounds must contain at least one parameter range")
    if trajectories <= 0:
        raise ValueError("trajectories must be positive")
    if levels < 2:
        raise ValueError("levels must be at least 2")

    normalized_bounds: list[tuple[float, float]] = []
    for lower, upper in bounds:
        lower_value = float(lower)
        upper_value = float(upper)
        if not (math.isfinite(lower_value) and math.isfinite(upper_value)):
            raise ValueError("sensitivity bounds must be finite")
        if lower_value >= upper_value:
            raise ValueError("each sensitivity bound must satisfy lower < upper")
        normalized_bounds.append((lower_value, upper_value))

    rng = random.Random(seed)
    effects: list[list[float]] = [[] for _ in normalized_bounds]
    evaluations = 0

    for _ in range(trajectories):
        baseline = [rng.uniform(lower, upper) for lower, upper in normalized_bounds]
        base_output = float(model(baseline))
        evaluations += 1
        if not math.isfinite(base_output):
            raise ValueError("model output must stay finite during global sensitivity analysis")

        for index, (lower, upper) in enumerate(normalized_bounds):
            step = (upper - lower) / (levels - 1)
            direction = 1.0 if baseline[index] + step <= upper else -1.0
            delta = direction * step
            perturbed = list(baseline)
            perturbed[index] += delta
            perturbed_output = float(model(perturbed))
            evaluations += 1
            if not math.isfinite(perturbed_output):
                raise ValueError(
                    f"model perturbation output for parameter {index} must stay finite"
                )
            effects[index].append((perturbed_output - base_output) / delta)

    summaries: list[GlobalParameterSensitivity] = []
    for index, values in enumerate(effects):
        mean_effect = sum(values) / len(values)
        mean_absolute = sum(abs(value) for value in values) / len(values)
        variance = sum((value - mean_effect) ** 2 for value in values) / len(values)
        monotonicity = 1.0 if mean_absolute == 0.0 else abs(mean_effect) / mean_absolute
        summaries.append(
            GlobalParameterSensitivity(
                index=index,
                mean_effect=mean_effect,
                mean_absolute_effect=mean_absolute,
                effect_std=math.sqrt(max(0.0, variance)),
                monotonicity=min(1.0, monotonicity),
            )
        )

    return GlobalSensitivityReport(
        trajectories=trajectories,
        levels=levels,
        evaluations=evaluations,
        parameters=tuple(summaries),
    )
