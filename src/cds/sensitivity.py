"""Local/global sensitivity, variance-based screening, and identifiability analysis."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

ScalarModel = Callable[[Sequence[float]], float]
VectorModel = Callable[[Sequence[float]], Sequence[float]]


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


@dataclass(frozen=True)
class VarianceParameterSensitivity:
    """Variance-based first-order and total-order sensitivity for one parameter."""

    index: int
    first_order: float
    total_order: float


@dataclass(frozen=True)
class VarianceSensitivityReport:
    """Monte-Carlo Sobol/Saltelli-style sensitivity report."""

    variance: float
    samples: int
    parameters: tuple[VarianceParameterSensitivity, ...]

    def most_influential(self) -> VarianceParameterSensitivity:
        """Return the parameter with the largest total-order influence."""
        return max(self.parameters, key=lambda item: abs(item.total_order))


@dataclass(frozen=True)
class IdentifiabilityReport:
    """Finite-difference Jacobian rank report around one parameter baseline."""

    rank: int
    parameter_count: int
    output_count: int
    identifiable: bool
    independent_parameters: tuple[bool, ...]
    jacobian: tuple[tuple[float, ...], ...]


def _validated_bounds(bounds: Sequence[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    if not bounds:
        raise ValueError("bounds must contain at least one parameter interval")
    validated: list[tuple[float, float]] = []
    for lower, upper in bounds:
        lo = float(lower)
        hi = float(upper)
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise ValueError("bounds must contain only finite values")
        if lo >= hi:
            raise ValueError("each lower bound must be smaller than its upper bound")
        validated.append((lo, hi))
    return tuple(validated)


def _scalar_output(model: ScalarModel, parameters: Sequence[float]) -> float:
    value = float(model(parameters))
    if not math.isfinite(value):
        raise ValueError("global sensitivity model outputs must be finite")
    return value


def variance_global_sensitivity(
    model: ScalarModel,
    bounds: Sequence[tuple[float, float]],
    *,
    samples: int = 2048,
    seed: int = 0,
    variance_tolerance: float = 1e-15,
) -> VarianceSensitivityReport:
    """Estimate variance-based global sensitivity with deterministic sampling.

    Two independent sample matrices ``A`` and ``B`` are drawn uniformly over
    the declared bounds. For each parameter a hybrid ``A_Bi`` matrix is used
    to estimate Saltelli-style first-order and Jansen total-order indices.
    Raw estimators are returned rather than silently clipping them to ``[0, 1]``.
    """
    if samples < 2:
        raise ValueError("samples must be at least 2")
    if variance_tolerance <= 0:
        raise ValueError("variance_tolerance must be positive")
    intervals = _validated_bounds(bounds)
    rng = random.Random(seed)

    def draw() -> tuple[float, ...]:
        return tuple(rng.uniform(lower, upper) for lower, upper in intervals)

    matrix_a = tuple(draw() for _ in range(samples))
    matrix_b = tuple(draw() for _ in range(samples))
    outputs_a = tuple(_scalar_output(model, row) for row in matrix_a)
    outputs_b = tuple(_scalar_output(model, row) for row in matrix_b)
    combined = outputs_a + outputs_b
    average = sum(combined) / len(combined)
    variance = sum((value - average) ** 2 for value in combined) / len(combined)
    if variance <= variance_tolerance:
        raise ValueError("model output variance is too small for global sensitivity")

    results: list[VarianceParameterSensitivity] = []
    for index in range(len(intervals)):
        hybrid_outputs: list[float] = []
        for sample_index in range(samples):
            hybrid = list(matrix_a[sample_index])
            hybrid[index] = matrix_b[sample_index][index]
            hybrid_outputs.append(_scalar_output(model, hybrid))
        first_order = (
            sum(outputs_b[row] * (hybrid_outputs[row] - outputs_a[row]) for row in range(samples))
            / samples
            / variance
        )
        total_order = sum((outputs_a[row] - hybrid_outputs[row]) ** 2 for row in range(samples)) / (
            2.0 * samples * variance
        )
        results.append(
            VarianceParameterSensitivity(
                index=index,
                first_order=first_order,
                total_order=total_order,
            )
        )

    return VarianceSensitivityReport(variance=variance, samples=samples, parameters=tuple(results))


def _vector_output(model: VectorModel, parameters: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in model(parameters))
    if not values:
        raise ValueError("identifiability model must return at least one output")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("identifiability model outputs must be finite")
    return values


def _column_independence(
    columns: Sequence[Sequence[float]],
    *,
    tolerance: float,
) -> tuple[bool, ...]:
    norms = [math.sqrt(sum(value * value for value in column)) for column in columns]
    scale = max([1.0, *norms])
    cutoff = tolerance * scale
    basis: list[list[float]] = []
    independent: list[bool] = []
    for column in columns:
        residual = [float(value) for value in column]
        for direction in basis:
            projection = sum(value * unit for value, unit in zip(residual, direction, strict=True))
            residual = [
                value - projection * unit for value, unit in zip(residual, direction, strict=True)
            ]
        norm = math.sqrt(sum(value * value for value in residual))
        is_independent = norm > cutoff
        independent.append(is_independent)
        if is_independent:
            basis.append([value / norm for value in residual])
    return tuple(independent)


def local_identifiability(
    model: VectorModel,
    parameters: Sequence[float],
    *,
    relative_step: float = 1e-6,
    absolute_step: float = 1e-8,
    rank_tolerance: float = 1e-9,
) -> IdentifiabilityReport:
    """Assess local parameter identifiability from finite-difference Jacobian rank."""
    if relative_step <= 0 or absolute_step <= 0:
        raise ValueError("relative_step and absolute_step must be positive")
    if rank_tolerance <= 0:
        raise ValueError("rank_tolerance must be positive")
    baseline = tuple(float(value) for value in parameters)
    if not baseline:
        raise ValueError("parameters must contain at least one value")
    if any(not math.isfinite(value) for value in baseline):
        raise ValueError("parameters must contain only finite values")

    baseline_output = _vector_output(model, baseline)
    columns: list[tuple[float, ...]] = []
    for index, parameter in enumerate(baseline):
        step = max(abs(parameter) * relative_step, absolute_step)
        lower = list(baseline)
        upper = list(baseline)
        lower[index] -= step
        upper[index] += step
        lower_output = _vector_output(model, lower)
        upper_output = _vector_output(model, upper)
        if len(lower_output) != len(baseline_output) or len(upper_output) != len(baseline_output):
            raise ValueError("identifiability model output length must remain constant")
        columns.append(
            tuple(
                (upper_output[row] - lower_output[row]) / (2.0 * step)
                for row in range(len(baseline_output))
            )
        )

    independent = _column_independence(columns, tolerance=rank_tolerance)
    rank = sum(independent)
    jacobian = tuple(
        tuple(columns[column][row] for column in range(len(columns)))
        for row in range(len(baseline_output))
    )
    return IdentifiabilityReport(
        rank=rank,
        parameter_count=len(baseline),
        output_count=len(baseline_output),
        identifiable=rank == len(baseline),
        independent_parameters=independent,
        jacobian=jacobian,
    )


__all__ = [
    "GlobalParameterSensitivity",
    "GlobalSensitivityReport",
    "IdentifiabilityReport",
    "ParameterSensitivity",
    "SensitivityReport",
    "VarianceParameterSensitivity",
    "VarianceSensitivityReport",
    "global_sensitivity",
    "local_identifiability",
    "local_sensitivity",
    "variance_global_sensitivity",
]
