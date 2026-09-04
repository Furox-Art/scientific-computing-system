"""Local sensitivity analysis with no external runtime dependencies."""

from __future__ import annotations

import math
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
    if relative_step <= 0 or absolute_step <= 0:
        raise ValueError("relative_step and absolute_step must be positive")

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
