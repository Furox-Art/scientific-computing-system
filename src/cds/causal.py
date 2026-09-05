"""Conservative, assumption-gated causal effect estimators.

This module does not infer causal structure from observational correlations.
Callers must declare causal assumptions explicitly and, for observational
back-door adjustment, provide the adjustment covariates themselves.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from cds.validation import CheckStatus, ValidationCheck, ValidationReport


@dataclass(frozen=True)
class CausalAssumptions:
    """Core identification assumptions that must be justified outside the estimator."""

    temporal_order: bool | None
    no_unmeasured_confounding: bool | None
    positivity: bool | None
    consistency: bool | None

    @property
    def identified(self) -> bool:
        """Whether every required assumption is explicitly declared true."""
        return all(
            value is True
            for value in (
                self.temporal_order,
                self.no_unmeasured_confounding,
                self.positivity,
                self.consistency,
            )
        )


@dataclass(frozen=True)
class CausalEffectEstimate:
    """Point estimate from an explicitly identified causal design."""

    effect: float
    method: str
    n_observations: int
    adjustment_count: int
    assumptions: CausalAssumptions | None


def audit_causal_assumptions(assumptions: CausalAssumptions) -> ValidationReport:
    """Convert explicit causal assumptions into PASS/WARNING/FAIL checks."""
    report = ValidationReport()
    for name, value in (
        ("temporal_order", assumptions.temporal_order),
        ("no_unmeasured_confounding", assumptions.no_unmeasured_confounding),
        ("positivity", assumptions.positivity),
        ("consistency", assumptions.consistency),
    ):
        if value is True:
            report.add(ValidationCheck(name, CheckStatus.PASS, f"{name} explicitly supported"))
        elif value is False:
            report.add(ValidationCheck(name, CheckStatus.FAIL, f"{name} explicitly violated"))
        else:
            report.add(ValidationCheck(name, CheckStatus.WARNING, f"{name} not established"))
    return report


def _finite_vector(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _invert(matrix: Sequence[Sequence[float]], *, tolerance: float = 1e-12) -> list[list[float]]:
    size = len(matrix)
    augmented = [
        [float(value) for value in row] + [1.0 if row_index == column else 0.0 for column in range(size)]
        for row_index, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= tolerance:
            raise ValueError("causal adjustment design matrix is singular")
        if pivot != column:
            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0.0:
                continue
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column], strict=True)
            ]
    return [row[size:] for row in augmented]


def _ols_coefficients(design: Sequence[Sequence[float]], outcome: Sequence[float]) -> tuple[float, ...]:
    if not design:
        raise ValueError("causal estimator requires at least one observation")
    columns = len(design[0])
    if columns == 0 or any(len(row) != columns for row in design):
        raise ValueError("causal design matrix must be rectangular and non-empty")
    if len(design) < columns:
        raise ValueError("causal design needs at least as many observations as coefficients")

    gram = [[0.0] * columns for _ in range(columns)]
    rhs = [0.0] * columns
    for row, target in zip(design, outcome, strict=True):
        for left in range(columns):
            rhs[left] += row[left] * target
            for right in range(columns):
                gram[left][right] += row[left] * row[right]
    inverse = _invert(gram)
    return tuple(sum(inverse[row][column] * rhs[column] for column in range(columns)) for row in range(columns))


def linear_backdoor_effect(
    outcome: Sequence[float],
    treatment: Sequence[float],
    covariates: Sequence[Sequence[float]],
    *,
    assumptions: CausalAssumptions,
) -> CausalEffectEstimate:
    """Estimate a linear back-door-adjusted treatment effect.

    Identification is fail-closed: every assumption must be explicitly true.
    The estimator fits ``outcome ~ 1 + treatment + covariates`` and returns the
    treatment coefficient. The caller remains responsible for supplying a
    causally valid adjustment set; this function never discovers one from data.
    """
    report = audit_causal_assumptions(assumptions)
    if not assumptions.identified:
        failures = [check.name for check in report.failures]
        warnings = [check.name for check in report.warnings]
        unresolved = failures + warnings
        raise ValueError("causal effect is not identified; unresolved assumptions: " + ", ".join(unresolved))

    y = _finite_vector(outcome, name="outcome")
    treatment_values = _finite_vector(treatment, name="treatment")
    if len(y) != len(treatment_values):
        raise ValueError("outcome and treatment must have equal length")
    if not y:
        raise ValueError("causal estimator requires at least one observation")
    if len(set(treatment_values)) < 2:
        raise ValueError("treatment must vary across observations")
    if len(covariates) != len(y):
        raise ValueError("covariates must have one row per observation")

    width = len(covariates[0]) if covariates else 0
    rows: list[list[float]] = []
    for index, raw_row in enumerate(covariates):
        if len(raw_row) != width:
            raise ValueError("covariate rows must have equal width")
        row = _finite_vector(raw_row, name=f"covariates row {index}")
        rows.append([1.0, treatment_values[index], *row])

    coefficients = _ols_coefficients(rows, y)
    return CausalEffectEstimate(
        effect=coefficients[1],
        method="linear-backdoor-adjustment",
        n_observations=len(y),
        adjustment_count=width,
        assumptions=assumptions,
    )


def randomized_mean_effect(
    treated_outcomes: Sequence[float],
    control_outcomes: Sequence[float],
    *,
    randomized: bool,
) -> CausalEffectEstimate:
    """Estimate an average treatment effect from a declared randomized design."""
    if not randomized:
        raise ValueError("randomized_mean_effect requires an explicitly randomized design")
    treated = _finite_vector(treated_outcomes, name="treated_outcomes")
    control = _finite_vector(control_outcomes, name="control_outcomes")
    if not treated or not control:
        raise ValueError("treated and control groups must both be non-empty")
    effect = sum(treated) / len(treated) - sum(control) / len(control)
    return CausalEffectEstimate(
        effect=effect,
        method="randomized-difference-in-means",
        n_observations=len(treated) + len(control),
        adjustment_count=0,
        assumptions=None,
    )


__all__ = [
    "CausalAssumptions",
    "CausalEffectEstimate",
    "audit_causal_assumptions",
    "linear_backdoor_effect",
    "randomized_mean_effect",
]
