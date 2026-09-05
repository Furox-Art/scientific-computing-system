"""Distribution-shift and out-of-distribution checks with explicit thresholds."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from cds.validation.report import CheckStatus, ValidationCheck


@dataclass(frozen=True)
class FeatureDrift:
    """Empirical distribution shift for one numeric feature."""

    index: int
    ks_distance: float
    drifted: bool


@dataclass(frozen=True)
class DriftReport:
    """Feature-wise two-sample empirical Kolmogorov-Smirnov distances."""

    threshold: float
    reference_rows: int
    current_rows: int
    features: tuple[FeatureDrift, ...]

    @property
    def any_drift(self) -> bool:
        return any(feature.drifted for feature in self.features)


@dataclass(frozen=True)
class OODObservation:
    """Maximum standardized reference-distance for one observation."""

    index: int
    max_abs_z: float
    out_of_distribution: bool


@dataclass(frozen=True)
class OODReport:
    """Reference-standardized OOD screening report."""

    z_threshold: float
    observations: tuple[OODObservation, ...]

    @property
    def any_ood(self) -> bool:
        return any(observation.out_of_distribution for observation in self.observations)


def _finite_vector(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    if any(not math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def empirical_ks_distance(reference: Sequence[float], current: Sequence[float]) -> float:
    """Return the exact empirical two-sample Kolmogorov-Smirnov distance."""
    left = sorted(_finite_vector(reference, name="reference"))
    right = sorted(_finite_vector(current, name="current"))
    left_count = len(left)
    right_count = len(right)
    left_index = 0
    right_index = 0
    distance = 0.0

    while left_index < left_count or right_index < right_count:
        if right_index >= right_count or (
            left_index < left_count and left[left_index] <= right[right_index]
        ):
            point = left[left_index]
        else:
            point = right[right_index]
        while left_index < left_count and left[left_index] <= point:
            left_index += 1
        while right_index < right_count and right[right_index] <= point:
            right_index += 1
        distance = max(
            distance,
            abs(left_index / left_count - right_index / right_count),
        )
    return distance


def _matrix(rows: Sequence[Sequence[float]], *, name: str) -> tuple[tuple[float, ...], ...]:
    if not rows:
        raise ValueError(f"{name} must contain at least one row")
    width = len(rows[0])
    if width == 0:
        raise ValueError(f"{name} rows must contain at least one feature")
    matrix: list[tuple[float, ...]] = []
    for index, row in enumerate(rows):
        if len(row) != width:
            raise ValueError(f"{name} rows must have equal width")
        values = tuple(float(value) for value in row)
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"{name} row {index} must contain only finite values")
        matrix.append(values)
    return tuple(matrix)


def feature_drift(
    reference: Sequence[Sequence[float]],
    current: Sequence[Sequence[float]],
    *,
    threshold: float = 0.2,
) -> DriftReport:
    """Compare each feature's empirical distribution using an explicit KS threshold."""
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and between 0 and 1")
    baseline = _matrix(reference, name="reference")
    observed = _matrix(current, name="current")
    if len(baseline[0]) != len(observed[0]):
        raise ValueError("reference and current matrices must have equal feature count")

    features: list[FeatureDrift] = []
    for index in range(len(baseline[0])):
        distance = empirical_ks_distance(
            [row[index] for row in baseline],
            [row[index] for row in observed],
        )
        features.append(FeatureDrift(index, distance, distance > threshold))
    return DriftReport(threshold, len(baseline), len(observed), tuple(features))


def screen_ood(
    reference: Sequence[Sequence[float]],
    observations: Sequence[Sequence[float]],
    *,
    z_threshold: float = 4.0,
) -> OODReport:
    """Flag observations far from the reference mean in standardized feature space.

    Reference standard deviations use the population convention. A zero-variance
    reference feature yields z=0 for the same value and z=+inf for any different
    value, preventing a constant training feature from silently accepting drift.
    """
    if not math.isfinite(z_threshold) or z_threshold <= 0.0:
        raise ValueError("z_threshold must be finite and positive")
    baseline = _matrix(reference, name="reference")
    current = _matrix(observations, name="observations")
    width = len(baseline[0])
    if len(current[0]) != width:
        raise ValueError("reference and observations must have equal feature count")

    means = tuple(sum(row[index] for row in baseline) / len(baseline) for index in range(width))
    deviations = tuple(
        math.sqrt(
            sum((row[index] - means[index]) ** 2 for row in baseline) / len(baseline)
        )
        for index in range(width)
    )

    screened: list[OODObservation] = []
    for row_index, row in enumerate(current):
        z_values: list[float] = []
        for index, value in enumerate(row):
            deviation = deviations[index]
            if deviation == 0.0:
                z_value = 0.0 if value == means[index] else math.inf
            else:
                z_value = abs(value - means[index]) / deviation
            z_values.append(z_value)
        maximum = max(z_values)
        screened.append(OODObservation(row_index, maximum, maximum > z_threshold))
    return OODReport(z_threshold, tuple(screened))


def drift_validation_check(report: DriftReport, *, fail_on_drift: bool = False) -> ValidationCheck:
    """Translate a drift report into the common validation gate format."""
    drifted = [feature.index for feature in report.features if feature.drifted]
    if not drifted:
        return ValidationCheck("distribution_drift", CheckStatus.PASS, "no feature exceeded drift threshold")
    status = CheckStatus.FAIL if fail_on_drift else CheckStatus.WARNING
    return ValidationCheck(
        "distribution_drift",
        status,
        f"features exceeded KS drift threshold: {drifted}",
    )


def ood_validation_check(report: OODReport, *, fail_on_ood: bool = False) -> ValidationCheck:
    """Translate an OOD report into the common validation gate format."""
    flagged = [observation.index for observation in report.observations if observation.out_of_distribution]
    if not flagged:
        return ValidationCheck("out_of_distribution", CheckStatus.PASS, "no observation exceeded OOD threshold")
    status = CheckStatus.FAIL if fail_on_ood else CheckStatus.WARNING
    return ValidationCheck(
        "out_of_distribution",
        status,
        f"observations exceeded OOD threshold: {flagged}",
    )


__all__ = [
    "DriftReport",
    "FeatureDrift",
    "OODObservation",
    "OODReport",
    "drift_validation_check",
    "empirical_ks_distance",
    "feature_drift",
    "ood_validation_check",
    "screen_ood",
]
