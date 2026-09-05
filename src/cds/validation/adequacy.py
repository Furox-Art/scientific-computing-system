"""Policy-driven data sufficiency checks for scientific workflows.

The module deliberately avoids universal sample-size heuristics. Callers declare
what their method requires, and CDS evaluates the observed data against those
requirements in a deterministic, auditable way.
"""

from __future__ import annotations

from dataclasses import dataclass

from cds.validation.report import CheckStatus, ValidationCheck, ValidationReport


@dataclass(frozen=True)
class DataProfile:
    """Minimal data facts needed for method-specific sufficiency checks."""

    observations: int
    total_values: int = 0
    missing_values: int = 0
    parameters: int = 0

    def __post_init__(self) -> None:
        if self.observations < 0:
            raise ValueError("observations must be non-negative")
        if self.total_values < 0:
            raise ValueError("total_values must be non-negative")
        if self.missing_values < 0:
            raise ValueError("missing_values must be non-negative")
        if self.missing_values > self.total_values:
            raise ValueError("missing_values cannot exceed total_values")
        if self.parameters < 0:
            raise ValueError("parameters must be non-negative")

    @property
    def missing_fraction(self) -> float:
        """Fraction of scalar values marked missing, with empty data defined as zero."""
        if self.total_values == 0:
            return 0.0
        return self.missing_values / self.total_values


@dataclass(frozen=True)
class DataRequirement:
    """Explicit sufficiency requirements declared by a scientific method."""

    min_observations: int = 1
    max_missing_fraction: float = 1.0
    min_observations_per_parameter: float | None = None

    def __post_init__(self) -> None:
        if self.min_observations < 0:
            raise ValueError("min_observations must be non-negative")
        if not 0.0 <= self.max_missing_fraction <= 1.0:
            raise ValueError("max_missing_fraction must lie in [0, 1]")
        if (
            self.min_observations_per_parameter is not None
            and self.min_observations_per_parameter <= 0
        ):
            raise ValueError("min_observations_per_parameter must be positive when provided")


def assess_data_adequacy(
    profile: DataProfile,
    requirement: DataRequirement,
    *,
    name_prefix: str = "data-adequacy",
) -> ValidationReport:
    """Evaluate data facts against caller-declared method requirements."""
    if not name_prefix:
        raise ValueError("name_prefix must not be empty")

    report = ValidationReport()

    observation_status = (
        CheckStatus.PASS
        if profile.observations >= requirement.min_observations
        else CheckStatus.FAIL
    )
    report.add(
        ValidationCheck(
            name=f"{name_prefix}:observations",
            status=observation_status,
            message=(
                "observation count satisfies the declared minimum"
                if observation_status is CheckStatus.PASS
                else "observation count is below the declared minimum"
            ),
            details={
                "observations": profile.observations,
                "minimum": requirement.min_observations,
            },
        )
    )

    missing_fraction = profile.missing_fraction
    missing_status = (
        CheckStatus.PASS
        if missing_fraction <= requirement.max_missing_fraction
        else CheckStatus.FAIL
    )
    report.add(
        ValidationCheck(
            name=f"{name_prefix}:missingness",
            status=missing_status,
            message=(
                "missingness satisfies the declared maximum"
                if missing_status is CheckStatus.PASS
                else "missingness exceeds the declared maximum"
            ),
            details={
                "missing_fraction": missing_fraction,
                "maximum": requirement.max_missing_fraction,
                "missing_values": profile.missing_values,
                "total_values": profile.total_values,
            },
        )
    )

    ratio_requirement = requirement.min_observations_per_parameter
    if ratio_requirement is not None:
        if profile.parameters == 0:
            report.add(
                ValidationCheck(
                    name=f"{name_prefix}:parameter-support",
                    status=CheckStatus.FAIL,
                    message=(
                        "parameter count is required to evaluate observations per parameter"
                    ),
                    details={
                        "observations": profile.observations,
                        "parameters": profile.parameters,
                        "minimum_ratio": ratio_requirement,
                    },
                )
            )
        else:
            ratio = profile.observations / profile.parameters
            ratio_status = CheckStatus.PASS if ratio >= ratio_requirement else CheckStatus.FAIL
            report.add(
                ValidationCheck(
                    name=f"{name_prefix}:parameter-support",
                    status=ratio_status,
                    message=(
                        "observations per parameter satisfy the declared minimum"
                        if ratio_status is CheckStatus.PASS
                        else "observations per parameter are below the declared minimum"
                    ),
                    details={
                        "observations": profile.observations,
                        "parameters": profile.parameters,
                        "ratio": ratio,
                        "minimum_ratio": ratio_requirement,
                    },
                )
            )

    return report
