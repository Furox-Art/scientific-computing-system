"""Decision gates that connect workflow plans to scientific validation.

A gate does not invent scientific thresholds or silently rewrite a plan. It
checks whether the caller supplied an explicit recommendation, whether required
independent checks actually ran, and whether any validation finding blocks or
qualifies a final conclusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from cds.validation import CheckStatus, ValidationReport
from cds.workflow.types import AnalysisPlan


class GateStatus(str, Enum):
    """Whether a workflow may present an unqualified final conclusion."""

    READY = "ready"
    REVIEW = "review"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class GatePolicy:
    """Explicit requirements for accepting a scientific workflow result."""

    require_recommendation: bool = True
    require_alternatives: bool = True
    require_validation: bool = True
    required_checks: tuple[str, ...] = ()
    warnings_require_review: bool = True

    def __post_init__(self) -> None:
        if any(not name for name in self.required_checks):
            raise ValueError("required check names must not be empty")
        if len(set(self.required_checks)) != len(self.required_checks):
            raise ValueError("required check names must be unique")


@dataclass(frozen=True)
class GateDecision:
    """Machine-readable outcome of a scientific decision gate."""

    status: GateStatus
    reasons: tuple[str, ...] = ()
    missing_checks: tuple[str, ...] = ()

    @property
    def allows_conclusion(self) -> bool:
        """True only when no scientific or policy qualification remains."""
        return self.status is GateStatus.READY


def evaluate_research_gate(
    plan: AnalysisPlan,
    reports: tuple[ValidationReport, ...],
    *,
    policy: GatePolicy | None = None,
) -> GateDecision:
    """Evaluate whether a planned analysis may state a final conclusion.

    Failures and missing mandatory checks block a conclusion. Warning-level
    findings trigger review by default. A recommendation is required by default
    because CDS should not silently execute an arbitrary method, while visible
    alternatives are treated as a review issue rather than a hard scientific
    failure.
    """
    active_policy = policy if policy is not None else GatePolicy()
    checks = tuple(check for report in reports for check in report.checks)
    check_names = {check.name for check in checks}

    blocking: list[str] = []
    review: list[str] = []

    recommendation = plan.recommendation
    if active_policy.require_recommendation and recommendation is None:
        blocking.append("analysis plan has no explicit method recommendation")
    elif (
        active_policy.require_alternatives
        and recommendation is not None
        and not recommendation.alternatives
    ):
        review.append("recommended method has no visible alternatives")

    if active_policy.require_validation and not checks:
        blocking.append("no scientific validation checks were supplied")

    missing_checks = tuple(
        name for name in active_policy.required_checks if name not in check_names
    )
    if missing_checks:
        blocking.append("required scientific validation checks are missing")

    failures = tuple(check for check in checks if check.status is CheckStatus.FAIL)
    if failures:
        names = ", ".join(check.name for check in failures)
        blocking.append(f"validation failures: {names}")

    if active_policy.warnings_require_review:
        warnings = tuple(check for check in checks if check.status is CheckStatus.WARNING)
        if warnings:
            names = ", ".join(check.name for check in warnings)
            review.append(f"validation warnings require review: {names}")

    if blocking:
        return GateDecision(
            status=GateStatus.BLOCKED,
            reasons=tuple(blocking + review),
            missing_checks=missing_checks,
        )
    if review:
        return GateDecision(
            status=GateStatus.REVIEW,
            reasons=tuple(review),
            missing_checks=missing_checks,
        )
    return GateDecision(status=GateStatus.READY, missing_checks=missing_checks)
