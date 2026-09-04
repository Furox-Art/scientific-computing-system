"""Validation result models used by scientific audit checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class CheckStatus(str, Enum):
    """Outcome of one scientific validation check."""

    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


@dataclass(frozen=True)
class ValidationCheck:
    """One machine-readable validation finding."""

    name: str
    status: CheckStatus
    message: str
    details: dict[str, object] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Collection of validation findings with a conservative pass rule."""

    checks: list[ValidationCheck] = field(default_factory=list)

    def add(self, check: ValidationCheck) -> None:
        """Append one check."""
        self.checks.append(check)

    @property
    def passed(self) -> bool:
        """True when no check failed."""
        return all(check.status is not CheckStatus.FAIL for check in self.checks)

    @property
    def warnings(self) -> tuple[ValidationCheck, ...]:
        """Warning-level findings."""
        return tuple(check for check in self.checks if check.status is CheckStatus.WARNING)

    @property
    def failures(self) -> tuple[ValidationCheck, ...]:
        """Failure-level findings."""
        return tuple(check for check in self.checks if check.status is CheckStatus.FAIL)
