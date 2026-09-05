"""Typed data structures for reproducible scientific workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class LanguageMode(str, Enum):
    """How explanations should be presented to the user."""

    EVERYDAY = "everyday"
    TECHNICAL = "technical"
    BOTH = "both"


class AnalysisKind(str, Enum):
    """Explicit scientific task families understood by the default autopilot."""

    PARAMETER_FIT = "parameter_fit"
    SIMULATION = "simulation"
    STATISTICAL_TEST = "statistical_test"
    SYMBOLIC_VERIFICATION = "symbolic_verification"
    CONSTRAINT_VERIFICATION = "constraint_verification"


class StepStatus(str, Enum):
    """Execution state of a workflow step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    RESUMED = "resumed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True)
class AnalysisRequest:
    """A scientific question plus execution-policy constraints."""

    question: str
    language: LanguageMode = LanguageMode.BOTH
    require_plan_approval: bool = True
    prefer_local: bool = True
    sensitive_data: bool = False
    analysis_kind: AnalysisKind | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("analysis question must not be empty")
        if self.seed is not None and (not isinstance(self.seed, int) or isinstance(self.seed, bool)):
            raise ValueError("seed must be an integer or null")


@dataclass(frozen=True)
class PlanStep:
    """One executable action in an analysis plan."""

    id: str
    description: str
    method: str
    rationale: str
    requires_approval: bool = False


@dataclass(frozen=True)
class Recommendation:
    """Ranked method recommendation with visible alternatives."""

    recommended: str
    rationale: str
    alternatives: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnalysisPlan:
    """Ordered, reviewable plan for a scientific analysis."""

    request: AnalysisRequest
    steps: tuple[PlanStep, ...]
    recommendation: Recommendation | None = None


@dataclass
class ExecutionEvent:
    """Immutable-in-practice audit event emitted during execution."""

    step_id: str
    status: StepStatus
    message: str


@dataclass
class ExecutionTrace:
    """Ordered audit trail for decisions and execution outcomes."""

    events: list[ExecutionEvent] = field(default_factory=list)

    def record(self, step_id: str, status: StepStatus, message: str) -> None:
        """Append an execution event."""
        self.events.append(ExecutionEvent(step_id=step_id, status=status, message=message))


@dataclass
class ScientificResult:
    """Layered scientific result plus reproducibility metadata."""

    summary: str
    details: dict[str, object] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    trace: ExecutionTrace = field(default_factory=ExecutionTrace)
