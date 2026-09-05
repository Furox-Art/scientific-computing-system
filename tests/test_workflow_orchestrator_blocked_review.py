"""Regression coverage for REVIEW methods under an already-blocked final gate."""

from __future__ import annotations

import pytest

import cds.workflow.orchestrator as orchestrator_module
from cds.validation import ValidationReport
from cds.workflow import (
    AnalysisPlan,
    AnalysisRequest,
    ConditionOperator,
    GateDecision,
    GatePolicy,
    GateStatus,
    MethodCandidate,
    MethodSelection,
    MethodSelectionContext,
    OrchestrationRecord,
    PlannedAction,
    PlanStep,
    ProblemProfile,
    ResearchBlueprint,
    ResearchOrchestrator,
    SelectionCondition,
)


def _profile(_request: AnalysisRequest) -> ProblemProfile:
    return ProblemProfile(
        label="regression",
        rationale="exercise blocked-gate preservation",
        selection_context=MethodSelectionContext(),
    )


def _planner(
    _request: AnalysisRequest,
    _profile_value: ProblemProfile,
    _selection: MethodSelection,
) -> ResearchBlueprint:
    step = PlanStep(
        id="analyze",
        description="run deterministic analysis",
        method="provisional",
        rationale="exercise the final gate transition",
    )
    return ResearchBlueprint(
        steps=(step,),
        actions=(PlannedAction("analyze", lambda _context: 1),),
    )


def test_review_method_preserves_existing_blocked_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A REVIEW method must append context without weakening a BLOCKED gate."""
    candidate = MethodCandidate(
        name="provisional",
        rationale="requires unresolved evidence",
        requirements=(
            SelectionCondition(
                "assumption-known",
                ConditionOperator.EQ,
                True,
                "assumption must be established",
            ),
        ),
    )

    def forced_blocked_gate(
        _plan: AnalysisPlan,
        _reports: tuple[ValidationReport, ...],
        *,
        policy: GatePolicy | None = None,
    ) -> GateDecision:
        del policy
        return GateDecision(status=GateStatus.BLOCKED, reasons=("forced scientific block",))

    monkeypatch.setattr(orchestrator_module, "evaluate_research_gate", forced_blocked_gate)
    orchestrator = ResearchOrchestrator(
        candidates=(candidate,),
        classifier=_profile,
        planner=_planner,
    )

    result = orchestrator.run(
        AnalysisRequest("blocked review case", require_plan_approval=False)
    )
    record = result.details["_orchestration"]
    assert isinstance(record, OrchestrationRecord)
    assert record.gate.status is GateStatus.BLOCKED
    assert record.gate.reasons == (
        "forced scientific block",
        "selected method still has unknown mandatory suitability evidence",
    )
    assert result.summary == "Scientific analysis completed, but the final conclusion is blocked."
