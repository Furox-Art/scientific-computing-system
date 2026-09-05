"""Coverage for an already-present method-review gate reason."""

from __future__ import annotations

import pytest

from cds.workflow import (
    AnalysisRequest,
    ConditionOperator,
    GateDecision,
    GateStatus,
    MethodCandidate,
    MethodSelection,
    MethodSelectionContext,
    PlannedAction,
    PlanStep,
    ProblemProfile,
    ResearchBlueprint,
    ResearchOrchestrator,
    SelectionCondition,
)
from cds.workflow.engine import ExecutionContext


def test_existing_method_review_reason_is_not_duplicated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A REVIEW method must preserve, not duplicate, an identical gate reason."""
    review_reason = "selected method still has unknown mandatory suitability evidence"
    candidate = MethodCandidate(
        name="provisional",
        rationale="method depends on unresolved evidence",
        requirements=(
            SelectionCondition(
                "assumption-known",
                ConditionOperator.EQ,
                True,
                "assumption must be established",
            ),
        ),
    )

    def classify(_request: AnalysisRequest) -> ProblemProfile:
        return ProblemProfile(
            "provisional-analysis",
            "exercise an unresolved method requirement",
            MethodSelectionContext(),
        )

    def planner(
        _request: AnalysisRequest,
        _profile: ProblemProfile,
        _selection: MethodSelection,
    ) -> ResearchBlueprint:
        step = PlanStep("analyze", "run analysis", "provisional", "deterministic test step")

        def analyze(_context: ExecutionContext) -> object:
            return 1

        return ResearchBlueprint(
            steps=(step,),
            actions=(PlannedAction("analyze", analyze),),
        )

    def gate_with_existing_reason(*_args: object, **_kwargs: object) -> GateDecision:
        return GateDecision(GateStatus.REVIEW, reasons=(review_reason,))

    monkeypatch.setattr(
        "cds.workflow.orchestrator.evaluate_research_gate",
        gate_with_existing_reason,
    )
    orchestrator = ResearchOrchestrator(
        candidates=(candidate,),
        classifier=classify,
        planner=planner,
    )

    result = orchestrator.run(
        AnalysisRequest("provisional analysis", require_plan_approval=False)
    )

    assert result.summary == "Scientific analysis completed but requires review before conclusion."
    assert result.warnings.count(review_reason) == 1
