"""Tests for approval-gated scientific workflow orchestration."""

from __future__ import annotations

import pytest

from cds.workflow import (
    AnalysisPlan,
    AnalysisRequest,
    ExecutionContext,
    LanguageMode,
    PlanStep,
    Recommendation,
    ResearchWorkflow,
    StepStatus,
)


def _plan(*, require_approval: bool = True) -> AnalysisPlan:
    request = AnalysisRequest(
        question="Estimate a parameter",
        language=LanguageMode.BOTH,
        require_plan_approval=require_approval,
    )
    return AnalysisPlan(
        request=request,
        steps=(
            PlanStep("load", "load data", "local", "keep data local"),
            PlanStep(
                "fit",
                "fit model",
                "advanced-fit",
                "compare robust methods",
                requires_approval=True,
            ),
            PlanStep("audit", "audit result", "validation", "independent checks"),
        ),
        recommendation=Recommendation(
            recommended="advanced-fit",
            rationale="supports diagnostics",
            alternatives=("least-squares",),
        ),
    )


def test_register_rejects_unknown_and_duplicate_steps() -> None:
    workflow = ResearchWorkflow(_plan())
    with pytest.raises(KeyError, match="unknown workflow step"):
        workflow.register("missing", lambda _ctx: None)

    workflow.register("load", lambda _ctx: [1, 2, 3])
    with pytest.raises(ValueError, match="already registered"):
        workflow.register("load", lambda _ctx: [4])


def test_plan_approval_is_required_before_execution() -> None:
    workflow = ResearchWorkflow(_plan())
    workflow.register("load", lambda _ctx: [1])
    result = workflow.execute()
    assert "not approved" in result.summary
    assert result.trace.events[-1].status is StepStatus.SKIPPED

    denied = workflow.execute(approve=lambda _step: False)
    assert "not approved" in denied.summary


def test_workflow_executes_in_order_and_skips_unregistered_step() -> None:
    workflow = ResearchWorkflow(_plan())
    workflow.register("load", lambda _ctx: [2, 4])

    def fit_action(ctx: ExecutionContext) -> int:
        loaded = ctx.values["load"]
        assert isinstance(loaded, list)
        return sum(int(value) for value in loaded)

    workflow.register("fit", fit_action)

    approved_ids: list[str] = []

    def approve(step: PlanStep | None) -> bool:
        approved_ids.append("plan" if step is None else step.id)
        return True

    result = workflow.execute(approve=approve)
    assert result.summary == "Analysis plan completed."
    assert result.details["fit"] == 6
    assert approved_ids == ["plan", "fit"]
    audit_event = next(event for event in result.trace.events if event.step_id == "audit")
    assert audit_event.status is StepStatus.SKIPPED


def test_step_level_approval_can_skip_material_change() -> None:
    workflow = ResearchWorkflow(_plan())
    workflow.register("load", lambda _ctx: "loaded")
    workflow.register("fit", lambda _ctx: "fit")
    workflow.register("audit", lambda _ctx: "audited")

    def approve(step: PlanStep | None) -> bool:
        return step is None or step.id != "fit"

    result = workflow.execute(approve=approve)
    assert "fit" not in result.details
    assert result.details["audit"] == "audited"
    fit_event = [event for event in result.trace.events if event.step_id == "fit"][-1]
    assert fit_event.status is StepStatus.SKIPPED


def test_failure_stops_workflow_and_records_error() -> None:
    workflow = ResearchWorkflow(_plan(require_approval=False))
    workflow.register("load", lambda _ctx: "loaded")

    def fail(_ctx: ExecutionContext) -> object:
        raise RuntimeError("boom")

    workflow.register("fit", fail)
    workflow.register("audit", lambda _ctx: "should not run")
    result = workflow.execute(approve=lambda _step: True)
    assert result.summary == "Analysis stopped at step 'fit'."
    assert result.details == {"load": "loaded"}
    assert result.warnings == ["boom"]
    assert result.trace.events[-1].status is StepStatus.FAILED


def test_custom_context_is_reused_without_plan_approval() -> None:
    workflow = ResearchWorkflow(_plan(require_approval=False))
    context = ExecutionContext(values={"seed": 7})

    def use_seed(ctx: ExecutionContext) -> int:
        seed = ctx.values["seed"]
        assert isinstance(seed, int)
        return seed + 1

    workflow.register("load", use_seed)
    result = workflow.execute(context=context)
    assert result.details["seed"] == 7
    assert result.details["load"] == 8
