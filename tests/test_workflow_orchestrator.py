"""Tests for the end-to-end scientific research orchestrator."""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import cast

import pytest

from cds.provenance import RunManifest
from cds.tools import ToolRegistry, ToolSpec
from cds.validation import CheckStatus, ValidationCheck, ValidationReport
from cds.workflow import (
    AnalysisRequest,
    ConditionOperator,
    GatePolicy,
    GateStatus,
    IndependentValidator,
    MethodCandidate,
    MethodSelection,
    MethodSelectionContext,
    OrchestrationRecord,
    PlannedAction,
    PlanStep,
    ProblemProfile,
    ResearchBlueprint,
    ResearchOrchestrator,
    ScientificResult,
    SelectionCondition,
    ToolPlannedAction,
)
from cds.workflow.engine import ExecutionContext
from cds.workflow.orchestrator import ValidatorAction


def _record(result: ScientificResult) -> OrchestrationRecord:
    record = result.details["_orchestration"]
    assert isinstance(record, OrchestrationRecord)
    return record


def _profile(_request: AnalysisRequest) -> ProblemProfile:
    return ProblemProfile(
        label="regression",
        rationale="the question asks for a fitted quantitative relationship",
        selection_context=MethodSelectionContext(),
    )


def _candidates() -> tuple[MethodCandidate, ...]:
    return (
        MethodCandidate(name="primary", rationale="primary deterministic method", base_score=2.0),
        MethodCandidate(name="alternate", rationale="independent alternative", base_score=1.0),
    )


def _pass_report(_context: ExecutionContext, _result: object) -> ValidationReport:
    return ValidationReport(
        checks=[ValidationCheck("independent", CheckStatus.PASS, "independent check passed")]
    )


def _normal_planner(
    _request: AnalysisRequest,
    _profile_value: ProblemProfile,
    _selection: MethodSelection,
) -> ResearchBlueprint:
    step = PlanStep(
        id="analyze",
        description="execute selected analysis",
        method="selected-method",
        rationale="run the selected method deterministically",
    )

    def analyze(context: ExecutionContext) -> object:
        context.values["side-effect"] = "recorded"
        return 42

    return ResearchBlueprint(steps=(step,), actions=(PlannedAction("analyze", analyze),))


def test_orchestrator_ready_pipeline_records_full_audit_trail() -> None:
    request = AnalysisRequest("fit relationship")
    orchestrator = ResearchOrchestrator(
        candidates=_candidates(),
        classifier=_profile,
        planner=_normal_planner,
        validators=(IndependentValidator("independent", _pass_report),),
        gate_policy=GatePolicy(required_checks=("independent",)),
    )

    def approve(_step: PlanStep | None) -> bool:
        return True

    result = orchestrator.run(request, approve=approve)
    record = _record(result)

    assert result.summary == "Scientific analysis completed and passed the final research gate."
    assert result.details["analyze"] == 42
    assert result.details["side-effect"] == "recorded"
    assert record.gate.status is GateStatus.READY
    assert record.selection.recommended is not None
    assert record.selection.recommended.candidate.name == "primary"
    assert record.plan is not None
    assert record.plan.recommendation is not None
    assert record.plan.recommendation.alternatives == ("alternate",)
    assert len(record.validation_reports) == 1
    assert [decision.action for decision in record.manifest.decisions] == [
        "classify problem as regression",
        "select primary",
        "final research gate",
    ]
    assert record.manifest.decisions[1].approved_by_user
    assert record.manifest.decisions[-1].rationale == "all configured research gates passed"


def test_orchestrator_uses_existing_context_manifest_and_no_required_approval() -> None:
    request = AnalysisRequest("fit relationship", require_plan_approval=False)
    context = ExecutionContext(values={"input": 5})
    manifest = RunManifest.create(
        request.question,
        run_id="fixed-run",
        created_utc="2026-09-05T00:00:00+00:00",
    )
    orchestrator = ResearchOrchestrator(
        candidates=_candidates(),
        classifier=_profile,
        planner=_normal_planner,
        validators=(IndependentValidator("independent", _pass_report),),
    )

    result = orchestrator.run(request, context=context, manifest=manifest)
    record = _record(result)

    assert result.details["input"] == 5
    assert record.manifest is manifest
    assert not record.manifest.decisions[1].approved_by_user


def test_all_methods_blocked_stops_before_planning() -> None:
    candidate = MethodCandidate(
        name="needs-capability",
        rationale="requires a missing capability",
        capabilities=("other",),
    )

    def classify(_request: AnalysisRequest) -> ProblemProfile:
        return ProblemProfile(
            "classification",
            "explicit capability requirement",
            MethodSelectionContext(required_capabilities=("required",)),
        )

    planner_called = False

    def planner(
        _request: AnalysisRequest,
        _profile_value: ProblemProfile,
        _selection: MethodSelection,
    ) -> ResearchBlueprint:
        nonlocal planner_called
        planner_called = True
        return _normal_planner(_request, _profile_value, _selection)

    orchestrator = ResearchOrchestrator(
        candidates=(candidate,),
        classifier=classify,
        planner=planner,
    )
    result = orchestrator.run(AnalysisRequest("blocked question"))
    record = _record(result)

    assert not planner_called
    assert record.plan is None
    assert record.gate.status is GateStatus.BLOCKED
    assert result.warnings == ["all candidate scientific methods are blocked"]


def test_tool_step_is_selected_loaded_and_recorded() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="math",
            module="math",
            distribution="distribution-that-does-not-exist",
            capabilities=("calculation",),
            purpose="stdlib numerical backend for testing",
        )
    )

    def planner(
        _request: AnalysisRequest,
        _profile_value: ProblemProfile,
        _selection: MethodSelection,
    ) -> ResearchBlueprint:
        step = PlanStep(
            id="calculate",
            description="calculate",
            method="backend",
            rationale="exercise capability selection",
        )

        def calculate(_context: ExecutionContext, module: ModuleType) -> object:
            sqrt = cast(Callable[[float], float], getattr(module, "sqrt"))
            return sqrt(81.0)

        return ResearchBlueprint(
            steps=(step,),
            tool_actions=(ToolPlannedAction("calculate", "calculation", calculate),),
        )

    orchestrator = ResearchOrchestrator(
        candidates=_candidates(),
        classifier=_profile,
        planner=planner,
        validators=(IndependentValidator("independent", _pass_report),),
        registry=registry,
    )
    result = orchestrator.run(AnalysisRequest("tool calculation", require_plan_approval=False))
    record = _record(result)

    assert result.details["calculate"] == 9.0
    assert record.manifest.tool_versions == {"math": "unknown"}
    assert any(
        decision.action == "select math for calculation" for decision in record.manifest.decisions
    )


def test_missing_tool_blocks_before_execution() -> None:
    def planner(
        _request: AnalysisRequest,
        _profile_value: ProblemProfile,
        _selection: MethodSelection,
    ) -> ResearchBlueprint:
        step = PlanStep("tool", "tool step", "tool", "needs unavailable tool")

        def use_tool(_context: ExecutionContext, _module: ModuleType) -> object:
            return "unused"

        return ResearchBlueprint(
            steps=(step,),
            tool_actions=(ToolPlannedAction("tool", "missing-capability", use_tool),),
        )

    orchestrator = ResearchOrchestrator(
        candidates=_candidates(),
        classifier=_profile,
        planner=planner,
        registry=ToolRegistry(),
    )
    result = orchestrator.run(AnalysisRequest("missing tool"))
    record = _record(result)

    assert record.gate.status is GateStatus.BLOCKED
    assert (
        result.summary == "Analysis is blocked because a required scientific tool is unavailable."
    )
    assert "tool registration failed" in result.warnings[0]


def test_denied_plan_approval_blocks_validation_and_conclusion() -> None:
    validator_called = False

    def validator(_context: ExecutionContext, _result: object) -> ValidationReport:
        nonlocal validator_called
        validator_called = True
        return ValidationReport()

    orchestrator = ResearchOrchestrator(
        candidates=_candidates(),
        classifier=_profile,
        planner=_normal_planner,
        validators=(IndependentValidator("never", validator),),
    )

    def deny(_step: PlanStep | None) -> bool:
        return False

    result = orchestrator.run(AnalysisRequest("approval required"), approve=deny)
    record = _record(result)

    assert not validator_called
    assert record.gate.status is GateStatus.BLOCKED
    assert "workflow execution incomplete: analyze" in record.gate.reasons
    assert not record.manifest.decisions[1].approved_by_user


def test_step_approval_without_callback_is_fail_closed() -> None:
    def planner(
        _request: AnalysisRequest,
        _profile_value: ProblemProfile,
        _selection: MethodSelection,
    ) -> ResearchBlueprint:
        step = PlanStep(
            "consequential",
            "consequential step",
            "method",
            "requires explicit approval",
            requires_approval=True,
        )
        return ResearchBlueprint(
            steps=(step,),
            actions=(PlannedAction("consequential", lambda _context: "ran"),),
        )

    orchestrator = ResearchOrchestrator(
        candidates=_candidates(),
        classifier=_profile,
        planner=planner,
    )
    result = orchestrator.run(AnalysisRequest("step approval", require_plan_approval=False))

    assert _record(result).gate.status is GateStatus.BLOCKED
    assert "consequential" not in result.details


def test_validator_failure_and_invalid_return_are_converted_to_blocking_checks() -> None:
    def exploding(_context: ExecutionContext, _result: object) -> ValidationReport:
        raise RuntimeError("audit crashed")

    def wrong_type(_context: ExecutionContext, _result: object) -> object:
        return "not a report"

    orchestrator = ResearchOrchestrator(
        candidates=_candidates(),
        classifier=_profile,
        planner=_normal_planner,
        validators=(
            IndependentValidator("exploding", exploding),
            IndependentValidator("wrong-type", cast(ValidatorAction, wrong_type)),
        ),
    )
    result = orchestrator.run(AnalysisRequest("validator failures", require_plan_approval=False))
    record = _record(result)

    assert record.gate.status is GateStatus.BLOCKED
    assert [report.failures[0].name for report in record.validation_reports] == [
        "validator:exploding",
        "validator:wrong-type",
    ]
    assert any("audit crashed" in warning for warning in result.warnings)
    assert any("validator must return ValidationReport" in warning for warning in result.warnings)


def test_warning_validator_produces_review() -> None:
    def warning(_context: ExecutionContext, _result: object) -> ValidationReport:
        return ValidationReport(
            checks=[ValidationCheck("diagnostic", CheckStatus.WARNING, "inspect residuals")]
        )

    orchestrator = ResearchOrchestrator(
        candidates=_candidates(),
        classifier=_profile,
        planner=_normal_planner,
        validators=(IndependentValidator("warning", warning),),
    )
    result = orchestrator.run(AnalysisRequest("warning case", require_plan_approval=False))

    assert _record(result).gate.status is GateStatus.REVIEW
    assert result.summary == "Scientific analysis completed but requires review before conclusion."
    assert "inspect residuals" in result.warnings


def test_review_method_cannot_be_promoted_to_ready_by_clean_validation() -> None:
    requirement = SelectionCondition(
        "assumption-known",
        ConditionOperator.EQ,
        True,
        "assumption must be established",
    )
    candidate = MethodCandidate(
        name="provisional",
        rationale="requires unresolved evidence",
        requirements=(requirement,),
    )
    orchestrator = ResearchOrchestrator(
        candidates=(candidate,),
        classifier=_profile,
        planner=_normal_planner,
        validators=(IndependentValidator("independent", _pass_report),),
        gate_policy=GatePolicy(require_alternatives=False),
    )
    result = orchestrator.run(AnalysisRequest("provisional case", require_plan_approval=False))
    record = _record(result)

    assert record.gate.status is GateStatus.REVIEW
    assert record.gate.reasons == (
        "selected method still has unknown mandatory suitability evidence",
    )


def test_review_method_reason_is_appended_to_existing_review() -> None:
    requirement = SelectionCondition("unknown", ConditionOperator.EQ, True, "unknown evidence")
    candidate = MethodCandidate(
        name="provisional",
        rationale="provisional method",
        requirements=(requirement,),
    )
    orchestrator = ResearchOrchestrator(
        candidates=(candidate,),
        classifier=_profile,
        planner=_normal_planner,
        validators=(IndependentValidator("independent", _pass_report),),
    )
    result = orchestrator.run(AnalysisRequest("double review", require_plan_approval=False))
    reasons = _record(result).gate.reasons

    assert reasons[0] == "recommended method has no visible alternatives"
    assert reasons[-1] == "selected method still has unknown mandatory suitability evidence"


def test_manifest_question_classifier_and_planner_contracts_are_enforced() -> None:
    request = AnalysisRequest("contract question")
    manifest = RunManifest.create(
        "different question",
        run_id="mismatch",
        created_utc="2026-09-05T00:00:00+00:00",
    )
    orchestrator = ResearchOrchestrator(
        candidates=_candidates(),
        classifier=_profile,
        planner=_normal_planner,
    )
    with pytest.raises(ValueError, match="manifest question"):
        orchestrator.run(request, manifest=manifest)

    def bad_classifier(_request: AnalysisRequest) -> object:
        return "not a profile"

    bad_classifier_orchestrator = ResearchOrchestrator(
        candidates=_candidates(),
        classifier=cast(Callable[[AnalysisRequest], ProblemProfile], bad_classifier),
        planner=_normal_planner,
    )
    with pytest.raises(TypeError, match="classifier must return ProblemProfile"):
        bad_classifier_orchestrator.run(request)

    def bad_planner(
        _request: AnalysisRequest,
        _profile_value: ProblemProfile,
        _selection: MethodSelection,
    ) -> object:
        return "not a blueprint"

    bad_planner_orchestrator = ResearchOrchestrator(
        candidates=_candidates(),
        classifier=_profile,
        planner=cast(
            Callable[[AnalysisRequest, ProblemProfile, MethodSelection], ResearchBlueprint],
            bad_planner,
        ),
    )
    with pytest.raises(TypeError, match="planner must return ResearchBlueprint"):
        bad_planner_orchestrator.run(request)


def test_orchestrator_and_component_validation() -> None:
    with pytest.raises(ValueError, match="at least one method"):
        ResearchOrchestrator(candidates=(), classifier=_profile, planner=_normal_planner)

    duplicate = MethodCandidate("same", "rationale")
    with pytest.raises(ValueError, match="candidate names"):
        ResearchOrchestrator(
            candidates=(duplicate, duplicate),
            classifier=_profile,
            planner=_normal_planner,
        )

    validator = IndependentValidator("same", _pass_report)
    with pytest.raises(ValueError, match="validator names"):
        ResearchOrchestrator(
            candidates=_candidates(),
            classifier=_profile,
            planner=_normal_planner,
            validators=(validator, validator),
        )

    with pytest.raises(ValueError, match="profile label"):
        ProblemProfile(" ", "rationale", MethodSelectionContext())
    with pytest.raises(ValueError, match="profile rationale"):
        ProblemProfile("label", " ", MethodSelectionContext())
    with pytest.raises(ValueError, match="planned action step id"):
        PlannedAction(" ", lambda _context: None)
    with pytest.raises(ValueError, match="tool action step id"):
        ToolPlannedAction(" ", "cap", lambda _context, _module: None)
    with pytest.raises(ValueError, match="tool action capability"):
        ToolPlannedAction("step", " ", lambda _context, _module: None)
    with pytest.raises(ValueError, match="preferred tool names"):
        ToolPlannedAction("step", "cap", lambda _context, _module: None, preferred=("",))
    with pytest.raises(ValueError, match="validator name"):
        IndependentValidator(" ", _pass_report)


def test_blueprint_rejects_incomplete_or_ambiguous_implementations() -> None:
    step = PlanStep("step", "description", "method", "rationale")
    other = PlanStep("other", "description", "method", "rationale")
    action = PlannedAction("step", lambda _context: None)

    with pytest.raises(ValueError, match="at least one step"):
        ResearchBlueprint(steps=())
    with pytest.raises(ValueError, match="step ids must not be empty"):
        ResearchBlueprint(
            steps=(PlanStep(" ", "description", "method", "rationale"),),
            actions=(PlannedAction(" ", lambda _context: None),),
        )
    with pytest.raises(ValueError, match="reserved"):
        ResearchBlueprint(
            steps=(PlanStep("plan", "description", "method", "rationale"),),
            actions=(PlannedAction("plan", lambda _context: None),),
        )
    with pytest.raises(ValueError, match="step ids must be unique"):
        ResearchBlueprint(steps=(step, step), actions=(action,))
    with pytest.raises(ValueError, match="exactly one implementation"):
        ResearchBlueprint(steps=(step,), actions=(action, action))
    with pytest.raises(ValueError, match="undeclared"):
        ResearchBlueprint(steps=(step,), actions=(PlannedAction("other", lambda _context: None),))
    with pytest.raises(ValueError, match="every plan step"):
        ResearchBlueprint(steps=(step, other), actions=(action,))
