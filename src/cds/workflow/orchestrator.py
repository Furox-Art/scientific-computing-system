"""End-to-end scientific research orchestration.

This layer connects classification, policy-enforced method ranking, plan construction,
approved/resumable execution, optional tools, independent validation, provenance,
and the final research gate. Scientific thresholds remain explicit inputs.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType

from cds.provenance import RunManifest, load_checkpoint, save_checkpoint
from cds.tools import ToolRegistry, default_registry
from cds.validation import CheckStatus, ValidationCheck, ValidationReport
from cds.workflow.engine import ApprovalCallback, ExecutionContext, ResearchWorkflow
from cds.workflow.gates import GateDecision, GatePolicy, GateStatus, evaluate_research_gate
from cds.workflow.selection import (
    MethodCandidate,
    MethodSelection,
    MethodSelectionContext,
    MethodStatus,
    SelectionPolicy,
    rank_methods,
)
from cds.workflow.tooling import register_tool_step
from cds.workflow.types import AnalysisPlan, AnalysisRequest, PlanStep, ScientificResult, StepStatus

StepAction = Callable[[ExecutionContext], object]
ToolAction = Callable[[ExecutionContext, ModuleType], object]
Classifier = Callable[[AnalysisRequest], "ProblemProfile"]
Planner = Callable[[AnalysisRequest, "ProblemProfile", MethodSelection], "ResearchBlueprint"]
ValidatorAction = Callable[[ExecutionContext, ScientificResult], ValidationReport]

_RESERVED_STEP_IDS = {"plan", "_orchestration"}


@dataclass(frozen=True)
class ProblemProfile:
    label: str
    rationale: str
    selection_context: MethodSelectionContext

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("problem profile label must not be empty")
        if not self.rationale.strip():
            raise ValueError("problem profile rationale must not be empty")


@dataclass(frozen=True)
class PlannedAction:
    step_id: str
    action: StepAction

    def __post_init__(self) -> None:
        if not self.step_id.strip():
            raise ValueError("planned action step id must not be empty")


@dataclass(frozen=True)
class ToolPlannedAction:
    step_id: str
    capability: str
    action: ToolAction
    preferred: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.step_id.strip():
            raise ValueError("tool action step id must not be empty")
        if not self.capability.strip():
            raise ValueError("tool action capability must not be empty")
        if any(not name.strip() for name in self.preferred):
            raise ValueError("preferred tool names must not be empty")


@dataclass(frozen=True)
class ResearchBlueprint:
    steps: tuple[PlanStep, ...]
    actions: tuple[PlannedAction, ...] = ()
    tool_actions: tuple[ToolPlannedAction, ...] = ()

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("research blueprint must contain at least one step")
        step_ids = tuple(step.id for step in self.steps)
        if any(not step_id.strip() for step_id in step_ids):
            raise ValueError("plan step ids must not be empty")
        if any(step_id in _RESERVED_STEP_IDS for step_id in step_ids):
            raise ValueError("plan step id is reserved by the orchestrator")
        if len(set(step_ids)) != len(step_ids):
            raise ValueError("plan step ids must be unique")
        implementation_ids = tuple(action.step_id for action in self.actions) + tuple(
            action.step_id for action in self.tool_actions
        )
        if len(set(implementation_ids)) != len(implementation_ids):
            raise ValueError("each plan step must have exactly one implementation")
        declared = set(step_ids)
        implemented = set(implementation_ids)
        if implemented - declared:
            raise ValueError("implementations reference undeclared plan steps")
        if declared - implemented:
            raise ValueError("every plan step must have an implementation")


@dataclass(frozen=True)
class IndependentValidator:
    name: str
    action: ValidatorAction

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("validator name must not be empty")


@dataclass(frozen=True)
class OrchestrationRecord:
    profile: ProblemProfile
    selection: MethodSelection
    plan: AnalysisPlan | None
    validation_reports: tuple[ValidationReport, ...]
    gate: GateDecision
    manifest: RunManifest


def _record_once(
    manifest: RunManifest,
    action: str,
    rationale: str,
    *,
    approved_by_user: bool,
) -> None:
    if any(
        decision.action == action and decision.rationale == rationale
        for decision in manifest.decisions
    ):
        return
    manifest.record_decision(
        action=action,
        rationale=rationale,
        approved_by_user=approved_by_user,
    )


def _sync_checkpoint_manifest(
    checkpoint_path: str | os.PathLike[str] | None,
    manifest: RunManifest,
) -> None:
    if checkpoint_path is None or not Path(checkpoint_path).exists():
        return
    _, state = load_checkpoint(checkpoint_path)
    save_checkpoint(checkpoint_path, manifest, state)


class ResearchOrchestrator:
    """Run a complete scientific-analysis pipeline with fail-closed semantics."""

    def __init__(
        self,
        *,
        candidates: tuple[MethodCandidate, ...],
        classifier: Classifier,
        planner: Planner,
        validators: tuple[IndependentValidator, ...] = (),
        selection_policy: SelectionPolicy | None = None,
        gate_policy: GatePolicy | None = None,
        registry: ToolRegistry | None = None,
    ) -> None:
        if not candidates:
            raise ValueError("at least one method candidate is required")
        names = tuple(candidate.name for candidate in candidates)
        if len(set(names)) != len(names):
            raise ValueError("method candidate names must be unique")
        validator_names = tuple(validator.name for validator in validators)
        if len(set(validator_names)) != len(validator_names):
            raise ValueError("validator names must be unique")
        self._candidates = candidates
        self._classifier = classifier
        self._planner = planner
        self._validators = validators
        self._selection_policy = selection_policy
        self._gate_policy = gate_policy
        self._registry = registry if registry is not None else default_registry()

    def run(
        self,
        request: AnalysisRequest,
        *,
        approve: ApprovalCallback | None = None,
        context: ExecutionContext | None = None,
        manifest: RunManifest | None = None,
        checkpoint_path: str | os.PathLike[str] | None = None,
        resume: bool = False,
    ) -> ScientificResult:
        """Classify, select, plan, execute/resume, validate, gate, and report."""
        if resume and checkpoint_path is None:
            raise ValueError("resume=True requires checkpoint_path")
        if resume:
            checkpoint_manifest, _ = load_checkpoint(checkpoint_path)
            run_manifest = manifest if manifest is not None else checkpoint_manifest
            if run_manifest.run_id != checkpoint_manifest.run_id:
                raise ValueError("supplied manifest does not match checkpoint run_id")
        else:
            run_manifest = (
                manifest
                if manifest is not None
                else RunManifest.create(request.question, seed=request.seed)
            )
        if run_manifest.question != request.question:
            raise ValueError("manifest question must match the analysis request")
        if request.seed is not None and run_manifest.seed not in (None, request.seed):
            raise ValueError("manifest seed must match the analysis request seed")
        if run_manifest.seed is None and request.seed is not None:
            run_manifest.seed = request.seed

        raw_profile = self._classifier(request)
        if not isinstance(raw_profile, ProblemProfile):
            raise TypeError("classifier must return ProblemProfile")
        effective_context = replace(
            raw_profile.selection_context,
            prefer_local=raw_profile.selection_context.prefer_local or request.prefer_local,
            sensitive_data=raw_profile.selection_context.sensitive_data or request.sensitive_data,
        )
        profile = ProblemProfile(raw_profile.label, raw_profile.rationale, effective_context)
        _record_once(
            run_manifest,
            action=f"classify problem as {profile.label}",
            rationale=profile.rationale,
            approved_by_user=False,
        )

        selection = rank_methods(
            self._candidates,
            profile.selection_context,
            policy=self._selection_policy,
        )
        selected = selection.recommended
        if selected is None:
            gate = GateDecision(
                status=GateStatus.BLOCKED,
                reasons=("all candidate scientific methods are blocked",),
            )
            _record_once(
                run_manifest,
                action="select scientific method",
                rationale="all candidate methods were blocked by explicit constraints or data policy",
                approved_by_user=False,
            )
            return _finalize_without_execution(
                profile=profile,
                selection=selection,
                plan=None,
                gate=gate,
                manifest=run_manifest,
                summary="Analysis is blocked because no candidate scientific method is eligible.",
            )

        blueprint = self._planner(request, profile, selection)
        if not isinstance(blueprint, ResearchBlueprint):
            raise TypeError("planner must return ResearchBlueprint")
        plan = AnalysisPlan(
            request=request,
            steps=blueprint.steps,
            recommendation=selection.to_recommendation(),
        )
        workflow = ResearchWorkflow(plan)
        execution_context = context if context is not None else ExecutionContext()
        for action in blueprint.actions:
            workflow.register(action.step_id, action.action)

        try:
            for tool_action in blueprint.tool_actions:
                register_tool_step(
                    workflow,
                    tool_action.step_id,
                    tool_action.capability,
                    tool_action.action,
                    registry=self._registry,
                    manifest=run_manifest,
                    preferred=tool_action.preferred,
                )
        except (KeyError, ModuleNotFoundError, ValueError) as exc:
            gate = GateDecision(
                status=GateStatus.BLOCKED,
                reasons=(f"tool registration failed: {type(exc).__name__}: {exc}",),
            )
            _record_once(
                run_manifest,
                action=f"select {selected.candidate.name}",
                rationale="method selected, but required tool registration failed",
                approved_by_user=False,
            )
            return _finalize_without_execution(
                profile=profile,
                selection=selection,
                plan=plan,
                gate=gate,
                manifest=run_manifest,
                summary="Analysis is blocked because a required scientific tool is unavailable.",
            )

        plan_approved = False

        def tracked_approval(step: PlanStep | None) -> bool:
            nonlocal plan_approved
            allowed = approve(step) if approve is not None else False
            if step is None and allowed:
                plan_approved = True
            return allowed

        approval = tracked_approval if approve is not None else None
        execution = workflow.execute(
            approve=approval,
            context=execution_context,
            checkpoint_path=checkpoint_path,
            resume=resume,
            manifest=run_manifest,
        )
        rationale = (
            plan.recommendation.rationale if plan.recommendation is not None else "selected"
        )
        _record_once(
            run_manifest,
            action=f"select {selected.candidate.name}",
            rationale=rationale,
            approved_by_user=plan_approved,
        )

        incomplete_steps = _incomplete_steps(plan, execution)
        if incomplete_steps:
            gate = GateDecision(
                status=GateStatus.BLOCKED,
                reasons=("workflow execution incomplete: " + ", ".join(incomplete_steps),),
            )
            _record_once(
                run_manifest,
                action="final research gate",
                rationale=gate.reasons[0],
                approved_by_user=False,
            )
            _sync_checkpoint_manifest(checkpoint_path, run_manifest)
            return _finalize_execution(
                execution,
                profile=profile,
                selection=selection,
                plan=plan,
                reports=(),
                gate=gate,
                manifest=run_manifest,
            )

        reports = self._run_validators(execution_context, execution)
        gate = evaluate_research_gate(plan, reports, policy=self._gate_policy)
        if selected.status is MethodStatus.REVIEW:
            review_reason = "selected method still has unknown mandatory suitability evidence"
            if gate.status is GateStatus.READY:
                gate = GateDecision(
                    status=GateStatus.REVIEW,
                    reasons=(review_reason,),
                    missing_checks=gate.missing_checks,
                )
            else:
                gate = GateDecision(
                    status=gate.status,
                    reasons=gate.reasons + (review_reason,),
                    missing_checks=gate.missing_checks,
                )

        gate_rationale = (
            "; ".join(gate.reasons) if gate.reasons else "all configured research gates passed"
        )
        _record_once(
            run_manifest,
            action="final research gate",
            rationale=gate_rationale,
            approved_by_user=False,
        )
        _sync_checkpoint_manifest(checkpoint_path, run_manifest)
        return _finalize_execution(
            execution,
            profile=profile,
            selection=selection,
            plan=plan,
            reports=reports,
            gate=gate,
            manifest=run_manifest,
        )

    def _run_validators(
        self,
        context: ExecutionContext,
        execution: ScientificResult,
    ) -> tuple[ValidationReport, ...]:
        reports: list[ValidationReport] = []
        for validator in self._validators:
            try:
                report = validator.action(context, execution)
                if not isinstance(report, ValidationReport):
                    raise TypeError("validator must return ValidationReport")
            except Exception as exc:
                report = ValidationReport(
                    checks=[
                        ValidationCheck(
                            name=f"validator:{validator.name}",
                            status=CheckStatus.FAIL,
                            message=f"independent validator failed: {type(exc).__name__}: {exc}",
                        )
                    ]
                )
            reports.append(report)
        return tuple(reports)


def _incomplete_steps(plan: AnalysisPlan, execution: ScientificResult) -> tuple[str, ...]:
    completed = {
        event.step_id
        for event in execution.trace.events
        if event.status in (StepStatus.COMPLETED, StepStatus.RESUMED)
    }
    return tuple(step.id for step in plan.steps if step.id not in completed)


def _finalize_without_execution(
    *,
    profile: ProblemProfile,
    selection: MethodSelection,
    plan: AnalysisPlan | None,
    gate: GateDecision,
    manifest: RunManifest,
    summary: str,
) -> ScientificResult:
    record = OrchestrationRecord(
        profile=profile,
        selection=selection,
        plan=plan,
        validation_reports=(),
        gate=gate,
        manifest=manifest,
    )
    return ScientificResult(
        summary=summary,
        details={"_orchestration": record},
        warnings=list(gate.reasons),
    )


def _finalize_execution(
    execution: ScientificResult,
    *,
    profile: ProblemProfile,
    selection: MethodSelection,
    plan: AnalysisPlan,
    reports: tuple[ValidationReport, ...],
    gate: GateDecision,
    manifest: RunManifest,
) -> ScientificResult:
    execution.details["_orchestration"] = OrchestrationRecord(
        profile=profile,
        selection=selection,
        plan=plan,
        validation_reports=reports,
        gate=gate,
        manifest=manifest,
    )
    findings = tuple(check for report in reports for check in report.checks)
    execution.warnings.extend(
        check.message for check in findings if check.status is not CheckStatus.PASS
    )
    if gate.status is not GateStatus.READY:
        execution.warnings.extend(gate.reasons)
    if gate.status is GateStatus.READY:
        execution.summary = "Scientific analysis completed and passed the final research gate."
    elif gate.status is GateStatus.REVIEW:
        execution.summary = "Scientific analysis completed but requires review before conclusion."
    else:
        execution.summary = "Scientific analysis completed, but the final conclusion is blocked."
    return execution
