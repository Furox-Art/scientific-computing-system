"""Deterministic, approval-gated orchestration with atomic checkpoint/resume."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field

from cds.provenance import RunManifest, load_checkpoint, save_checkpoint, sha256_text
from cds.workflow.types import AnalysisPlan, ExecutionTrace, PlanStep, ScientificResult, StepStatus

StepAction = Callable[["ExecutionContext"], object]
ApprovalCallback = Callable[[PlanStep | None], bool]


@dataclass
class ExecutionContext:
    """Mutable values shared between approved workflow steps."""

    values: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RegisteredStep:
    """Executable implementation bound to a plan-step identifier."""

    plan_step: PlanStep
    action: StepAction


def _plan_fingerprint(plan: AnalysisPlan) -> str:
    payload = {
        "question": plan.request.question,
        "steps": [
            {
                "id": step.id,
                "description": step.description,
                "method": step.method,
                "rationale": step.rationale,
                "requires_approval": step.requires_approval,
            }
            for step in plan.steps
        ],
    }
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _checkpoint_state(
    plan: AnalysisPlan,
    context: ExecutionContext,
    completed: set[str],
    *,
    failed_step: str | None = None,
) -> dict[str, object]:
    try:
        json.dumps(context.values, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "automatic workflow checkpointing requires JSON-serializable context values"
        ) from exc
    return {
        "plan_fingerprint": _plan_fingerprint(plan),
        "completed_steps": sorted(completed),
        "context": dict(context.values),
        "failed_step": failed_step,
    }


class ResearchWorkflow:
    """Execute a reviewed analysis plan with audit and resumability semantics."""

    def __init__(self, plan: AnalysisPlan) -> None:
        self.plan = plan
        self._steps: dict[str, RegisteredStep] = {}

    def register(self, step_id: str, action: StepAction) -> None:
        plan_step = next((step for step in self.plan.steps if step.id == step_id), None)
        if plan_step is None:
            raise KeyError(f"unknown workflow step {step_id!r}")
        if step_id in self._steps:
            raise ValueError(f"workflow step {step_id!r} is already registered")
        self._steps[step_id] = RegisteredStep(plan_step=plan_step, action=action)

    def execute(
        self,
        *,
        approve: ApprovalCallback | None = None,
        context: ExecutionContext | None = None,
        checkpoint_path: str | os.PathLike[str] | None = None,
        resume: bool = False,
        manifest: RunManifest | None = None,
    ) -> ScientificResult:
        """Execute registered steps, optionally checkpointing after every completed step.

        ``resume=True`` requires ``checkpoint_path``.  A checkpoint is accepted
        only when its question and plan fingerprint match this workflow.  The
        caller must approve the plan again in a resumed process when plan
        approval is required; already completed steps are never re-executed.
        """
        if resume and checkpoint_path is None:
            raise ValueError("resume=True requires checkpoint_path")

        ctx = context if context is not None else ExecutionContext()
        trace = ExecutionTrace()
        completed: set[str] = set()
        run_manifest = manifest

        if resume:
            loaded_manifest, state = load_checkpoint(checkpoint_path)
            if loaded_manifest.question != self.plan.request.question:
                raise ValueError("checkpoint question does not match workflow plan")
            if manifest is not None and manifest.run_id != loaded_manifest.run_id:
                raise ValueError("checkpoint manifest run_id does not match supplied manifest")
            fingerprint = state.get("plan_fingerprint")
            if fingerprint != _plan_fingerprint(self.plan):
                raise ValueError("checkpoint plan fingerprint does not match workflow plan")
            saved_context = state.get("context")
            saved_completed = state.get("completed_steps")
            if not isinstance(saved_context, dict) or not isinstance(saved_completed, list):
                raise ValueError("checkpoint workflow state is malformed")
            if any(not isinstance(step_id, str) for step_id in saved_completed):
                raise ValueError("checkpoint completed_steps must contain strings")
            ctx.values.update(saved_context)
            completed = set(saved_completed)
            declared = {step.id for step in self.plan.steps}
            if not completed <= declared:
                raise ValueError("checkpoint contains completed steps not present in the plan")
            run_manifest = loaded_manifest

        if run_manifest is None and checkpoint_path is not None:
            run_manifest = RunManifest.create(
                self.plan.request.question,
                seed=self.plan.request.seed,
            )
        if run_manifest is not None and run_manifest.question != self.plan.request.question:
            raise ValueError("manifest question must match workflow plan")

        if self.plan.request.require_plan_approval:
            if approve is None or not approve(None):
                trace.record("plan", StepStatus.SKIPPED, "analysis plan was not approved")
                return ScientificResult(
                    summary="Analysis was not executed because the plan was not approved.",
                    details=dict(ctx.values),
                    trace=trace,
                )
            trace.record("plan", StepStatus.COMPLETED, "analysis plan approved")

        for step in self.plan.steps:
            if step.id in completed:
                trace.record(step.id, StepStatus.RESUMED, "restored from workflow checkpoint")
                continue

            registered = self._steps.get(step.id)
            if registered is None:
                trace.record(step.id, StepStatus.SKIPPED, "no implementation registered")
                continue
            if step.requires_approval and (approve is None or not approve(step)):
                trace.record(step.id, StepStatus.SKIPPED, "step approval was not granted")
                continue

            trace.record(step.id, StepStatus.RUNNING, step.description)
            try:
                ctx.values[step.id] = registered.action(ctx)
            except Exception as exc:
                trace.record(step.id, StepStatus.FAILED, f"{type(exc).__name__}: {exc}")
                if checkpoint_path is not None and run_manifest is not None:
                    state = _checkpoint_state(self.plan, ctx, completed, failed_step=step.id)
                    save_checkpoint(checkpoint_path, run_manifest, state)
                return ScientificResult(
                    summary=f"Analysis stopped at step {step.id!r}.",
                    details=dict(ctx.values),
                    warnings=[str(exc)],
                    trace=trace,
                )

            completed.add(step.id)
            if checkpoint_path is not None and run_manifest is not None:
                try:
                    state = _checkpoint_state(self.plan, ctx, completed)
                    save_checkpoint(checkpoint_path, run_manifest, state)
                except (TypeError, ValueError) as exc:
                    completed.remove(step.id)
                    trace.record(step.id, StepStatus.FAILED, f"checkpoint: {exc}")
                    return ScientificResult(
                        summary=f"Analysis stopped because step {step.id!r} could not be checkpointed.",
                        details=dict(ctx.values),
                        warnings=[str(exc)],
                        trace=trace,
                    )
            trace.record(step.id, StepStatus.COMPLETED, "completed")

        return ScientificResult(
            summary="Analysis plan completed.",
            details=dict(ctx.values),
            trace=trace,
        )
