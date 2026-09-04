"""Deterministic orchestration primitives for scientific analyses."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

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


class ResearchWorkflow:
    """Execute a user-reviewed analysis plan with an explicit audit trail.

    The engine intentionally does not decide scientific questions by itself.
    Callers construct a plan, expose it to the user, and register deterministic
    step implementations. Approval is checked once for the overall plan and
    again for individual steps marked as materially consequential.
    """

    def __init__(self, plan: AnalysisPlan) -> None:
        self.plan = plan
        self._steps: dict[str, RegisteredStep] = {}

    def register(self, step_id: str, action: StepAction) -> None:
        """Bind an implementation to one declared plan step."""
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
    ) -> ScientificResult:
        """Execute registered steps in plan order.

        If the request requires plan approval, ``approve(None)`` must return
        ``True`` before any work starts. A step with ``requires_approval=True``
        similarly calls ``approve(step)`` immediately before execution.
        """
        ctx = context if context is not None else ExecutionContext()
        trace = ExecutionTrace()

        if self.plan.request.require_plan_approval:
            if approve is None or not approve(None):
                trace.record("plan", StepStatus.SKIPPED, "analysis plan was not approved")
                return ScientificResult(
                    summary="Analysis was not executed because the plan was not approved.",
                    trace=trace,
                )
            trace.record("plan", StepStatus.COMPLETED, "analysis plan approved")

        for step in self.plan.steps:
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
                return ScientificResult(
                    summary=f"Analysis stopped at step {step.id!r}.",
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
