"""Capability-based optional-tool selection for scientific workflows."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import ModuleType

from cds.provenance import RunManifest
from cds.tools import ToolRegistry, ToolStatus, default_registry
from cds.workflow.engine import ExecutionContext, ResearchWorkflow

ToolStepAction = Callable[[ExecutionContext, ModuleType], object]


@dataclass(frozen=True)
class ToolSelection:
    """Auditable choice of an installed backend for one scientific capability."""

    capability: str
    tool: str
    version: str | None
    alternatives: tuple[str, ...]
    rationale: str


def select_tool(
    capability: str,
    *,
    registry: ToolRegistry | None = None,
    preferred: Sequence[str] = (),
) -> ToolSelection:
    """Select an installed backend deterministically, honoring explicit preferences."""
    if not capability.strip():
        raise ValueError("capability must not be empty")
    if any(not name.strip() for name in preferred):
        raise ValueError("preferred tool names must not be empty")

    tools = registry if registry is not None else default_registry()
    installed = list(tools.recommend(capability, installed_only=True))
    if not installed:
        advertised = tuple(
            status.spec.name for status in tools.recommend(capability, installed_only=False)
        )
        suffix = f"; known backends: {', '.join(advertised)}" if advertised else ""
        raise ModuleNotFoundError(f"no installed tool provides capability {capability!r}{suffix}")

    preference_rank = {name: index for index, name in enumerate(preferred)}
    fallback_rank = len(preference_rank)
    installed.sort(
        key=lambda status: (
            preference_rank.get(status.spec.name, fallback_rank),
            status.spec.name,
        )
    )
    chosen: ToolStatus = installed[0]
    alternatives = tuple(status.spec.name for status in installed[1:])
    rationale = f"selected installed backend {chosen.spec.name!r} for capability {capability!r}"
    if chosen.spec.name in preference_rank:
        rationale += " because it was explicitly preferred"
    elif alternatives:
        rationale += "; alternatives remain available: " + ", ".join(alternatives)

    return ToolSelection(
        capability=capability,
        tool=chosen.spec.name,
        version=chosen.version,
        alternatives=alternatives,
        rationale=rationale,
    )


def register_tool_step(
    workflow: ResearchWorkflow,
    step_id: str,
    capability: str,
    action: ToolStepAction,
    *,
    registry: ToolRegistry | None = None,
    manifest: RunManifest | None = None,
    preferred: Sequence[str] = (),
) -> ToolSelection:
    """Bind a workflow step to a selected optional backend with provenance recording."""
    plan_step = next((step for step in workflow.plan.steps if step.id == step_id), None)
    if plan_step is None:
        raise KeyError(f"unknown workflow step {step_id!r}")

    tools = registry if registry is not None else default_registry()
    selection = select_tool(capability, registry=tools, preferred=preferred)
    covered_by_approval = workflow.plan.request.require_plan_approval or plan_step.requires_approval

    def execute_with_tool(context: ExecutionContext) -> object:
        module = tools.load(selection.tool)
        if manifest is not None:
            manifest.record_tool(selection.tool, selection.version or "unknown")
            manifest.record_decision(
                action=f"select {selection.tool} for {capability}",
                rationale=selection.rationale,
                approved_by_user=covered_by_approval,
            )
        return action(context, module)

    workflow.register(step_id, execute_with_tool)
    return selection
