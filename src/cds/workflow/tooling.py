"""Capability-based optional-tool selection for scientific workflows."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import ModuleType

from cds.provenance import RunManifest
from cds.tools import ToolLocality, ToolRegistry, ToolStatus, default_registry
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
    locality: ToolLocality = ToolLocality.LOCAL
    data_egress: bool = False


def select_tool(
    capability: str,
    *,
    registry: ToolRegistry | None = None,
    preferred: Sequence[str] = (),
    prefer_local: bool = True,
    sensitive_data: bool = False,
    allow_remote_fallback: bool = False,
) -> ToolSelection:
    """Select an installed backend under explicit locality/data-egress policy.

    Sensitive data is never allowed to reach a tool that declares data egress.
    When local execution is preferred, local tools form the eligible set. A
    remote-only fallback requires the caller to explicitly enable it.
    """
    if not capability.strip():
        raise ValueError("capability must not be empty")
    if any(not name.strip() for name in preferred):
        raise ValueError("preferred tool names must not be empty")
    if sensitive_data and allow_remote_fallback:
        raise ValueError("sensitive data cannot allow remote fallback")

    tools = registry if registry is not None else default_registry()
    installed = list(tools.recommend(capability, installed_only=True))
    if not installed:
        advertised = tuple(
            status.spec.name for status in tools.recommend(capability, installed_only=False)
        )
        suffix = f"; known backends: {', '.join(advertised)}" if advertised else ""
        raise ModuleNotFoundError(f"no installed tool provides capability {capability!r}{suffix}")

    if sensitive_data:
        eligible = [status for status in installed if not status.spec.data_egress]
        if not eligible:
            raise PermissionError(
                f"capability {capability!r} is available only through data-egress tools, "
                "which are forbidden for sensitive data"
            )
    else:
        eligible = installed

    if prefer_local:
        local = [status for status in eligible if status.spec.locality is ToolLocality.LOCAL]
        if local:
            eligible = local
        elif not allow_remote_fallback:
            raise PermissionError(
                f"no installed local tool provides capability {capability!r}; "
                "remote fallback requires allow_remote_fallback=True"
            )

    preference_rank = {name: index for index, name in enumerate(preferred)}
    fallback_rank = len(preference_rank)
    eligible.sort(
        key=lambda status: (
            preference_rank.get(status.spec.name, fallback_rank),
            status.spec.name,
        )
    )
    chosen: ToolStatus = eligible[0]
    alternatives = tuple(status.spec.name for status in eligible[1:])
    rationale = (
        f"selected installed {chosen.spec.locality.value} backend {chosen.spec.name!r} "
        f"for capability {capability!r}"
    )
    if chosen.spec.name in preference_rank:
        rationale += " because it was explicitly preferred within the policy-eligible set"
    elif alternatives:
        rationale += "; policy-eligible alternatives remain available: " + ", ".join(alternatives)
    if chosen.spec.data_egress:
        rationale += "; remote data egress was explicitly permitted by the request policy"

    return ToolSelection(
        capability=capability,
        tool=chosen.spec.name,
        version=chosen.version,
        alternatives=alternatives,
        rationale=rationale,
        locality=chosen.spec.locality,
        data_egress=chosen.spec.data_egress,
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
    """Bind a workflow step to a policy-eligible backend with provenance recording."""
    plan_step = next((step for step in workflow.plan.steps if step.id == step_id), None)
    if plan_step is None:
        raise KeyError(f"unknown workflow step {step_id!r}")

    tools = registry if registry is not None else default_registry()
    request = workflow.plan.request
    selection = select_tool(
        capability,
        registry=tools,
        preferred=preferred,
        prefer_local=request.prefer_local,
        sensitive_data=request.sensitive_data,
        allow_remote_fallback=request.allow_remote_fallback,
    )
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
            manifest.metadata[f"tool.{selection.tool}.locality"] = selection.locality.value
            manifest.metadata[f"tool.{selection.tool}.data_egress"] = str(
                selection.data_egress
            ).lower()
        return action(context, module)

    workflow.register(step_id, execute_with_tool)
    return selection
