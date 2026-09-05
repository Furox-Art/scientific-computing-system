"""Conservative default recipes for common scientific workflow kinds.

The autopilot is intentionally explicit: callers select :class:`AnalysisKind`.
It never guesses a scientific task family from free text.  Once the kind is
known it supplies a default method catalog, planner, optional-tool binding,
execution validation, and research gate.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import cast

from cds.modeling import MathModel, fit_parameters_advanced
from cds.tools import ToolRegistry, default_registry
from cds.validation import CheckStatus, ValidationCheck, ValidationReport
from cds.workflow.engine import ExecutionContext
from cds.workflow.gates import GatePolicy
from cds.workflow.orchestrator import (
    IndependentValidator,
    PlannedAction,
    ProblemProfile,
    ResearchBlueprint,
    ResearchOrchestrator,
    ToolPlannedAction,
)
from cds.workflow.selection import MethodCandidate, MethodSelection, MethodSelectionContext
from cds.workflow.types import AnalysisKind, AnalysisRequest, PlanStep, ScientificResult

_KIND_CAPABILITY = {
    AnalysisKind.PARAMETER_FIT: "parameter-fit",
    AnalysisKind.SIMULATION: "simulation",
    AnalysisKind.STATISTICAL_TEST: "statistical-test",
    AnalysisKind.SYMBOLIC_VERIFICATION: "symbolic-verification",
    AnalysisKind.CONSTRAINT_VERIFICATION: "constraint-verification",
}


def default_method_catalog() -> tuple[MethodCandidate, ...]:
    """Return the local-first method recipes shipped with CDS."""
    return (
        MethodCandidate(
            "advanced-fit",
            "dependency-free advanced fitting with diagnostics and replayable observations",
            capabilities=("parameter-fit",),
        ),
        MethodCandidate(
            "local-simulation",
            "execute a caller-supplied deterministic simulation inside the workflow context",
            capabilities=("simulation",),
        ),
        MethodCandidate(
            "local-statistical-test",
            "execute a caller-supplied statistical test while retaining approval/provenance gates",
            capabilities=("statistical-test",),
        ),
        MethodCandidate(
            "sympy-identity",
            "verify a symbolic identity through the optional local SymPy backend",
            capabilities=("symbolic-verification",),
            required_tools=("sympy",),
        ),
        MethodCandidate(
            "z3-constraints",
            "verify satisfiability through the optional local Z3 SMT backend",
            capabilities=("constraint-verification",),
            required_tools=("z3",),
        ),
    )


def _installed_tools(registry: ToolRegistry) -> tuple[str, ...]:
    return tuple(status.spec.name for status in registry.statuses() if status.available)


def _classifier(registry: ToolRegistry) -> Callable[[AnalysisRequest], ProblemProfile]:
    def classify(request: AnalysisRequest) -> ProblemProfile:
        if request.analysis_kind is None:
            raise ValueError(
                "default autopilot requires AnalysisRequest.analysis_kind; free-text method guessing is disabled"
            )
        capability = _KIND_CAPABILITY[request.analysis_kind]
        return ProblemProfile(
            label=request.analysis_kind.value,
            rationale=f"caller explicitly selected scientific task family {request.analysis_kind.value!r}",
            selection_context=MethodSelectionContext(
                required_capabilities=(capability,),
                available_tools=_installed_tools(registry),
            ),
        )

    return classify


def _fit_action(context: ExecutionContext) -> object:
    model = context.values.get("model")
    names = context.values.get("parameter_names")
    observations = context.values.get("observations")
    options = context.values.get("fit_options", {})
    if not isinstance(model, MathModel):
        raise TypeError("parameter-fit autopilot requires context['model'] as MathModel")
    if not isinstance(names, (list, tuple)) or any(not isinstance(name, str) for name in names):
        raise TypeError("parameter-fit autopilot requires context['parameter_names'] as strings")
    if observations is None:
        raise KeyError("parameter-fit autopilot requires context['observations']")
    if not isinstance(options, dict):
        raise TypeError("context['fit_options'] must be a dictionary when provided")
    return fit_parameters_advanced(
        model,
        cast(object, observations),  # runtime fitting API validates replayability
        cast(Sequence[str], names),
        **cast(dict[str, object], options),
    )


def _call_context_callable(context: ExecutionContext, key: str) -> object:
    action = context.values.get(key)
    if not callable(action):
        raise TypeError(f"autopilot requires context[{key!r}] to be callable")
    return cast(Callable[[ExecutionContext], object], action)(context)


def _sympy_action(context: ExecutionContext, module: ModuleType) -> object:
    left = context.values.get("left")
    right = context.values.get("right")
    if not isinstance(left, str) or not left.strip() or not isinstance(right, str) or not right.strip():
        raise TypeError("symbolic autopilot requires non-empty string context['left'] and context['right']")
    left_expr = module.sympify(left)
    right_expr = module.sympify(right)
    return bool(module.simplify(left_expr - right_expr) == 0)


def _z3_action(context: ExecutionContext, module: ModuleType) -> object:
    builder = context.values.get("constraint_builder")
    if not callable(builder):
        raise TypeError("constraint autopilot requires callable context['constraint_builder']")
    constraints = cast(Callable[[ModuleType], Sequence[object]], builder)(module)
    solver = module.Solver()
    solver.add(*tuple(constraints))
    return str(solver.check()).lower()


def _planner(
    _request: AnalysisRequest,
    _profile: ProblemProfile,
    selection: MethodSelection,
) -> ResearchBlueprint:
    selected = selection.recommended
    if selected is None:
        raise ValueError("autopilot planner received no eligible method")
    name = selected.candidate.name
    step = PlanStep(
        id="analyze",
        description=f"execute {name}",
        method=name,
        rationale=selected.candidate.rationale,
    )
    if name == "advanced-fit":
        return ResearchBlueprint(steps=(step,), actions=(PlannedAction("analyze", _fit_action),))
    if name == "local-simulation":
        return ResearchBlueprint(
            steps=(step,),
            actions=(PlannedAction("analyze", lambda context: _call_context_callable(context, "simulation")),),
        )
    if name == "local-statistical-test":
        return ResearchBlueprint(
            steps=(step,),
            actions=(
                PlannedAction(
                    "analyze",
                    lambda context: _call_context_callable(context, "statistical_test"),
                ),
            ),
        )
    if name == "sympy-identity":
        return ResearchBlueprint(
            steps=(step,),
            tool_actions=(ToolPlannedAction("analyze", "symbolic", _sympy_action, preferred=("sympy",)),),
        )
    if name == "z3-constraints":
        return ResearchBlueprint(
            steps=(step,),
            tool_actions=(
                ToolPlannedAction("analyze", "formal-verification", _z3_action, preferred=("z3",)),
            ),
        )
    raise ValueError(f"no default autopilot recipe for method {name!r}")


def _execution_validator(_context: ExecutionContext, result: ScientificResult) -> ValidationReport:
    if "analyze" not in result.details:
        return ValidationReport(
            checks=[ValidationCheck("autopilot-execution", CheckStatus.FAIL, "analysis output is missing")]
        )
    output = result.details["analyze"]
    if isinstance(output, float) and not math.isfinite(output):
        return ValidationReport(
            checks=[
                ValidationCheck(
                    "autopilot-execution",
                    CheckStatus.FAIL,
                    "analysis output is non-finite",
                )
            ]
        )
    return ValidationReport(
        checks=[ValidationCheck("autopilot-execution", CheckStatus.PASS, "analysis recipe completed")]
    )


def default_research_orchestrator(*, registry: ToolRegistry | None = None) -> ResearchOrchestrator:
    """Create CDS's conservative explicit-kind scientific autopilot."""
    tools = registry if registry is not None else default_registry()
    return ResearchOrchestrator(
        candidates=default_method_catalog(),
        classifier=_classifier(tools),
        planner=_planner,
        validators=(IndependentValidator("autopilot-execution", _execution_validator),),
        gate_policy=GatePolicy(require_alternatives=False, required_checks=("autopilot-execution",)),
        registry=tools,
    )
