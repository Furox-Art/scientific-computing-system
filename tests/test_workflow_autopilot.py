"""Tests for the conservative default research autopilot."""

from __future__ import annotations

import pytest

from cds.modeling import MathModel, Variable
from cds.workflow import (
    AnalysisKind,
    AnalysisRequest,
    ExecutionContext,
    GateStatus,
    OrchestrationRecord,
    default_research_orchestrator,
)


def _record(result) -> OrchestrationRecord:
    value = result.details["_orchestration"]
    assert isinstance(value, OrchestrationRecord)
    return value


def test_autopilot_requires_explicit_analysis_kind() -> None:
    with pytest.raises(ValueError, match="analysis_kind"):
        default_research_orchestrator().run(
            AnalysisRequest("do science", require_plan_approval=False)
        )


def test_simulation_autopilot_runs_caller_callable_and_passes_gate() -> None:
    context = ExecutionContext(values={"base": 4})
    context.values["simulation"] = lambda ctx: int(ctx.values["base"]) ** 2
    result = default_research_orchestrator().run(
        AnalysisRequest(
            "run declared simulation",
            analysis_kind=AnalysisKind.SIMULATION,
            require_plan_approval=False,
        ),
        context=context,
    )
    assert result.details["analyze"] == 16
    assert _record(result).gate.status is GateStatus.READY


def test_statistical_autopilot_runs_explicit_statistical_callable() -> None:
    context = ExecutionContext(values={})
    context.values["statistical_test"] = lambda _ctx: {"statistic": 3.2, "p_value": 0.01}
    result = default_research_orchestrator().run(
        AnalysisRequest(
            "run declared statistical test",
            analysis_kind=AnalysisKind.STATISTICAL_TEST,
            require_plan_approval=False,
        ),
        context=context,
    )
    assert result.details["analyze"] == {"statistic": 3.2, "p_value": 0.01}
    assert _record(result).gate.status is GateStatus.READY


def test_parameter_fit_autopilot_uses_advanced_fit() -> None:
    model = MathModel(name="line", parameters={"a": 0.0}, variables=["x"])
    model.add_equation("y", Variable("a") * Variable("x"))
    context = ExecutionContext(
        values={
            "model": model,
            "parameter_names": ["a"],
            "observations": [({"x": 1.0}, 2.0), ({"x": 2.0}, 4.0), ({"x": 3.0}, 6.0)],
            "fit_options": {
                "x0": [1.0],
                "optimizer": "gradient_descent",
                "lr": 0.01,
                "max_iter": 1500,
            },
        }
    )
    result = default_research_orchestrator().run(
        AnalysisRequest(
            "fit declared model",
            analysis_kind=AnalysisKind.PARAMETER_FIT,
            require_plan_approval=False,
        ),
        context=context,
    )
    fit = result.details["analyze"]
    assert hasattr(fit, "parameters")
    assert fit.parameters["a"] == pytest.approx(2.0, abs=2e-2)
    assert _record(result).gate.status is GateStatus.READY


def test_autopilot_fails_closed_when_required_context_is_missing() -> None:
    result = default_research_orchestrator().run(
        AnalysisRequest(
            "missing simulation",
            analysis_kind=AnalysisKind.SIMULATION,
            require_plan_approval=False,
        )
    )
    assert "stopped" in result.summary or "blocked" in result.summary
    assert _record(result).gate.status is GateStatus.BLOCKED
