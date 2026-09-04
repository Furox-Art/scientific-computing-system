"""Tests for capability-based workflow tool orchestration."""

from __future__ import annotations

from importlib import metadata
from types import ModuleType

import pytest

from cds.provenance import RunManifest
from cds.tools import ToolRegistry, ToolSpec
from cds.workflow import AnalysisPlan, AnalysisRequest, PlanStep, ResearchWorkflow
from cds.workflow.engine import ExecutionContext
from cds.workflow.tooling import register_tool_step, select_tool


def _tool_registry(
    monkeypatch: pytest.MonkeyPatch,
    *,
    available: set[str],
    versions: dict[str, str | None],
) -> ToolRegistry:
    registry = ToolRegistry()
    for name in ("alpha", "beta"):
        registry.register(
            ToolSpec(
                name=name,
                module=f"{name}_module",
                distribution=f"{name}-dist",
                capabilities=("fit", "verification" if name == "beta" else "fit"),
                purpose=f"{name} backend",
            )
        )

    def fake_find_spec(module: str) -> object | None:
        name = module.removesuffix("_module")
        return object() if name in available else None

    def fake_version(distribution: str) -> str:
        name = distribution.removesuffix("-dist")
        version = versions.get(name)
        if version is None:
            raise metadata.PackageNotFoundError(distribution)
        return version

    def fake_import(module: str) -> ModuleType:
        return ModuleType(module)

    monkeypatch.setattr("cds.tools.registry.importlib.util.find_spec", fake_find_spec)
    monkeypatch.setattr("cds.tools.registry.metadata.version", fake_version)
    monkeypatch.setattr("cds.tools.registry.importlib.import_module", fake_import)
    return registry


def _plan(*, plan_approval: bool, step_approval: bool) -> AnalysisPlan:
    return AnalysisPlan(
        request=AnalysisRequest("fit a model", require_plan_approval=plan_approval),
        steps=(
            PlanStep(
                id="fit",
                description="fit model",
                method="external-optimizer",
                rationale="use a verified numerical backend",
                requires_approval=step_approval,
            ),
        ),
    )


def test_select_tool_preference_fallback_and_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _tool_registry(
        monkeypatch,
        available={"alpha", "beta"},
        versions={"alpha": "1.0", "beta": "2.0"},
    )
    preferred = select_tool("fit", registry=registry, preferred=("beta",))
    assert preferred.tool == "beta"
    assert preferred.version == "2.0"
    assert preferred.alternatives == ("alpha",)
    assert "explicitly preferred" in preferred.rationale

    fallback = select_tool("fit", registry=registry, preferred=("missing",))
    assert fallback.tool == "alpha"
    assert fallback.alternatives == ("beta",)
    assert "alternatives remain available" in fallback.rationale

    with pytest.raises(ValueError, match="capability must not be empty"):
        select_tool(" ", registry=registry)
    with pytest.raises(ValueError, match="preferred tool names"):
        select_tool("fit", registry=registry, preferred=("",))


def test_select_tool_reports_missing_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _tool_registry(monkeypatch, available=set(), versions={})
    with pytest.raises(ModuleNotFoundError, match="known backends: alpha, beta"):
        select_tool("fit", registry=registry)
    with pytest.raises(ModuleNotFoundError, match="capability 'unknown'"):
        select_tool("unknown", registry=registry)


def test_register_tool_step_records_version_and_approved_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _tool_registry(
        monkeypatch,
        available={"alpha"},
        versions={"alpha": "1.2.3"},
    )
    workflow = ResearchWorkflow(_plan(plan_approval=False, step_approval=True))
    manifest = RunManifest.create(
        "fit a model",
        run_id="run-1",
        created_utc="2026-09-05T00:00:00+00:00",
    )

    def action(_context: ExecutionContext, module: ModuleType) -> object:
        return module.__name__

    selection = register_tool_step(
        workflow,
        "fit",
        "fit",
        action,
        registry=registry,
        manifest=manifest,
    )
    assert selection.tool == "alpha"

    def approve(_step: PlanStep | None) -> bool:
        return True

    result = workflow.execute(approve=approve)
    assert result.details["fit"] == "alpha_module"
    assert manifest.tool_versions == {"alpha": "1.2.3"}
    assert len(manifest.decisions) == 1
    assert manifest.decisions[0].approved_by_user
    assert "select alpha for fit" == manifest.decisions[0].action


def test_register_tool_step_unknown_version_and_unapproved_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _tool_registry(monkeypatch, available={"alpha"}, versions={"alpha": None})
    workflow = ResearchWorkflow(_plan(plan_approval=False, step_approval=False))
    manifest = RunManifest.create(
        "fit a model",
        run_id="run-2",
        created_utc="2026-09-05T00:00:00+00:00",
    )

    def action(_context: ExecutionContext, module: ModuleType) -> object:
        return module.__name__

    register_tool_step(workflow, "fit", "fit", action, registry=registry, manifest=manifest)
    result = workflow.execute()
    assert result.summary == "Analysis plan completed."
    assert manifest.tool_versions == {"alpha": "unknown"}
    assert not manifest.decisions[0].approved_by_user

    with pytest.raises(KeyError, match="unknown workflow step"):
        register_tool_step(workflow, "missing", "fit", action, registry=registry)


def test_default_registry_plan_approval_and_manifest_free_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _tool_registry(monkeypatch, available={"alpha"}, versions={"alpha": "3.0"})

    def fake_default_registry() -> ToolRegistry:
        return registry

    monkeypatch.setattr("cds.workflow.tooling.default_registry", fake_default_registry)
    selection = select_tool("fit")
    assert selection.tool == "alpha"
    assert selection.alternatives == ()

    workflow = ResearchWorkflow(_plan(plan_approval=True, step_approval=False))

    def action(_context: ExecutionContext, module: ModuleType) -> object:
        return module.__name__

    register_tool_step(workflow, "fit", "fit", action)

    def approve(_step: PlanStep | None) -> bool:
        return True

    result = workflow.execute(approve=approve)
    assert result.details["fit"] == "alpha_module"
