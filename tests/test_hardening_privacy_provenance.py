"""Hardening tests for privacy policy, provenance binding, and fail-closed orchestration."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import ModuleType

import pytest

from cds.provenance import RunManifest, canonical_sha256, detect_git_sha, sha256_file
from cds.tools import ToolLocality, ToolRegistry, ToolSpec
from cds.workflow import (
    AnalysisPlan,
    AnalysisRequest,
    GateStatus,
    MethodCandidate,
    MethodSelection,
    MethodSelectionContext,
    OrchestrationRecord,
    PlanStep,
    ProblemProfile,
    ResearchBlueprint,
    ResearchOrchestrator,
    ResearchWorkflow,
    ToolPlannedAction,
)
from cds.workflow.engine import ExecutionContext
from cds.workflow.tooling import register_tool_step, select_tool


def _installed_registry(*specs: ToolSpec) -> ToolRegistry:
    registry = ToolRegistry()
    for spec in specs:
        registry.register(spec)
    return registry


def _local_spec() -> ToolSpec:
    return ToolSpec(
        name="local-fit",
        module="math",
        distribution="distribution-that-does-not-exist",
        capabilities=("fit",),
        purpose="local fitting backend",
    )


def _remote_spec() -> ToolSpec:
    return ToolSpec(
        name="remote-fit",
        module="math",
        distribution="distribution-that-does-not-exist",
        capabilities=("fit",),
        purpose="remote fitting backend",
        locality=ToolLocality.REMOTE,
        data_egress=True,
    )


def test_privacy_contract_rejects_inconsistent_tool_and_request_policies() -> None:
    with pytest.raises(ValueError, match="remote tools"):
        ToolSpec(
            "remote",
            "remote_module",
            "remote-dist",
            ("fit",),
            "remote",
            locality=ToolLocality.REMOTE,
        )
    with pytest.raises(ValueError, match="local tools"):
        ToolSpec(
            "local",
            "local_module",
            "local-dist",
            ("fit",),
            "local",
            data_egress=True,
        )
    with pytest.raises(ValueError, match="sensitive_data"):
        AnalysisRequest(
            "private fit",
            sensitive_data=True,
            allow_remote_fallback=True,
        )


def test_tool_selection_prefers_local_and_requires_explicit_remote_fallback() -> None:
    mixed = _installed_registry(_remote_spec(), _local_spec())
    selected = select_tool("fit", registry=mixed)
    assert selected.tool == "local-fit"
    assert selected.locality is ToolLocality.LOCAL
    assert not selected.data_egress

    remote_only = _installed_registry(_remote_spec())
    with pytest.raises(PermissionError, match="remote fallback requires"):
        select_tool("fit", registry=remote_only)

    remote = select_tool(
        "fit",
        registry=remote_only,
        allow_remote_fallback=True,
    )
    assert remote.tool == "remote-fit"
    assert remote.locality is ToolLocality.REMOTE
    assert remote.data_egress
    assert "explicitly permitted" in remote.rationale

    with pytest.raises(PermissionError, match="forbidden for sensitive data"):
        select_tool(
            "fit",
            registry=remote_only,
            prefer_local=False,
            sensitive_data=True,
        )
    with pytest.raises(ValueError, match="sensitive data"):
        select_tool(
            "fit",
            registry=remote_only,
            sensitive_data=True,
            allow_remote_fallback=True,
        )


def test_remote_tool_execution_records_egress_provenance() -> None:
    registry = _installed_registry(_remote_spec())
    request = AnalysisRequest(
        "remote fit",
        require_plan_approval=False,
        prefer_local=True,
        allow_remote_fallback=True,
    )
    plan = AnalysisPlan(
        request=request,
        steps=(PlanStep("fit", "fit model", "remote", "remote backend required"),),
    )
    workflow = ResearchWorkflow(plan)
    manifest = RunManifest.create(
        request.question,
        run_id="privacy-run",
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
    result = workflow.execute()

    assert result.details["fit"] == "math"
    assert selection.locality is ToolLocality.REMOTE
    assert manifest.tool_versions == {"remote-fit": "unknown"}
    assert manifest.metadata["tool.remote-fit.locality"] == "remote"
    assert manifest.metadata["tool.remote-fit.data_egress"] == "true"
    assert not manifest.decisions[0].approved_by_user


def _profile(_request: AnalysisRequest) -> ProblemProfile:
    return ProblemProfile(
        "fit",
        "quantitative fitting problem",
        MethodSelectionContext(),
    )


def _candidate() -> tuple[MethodCandidate, ...]:
    return (MethodCandidate("fit-method", "deterministic fitting method"),)


def _tool_planner(
    _request: AnalysisRequest,
    _profile_value: ProblemProfile,
    _selection: MethodSelection,
) -> ResearchBlueprint:
    step = PlanStep("fit", "fit model", "selected tool", "execute selected fitting backend")

    def action(_context: ExecutionContext, module: ModuleType) -> object:
        return module.__name__

    return ResearchBlueprint(
        steps=(step,),
        tool_actions=(ToolPlannedAction("fit", "fit", action),),
    )


def _orchestration_record(result: object) -> OrchestrationRecord:
    from cds.workflow import ScientificResult

    assert isinstance(result, ScientificResult)
    record = result.details["_orchestration"]
    assert isinstance(record, OrchestrationRecord)
    return record


def test_orchestrator_binds_plan_hash_automatically() -> None:
    registry = _installed_registry(_local_spec())
    request = AnalysisRequest("fit locally", require_plan_approval=False)
    result = ResearchOrchestrator(
        candidates=_candidate(),
        classifier=_profile,
        planner=_tool_planner,
        registry=registry,
    ).run(request)
    record = _orchestration_record(result)

    assert record.plan is not None
    assert record.manifest.metadata["plan.sha256"] == canonical_sha256(record.plan)
    assert len(record.manifest.metadata["plan.sha256"]) == 64


def test_orchestrator_blocks_sensitive_remote_only_backend_and_keeps_plan_hash() -> None:
    registry = _installed_registry(_remote_spec())
    request = AnalysisRequest(
        "fit sensitive data",
        require_plan_approval=False,
        prefer_local=False,
        sensitive_data=True,
    )
    result = ResearchOrchestrator(
        candidates=_candidate(),
        classifier=_profile,
        planner=_tool_planner,
        registry=registry,
    ).run(request)
    record = _orchestration_record(result)

    assert record.gate.status is GateStatus.BLOCKED
    assert record.plan is not None
    assert record.manifest.metadata["plan.sha256"] == canonical_sha256(record.plan)
    assert "PermissionError" in result.warnings[0]
    assert "sensitive data" in result.warnings[0]
    assert "fit" not in result.details


def test_provenance_canonical_hash_and_environment_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert canonical_sha256({"b": [2, 3], "a": 1}) == canonical_sha256({"a": 1, "b": (2, 3)})
    with pytest.raises(TypeError, match="unsupported canonical provenance value"):
        canonical_sha256(object())
    with pytest.raises(ValueError):
        canonical_sha256({"bad": float("nan")})

    runtime_lock = tmp_path / "requirements.lock"
    dev_lock = tmp_path / "requirements-dev.lock"
    extra_lock = tmp_path / "solver.lock"
    runtime_lock.write_text("runtime\n", encoding="utf-8")
    dev_lock.write_text("dev\n", encoding="utf-8")
    extra_lock.write_text("solver\n", encoding="utf-8")
    monkeypatch.setenv("GITHUB_SHA", "ABCDEF1234567")

    manifest = RunManifest.create(
        "provenance",
        project_root=tmp_path,
        run_id="provenance-run",
        created_utc="2026-09-05T00:00:00+00:00",
    )
    assert manifest.metadata["source.git_sha"] == "abcdef1234567"
    assert manifest.metadata["lock.requirements.lock.sha256"] == sha256_file(runtime_lock)
    assert manifest.metadata["lock.requirements-dev.lock.sha256"] == sha256_file(dev_lock)
    digest = manifest.record_environment_lock("solver", extra_lock)
    assert digest == sha256_file(extra_lock)
    assert manifest.metadata["lock.solver.sha256"] == digest
    with pytest.raises(ValueError, match="lock name"):
        manifest.record_environment_lock(" ", extra_lock)


def test_detect_git_sha_fallback_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("GITHUB_SHA", "CI_COMMIT_SHA", "GIT_COMMIT"):
        monkeypatch.delenv(key, raising=False)

    def successful_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["git", "rev-parse", "HEAD"],
            returncode=0,
            stdout="1234567ABCDEF\n",
            stderr="",
        )

    monkeypatch.setattr("cds.provenance.manifest.subprocess.run", successful_run)
    assert detect_git_sha() == "1234567abcdef"

    def invalid_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["git", "rev-parse", "HEAD"],
            returncode=1,
            stdout="not-a-sha\n",
            stderr="fatal",
        )

    monkeypatch.setattr("cds.provenance.manifest.subprocess.run", invalid_run)
    assert detect_git_sha() is None

    def failing_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise OSError("git unavailable")

    monkeypatch.setattr("cds.provenance.manifest.subprocess.run", failing_run)
    assert detect_git_sha() is None
