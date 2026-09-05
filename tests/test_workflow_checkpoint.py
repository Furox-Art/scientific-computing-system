"""Tests for workflow checkpointing and exact resume semantics."""

from __future__ import annotations

from pathlib import Path

import pytest

from cds.provenance import RunManifest, load_checkpoint, save_checkpoint
from cds.workflow import AnalysisPlan, AnalysisRequest, PlanStep, ResearchWorkflow, StepStatus


def _plan() -> AnalysisPlan:
    return AnalysisPlan(
        request=AnalysisRequest("checkpoint experiment", require_plan_approval=False, seed=17),
        steps=(
            PlanStep("first", "first", "local", "first deterministic step"),
            PlanStep("second", "second", "local", "second deterministic step"),
            PlanStep("third", "third", "local", "third deterministic step"),
        ),
    )


def test_failed_run_resumes_without_reexecuting_completed_steps(tmp_path: Path) -> None:
    checkpoint = tmp_path / "run.json"
    calls: list[str] = []
    fail = True
    workflow = ResearchWorkflow(_plan())

    def first(_ctx):
        calls.append("first")
        return 10

    def second(ctx):
        nonlocal fail
        calls.append("second")
        if fail:
            raise RuntimeError("interrupted")
        return int(ctx.values["first"]) + 5

    def third(ctx):
        calls.append("third")
        return int(ctx.values["second"]) * 2

    workflow.register("first", first)
    workflow.register("second", second)
    workflow.register("third", third)
    first_result = workflow.execute(checkpoint_path=checkpoint)
    assert first_result.summary == "Analysis stopped at step 'second'."
    assert calls == ["first", "second"]
    manifest, state = load_checkpoint(checkpoint)
    assert manifest.seed == 17
    assert state["completed_steps"] == ["first"]
    assert state["failed_step"] == "second"

    fail = False
    resumed = workflow.execute(checkpoint_path=checkpoint, resume=True)
    assert calls == ["first", "second", "second", "third"]
    assert resumed.details["first"] == 10
    assert resumed.details["second"] == 15
    assert resumed.details["third"] == 30
    first_event = next(event for event in resumed.trace.events if event.step_id == "first")
    assert first_event.status is StepStatus.RESUMED


def test_resume_rejects_changed_plan_and_manifest(tmp_path: Path) -> None:
    checkpoint = tmp_path / "run.json"
    workflow = ResearchWorkflow(_plan())
    for step in _plan().steps:
        workflow.register(step.id, lambda _ctx, value=step.id: value)
    workflow.execute(checkpoint_path=checkpoint)

    changed = AnalysisPlan(
        request=AnalysisRequest("checkpoint experiment", require_plan_approval=False),
        steps=(PlanStep("different", "different", "local", "changed"),),
    )
    with pytest.raises(ValueError, match="fingerprint"):
        ResearchWorkflow(changed).execute(checkpoint_path=checkpoint, resume=True)

    wrong = RunManifest.create("checkpoint experiment", run_id="other")
    with pytest.raises(ValueError, match="run_id"):
        workflow.execute(checkpoint_path=checkpoint, resume=True, manifest=wrong)


def test_resume_requires_path_and_valid_state(tmp_path: Path) -> None:
    workflow = ResearchWorkflow(_plan())
    with pytest.raises(ValueError, match="checkpoint_path"):
        workflow.execute(resume=True)

    checkpoint = tmp_path / "bad.json"
    manifest = RunManifest.create("checkpoint experiment")
    save_checkpoint(
        checkpoint,
        manifest,
        {"plan_fingerprint": "wrong", "completed_steps": [], "context": {}},
    )
    with pytest.raises(ValueError, match="fingerprint"):
        workflow.execute(checkpoint_path=checkpoint, resume=True)


def test_checkpoint_rejects_non_json_context(tmp_path: Path) -> None:
    checkpoint = tmp_path / "run.json"
    plan = AnalysisPlan(
        request=AnalysisRequest("non-json", require_plan_approval=False),
        steps=(PlanStep("bad", "bad", "local", "returns non-json"),),
    )
    workflow = ResearchWorkflow(plan)
    workflow.register("bad", lambda _ctx: object())
    result = workflow.execute(checkpoint_path=checkpoint)
    assert "could not be checkpointed" in result.summary
    assert result.trace.events[-1].status is StepStatus.FAILED
    assert "JSON-serializable" in result.warnings[0]
