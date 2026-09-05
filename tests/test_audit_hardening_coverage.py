from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import pytest

import cds.provenance.manifest as provenance_manifest
from cds.data_io.streaming import StreamingLinearAccumulator, _solve_dense
from cds.provenance import RunManifest
from cds.sensitivity import GlobalSensitivityReport, global_sensitivity
from cds.tools import ToolRegistry, ToolSpec
from cds.validation import CheckStatus, check_residual_diagnostics
from cds.workflow import AnalysisRequest
from cds.workflow.tooling import select_tool


def test_streaming_extend_and_invalid_dense_dimensions() -> None:
    accumulator = StreamingLinearAccumulator()
    accumulator.extend([([1.0], 2.0)])
    assert accumulator.rows_seen == 1

    with pytest.raises(ValueError, match="dimensions are inconsistent"):
        _solve_dense([], [])


def test_manifest_creation_without_locks_or_git_sha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provenance_manifest, "detect_git_sha", lambda: None)
    manifest = RunManifest.create(
        "audit",
        project_root=tmp_path,
        run_id="audit-run",
        created_utc="2026-09-05T00:00:00+00:00",
    )
    assert manifest.run_id == "audit-run"
    assert "source.git_sha" not in manifest.metadata
    assert not any(key.startswith("lock.") for key in manifest.metadata)


def test_empty_global_sensitivity_report_has_no_most_influential_parameter() -> None:
    report = GlobalSensitivityReport(trajectories=0, levels=2, evaluations=0, parameters=())
    assert report.most_influential() is None


def test_global_sensitivity_rejects_nonfinite_perturbation_output() -> None:
    calls = 0

    def model(_values: Sequence[float]) -> float:
        nonlocal calls
        calls += 1
        return 0.0 if calls == 1 else math.nan

    with pytest.raises(ValueError, match="perturbation output"):
        global_sensitivity(model, [(0.0, 1.0)], trajectories=1, levels=2, seed=0)


def test_residual_diagnostics_handles_one_zero_variance_half() -> None:
    check = check_residual_diagnostics([0.0, 0.0, 1.0, -1.0])
    assert check.status is CheckStatus.WARNING
    assert check.details["variance_ratio"] == math.inf


def _local_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="stdlib-math",
            module="math",
            distribution="pip",
            capabilities=("audit-capability",),
            purpose="coverage fixture",
        )
    )
    return registry


def test_tool_selection_covers_sensitive_local_and_nonlocal_preference_paths() -> None:
    registry = _local_registry()

    sensitive = select_tool(
        "audit-capability",
        registry=registry,
        sensitive_data=True,
    )
    assert sensitive.tool == "stdlib-math"
    assert sensitive.data_egress is False

    unrestricted = select_tool(
        "audit-capability",
        registry=registry,
        prefer_local=False,
    )
    assert unrestricted.tool == "stdlib-math"


def test_analysis_request_rejects_sensitive_remote_fallback() -> None:
    with pytest.raises(ValueError, match="sensitive_data cannot allow remote fallback"):
        AnalysisRequest(
            question="audit",
            sensitive_data=True,
            allow_remote_fallback=True,
        )
