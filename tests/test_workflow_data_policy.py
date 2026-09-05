"""Tests for execution-location and sensitive-data enforcement."""

from __future__ import annotations

import importlib.util

import pytest

from cds.policy import DataHandling, ExecutionLocation
from cds.tools import ToolRegistry, ToolSpec
from cds.workflow import (
    AnalysisRequest,
    MethodCandidate,
    MethodSelectionContext,
    rank_methods,
)
from cds.workflow.tooling import select_tool


def test_prefer_local_changes_method_ranking() -> None:
    remote = MethodCandidate(
        "remote",
        "remote backend",
        base_score=1.0,
        execution_location=ExecutionLocation.REMOTE,
        data_handling=DataHandling.NO_RETENTION,
    )
    local = MethodCandidate("local", "local backend")
    neutral = rank_methods((remote, local), MethodSelectionContext())
    assert neutral.recommended is not None
    assert neutral.recommended.candidate.name == "remote"

    preferred = rank_methods(
        (remote, local),
        MethodSelectionContext(prefer_local=True),
    )
    assert preferred.recommended is not None
    assert preferred.recommended.candidate.name == "local"
    assert "local execution preferred" in "; ".join(preferred.recommended.reasons)


def test_sensitive_data_hard_blocks_remote_or_external_handling() -> None:
    local = MethodCandidate("local", "local safe")
    remote = MethodCandidate(
        "remote",
        "remote",
        execution_location=ExecutionLocation.REMOTE,
        data_handling=DataHandling.NO_RETENTION,
    )
    external = MethodCandidate(
        "external",
        "external",
        data_handling=DataHandling.EXTERNAL_UNSPECIFIED,
    )
    ranked = rank_methods(
        (remote, external, local),
        MethodSelectionContext(sensitive_data=True),
    )
    assert ranked.recommended is not None
    assert ranked.recommended.candidate.name == "local"
    blocked = {item.candidate.name: item for item in ranked.ranked if item.policy_blocked}
    assert set(blocked) == {"remote", "external"}


def test_tool_selection_enforces_same_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "remote_math",
            "remote_math_fake",
            "remote-math",
            ("calc",),
            "remote test backend",
            execution_location=ExecutionLocation.REMOTE,
            data_handling=DataHandling.NO_RETENTION,
        )
    )
    registry.register(
        ToolSpec(
            "local_math",
            "local_math_fake",
            "local-math",
            ("calc",),
            "local test backend",
        )
    )

    original = importlib.util.find_spec

    def fake_find_spec(name: str, package: str | None = None):
        if name in {"remote_math_fake", "local_math_fake"}:
            return object()
        return original(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    preferred = select_tool("calc", registry=registry, prefer_local=True)
    assert preferred.tool == "local_math"
    sensitive = select_tool("calc", registry=registry, sensitive_data=True)
    assert sensitive.tool == "local_math"


def test_sensitive_tool_policy_fails_closed_when_only_remote_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "remote",
            "remote_fake",
            "remote-dist",
            ("calc",),
            "remote",
            execution_location=ExecutionLocation.REMOTE,
            data_handling=DataHandling.NO_RETENTION,
        )
    )
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name, package=None: object())
    with pytest.raises(ModuleNotFoundError, match="sensitive-data"):
        select_tool("calc", registry=registry, sensitive_data=True)


def test_analysis_request_validates_identity_fields() -> None:
    with pytest.raises(ValueError, match="question"):
        AnalysisRequest("   ")
    with pytest.raises(ValueError, match="seed"):
        AnalysisRequest("q", seed=True)  # type: ignore[arg-type]
