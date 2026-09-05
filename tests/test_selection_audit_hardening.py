"""Regression tests for fail-closed scientific method selection inputs."""

import math

import pytest

from cds.workflow import (
    ConditionOperator,
    ConditionStatus,
    MethodCandidate,
    MethodPreference,
    MethodSelectionContext,
    SelectionCondition,
    SelectionPolicy,
)


def test_nonfinite_or_boolean_scores_are_rejected() -> None:
    condition = SelectionCondition("n", ConditionOperator.GT, 1)
    for invalid in (math.nan, math.inf, -math.inf, True):
        with pytest.raises(ValueError):
            MethodPreference(condition, invalid)
        with pytest.raises(ValueError):
            MethodCandidate("method", "rationale", base_score=invalid)
        with pytest.raises(ValueError):
            SelectionPolicy(preferred_capability_weight=invalid)


def test_whitespace_and_duplicate_selection_metadata_are_rejected() -> None:
    with pytest.raises(ValueError):
        SelectionCondition("   ", ConditionOperator.EQ, True)
    with pytest.raises(ValueError):
        MethodCandidate("   ", "rationale")
    with pytest.raises(ValueError):
        MethodCandidate("method", "   ")
    with pytest.raises(ValueError):
        MethodCandidate("method", "rationale", capabilities=("fit", "fit"))
    with pytest.raises(ValueError):
        MethodSelectionContext(required_capabilities=("fit", "fit"))
    with pytest.raises(ValueError):
        MethodSelectionContext(facts=(("n", 10), ("n", 11)))
    with pytest.raises(ValueError):
        MethodSelectionContext(facts=(("   ", 10),))


def test_nonfinite_ordered_fact_is_unknown_not_false_evidence() -> None:
    context = MethodSelectionContext.from_facts({"sample_size": math.nan})
    condition = SelectionCondition("sample_size", ConditionOperator.GTE, 30)
    assert condition.evaluate(context) is ConditionStatus.UNKNOWN


def test_boolean_is_not_silently_treated_as_numeric_ordered_fact() -> None:
    context = MethodSelectionContext.from_facts({"sample_size": True})
    condition = SelectionCondition("sample_size", ConditionOperator.GTE, 1)
    assert condition.evaluate(context) is ConditionStatus.UNKNOWN
