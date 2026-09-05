"""Branch-completeness tests for method selection."""

from __future__ import annotations

import pytest

import cds.workflow.selection as selection_module
from cds.workflow import (
    ConditionOperator,
    ConditionStatus,
    MethodCandidate,
    MethodPreference,
    MethodSelectionContext,
    MethodStatus,
    SelectionCondition,
    rank_methods,
)


def test_alternatives_exclude_blocked_candidates_when_recommendation_exists() -> None:
    eligible = MethodCandidate(name="eligible", rationale="eligible rationale")
    blocked = MethodCandidate(
        name="blocked",
        rationale="blocked rationale",
        capabilities=("other",),
    )
    context = MethodSelectionContext(required_capabilities=("required",))
    eligible_with_capability = MethodCandidate(
        name="eligible-with-capability",
        rationale="eligible rationale",
        capabilities=("required",),
    )

    selection = rank_methods((eligible, blocked, eligible_with_capability), context)

    assert selection.recommended is not None
    assert selection.recommended.candidate.name == "eligible-with-capability"
    assert selection.alternatives == ()
    assert [item.status for item in selection.ranked].count(MethodStatus.BLOCKED) == 2


def test_passed_requirement_and_failed_soft_preference_are_both_handled() -> None:
    requirement = SelectionCondition("supported", ConditionOperator.EQ, True, "supported")
    preference = MethodPreference(
        SelectionCondition("fast", ConditionOperator.EQ, True, "fast preferred"),
        weight=3.0,
    )
    candidate = MethodCandidate(
        name="method",
        rationale="method rationale",
        requirements=(requirement,),
        preferences=(preference,),
    )
    context = MethodSelectionContext.from_facts({"supported": True, "fast": False})

    ranked = rank_methods((candidate,), context).ranked[0]

    assert ranked.status is MethodStatus.ELIGIBLE
    assert ranked.failed_requirements == ()
    assert ranked.unknown_requirements == ()
    assert ranked.matched_preferences == ()
    assert ranked.unknown_preferences == ()
    assert ranked.score == 0.0


@pytest.mark.parametrize(
    ("operator", "value", "expected", "expected_status"),
    [
        (ConditionOperator.LT, "alpha", "beta", ConditionStatus.PASS),
        (ConditionOperator.LTE, "alpha", "alpha", ConditionStatus.PASS),
        (ConditionOperator.GT, "beta", "alpha", ConditionStatus.PASS),
        (ConditionOperator.GTE, "beta", "beta", ConditionStatus.PASS),
        (ConditionOperator.LT, "beta", "alpha", ConditionStatus.FAIL),
        (ConditionOperator.LTE, "beta", "alpha", ConditionStatus.FAIL),
        (ConditionOperator.GT, "alpha", "beta", ConditionStatus.FAIL),
        (ConditionOperator.GTE, "alpha", "beta", ConditionStatus.FAIL),
    ],
)
def test_ordered_string_conditions(
    operator: ConditionOperator,
    value: str,
    expected: str,
    expected_status: ConditionStatus,
) -> None:
    condition = SelectionCondition("label", operator, expected)
    context = MethodSelectionContext.from_facts({"label": value})
    assert condition.evaluate(context) is expected_status


@pytest.mark.parametrize("operator", [ConditionOperator.IN, ConditionOperator.NOT_IN])
def test_membership_runtime_guard_degrades_corrupt_condition_to_unknown(
    operator: ConditionOperator,
) -> None:
    condition = SelectionCondition("kind", operator, ("a", "b"))
    object.__setattr__(condition, "expected", "not-a-tuple")

    status = condition.evaluate(MethodSelectionContext.from_facts({"kind": "a"}))

    assert status is ConditionStatus.UNKNOWN


def test_ordered_comparison_runtime_guard_rejects_mixed_types() -> None:
    with pytest.raises(TypeError, match="same type"):
        selection_module._compare_ordered(1.0, ConditionOperator.LT, "2")
