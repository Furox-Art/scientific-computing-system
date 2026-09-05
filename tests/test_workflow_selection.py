"""Tests for transparent scientific method ranking."""

from __future__ import annotations

import pytest

from cds.workflow import (
    ConditionOperator,
    ConditionStatus,
    MethodCandidate,
    MethodPreference,
    MethodSelectionContext,
    MethodStatus,
    SelectionCondition,
    SelectionPolicy,
    rank_methods,
)


def _candidate(
    name: str,
    *,
    requirements: tuple[SelectionCondition, ...] = (),
    preferences: tuple[MethodPreference, ...] = (),
    capabilities: tuple[str, ...] = ("regression",),
    traits: tuple[str, ...] = (),
    required_tools: tuple[str, ...] = (),
    base_score: float = 0.0,
) -> MethodCandidate:
    return MethodCandidate(
        name=name,
        rationale=f"{name} rationale",
        requirements=requirements,
        preferences=preferences,
        capabilities=capabilities,
        traits=traits,
        required_tools=required_tools,
        base_score=base_score,
    )


def test_rank_methods_prefers_explicit_matches_and_preserves_alternatives() -> None:
    low_missing = SelectionCondition(
        "missing_fraction",
        ConditionOperator.LTE,
        0.1,
        "limited missingness",
    )
    linear = _candidate(
        "linear",
        preferences=(MethodPreference(low_missing, weight=2.0),),
        capabilities=("regression", "interpretable"),
        traits=("local",),
    )
    flexible = _candidate("flexible", capabilities=("regression",))
    context = MethodSelectionContext.from_facts(
        {"missing_fraction": 0.02},
        required_capabilities=("regression",),
        preferred_capabilities=("interpretable",),
        preferred_traits=("local",),
    )

    selection = rank_methods((flexible, linear), context)

    assert [item.candidate.name for item in selection.ranked] == ["linear", "flexible"]
    assert selection.recommended is not None
    assert selection.recommended.candidate.name == "linear"
    assert selection.recommended.status is MethodStatus.ELIGIBLE
    assert selection.recommended.score == pytest.approx(3.5)
    assert selection.recommended.matched_preferences == ("limited missingness",)
    assert "preferred capabilities matched: interpretable" in selection.recommended.reasons
    assert "preferred traits matched: local" in selection.recommended.reasons
    assert [item.candidate.name for item in selection.alternatives] == ["flexible"]

    recommendation = selection.to_recommendation()
    assert recommendation.recommended == "linear"
    assert recommendation.alternatives == ("flexible",)
    assert "Selection evidence:" in recommendation.rationale


def test_failed_requirement_blocks_candidate() -> None:
    enough_data = SelectionCondition(
        "observations",
        ConditionOperator.GTE,
        30,
        "minimum observations",
    )
    candidate = _candidate("needs-data", requirements=(enough_data,))

    selection = rank_methods(
        (candidate,),
        MethodSelectionContext.from_facts({"observations": 12}),
    )

    ranked = selection.ranked[0]
    assert ranked.status is MethodStatus.BLOCKED
    assert ranked.failed_requirements == ("minimum observations",)
    assert "mandatory requirements failed: minimum observations" in ranked.reasons
    assert selection.recommended is None
    assert selection.alternatives == ()
    with pytest.raises(ValueError, match="no non-blocked method"):
        selection.to_recommendation()


def test_missing_requirement_produces_provisional_review() -> None:
    normality = SelectionCondition("normality_supported", ConditionOperator.EQ, True)
    candidate = _candidate("parametric", requirements=(normality,))

    selection = rank_methods((candidate,), MethodSelectionContext())

    ranked = selection.ranked[0]
    assert ranked.status is MethodStatus.REVIEW
    assert ranked.unknown_requirements == ("normality_supported",)
    recommendation = selection.to_recommendation()
    assert recommendation.recommended == "parametric"
    assert recommendation.rationale.startswith("Provisional recommendation; review required.")


def test_unknown_requirement_can_be_configured_as_blocking() -> None:
    requirement = SelectionCondition("assumption", ConditionOperator.EQ, True)
    candidate = _candidate("strict", requirements=(requirement,))

    selection = rank_methods(
        (candidate,),
        MethodSelectionContext(),
        policy=SelectionPolicy(unknown_requirements_block=True),
    )

    assert selection.ranked[0].status is MethodStatus.BLOCKED


def test_confirmed_eligible_method_ranks_before_higher_scoring_review_method() -> None:
    unknown = SelectionCondition("unknown_fact", ConditionOperator.EQ, True)
    review = _candidate("review", requirements=(unknown,), base_score=100.0)
    eligible = _candidate("eligible", base_score=0.0)

    selection = rank_methods((review, eligible), MethodSelectionContext())

    assert [item.candidate.name for item in selection.ranked] == ["eligible", "review"]


def test_missing_capability_tool_and_prohibited_trait_are_blocking() -> None:
    candidate = _candidate(
        "external",
        capabilities=("classification",),
        traits=("remote",),
        required_tools=("solver",),
    )
    context = MethodSelectionContext(
        required_capabilities=("regression",),
        available_tools=(),
        prohibited_traits=("remote",),
    )

    ranked = rank_methods((candidate,), context).ranked[0]

    assert ranked.status is MethodStatus.BLOCKED
    assert ranked.missing_capabilities == ("regression",)
    assert ranked.missing_tools == ("solver",)
    assert ranked.prohibited_traits == ("remote",)
    assert "missing required capabilities: regression" in ranked.reasons
    assert "missing required tools: solver" in ranked.reasons
    assert "prohibited traits present: remote" in ranked.reasons


def test_unknown_soft_preference_is_visible_but_not_blocking() -> None:
    preference = MethodPreference(
        SelectionCondition("outliers", ConditionOperator.EQ, True, "outliers present"),
        weight=4.0,
    )
    candidate = _candidate("robust", preferences=(preference,))

    ranked = rank_methods((candidate,), MethodSelectionContext()).ranked[0]

    assert ranked.status is MethodStatus.ELIGIBLE
    assert ranked.score == 0.0
    assert ranked.unknown_preferences == ("outliers present",)
    assert "soft preferences unknown: outliers present" in ranked.reasons


@pytest.mark.parametrize(
    ("operator", "value", "expected", "status"),
    [
        (ConditionOperator.EQ, "continuous", "continuous", ConditionStatus.PASS),
        (ConditionOperator.NE, "continuous", "binary", ConditionStatus.PASS),
        (ConditionOperator.LT, 2, 3, ConditionStatus.PASS),
        (ConditionOperator.LTE, 3, 3, ConditionStatus.PASS),
        (ConditionOperator.GT, 4, 3, ConditionStatus.PASS),
        (ConditionOperator.GTE, 3, 3, ConditionStatus.PASS),
        (ConditionOperator.IN, "continuous", ("continuous", "binary"), ConditionStatus.PASS),
        (ConditionOperator.NOT_IN, "count", ("continuous", "binary"), ConditionStatus.PASS),
        (ConditionOperator.EQ, "continuous", "binary", ConditionStatus.FAIL),
    ],
)
def test_condition_operators(
    operator: ConditionOperator,
    value: object,
    expected: object,
    status: ConditionStatus,
) -> None:
    condition = SelectionCondition("value", operator, expected)
    context = MethodSelectionContext.from_facts({"value": value})
    assert condition.evaluate(context) is status


def test_condition_returns_unknown_for_missing_or_incomparable_fact() -> None:
    missing = SelectionCondition("missing", ConditionOperator.EQ, True)
    incomparable = SelectionCondition("value", ConditionOperator.GT, 3)

    assert missing.evaluate(MethodSelectionContext()) is ConditionStatus.UNKNOWN
    assert (
        incomparable.evaluate(MethodSelectionContext.from_facts({"value": "text"}))
        is ConditionStatus.UNKNOWN
    )


def test_context_fact_and_deterministic_mapping_conversion() -> None:
    context = MethodSelectionContext.from_facts(
        {"z": 1, "a": 2},
        required_capabilities=("fit",),
        preferred_capabilities=("interpretable",),
        available_tools=("python",),
        prohibited_traits=("remote",),
        preferred_traits=("local",),
    )
    assert context.facts == (("a", 2), ("z", 1))
    assert context.fact("a") == 2
    assert context.required_capabilities == ("fit",)
    assert context.preferred_capabilities == ("interpretable",)
    assert context.available_tools == ("python",)
    assert context.prohibited_traits == ("remote",)
    assert context.preferred_traits == ("local",)
    assert context.fact("missing") is not None


def test_ranking_ties_are_deterministic_by_name() -> None:
    selection = rank_methods(
        (_candidate("zeta"), _candidate("alpha")),
        MethodSelectionContext(),
    )
    assert [item.candidate.name for item in selection.ranked] == ["alpha", "zeta"]


def test_candidate_rejects_empty_name_and_rationale() -> None:
    with pytest.raises(ValueError, match="method name"):
        MethodCandidate(name="", rationale="rationale")
    with pytest.raises(ValueError, match="method rationale"):
        MethodCandidate(name="method", rationale="")


def test_candidate_rejects_invalid_capabilities() -> None:
    with pytest.raises(ValueError, match="capabilities"):
        MethodCandidate(name="method", rationale="rationale", capabilities=("",))
    with pytest.raises(ValueError, match="capabilities"):
        MethodCandidate(name="method", rationale="rationale", capabilities=("a", "a"))


def test_candidate_rejects_invalid_traits() -> None:
    with pytest.raises(ValueError, match="traits"):
        MethodCandidate(name="method", rationale="rationale", traits=("",))
    with pytest.raises(ValueError, match="traits"):
        MethodCandidate(name="method", rationale="rationale", traits=("a", "a"))


def test_candidate_rejects_invalid_required_tools() -> None:
    with pytest.raises(ValueError, match="required tools"):
        MethodCandidate(name="method", rationale="rationale", required_tools=("",))
    with pytest.raises(ValueError, match="required tools"):
        MethodCandidate(name="method", rationale="rationale", required_tools=("a", "a"))


def test_selection_input_validation() -> None:
    with pytest.raises(ValueError, match="at least one"):
        rank_methods((), MethodSelectionContext())
    with pytest.raises(ValueError, match="names must be unique"):
        rank_methods(
            (_candidate("same"), _candidate("same")),
            MethodSelectionContext(),
        )


def test_condition_and_policy_validation() -> None:
    with pytest.raises(ValueError, match="condition key"):
        SelectionCondition("", ConditionOperator.EQ, True)
    with pytest.raises(ValueError, match="membership conditions"):
        SelectionCondition("kind", ConditionOperator.IN, "continuous")
    with pytest.raises(ValueError, match="preference weight"):
        MethodPreference(SelectionCondition("x", ConditionOperator.EQ, 1), weight=0)
    with pytest.raises(ValueError, match="capability weight"):
        SelectionPolicy(preferred_capability_weight=-1)
    with pytest.raises(ValueError, match="trait weight"):
        SelectionPolicy(preferred_trait_weight=-1)


def test_custom_zero_weights_leave_only_base_and_preference_scores() -> None:
    preference = MethodPreference(SelectionCondition("x", ConditionOperator.EQ, 1), weight=2)
    candidate = _candidate(
        "method",
        preferences=(preference,),
        capabilities=("regression", "fast"),
        traits=("local",),
        base_score=3,
    )
    context = MethodSelectionContext.from_facts(
        {"x": 1},
        preferred_capabilities=("fast",),
        preferred_traits=("local",),
    )
    policy = SelectionPolicy(preferred_capability_weight=0, preferred_trait_weight=0)

    ranked = rank_methods((candidate,), context, policy=policy).ranked[0]
    assert ranked.score == 5
