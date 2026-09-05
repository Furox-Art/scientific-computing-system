"""Transparent, policy-driven scientific method selection.

The selector deliberately avoids hard-coded universal scientific thresholds.
Callers describe each candidate's requirements and soft preferences explicitly;
the engine then evaluates those rules against supplied facts, records uncertainty,
and returns a deterministic ranking with visible alternatives.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from cds.workflow.types import Recommendation

_MISSING = object()


class ConditionOperator(str, Enum):
    """Supported declarative comparisons for method-selection facts."""

    EQ = "eq"
    NE = "ne"
    LT = "lt"
    LTE = "lte"
    GT = "gt"
    GTE = "gte"
    IN = "in"
    NOT_IN = "not_in"


class ConditionStatus(str, Enum):
    """Outcome of evaluating one explicit condition."""

    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


class MethodStatus(str, Enum):
    """Eligibility state of one candidate method."""

    ELIGIBLE = "eligible"
    REVIEW = "review"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class MethodSelectionContext:
    """Observed facts and explicit analysis constraints used for ranking."""

    facts: tuple[tuple[str, object], ...] = ()
    required_capabilities: tuple[str, ...] = ()
    preferred_capabilities: tuple[str, ...] = ()
    available_tools: tuple[str, ...] = ()
    prohibited_traits: tuple[str, ...] = ()
    preferred_traits: tuple[str, ...] = ()

    @classmethod
    def from_facts(
        cls,
        facts: Mapping[str, object],
        **kwargs: object,
    ) -> MethodSelectionContext:
        """Create a deterministic context from a mapping of observed facts."""
        return cls(facts=tuple(sorted(facts.items())), **kwargs)

    def fact(self, key: str) -> object:
        """Return a fact value or an internal missing sentinel."""
        for name, value in self.facts:
            if name == key:
                return value
        return _MISSING


@dataclass(frozen=True)
class SelectionCondition:
    """One declarative requirement or preference condition."""

    key: str
    operator: ConditionOperator
    expected: object
    description: str = ""

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("condition key must not be empty")
        if self.operator in {ConditionOperator.IN, ConditionOperator.NOT_IN} and not isinstance(
            self.expected, tuple
        ):
            raise ValueError("membership conditions require tuple expected values")

    @property
    def label(self) -> str:
        """Human-readable label for audit output."""
        return self.description or self.key

    def evaluate(self, context: MethodSelectionContext) -> ConditionStatus:
        """Evaluate this condition without converting missing data into failure."""
        value = context.fact(self.key)
        if value is _MISSING:
            return ConditionStatus.UNKNOWN

        try:
            matched = _compare(value, self.operator, self.expected)
        except TypeError:
            return ConditionStatus.UNKNOWN
        return ConditionStatus.PASS if matched else ConditionStatus.FAIL


@dataclass(frozen=True)
class MethodPreference:
    """Soft criterion that contributes transparent weight when satisfied."""

    condition: SelectionCondition
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError("preference weight must be greater than zero")


@dataclass(frozen=True)
class MethodCandidate:
    """Method metadata and caller-defined scientific suitability rules."""

    name: str
    rationale: str
    capabilities: tuple[str, ...] = ()
    traits: tuple[str, ...] = ()
    required_tools: tuple[str, ...] = ()
    requirements: tuple[SelectionCondition, ...] = ()
    preferences: tuple[MethodPreference, ...] = ()
    base_score: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("method name must not be empty")
        if not self.rationale:
            raise ValueError("method rationale must not be empty")
        for field_name, values in (
            ("capabilities", self.capabilities),
            ("traits", self.traits),
            ("required tools", self.required_tools),
        ):
            if any(not value for value in values):
                raise ValueError(f"{field_name} must not contain empty values")
            if len(set(values)) != len(values):
                raise ValueError(f"{field_name} must be unique")


@dataclass(frozen=True)
class SelectionPolicy:
    """Weights and uncertainty policy for deterministic ranking."""

    preferred_capability_weight: float = 1.0
    preferred_trait_weight: float = 0.5
    unknown_requirements_block: bool = False

    def __post_init__(self) -> None:
        if self.preferred_capability_weight < 0:
            raise ValueError("preferred capability weight must not be negative")
        if self.preferred_trait_weight < 0:
            raise ValueError("preferred trait weight must not be negative")


@dataclass(frozen=True)
class RankedMethod:
    """Auditable score and eligibility explanation for one candidate."""

    candidate: MethodCandidate
    status: MethodStatus
    score: float
    reasons: tuple[str, ...] = ()
    matched_preferences: tuple[str, ...] = ()
    unknown_preferences: tuple[str, ...] = ()
    failed_requirements: tuple[str, ...] = ()
    unknown_requirements: tuple[str, ...] = ()
    missing_capabilities: tuple[str, ...] = ()
    missing_tools: tuple[str, ...] = ()
    prohibited_traits: tuple[str, ...] = ()


@dataclass(frozen=True)
class MethodSelection:
    """Full ranking, including blocked candidates and visible alternatives."""

    ranked: tuple[RankedMethod, ...]

    @property
    def recommended(self) -> RankedMethod | None:
        """Best non-blocked candidate, or ``None`` when all methods are blocked."""
        return next((item for item in self.ranked if item.status is not MethodStatus.BLOCKED), None)

    @property
    def alternatives(self) -> tuple[RankedMethod, ...]:
        """Other non-blocked candidates retained for user review."""
        recommended = self.recommended
        if recommended is None:
            return ()
        return tuple(
            item
            for item in self.ranked
            if item is not recommended and item.status is not MethodStatus.BLOCKED
        )

    def to_recommendation(self) -> Recommendation:
        """Convert the ranking into the workflow's existing recommendation type."""
        recommended = self.recommended
        if recommended is None:
            raise ValueError("no non-blocked method is available for recommendation")

        rationale = recommended.candidate.rationale
        if recommended.status is MethodStatus.REVIEW:
            rationale = f"Provisional recommendation; review required. {rationale}"
        if recommended.reasons:
            rationale = f"{rationale} Selection evidence: {'; '.join(recommended.reasons)}"

        return Recommendation(
            recommended=recommended.candidate.name,
            rationale=rationale,
            alternatives=tuple(item.candidate.name for item in self.alternatives),
        )


def rank_methods(
    candidates: tuple[MethodCandidate, ...],
    context: MethodSelectionContext,
    *,
    policy: SelectionPolicy | None = None,
) -> MethodSelection:
    """Rank candidate methods using explicit constraints and supplied facts.

    Hard failures block a method. Missing mandatory evidence produces ``REVIEW``
    by default rather than silently accepting or rejecting the method. Soft
    preferences only add their declared weights when satisfied.
    """
    if not candidates:
        raise ValueError("at least one method candidate is required")
    names = tuple(candidate.name for candidate in candidates)
    if len(set(names)) != len(names):
        raise ValueError("method candidate names must be unique")

    active_policy = policy if policy is not None else SelectionPolicy()
    ranked = tuple(_rank_candidate(candidate, context, active_policy) for candidate in candidates)
    status_order = {
        MethodStatus.ELIGIBLE: 0,
        MethodStatus.REVIEW: 1,
        MethodStatus.BLOCKED: 2,
    }
    ordered = tuple(
        sorted(
            ranked,
            key=lambda item: (status_order[item.status], -item.score, item.candidate.name),
        )
    )
    return MethodSelection(ranked=ordered)


def _rank_candidate(
    candidate: MethodCandidate,
    context: MethodSelectionContext,
    policy: SelectionPolicy,
) -> RankedMethod:
    missing_capabilities = tuple(
        capability
        for capability in context.required_capabilities
        if capability not in candidate.capabilities
    )
    missing_tools = tuple(
        tool for tool in candidate.required_tools if tool not in context.available_tools
    )
    prohibited_traits = tuple(
        trait for trait in candidate.traits if trait in context.prohibited_traits
    )

    failed_requirements: list[str] = []
    unknown_requirements: list[str] = []
    for requirement in candidate.requirements:
        status = requirement.evaluate(context)
        if status is ConditionStatus.FAIL:
            failed_requirements.append(requirement.label)
        elif status is ConditionStatus.UNKNOWN:
            unknown_requirements.append(requirement.label)

    score = candidate.base_score
    matched_preferences: list[str] = []
    unknown_preferences: list[str] = []
    for preference in candidate.preferences:
        status = preference.condition.evaluate(context)
        if status is ConditionStatus.PASS:
            score += preference.weight
            matched_preferences.append(preference.condition.label)
        elif status is ConditionStatus.UNKNOWN:
            unknown_preferences.append(preference.condition.label)

    matched_capabilities = tuple(
        capability
        for capability in context.preferred_capabilities
        if capability in candidate.capabilities
    )
    matched_traits = tuple(
        trait for trait in context.preferred_traits if trait in candidate.traits
    )
    score += len(matched_capabilities) * policy.preferred_capability_weight
    score += len(matched_traits) * policy.preferred_trait_weight

    reasons: list[str] = []
    if matched_capabilities:
        reasons.append(f"preferred capabilities matched: {', '.join(matched_capabilities)}")
    if matched_traits:
        reasons.append(f"preferred traits matched: {', '.join(matched_traits)}")
    if matched_preferences:
        reasons.append(f"soft preferences matched: {', '.join(matched_preferences)}")
    if unknown_preferences:
        reasons.append(f"soft preferences unknown: {', '.join(unknown_preferences)}")
    if missing_capabilities:
        reasons.append(f"missing required capabilities: {', '.join(missing_capabilities)}")
    if missing_tools:
        reasons.append(f"missing required tools: {', '.join(missing_tools)}")
    if prohibited_traits:
        reasons.append(f"prohibited traits present: {', '.join(prohibited_traits)}")
    if failed_requirements:
        reasons.append(f"mandatory requirements failed: {', '.join(failed_requirements)}")
    if unknown_requirements:
        reasons.append(f"mandatory requirements unknown: {', '.join(unknown_requirements)}")

    hard_block = bool(
        missing_capabilities or missing_tools or prohibited_traits or failed_requirements
    )
    if policy.unknown_requirements_block and unknown_requirements:
        hard_block = True

    if hard_block:
        status = MethodStatus.BLOCKED
    elif unknown_requirements:
        status = MethodStatus.REVIEW
    else:
        status = MethodStatus.ELIGIBLE

    return RankedMethod(
        candidate=candidate,
        status=status,
        score=score,
        reasons=tuple(reasons),
        matched_preferences=tuple(matched_preferences),
        unknown_preferences=tuple(unknown_preferences),
        failed_requirements=tuple(failed_requirements),
        unknown_requirements=tuple(unknown_requirements),
        missing_capabilities=missing_capabilities,
        missing_tools=missing_tools,
        prohibited_traits=prohibited_traits,
    )


def _compare(value: object, operator: ConditionOperator, expected: object) -> bool:
    if operator is ConditionOperator.EQ:
        return value == expected
    if operator is ConditionOperator.NE:
        return value != expected
    if operator is ConditionOperator.LT:
        return value < expected  # type: ignore[operator]
    if operator is ConditionOperator.LTE:
        return value <= expected  # type: ignore[operator]
    if operator is ConditionOperator.GT:
        return value > expected  # type: ignore[operator]
    if operator is ConditionOperator.GTE:
        return value >= expected  # type: ignore[operator]
    if operator is ConditionOperator.IN:
        return value in expected  # type: ignore[operator]
    return value not in expected  # type: ignore[operator]
