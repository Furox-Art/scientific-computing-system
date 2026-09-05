"""Transparent, policy-driven scientific method selection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from cds.policy import DataHandling, ExecutionLocation
from cds.workflow.types import Recommendation

_MISSING = object()


class ConditionOperator(str, Enum):
    EQ = "eq"
    NE = "ne"
    LT = "lt"
    LTE = "lte"
    GT = "gt"
    GTE = "gte"
    IN = "in"
    NOT_IN = "not_in"


class ConditionStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


class MethodStatus(str, Enum):
    ELIGIBLE = "eligible"
    REVIEW = "review"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class MethodSelectionContext:
    """Observed facts and explicit analysis/data-boundary constraints."""

    facts: tuple[tuple[str, object], ...] = ()
    required_capabilities: tuple[str, ...] = ()
    preferred_capabilities: tuple[str, ...] = ()
    available_tools: tuple[str, ...] = ()
    prohibited_traits: tuple[str, ...] = ()
    preferred_traits: tuple[str, ...] = ()
    prefer_local: bool = False
    sensitive_data: bool = False

    @classmethod
    def from_facts(
        cls,
        facts: Mapping[str, object],
        *,
        required_capabilities: tuple[str, ...] = (),
        preferred_capabilities: tuple[str, ...] = (),
        available_tools: tuple[str, ...] = (),
        prohibited_traits: tuple[str, ...] = (),
        preferred_traits: tuple[str, ...] = (),
        prefer_local: bool = False,
        sensitive_data: bool = False,
    ) -> MethodSelectionContext:
        return cls(
            facts=tuple(sorted(facts.items())),
            required_capabilities=required_capabilities,
            preferred_capabilities=preferred_capabilities,
            available_tools=available_tools,
            prohibited_traits=prohibited_traits,
            preferred_traits=preferred_traits,
            prefer_local=prefer_local,
            sensitive_data=sensitive_data,
        )

    def fact(self, key: str) -> object:
        for name, value in self.facts:
            if name == key:
                return value
        return _MISSING


@dataclass(frozen=True)
class SelectionCondition:
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
        return self.description or self.key

    def evaluate(self, context: MethodSelectionContext) -> ConditionStatus:
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
    condition: SelectionCondition
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError("preference weight must be greater than zero")


@dataclass(frozen=True)
class MethodCandidate:
    """Scientific method metadata, suitability rules, and data boundary."""

    name: str
    rationale: str
    capabilities: tuple[str, ...] = ()
    traits: tuple[str, ...] = ()
    required_tools: tuple[str, ...] = ()
    requirements: tuple[SelectionCondition, ...] = ()
    preferences: tuple[MethodPreference, ...] = ()
    base_score: float = 0.0
    execution_location: ExecutionLocation = ExecutionLocation.LOCAL
    data_handling: DataHandling = DataHandling.LOCAL_ONLY

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
    preferred_capability_weight: float = 1.0
    preferred_trait_weight: float = 0.5
    local_preference_weight: float = 2.0
    unknown_requirements_block: bool = False

    def __post_init__(self) -> None:
        if self.preferred_capability_weight < 0:
            raise ValueError("preferred capability weight must not be negative")
        if self.preferred_trait_weight < 0:
            raise ValueError("preferred trait weight must not be negative")
        if self.local_preference_weight < 0:
            raise ValueError("local preference weight must not be negative")


@dataclass(frozen=True)
class RankedMethod:
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
    policy_blocked: bool = False


@dataclass(frozen=True)
class MethodSelection:
    ranked: tuple[RankedMethod, ...]

    @property
    def recommended(self) -> RankedMethod | None:
        return next((item for item in self.ranked if item.status is not MethodStatus.BLOCKED), None)

    @property
    def alternatives(self) -> tuple[RankedMethod, ...]:
        recommended = self.recommended
        if recommended is None:
            return ()
        return tuple(
            item
            for item in self.ranked
            if item is not recommended and item.status is not MethodStatus.BLOCKED
        )

    def to_recommendation(self) -> Recommendation:
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
    if not candidates:
        raise ValueError("at least one method candidate is required")
    names = tuple(candidate.name for candidate in candidates)
    if len(set(names)) != len(names):
        raise ValueError("method candidate names must be unique")
    active_policy = policy if policy is not None else SelectionPolicy()
    ranked = tuple(_rank_candidate(candidate, context, active_policy) for candidate in candidates)
    status_order = {MethodStatus.ELIGIBLE: 0, MethodStatus.REVIEW: 1, MethodStatus.BLOCKED: 2}
    return MethodSelection(
        ranked=tuple(
            sorted(
                ranked,
                key=lambda item: (status_order[item.status], -item.score, item.candidate.name),
            )
        )
    )


def _rank_candidate(
    candidate: MethodCandidate,
    context: MethodSelectionContext,
    policy: SelectionPolicy,
) -> RankedMethod:
    missing_capabilities = tuple(
        capability for capability in context.required_capabilities if capability not in candidate.capabilities
    )
    missing_tools = tuple(tool for tool in candidate.required_tools if tool not in context.available_tools)
    prohibited_traits = tuple(trait for trait in candidate.traits if trait in context.prohibited_traits)

    failed_requirements: list[str] = []
    unknown_requirements: list[str] = []
    for requirement in candidate.requirements:
        condition_status = requirement.evaluate(context)
        if condition_status is ConditionStatus.FAIL:
            failed_requirements.append(requirement.label)
        elif condition_status is ConditionStatus.UNKNOWN:
            unknown_requirements.append(requirement.label)

    score = candidate.base_score
    matched_preferences: list[str] = []
    unknown_preferences: list[str] = []
    for preference in candidate.preferences:
        condition_status = preference.condition.evaluate(context)
        if condition_status is ConditionStatus.PASS:
            score += preference.weight
            matched_preferences.append(preference.condition.label)
        elif condition_status is ConditionStatus.UNKNOWN:
            unknown_preferences.append(preference.condition.label)

    matched_capabilities = tuple(
        capability
        for capability in context.preferred_capabilities
        if capability in candidate.capabilities
    )
    matched_traits = tuple(trait for trait in context.preferred_traits if trait in candidate.traits)
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

    if context.prefer_local and candidate.execution_location is ExecutionLocation.LOCAL:
        score += policy.local_preference_weight
        reasons.append("local execution preferred by analysis request")

    policy_blocked = context.sensitive_data and (
        candidate.execution_location is not ExecutionLocation.LOCAL
        or candidate.data_handling is not DataHandling.LOCAL_ONLY
    )
    if policy_blocked:
        reasons.append(
            "sensitive-data policy requires local execution with local-only data handling"
        )
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
        policy_blocked
        or missing_capabilities
        or missing_tools
        or prohibited_traits
        or failed_requirements
    )
    if policy.unknown_requirements_block and unknown_requirements:
        hard_block = True

    if hard_block:
        method_status = MethodStatus.BLOCKED
    elif unknown_requirements:
        method_status = MethodStatus.REVIEW
    else:
        method_status = MethodStatus.ELIGIBLE

    return RankedMethod(
        candidate=candidate,
        status=method_status,
        score=score,
        reasons=tuple(reasons),
        matched_preferences=tuple(matched_preferences),
        unknown_preferences=tuple(unknown_preferences),
        failed_requirements=tuple(failed_requirements),
        unknown_requirements=tuple(unknown_requirements),
        missing_capabilities=missing_capabilities,
        missing_tools=missing_tools,
        prohibited_traits=prohibited_traits,
        policy_blocked=policy_blocked,
    )


def _compare(value: object, operator: ConditionOperator, expected: object) -> bool:
    if operator is ConditionOperator.EQ:
        return value == expected
    if operator is ConditionOperator.NE:
        return value != expected
    if operator is ConditionOperator.IN:
        if not isinstance(expected, tuple):
            raise TypeError("membership comparison requires a tuple")
        return value in expected
    if operator is ConditionOperator.NOT_IN:
        if not isinstance(expected, tuple):
            raise TypeError("membership comparison requires a tuple")
        return value not in expected
    if isinstance(value, (int, float)) and isinstance(expected, (int, float)):
        return _compare_ordered(float(value), operator, float(expected))
    if isinstance(value, str) and isinstance(expected, str):
        return _compare_ordered(value, operator, expected)
    raise TypeError("ordered comparison requires compatible numeric or string values")


def _compare_ordered(
    left: float | str,
    operator: ConditionOperator,
    right: float | str,
) -> bool:
    if isinstance(left, float) and isinstance(right, float):
        if operator is ConditionOperator.LT:
            return left < right
        if operator is ConditionOperator.LTE:
            return left <= right
        if operator is ConditionOperator.GT:
            return left > right
        return left >= right
    if isinstance(left, str) and isinstance(right, str):
        if operator is ConditionOperator.LT:
            return left < right
        if operator is ConditionOperator.LTE:
            return left <= right
        if operator is ConditionOperator.GT:
            return left > right
        return left >= right
    raise TypeError("ordered comparison requires values of the same type")
