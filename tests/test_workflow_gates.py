"""Tests for validation-backed scientific conclusion gates."""

from __future__ import annotations

import pytest

from cds.validation import CheckStatus, ValidationCheck, ValidationReport
from cds.workflow import (
    AnalysisPlan,
    AnalysisRequest,
    GatePolicy,
    GateStatus,
    Recommendation,
    evaluate_research_gate,
)


def _plan(*, recommendation: Recommendation | None = None) -> AnalysisPlan:
    return AnalysisPlan(
        request=AnalysisRequest(question="Choose a defensible model"),
        steps=(),
        recommendation=(
            recommendation
            if recommendation is not None
            else Recommendation(
                recommended="robust-fit",
                rationale="resists influential observations",
                alternatives=("least-squares", "bayesian-fit"),
            )
        ),
    )


def _report(*checks: ValidationCheck) -> ValidationReport:
    return ValidationReport(checks=list(checks))


def test_gate_ready_only_after_required_validation_passes() -> None:
    report = _report(
        ValidationCheck("data-adequacy:observations", CheckStatus.PASS, "enough data"),
        ValidationCheck("cross-method-agreement", CheckStatus.PASS, "methods agree"),
    )
    policy = GatePolicy(required_checks=("cross-method-agreement",))

    decision = evaluate_research_gate(_plan(), (report,), policy=policy)

    assert decision.status is GateStatus.READY
    assert decision.allows_conclusion
    assert decision.reasons == ()
    assert decision.missing_checks == ()


def test_gate_blocks_validation_failures_and_preserves_review_reasons() -> None:
    report = _report(
        ValidationCheck("finite", CheckStatus.FAIL, "non-finite output"),
        ValidationCheck("stability", CheckStatus.WARNING, "sensitive result"),
    )

    decision = evaluate_research_gate(_plan(), (report,))

    assert decision.status is GateStatus.BLOCKED
    assert not decision.allows_conclusion
    assert "validation failures: finite" in decision.reasons
    assert "validation warnings require review: stability" in decision.reasons


def test_gate_blocks_when_validation_is_missing() -> None:
    decision = evaluate_research_gate(_plan(), ())
    assert decision.status is GateStatus.BLOCKED
    assert decision.reasons == ("no scientific validation checks were supplied",)


def test_gate_blocks_missing_required_checks() -> None:
    report = _report(ValidationCheck("finite", CheckStatus.PASS, "finite"))
    decision = evaluate_research_gate(
        _plan(),
        (report,),
        policy=GatePolicy(required_checks=("finite", "independent-audit")),
    )

    assert decision.status is GateStatus.BLOCKED
    assert decision.missing_checks == ("independent-audit",)
    assert "required scientific validation checks are missing" in decision.reasons


def test_gate_blocks_plan_without_explicit_recommendation() -> None:
    plan = AnalysisPlan(request=AnalysisRequest(question="test"), steps=())
    report = _report(ValidationCheck("finite", CheckStatus.PASS, "finite"))

    decision = evaluate_research_gate(plan, (report,))

    assert decision.status is GateStatus.BLOCKED
    assert "no explicit method recommendation" in decision.reasons[0]


def test_gate_reviews_recommendation_without_alternatives() -> None:
    plan = _plan(
        recommendation=Recommendation(
            recommended="one-method",
            rationale="caller preference",
        )
    )
    report = _report(ValidationCheck("finite", CheckStatus.PASS, "finite"))

    decision = evaluate_research_gate(plan, (report,))

    assert decision.status is GateStatus.REVIEW
    assert not decision.allows_conclusion
    assert decision.reasons == ("recommended method has no visible alternatives",)


def test_gate_can_allow_warnings_and_optional_policy_requirements() -> None:
    plan = AnalysisPlan(request=AnalysisRequest(question="test"), steps=())
    report = _report(ValidationCheck("stability", CheckStatus.WARNING, "review"))
    policy = GatePolicy(
        require_recommendation=False,
        require_alternatives=False,
        warnings_require_review=False,
    )

    decision = evaluate_research_gate(plan, (report,), policy=policy)

    assert decision.status is GateStatus.READY
    assert decision.allows_conclusion


def test_gate_can_disable_validation_requirement() -> None:
    decision = evaluate_research_gate(
        _plan(),
        (),
        policy=GatePolicy(require_validation=False),
    )
    assert decision.status is GateStatus.READY


@pytest.mark.parametrize(
    "required_checks",
    [("",), ("finite", "finite")],
)
def test_gate_policy_rejects_invalid_required_check_names(
    required_checks: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="required check names"):
        GatePolicy(required_checks=required_checks)
