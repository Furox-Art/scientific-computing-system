"""Evaluator regression tests for statistical-method findings from the audit."""

from __future__ import annotations

import math
from typing import cast

import pytest

from cds.core.models import Hypothesis
from cds.hypothesis import Domain, EvaluationData, HypothesisEvaluator, generate_hypotheses


def _hypothesis() -> Hypothesis:
    return generate_hypotheses("audit hypothesis", Domain.GENERAL_SCIENCE, n=1)[0]


def test_evaluator_validates_alpha_and_dispatch_contract() -> None:
    invalid_alphas = (0.0, 1.0, math.nan, math.inf, True, cast(float, "bad"))
    for alpha in invalid_alphas:
        with pytest.raises(ValueError, match="alpha"):
            HypothesisEvaluator(alpha=alpha)

    evaluator = HypothesisEvaluator()
    hypothesis = _hypothesis()
    ambiguous = cast(
        EvaluationData,
        {
            "groups": [[1.0, 2.0], [2.0, 3.0]],
            "paired": ([1.0, 2.0], [2.0, 3.0]),
        },
    )
    with pytest.raises(ValueError, match="exactly one"):
        evaluator.evaluate(hypothesis, ambiguous)
    with pytest.raises(ValueError, match="requires popmean"):
        evaluator.evaluate(hypothesis, cast(EvaluationData, {"one_sample": [1.0, 2.0]}))
    with pytest.raises(ValueError, match="requires observed"):
        evaluator.evaluate(hypothesis, {"chi_square_gof": {}})
    with pytest.raises(ValueError, match="labels"):
        evaluator.evaluate(
            hypothesis,
            {"groups": [[1.0, 2.0], [3.0, 4.0]], "labels": ["only-one"]},
        )
    correctly_labeled = evaluator.evaluate(
        _hypothesis(),
        {"groups": [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]], "labels": ["a", "b"]},
    )
    assert correctly_labeled.method_name == "Two-sample t-test"


def test_evaluator_paired_method_and_evidence_are_explicit() -> None:
    evaluator = HypothesisEvaluator()
    result = evaluator.evaluate(
        _hypothesis(),
        {"paired": ([1.0, 2.0, 3.0, 4.0], [10.0, 11.0, 12.0, 13.0])},
    )
    assert result.method_name == "Paired t-test"
    assert result.evidence_interpretation == "supported"
    assert "does not validate the full scientific hypothesis" in result.conclusion

    inconclusive = evaluator.evaluate(
        _hypothesis(),
        {"groups": [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]},
    )
    assert inconclusive.evidence_interpretation == "inconclusive"
    assert "does not demonstrate" in inconclusive.conclusion
