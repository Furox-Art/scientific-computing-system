"""Regression coverage for fail-closed audit guards added during hardening."""

from __future__ import annotations

import math
from typing import cast

import pytest

from cds.core.models import Hypothesis
from cds.hypothesis import Domain, HypothesisEvaluator, generate_hypotheses
from cds.provenance import DecisionRecord, RunManifest
from cds.stats import (
    bonferroni_corrected_alpha,
    chi_square_gof,
    chi_square_independence,
    cohens_d,
    cramers_v,
    eta_squared_from_f,
    one_sample_ttest,
    one_way_anova,
    paired_cohens_d,
    paired_ttest,
    two_sample_ttest,
)


def _hypothesis() -> Hypothesis:
    return generate_hypotheses("audit coverage", Domain.GENERAL_SCIENCE, n=1)[0]


def test_evaluator_result_guards_reject_nonfinite_or_invalid_statistics() -> None:
    evaluator = HypothesisEvaluator(alpha=0.05)
    hypothesis = _hypothesis()

    with pytest.raises(ValueError, match="effective alpha"):
        evaluator._build_result(hypothesis, "test", 0.0, 0.5, alpha=math.nan)
    with pytest.raises(ValueError, match="statistic"):
        evaluator._build_result(hypothesis, "test", math.nan, 0.5)
    with pytest.raises(ValueError, match="p-value"):
        evaluator._build_result(hypothesis, "test", 0.0, math.nan)
    with pytest.raises(ValueError, match="p-value"):
        evaluator._build_result(hypothesis, "test", 0.0, 1.1)
    with pytest.raises(ValueError, match="effect size"):
        evaluator._build_result(hypothesis, "test", 0.0, 0.5, effect_size=math.nan)


def _valid_manifest(**kwargs: object) -> RunManifest:
    values: dict[str, object] = {
        "question": "question",
        "run_id": "run",
        "created_utc": "2026-09-05T12:00:00+00:00",
    }
    values.update(kwargs)
    return RunManifest(
        question=cast(str, values["question"]),
        run_id=cast(str, values["run_id"]),
        created_utc=cast(str, values["created_utc"]),
        seed=cast(int | None, values.get("seed")),
        data_hashes=cast(dict[str, str], values.get("data_hashes", {})),
        tool_versions=cast(dict[str, str], values.get("tool_versions", {})),
        environment=cast(dict[str, str], values.get("environment", {})),
        decisions=cast(list[DecisionRecord], values.get("decisions", [])),
        metadata=cast(dict[str, str], values.get("metadata", {})),
    )


def test_provenance_identity_and_type_guards_cover_each_rejection_path() -> None:
    with pytest.raises(ValueError, match="approved_by_user"):
        DecisionRecord("action", "reason", cast(bool, 1))
    with pytest.raises(ValueError, match="question"):
        _valid_manifest(question=cast(str, 1))
    with pytest.raises(ValueError, match="run_id"):
        _valid_manifest(run_id=cast(str, 1))
    with pytest.raises(ValueError, match="created_utc"):
        _valid_manifest(created_utc="not-a-date")
    with pytest.raises(ValueError, match="timezone-aware"):
        _valid_manifest(created_utc="2026-09-05T12:00:00")
    with pytest.raises(ValueError, match="seed"):
        _valid_manifest(seed=True)


def test_provenance_hash_mapping_and_decision_guards_cover_compound_conditions() -> None:
    digest = "0" * 64
    with pytest.raises(ValueError, match="data hash names"):
        _valid_manifest(data_hashes={cast(str, 1): digest})
    with pytest.raises(ValueError, match="data hash names"):
        _valid_manifest(data_hashes={" ": digest})
    with pytest.raises(ValueError, match="SHA-256"):
        _valid_manifest(data_hashes={"data": cast(str, 1)})
    with pytest.raises(ValueError, match="SHA-256"):
        _valid_manifest(data_hashes={"data": "short"})
    with pytest.raises(ValueError, match="SHA-256"):
        _valid_manifest(data_hashes={"data": "z" * 64})

    with pytest.raises(ValueError, match="tool_versions"):
        _valid_manifest(tool_versions={cast(str, 1): "1.0"})
    with pytest.raises(ValueError, match="tool_versions"):
        _valid_manifest(tool_versions={" ": "1.0"})
    with pytest.raises(ValueError, match="tool_versions"):
        _valid_manifest(tool_versions={"tool": cast(str, 1)})
    with pytest.raises(ValueError, match="tool_versions"):
        _valid_manifest(tool_versions={"tool": " "})
    with pytest.raises(ValueError, match="decisions"):
        _valid_manifest(decisions=[cast(DecisionRecord, object())])


def test_statistical_sample_and_count_guards_reject_nonfinite_inputs() -> None:
    with pytest.raises(ValueError, match="finite"):
        one_sample_ttest([1.0, math.inf])
    with pytest.raises(ValueError, match="population mean"):
        one_sample_ttest([1.0, 2.0], popmean=math.nan)
    with pytest.raises(ValueError, match="finite"):
        two_sample_ttest([1.0, math.nan], [2.0, 3.0])
    with pytest.raises(ValueError, match="finite"):
        paired_ttest([1.0, 2.0], [2.0, math.inf])

    with pytest.raises(ValueError, match="finite"):
        chi_square_gof([1.0, math.nan], [1.0, 1.0])
    with pytest.raises(ValueError, match="non-negative"):
        chi_square_gof([-1.0, 2.0], [0.5, 0.5])
    with pytest.raises(ValueError, match="finite"):
        chi_square_gof([1.0, 1.0], [1.0, math.inf])
    with pytest.raises(ValueError, match="positive"):
        chi_square_gof([1.0, 1.0], [0.0, 2.0])


def test_statistical_ttest_edge_guards_and_paired_limits() -> None:
    with pytest.raises(ValueError, match="zero variance"):
        two_sample_ttest([1.0, 1.0], [2.0, 2.0], equal_var=True)
    with pytest.raises(ValueError, match="zero variance"):
        two_sample_ttest([1.0, 1.0], [2.0, 2.0], equal_var=False)
    with pytest.raises(ValueError, match="same length"):
        paired_ttest([1.0, 2.0], [1.0, 2.0, 3.0])

    identical = paired_ttest([1.0, 2.0], [1.0, 2.0])
    assert identical.statistic == 0.0
    assert identical.p_value == 1.0
    shifted = paired_ttest([2.0, 3.0], [1.0, 2.0])
    assert math.isinf(shifted.statistic)
    assert shifted.p_value == 0.0


def test_statistical_gof_anova_and_effect_size_fail_closed_paths() -> None:
    with pytest.raises(ValueError, match="equal totals"):
        chi_square_gof([2.0, 2.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="positive"):
        chi_square_independence([[0.0, 0.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="finite"):
        chi_square_independence([[1.0, math.inf], [2.0, 3.0]])
    with pytest.raises(ValueError, match="non-negative"):
        chi_square_independence([[1.0, -1.0], [2.0, 3.0]])

    with pytest.raises(ValueError, match="more observations"):
        one_way_anova([1.0], [2.0])
    with pytest.raises(ValueError, match="finite"):
        one_way_anova([1.0, math.nan], [2.0, 3.0])
    with pytest.raises(ValueError, match="zero within-group variance"):
        one_way_anova([1.0, 1.0], [2.0, 2.0])

    with pytest.raises(ValueError, match="finite"):
        cohens_d([1.0, math.inf], [2.0, 3.0])
    with pytest.raises(ValueError, match="same length"):
        paired_cohens_d([1.0, 2.0], [1.0, 2.0, 3.0])
    assert paired_cohens_d([1.0, 2.0], [1.0, 2.0]) == 0.0
    assert math.isinf(paired_cohens_d([2.0, 3.0], [1.0, 2.0]))


def test_statistical_effect_and_alpha_validation_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="F statistic"):
        eta_squared_from_f(math.nan, 1, 2)
    with pytest.raises(ValueError, match="df1"):
        eta_squared_from_f(1.0, 0, 2)
    with pytest.raises(ValueError, match="finite and positive"):
        cramers_v([[math.inf, 1.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="alpha"):
        bonferroni_corrected_alpha(math.nan, 2)
