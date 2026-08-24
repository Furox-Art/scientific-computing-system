"""Tests for the hypothesis-mining engine (cds.hypothesis.mining)."""

from __future__ import annotations

import random

import pytest

from cds.core.models import Domain, Hypothesis, HypothesisStatus
from cds.hypothesis import HypothesisEvaluator
from cds.hypothesis.mining import (
    MinedHypothesis,
    _classify_strength,
    _pearson_p_value,
    mine_correlations,
)
from cds.stats.descriptive import correlation
from cds.stats.hypothesis_tests import t_sf


def _snapshot(records: list[MinedHypothesis]) -> list[tuple[str, str, float, float, str, str]]:
    return [
        (
            hit.feature_a,
            hit.feature_b,
            hit.correlation,
            hit.p_value,
            hit.strength,
            hit.hypothesis.statement,
        )
        for hit in records
    ]


def test_planted_linear_relation_surfaces_as_strong_and_significant() -> None:
    data = {
        "signal": [1.0, 2.0, 3.0, 4.0, 5.0],
        "response": [3.1, 6.2, 9.3, 12.4, 15.5],
    }
    mined = mine_correlations(data)
    assert len(mined) == 1
    hit = mined[0]
    assert (hit.feature_a, hit.feature_b) == ("signal", "response")
    assert hit.strength == "strong"
    assert hit.correlation == pytest.approx(1.0)
    assert hit.p_value < 0.05
    assert isinstance(hit.hypothesis, Hypothesis)
    assert hit.hypothesis.status == HypothesisStatus.TESTABLE


def test_perfect_correlation_yields_exactly_zero_p_value() -> None:
    mined = mine_correlations({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})
    assert len(mined) == 1
    assert mined[0].correlation == 1.0
    assert mined[0].p_value == 0.0
    assert mined[0].strength == "strong"


def test_negative_relation_reports_negative_direction() -> None:
    data = {"dose": [0.0, 1.0, 2.0, 3.0, 4.0], "effect": [10.0, 8.0, 6.0, 4.0, 2.0]}
    mined = mine_correlations(data)
    assert len(mined) == 1
    hit = mined[0]
    assert hit.correlation < 0
    assert hit.correlation == pytest.approx(-1.0)
    assert hit.strength == "strong"
    assert "negative" in hit.hypothesis.statement
    assert hit.hypothesis.metadata["sign"] == "negative"
    assert hit.hypothesis.confidence == pytest.approx(0.85)


def test_results_sorted_by_abs_r_with_tiers_and_stable_ties() -> None:
    ramp = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    weave = [7.0, 2.0, 5.0, 4.0, 3.0, 6.0, 1.0]
    data = {"s": ramp * 4, "t": [2.0 * v for v in ramp * 4], "u": weave * 4}
    mined = mine_correlations(data)
    assert [(hit.feature_a, hit.feature_b) for hit in mined] == [
        ("s", "t"),
        ("s", "u"),
        ("t", "u"),
    ]
    assert [hit.strength for hit in mined] == ["strong", "moderate", "moderate"]
    abs_rs = [abs(hit.correlation) for hit in mined]
    assert abs_rs == sorted(abs_rs, reverse=True)
    assert abs_rs[0] == pytest.approx(1.0)
    assert abs_rs[1] == pytest.approx(abs(-12.0 / 28.0))
    assert mined[1].p_value < 0.05


def test_independent_noise_columns_produce_no_hits_under_strict_thresholds() -> None:
    rng = random.Random(1234)
    data = {f"noise_{k}": [rng.random() for _ in range(50)] for k in range(3)}
    assert mine_correlations(data, min_abs_r=0.95, alpha=0.01) == []


def test_mining_is_deterministic_across_calls() -> None:
    data = {
        "s": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "t": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        "u": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
    }
    first = mine_correlations(data)
    second = mine_correlations(data)
    assert first is not second
    assert _snapshot(first) == _snapshot(second)
    assert [hit.hypothesis.id for hit in first] == [hit.hypothesis.id for hit in second]


def test_emitted_hypothesis_carries_expected_statement_text() -> None:
    mined = mine_correlations({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})
    expected = (
        "x shows a strong positive linear association with y (Pearson r=1.0000, p=0.000e+00, n=3)."
    )
    assert mined[0].hypothesis.statement == expected
    assert mined[0].hypothesis.research_question == "Are x and y linearly associated?"


def test_emitted_hypothesis_payload_is_derived_from_the_numbers() -> None:
    data = {"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]}
    hypo = mine_correlations(data)[0].hypothesis
    assert hypo.domain == Domain.GENERAL_SCIENCE
    assert hypo.id == "CORR-x-vs-y"
    assert hypo.rationale is not None
    assert "r=1.0000" in hypo.rationale
    assert "n=3" in hypo.rationale
    assert "|r| >= 0.3 and p < 0.05" in hypo.rationale
    assert any("approximately linear" in a for a in hypo.assumptions)
    assert any("|r| >= 0.3" in p for p in hypo.predictions)
    assert any("alpha=0.05" in p for p in hypo.predictions)
    assert hypo.tags == ["correlation", "mined", "strong"]
    assert hypo.sources == ["cds.hypothesis.mining.mine_correlations"]
    assert hypo.metadata["n"] == "3"
    assert hypo.metadata["degrees_of_freedom"] == "1"
    assert hypo.metadata["pearson_r"] == "1.000000"


def test_constant_column_skipped_silently() -> None:
    data = {
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "const": [7.0, 7.0, 7.0, 7.0, 7.0],
        "b": [2.0, 4.0, 6.0, 8.0, 10.0],
    }
    mined = mine_correlations(data)
    assert [(hit.feature_a, hit.feature_b) for hit in mined] == [("a", "b")]


def test_fewer_than_two_usable_columns_after_constant_skip_raises() -> None:
    with pytest.raises(ValueError, match="at least 2 non-constant columns"):
        mine_correlations({"a": [1.0, 2.0, 3.0], "dead": [5.0, 5.0, 5.0]})


def test_empty_data_raises() -> None:
    with pytest.raises(ValueError, match="at least 2 numeric columns"):
        mine_correlations({})


def test_single_column_raises_after_filtering() -> None:
    with pytest.raises(ValueError, match="at least 2 non-constant columns"):
        mine_correlations({"only": [1.0, 2.0, 3.0]})


def test_column_shorter_than_three_observations_raises() -> None:
    with pytest.raises(ValueError, match="needs at least 3 observations"):
        mine_correlations({"a": [1.0, 2.0], "b": [3.0, 4.0]})


def test_mismatched_column_lengths_raise() -> None:
    with pytest.raises(ValueError, match="'b' has 5, expected 4"):
        mine_correlations({"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 2.0, 3.0, 4.0, 5.0]})


@pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.1, 1.5])
def test_alpha_out_of_range_raises_both_directions(bad_alpha: float) -> None:
    data = {"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.5]}
    with pytest.raises(ValueError, match="alpha must be strictly between"):
        mine_correlations(data, alpha=bad_alpha)


@pytest.mark.parametrize("bad_min_abs_r", [-0.1, 1.0, 1.5])
def test_min_abs_r_out_of_range_raises_both_directions(bad_min_abs_r: float) -> None:
    data = {"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.5]}
    with pytest.raises(ValueError, match="min_abs_r must lie between"):
        mine_correlations(data, min_abs_r=bad_min_abs_r)


def test_max_features_caps_considered_columns() -> None:
    data = {
        "f1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "f2": [2.0, 4.0, 6.0, 8.0, 10.0],
        "f3": [1.0, 3.0, 2.0, 5.0, 4.0],
        "f4": [9.0, 7.0, 8.0, 5.0, 6.0],
    }
    capped = mine_correlations(data, max_features=2)
    assert [(hit.feature_a, hit.feature_b) for hit in capped] == [("f1", "f2")]
    uncapped = mine_correlations(data)
    assert len(uncapped) > len(capped)


def test_significant_but_below_min_abs_r_pair_is_rejected_by_effect_size() -> None:
    rng = random.Random(99)
    data = {
        "a": [rng.random() for _ in range(60)],
        "b": [rng.random() for _ in range(60)],
    }
    assert mine_correlations(data, min_abs_r=0.9) == []
    r = correlation(data["a"], data["b"])
    assert abs(r) < 0.9


def test_strong_but_nonsignificant_pair_is_rejected_by_alpha() -> None:
    x = [1.0, 2.0, 3.0, 4.0]
    y = [1.0, 2.0, 3.0, 8.0]
    r = correlation(x, y)
    assert abs(r) >= 0.3
    assert _pearson_p_value(r, len(x)) >= 0.05
    assert mine_correlations({"x": x, "y": y}) == []


def test_strength_tier_boundaries_at_exact_thresholds() -> None:
    assert _classify_strength(0.199) == "weak"
    assert _classify_strength(0.2) == "moderate"
    assert _classify_strength(0.35) == "moderate"
    assert _classify_strength(0.499) == "moderate"
    assert _classify_strength(0.5) == "strong"
    assert _classify_strength(0.999) == "strong"


def test_p_value_matches_manual_t_transformation() -> None:
    r, n = 0.5, 10
    t_stat = abs(r) * ((n - 2) / (1.0 - r * r)) ** 0.5
    expected = t_sf(t_stat, float(n - 2))
    assert _pearson_p_value(r, n) == pytest.approx(expected)


def test_mined_hypothesis_flows_into_existing_evaluator_pipeline() -> None:
    mined = mine_correlations(
        {"dose": [1.0, 2.0, 3.0, 4.0, 5.0], "effect": [2.0, 4.0, 6.0, 8.0, 10.0]}
    )
    hypo = mined[0].hypothesis
    result = HypothesisEvaluator(alpha=0.05).evaluate(
        hypo,
        {"groups": [[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]]},
    )
    assert result.is_significant
    assert hypo.status == HypothesisStatus.VALIDATED
