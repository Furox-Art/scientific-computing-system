"""Empirical evaluation of structured scientific hypotheses.

Dispatch is explicit and fail-closed. Paired observations use a true paired
analysis rather than an independent two-sample test. ``EvaluationResult``
reports statistical evidence without mutating the hypothesis lifecycle: one
p-value is evidence about a tested prediction, not proof or refutation of the
full scientific hypothesis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypedDict

from cds.core.models import Hypothesis
from cds.stats.descriptive import mean as _mean
from cds.stats.descriptive import stdev as _stdev
from cds.stats.hypothesis_tests import (
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


@dataclass
class EvaluationResult:
    """Detailed outcome of one statistical evaluation."""

    hypothesis_id: str
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    conclusion: str
    effect_size: float | None = None
    effect_size_label: str | None = None
    evidence_interpretation: str = "inconclusive"
    method_name: str | None = None


class ChiSquareGofPayload(TypedDict, total=False):
    observed: list[float]
    expected: list[float]


class EvaluationData(TypedDict, total=False):
    groups: list[list[float]]
    labels: list[str]
    one_sample: list[float]
    popmean: float
    chi_square_gof: ChiSquareGofPayload
    chi_square_independence: list[list[float]]
    paired: tuple[list[float], list[float]]


_DISPATCH_KEYS = (
    "groups",
    "one_sample",
    "chi_square_gof",
    "chi_square_independence",
    "paired",
)


class HypothesisEvaluator:
    """Match structured evaluation data to deterministic statistical tests."""

    def __init__(self, alpha: float = 0.05):
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise ValueError("alpha must be a finite number in the open interval (0, 1)")
        normalized = float(alpha)
        if not math.isfinite(normalized) or not 0.0 < normalized < 1.0:
            raise ValueError("alpha must be a finite number in the open interval (0, 1)")
        self.alpha = normalized

    def _build_result(
        self,
        hypothesis: Hypothesis,
        test_name: str,
        statistic: float,
        p_value: float,
        effect_size: float | None = None,
        effect_size_label: str | None = None,
        alpha: float | None = None,
        *,
        method_name: str | None = None,
    ) -> EvaluationResult:
        effective_alpha = self.alpha if alpha is None else alpha
        if not math.isfinite(effective_alpha) or not 0.0 < effective_alpha < 1.0:
            raise ValueError("effective alpha must be finite and in the open interval (0, 1)")
        if math.isnan(statistic):
            raise ValueError("test statistic must not be NaN")
        if not math.isfinite(p_value) or not 0.0 <= p_value <= 1.0:
            raise ValueError("p-value must be finite and in the closed interval [0, 1]")
        if effect_size is not None and math.isnan(effect_size):
            raise ValueError("effect size must not be NaN")

        is_significant = p_value < effective_alpha
        effect_clause = ""
        if effect_size is not None and effect_size_label is not None:
            effect_clause = f" Effect size: {effect_size_label} = {effect_size:.3f}."

        if is_significant:
            interpretation = "supported"
            conclusion = (
                f"Statistical evidence supports the tested prediction at alpha={effective_alpha} "
                f"({method_name or test_name}). A single test does not validate the full scientific "
                f"hypothesis.{effect_clause}"
            )
        else:
            interpretation = "inconclusive"
            conclusion = (
                f"The data did not provide sufficient statistical evidence at alpha={effective_alpha} "
                f"({method_name or test_name}). Non-significance does not demonstrate that the "
                f"scientific hypothesis is false.{effect_clause}"
            )

        return EvaluationResult(
            hypothesis_id=hypothesis.id,
            test_name=test_name,
            statistic=statistic,
            p_value=p_value,
            is_significant=is_significant,
            conclusion=conclusion,
            effect_size=effect_size,
            effect_size_label=effect_size_label,
            evidence_interpretation=interpretation,
            method_name=method_name or test_name,
        )

    def compare_groups(
        self,
        hypothesis: Hypothesis,
        groups: list[list[float]],
        labels: list[str] | None = None,
    ) -> EvaluationResult:
        if len(groups) < 2:
            raise ValueError("Evaluation requires at least 2 groups of data.")
        if labels is not None and len(labels) != len(groups):
            raise ValueError("labels must have the same length as groups")

        if len(groups) == 2:
            result = two_sample_ttest(groups[0], groups[1])
            effect = cohens_d(groups[0], groups[1])
            return self._build_result(
                hypothesis,
                "Two-sample t-test",
                result.statistic,
                result.p_value,
                effect,
                "Cohen's d",
            )

        result = one_way_anova(*groups)
        df1 = len(groups) - 1
        total_count = sum(len(group) for group in groups)
        df2 = total_count - len(groups)
        effect = eta_squared_from_f(result.statistic, df1, df2)
        return self._build_result(
            hypothesis,
            "One-way ANOVA",
            result.statistic,
            result.p_value,
            effect,
            "eta-squared",
        )

    def compare_to_reference(
        self,
        hypothesis: Hypothesis,
        sample: list[float],
        popmean: float,
    ) -> EvaluationResult:
        if len(sample) < 2:
            raise ValueError("One-sample evaluation requires at least 2 observations.")
        result = one_sample_ttest(sample, popmean)
        sample_sd = _stdev(sample, ddof=1)
        effect = (_mean(sample) - popmean) / sample_sd if sample_sd > 0 else None
        return self._build_result(
            hypothesis,
            "One-sample t-test",
            result.statistic,
            result.p_value,
            effect,
            "Cohen's d" if effect is not None else None,
        )

    def compare_paired(
        self,
        hypothesis: Hypothesis,
        first: list[float],
        second: list[float],
    ) -> EvaluationResult:
        result = paired_ttest(first, second)
        effect = paired_cohens_d(first, second)
        return self._build_result(
            hypothesis,
            "Paired t-test",
            result.statistic,
            result.p_value,
            effect,
            "Cohen's dz",
        )

    def goodness_of_fit(
        self,
        hypothesis: Hypothesis,
        observed: list[float],
        expected: list[float] | None = None,
    ) -> EvaluationResult:
        if len(observed) < 2:
            raise ValueError("Goodness-of-fit requires at least 2 categories.")
        if expected is None:
            total = sum(observed)
            expected = [total / len(observed)] * len(observed)
        result = chi_square_gof(observed, expected)
        return self._build_result(
            hypothesis,
            "Chi-square goodness-of-fit",
            result.statistic,
            result.p_value,
        )

    def test_independence(
        self,
        hypothesis: Hypothesis,
        table: list[list[float]],
    ) -> EvaluationResult:
        if len(table) < 2 or any(len(row) < 2 for row in table):
            raise ValueError("Independence test requires a 2x2 or larger contingency table.")
        result = chi_square_independence(table)
        effect = cramers_v(table)
        return self._build_result(
            hypothesis,
            "Chi-square independence",
            result.statistic,
            result.p_value,
            effect,
            "Cramer's V",
        )

    def evaluate(self, hypothesis: Hypothesis, data: EvaluationData) -> EvaluationResult:
        present = [key for key in _DISPATCH_KEYS if key in data]
        if not present:
            raise ValueError(
                "Unsupported data format for evaluation. Provide one of: 'groups', 'one_sample' "
                "(with 'popmean'), 'chi_square_gof', 'chi_square_independence', or 'paired'."
            )
        if len(present) != 1:
            raise ValueError("evaluation data must contain exactly one statistical dispatch key")

        dispatch = present[0]
        if dispatch == "groups":
            return self.compare_groups(hypothesis, data["groups"], data.get("labels"))
        if dispatch == "one_sample":
            if "popmean" not in data:
                raise ValueError("one_sample evaluation requires popmean")
            return self.compare_to_reference(hypothesis, data["one_sample"], data["popmean"])
        if dispatch == "chi_square_gof":
            payload = data["chi_square_gof"]
            if "observed" not in payload:
                raise ValueError("chi_square_gof evaluation requires observed counts")
            return self.goodness_of_fit(
                hypothesis,
                payload["observed"],
                payload.get("expected"),
            )
        if dispatch == "chi_square_independence":
            return self.test_independence(hypothesis, data["chi_square_independence"])

        first, second = data["paired"]
        return self.compare_paired(hypothesis, list(first), list(second))

    def evaluate_batch(
        self,
        hypotheses: list[Hypothesis],
        data_items: list[EvaluationData],
    ) -> list[EvaluationResult]:
        if len(hypotheses) != len(data_items):
            raise ValueError("hypotheses and data_items must have the same length")
        if not hypotheses:
            raise ValueError("evaluate_batch requires at least one hypothesis")

        raw_results = [
            self.evaluate(hypothesis, data) for hypothesis, data in zip(hypotheses, data_items)
        ]
        corrected_alpha = bonferroni_corrected_alpha(self.alpha, len(hypotheses))
        corrected: list[EvaluationResult] = []
        for hypothesis, raw in zip(hypotheses, raw_results):
            corrected.append(
                self._build_result(
                    hypothesis,
                    raw.test_name,
                    raw.statistic,
                    raw.p_value,
                    raw.effect_size,
                    raw.effect_size_label,
                    alpha=corrected_alpha,
                    method_name=raw.method_name,
                )
            )
        return corrected
