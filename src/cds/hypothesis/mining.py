"""Correlation-driven hypothesis mining over tabular numeric data.

Every usable pairwise Pearson test belongs to one statistical family.  Raw
p-values are therefore computed for *all* usable pairs first, adjusted as one
family, and only then combined with the caller's practical effect-size cutoff.
This prevents effect-size pre-filtering from shrinking the multiplicity burden
and inflating the false-positive rate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from cds.core.models import Domain, Hypothesis, HypothesisStatus
from cds.stats.descriptive import correlation
from cds.stats.hypothesis_tests import t_sf
from cds.stats.multiple_testing import CorrectionMethod, adjust_p_values

__all__ = ["MinedHypothesis", "mine_correlations"]

WEAK_THRESHOLD = 0.2
STRONG_THRESHOLD = 0.5
_STRENGTH_CONFIDENCE: dict[str, float] = {"weak": 0.4, "moderate": 0.6, "strong": 0.85}


@dataclass
class MinedHypothesis:
    """A multiplicity-controlled pairwise association surfaced by mining."""

    feature_a: str
    feature_b: str
    correlation: float
    p_value: float
    adjusted_p_value: float
    correction: CorrectionMethod
    strength: str
    hypothesis: Hypothesis


@dataclass(frozen=True)
class _PairTest:
    feature_a: str
    feature_b: str
    correlation: float
    p_value: float


def _classify_strength(abs_r: float) -> str:
    if abs_r >= STRONG_THRESHOLD:
        return "strong"
    if abs_r >= WEAK_THRESHOLD:
        return "moderate"
    return "weak"


def _is_constant(values: list[float]) -> bool:
    return all(value == values[0] for value in values)


def _pearson_p_value(r: float, n: int) -> float:
    denominator = max(0.0, 1.0 - r * r)
    if denominator == 0.0:
        return 0.0
    t_stat = abs(r) * math.sqrt((n - 2) / denominator)
    return t_sf(t_stat, float(n - 2))


def _build_hypothesis(
    *,
    feature_a: str,
    feature_b: str,
    r: float,
    p_value: float,
    adjusted_p_value: float,
    correction: CorrectionMethod,
    family_size: int,
    n: int,
    strength: str,
    alpha: float,
    min_abs_r: float,
) -> Hypothesis:
    direction = "positive" if r > 0 else "negative"
    statement = (
        f"{feature_a} shows a {strength} {direction} linear association with {feature_b} "
        f"(Pearson r={r:.4f}, raw p={p_value:.3e}, adjusted p={adjusted_p_value:.3e}, n={n})."
    )
    multiplicity = (
        "without additional multiplicity adjustment"
        if correction == "none"
        else f"after {correction} correction across {family_size} pairwise tests"
    )
    return Hypothesis(
        id=f"CORR-{feature_a}-vs-{feature_b}",
        statement=statement,
        domain=Domain.GENERAL_SCIENCE,
        research_question=f"Are {feature_a} and {feature_b} linearly associated?",
        rationale=(
            f"Pearson r={r:.4f} over n={n} paired observations has raw p={p_value:.3e} "
            f"and adjusted p={adjusted_p_value:.3e} {multiplicity}; it clears the "
            f"pre-registered thresholds |r| >= {min_abs_r} and adjusted p < {alpha}."
        ),
        assumptions=[
            "Paired observations are independent across rows.",
            f"The relationship between {feature_a} and {feature_b} is approximately linear.",
            "Neither variable is truncated to a range that would distort Pearson's r.",
            f"The declared multiple-testing family contains {family_size} usable pairwise tests.",
        ],
        predictions=[
            f"A fresh comparable sample reproduces a {strength} {direction} correlation "
            f"with |r| >= {min_abs_r}.",
            f"Re-testing the declared family at alpha={alpha} using {correction} control again "
            f"supports the {feature_a}/{feature_b} association.",
        ],
        status=HypothesisStatus.TESTABLE,
        confidence=_STRENGTH_CONFIDENCE[strength],
        tags=["correlation", "mined", strength, f"multiplicity:{correction}"],
        sources=["cds.hypothesis.mining.mine_correlations"],
        metadata={
            "feature_a": feature_a,
            "feature_b": feature_b,
            "pearson_r": f"{r:.6f}",
            "p_value": f"{p_value:.6e}",
            "adjusted_p_value": f"{adjusted_p_value:.6e}",
            "multiple_testing_correction": correction,
            "family_size": str(family_size),
            "sign": direction,
            "n": str(n),
            "degrees_of_freedom": str(n - 2),
        },
    )


def mine_correlations(
    data: dict[str, list[float]],
    *,
    alpha: float = 0.05,
    min_abs_r: float = 0.3,
    max_features: int = 8,
    correction: CorrectionMethod = "fdr_bh",
) -> list[MinedHypothesis]:
    """Mine pairwise Pearson associations with explicit multiplicity control.

    All usable pairs among the first ``max_features`` columns are tested.  The
    resulting raw p-values are adjusted as one family using ``correction``.
    A pair is emitted only when both ``abs(r) >= min_abs_r`` and its *adjusted*
    p-value is below ``alpha``.
    """
    if not data:
        raise ValueError("correlation mining needs at least 2 numeric columns, got none")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be strictly between 0.0 and 1.0, got {alpha}")
    if not 0.0 <= min_abs_r < 1.0:
        raise ValueError(
            f"min_abs_r must lie between 0.0 inclusive and 1.0 exclusive, got {min_abs_r}"
        )
    if max_features < 2:
        raise ValueError("max_features must be at least 2")

    expected_length: int | None = None
    for name, values in data.items():
        if len(values) < 3:
            raise ValueError(f"column '{name}' needs at least 3 observations, got {len(values)}")
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError(f"column '{name}' must contain only finite numeric values")
        if expected_length is None:
            expected_length = len(values)
        elif len(values) != expected_length:
            raise ValueError(
                f"columns must share one length: '{name}' has {len(values)}, expected {expected_length}"
            )

    columns = list(data.items())[:max_features]
    usable = [(name, values) for name, values in columns if not _is_constant(values)]
    if len(usable) < 2:
        raise ValueError(
            f"correlation mining needs at least 2 non-constant columns, got {len(usable)}"
        )

    sample_size = len(usable[0][1])
    tests: list[_PairTest] = []
    for index_a in range(len(usable)):
        name_a, values_a = usable[index_a]
        for index_b in range(index_a + 1, len(usable)):
            name_b, values_b = usable[index_b]
            r = correlation(values_a, values_b)
            tests.append(_PairTest(name_a, name_b, r, _pearson_p_value(r, sample_size)))

    adjusted = adjust_p_values(tuple(test.p_value for test in tests), method=correction)
    family_size = len(tests)
    mined: list[MinedHypothesis] = []
    for test, adjusted_p in zip(tests, adjusted):
        if abs(test.correlation) < min_abs_r or adjusted_p >= alpha:
            continue
        strength = _classify_strength(abs(test.correlation))
        hypothesis = _build_hypothesis(
            feature_a=test.feature_a,
            feature_b=test.feature_b,
            r=test.correlation,
            p_value=test.p_value,
            adjusted_p_value=adjusted_p,
            correction=correction,
            family_size=family_size,
            n=sample_size,
            strength=strength,
            alpha=alpha,
            min_abs_r=min_abs_r,
        )
        mined.append(
            MinedHypothesis(
                feature_a=test.feature_a,
                feature_b=test.feature_b,
                correlation=test.correlation,
                p_value=test.p_value,
                adjusted_p_value=adjusted_p,
                correction=correction,
                strength=strength,
                hypothesis=hypothesis,
            )
        )

    mined.sort(key=lambda hit: (-abs(hit.correlation), hit.feature_a, hit.feature_b))
    return mined
