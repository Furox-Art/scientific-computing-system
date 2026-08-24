"""Correlation-driven hypothesis mining over tabular numeric data.

This module is the hypothesis-mining engine of CDS: given a plain
``dict[str, list[float]]`` dataset it scans every pairwise combination of
numeric columns, computes the Pearson correlation coefficient, judges its
statistical significance, and emits :class:`MinedHypothesis` records whose
embedded :class:`~cds.core.models.Hypothesis` objects plug directly into the
existing evaluator/statistics pipeline
(:class:`~cds.hypothesis.evaluator.HypothesisEvaluator`).

Strength tiers (Cohen-style conventions applied to ``|r|``):

- ``"weak"``     -- ``|r| < 0.2``
- ``"moderate"`` -- ``0.2 <= |r| < 0.5``
- ``"strong"``   -- ``|r| >= 0.5``

Boundary values belong to the higher tier: ``|r| = 0.2`` classifies as
``"moderate"`` and ``|r| = 0.5`` classifies as ``"strong"``.

Significance follows the standard Student-t transformation of Pearson's
coefficient, ``t = r * sqrt((n - 2) / (1 - r^2))`` with ``n - 2`` degrees
of freedom and a two-sided p-value obtained from the existing survival
function :func:`cds.stats.hypothesis_tests.t_sf`. A perfect sample
correlation (``|r| = 1``) short-circuits to ``p = 0.0`` because the t
statistic diverges.

Constant columns (zero variance, detected as every value identical to the
first) are skipped silently: they cannot enter a meaningful Pearson
correlation, so they are excluded from consideration before pairing and do
not raise or otherwise surface to the caller.

References:
    - Fisher, R. A. (1921). "On the 'probable error' of a coefficient of
      correlation deduced from a small sample." Metron, 1(4), 3-32.
      (t-transformation of Pearson's r)
    - Student [W. S. Gosset] (1908). Biometrika, 6(1), 1-25.
      (t distribution / two-tailed tail probability)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from cds.core.models import Domain, Hypothesis, HypothesisStatus
from cds.stats.descriptive import correlation
from cds.stats.hypothesis_tests import t_sf

__all__ = ["MinedHypothesis", "mine_correlations"]

WEAK_THRESHOLD = 0.2
"""Lower bound of the "moderate" tier; below it a correlation is "weak"."""

STRONG_THRESHOLD = 0.5
"""Lower bound of the "strong" tier (inclusive)."""

_STRENGTH_CONFIDENCE: dict[str, float] = {"weak": 0.4, "moderate": 0.6, "strong": 0.85}


@dataclass
class MinedHypothesis:
    """A statistically significant pairwise association surfaced by mining.

    Attributes:
        feature_a: Name of the first numeric column.
        feature_b: Name of the second numeric column.
        correlation: Sample Pearson correlation coefficient in ``[-1.0, 1.0]``.
        p_value: Two-sided p-value testing H0: population correlation is zero.
        strength: Qualitative effect-size tier — one of ``"weak"``,
            ``"moderate"``, ``"strong"`` (thresholds 0.2 / 0.5 on ``|r|``,
            boundaries included in the higher tier).
        hypothesis: Fully populated :class:`~cds.core.models.Hypothesis`
            carrying statement, rationale, assumptions and predictions derived
            from the observed numbers, ready for the evaluator pipeline.
    """

    feature_a: str
    feature_b: str
    correlation: float
    p_value: float
    strength: str
    hypothesis: Hypothesis


def _classify_strength(abs_r: float) -> str:
    """Map an absolute correlation onto its documented qualitative tier.

    Args:
        abs_r: Absolute value of a Pearson correlation coefficient.

    Returns:
        ``"strong"`` when ``abs_r >= 0.5``, ``"moderate"`` when
        ``abs_r >= 0.2``, and ``"weak"`` otherwise. Boundary values join the
        higher tier.
    """
    if abs_r >= STRONG_THRESHOLD:
        return "strong"
    if abs_r >= WEAK_THRESHOLD:
        return "moderate"
    return "weak"


def _is_constant(values: list[float]) -> bool:
    """Return whether every entry equals the first (a zero-variance column).

    Args:
        values: Column values (guaranteed non-empty by upstream validation).

    Returns:
        True if the column is constant and therefore unusable for Pearson
        correlation.
    """
    return all(value == values[0] for value in values)


def _pearson_p_value(r: float, n: int) -> float:
    """Two-sided p-value for H0: the population Pearson correlation is zero.

    Uses the classical transformation ``t = |r| * sqrt((n - 2) / (1 - r^2))``
    on ``n - 2`` degrees of freedom evaluated with the existing
    :func:`cds.stats.hypothesis_tests.t_sf` survival function.

    Args:
        r: Sample Pearson correlation coefficient.
        n: Number of paired observations (must be at least 3).

    Returns:
        Two-sided p-value in ``[0.0, 1.0]``; exactly ``0.0`` when the sample
        correlation is perfect (``|r| = 1``), where the t statistic diverges.
    """
    denominator = max(0.0, 1.0 - r * r)
    if denominator == 0.0:
        return 0.0
    df = float(n - 2)
    t_stat = abs(r) * math.sqrt((n - 2) / denominator)
    return t_sf(t_stat, df)


def _build_hypothesis(
    *,
    feature_a: str,
    feature_b: str,
    r: float,
    p_value: float,
    n: int,
    strength: str,
    alpha: float,
    min_abs_r: float,
) -> Hypothesis:
    """Assemble a fully populated :class:`~cds.core.models.Hypothesis`.

    Every text field is derived from the observed statistics so the emitted
    object is self-describing and falsifiable: the statement quotes ``r``,
    ``p`` and ``n``; assumptions state the preconditions of Pearson's
    coefficient; predictions commit to sign, tier and threshold so future
    data can refute them. Confidence follows a fixed per-tier mapping
    (weak 0.4 / moderate 0.6 / strong 0.85).

    Args:
        feature_a: First column name.
        feature_b: Second column name.
        r: Observed Pearson correlation coefficient.
        p_value: Two-sided p-value for the association.
        n: Number of paired observations.
        strength: Classified effect-size tier.
        alpha: Significance level used during mining.
        min_abs_r: Minimum absolute-correlation threshold used during mining.

    Returns:
        A :class:`~cds.core.models.Hypothesis` in ``TESTABLE`` status with
        deterministic id, tags, sources and string-valued metadata.
    """
    direction = "positive" if r > 0 else "negative"
    statement = (
        f"{feature_a} shows a {strength} {direction} linear association with "
        f"{feature_b} (Pearson r={r:.4f}, p={p_value:.3e}, n={n})."
    )
    return Hypothesis(
        id=f"CORR-{feature_a}-vs-{feature_b}",
        statement=statement,
        domain=Domain.GENERAL_SCIENCE,
        research_question=f"Are {feature_a} and {feature_b} linearly associated?",
        rationale=(
            f"Pearson r={r:.4f} over n={n} paired observations "
            f"(two-sided p={p_value:.3e}) clears the pre-registered thresholds "
            f"|r| >= {min_abs_r} and p < {alpha}."
        ),
        assumptions=[
            "Paired observations are independent across rows.",
            f"The relationship between {feature_a} and {feature_b} is approximately linear.",
            "Neither variable is truncated to a range that would distort Pearson's r.",
        ],
        predictions=[
            f"A fresh comparable sample reproduces a {strength} {direction} "
            f"correlation with |r| >= {min_abs_r}.",
            f"Re-testing at alpha={alpha} again rejects independence between "
            f"{feature_a} and {feature_b}.",
        ],
        status=HypothesisStatus.TESTABLE,
        confidence=_STRENGTH_CONFIDENCE[strength],
        tags=["correlation", "mined", strength],
        sources=["cds.hypothesis.mining.mine_correlations"],
        metadata={
            "feature_a": feature_a,
            "feature_b": feature_b,
            "pearson_r": f"{r:.6f}",
            "p_value": f"{p_value:.6e}",
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
) -> list[MinedHypothesis]:
    """Mine pairwise Pearson-correlation hypotheses from numeric columns.

    Only the first ``max_features`` columns (in insertion order) are
    considered. Constant columns among them are skipped silently. Each
    surviving pair yields a candidate only if ``abs(r) >= min_abs_r`` and
    the two-sided significance test gives ``p < alpha``; results are sorted
    by ``|r|`` descending (ties keep deterministic feature-name order).

    Strength tiers follow the documented 0.2 / 0.5 thresholds on ``|r|``
    with boundaries assigned to the higher tier (see module docstring).

    Args:
        data: Mapping of column names to equal-length numeric columns.
        alpha: Significance level in the open interval ``(0.0, 1.0)``.
        min_abs_r: Inclusive lower bound on ``abs(r)``, within ``[0.0, 1.0)``.
        max_features: Maximum number of leading columns to consider.

    Returns:
        One :class:`MinedHypothesis` per qualifying pair, strongest first.

    Raises:
        ValueError: If fewer than two columns are provided after filtering
            out constant columns, any column holds fewer than 3 observations,
            column lengths are mismatched, ``alpha`` falls outside
            ``(0.0, 1.0)``, or ``min_abs_r`` falls outside ``[0.0, 1.0)``.
    """
    if not data:
        raise ValueError("correlation mining needs at least 2 numeric columns, got none")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be strictly between 0.0 and 1.0, got {alpha}")
    if not 0.0 <= min_abs_r < 1.0:
        raise ValueError(
            f"min_abs_r must lie between 0.0 inclusive and 1.0 exclusive, got {min_abs_r}"
        )

    expected_length: int | None = None
    for name, values in data.items():
        if len(values) < 3:
            message = f"column '{name}' needs at least 3 observations, got {len(values)}"
            raise ValueError(message)
        if expected_length is None:
            expected_length = len(values)
        elif len(values) != expected_length:
            raise ValueError(
                f"columns must share one length: '{name}' has "
                f"{len(values)}, expected {expected_length}"
            )

    columns = list(data.items())[:max_features]
    usable: list[tuple[str, list[float]]] = []
    for name, values in columns:
        if not _is_constant(values):
            usable.append((name, values))
    if len(usable) < 2:
        raise ValueError(
            f"correlation mining needs at least 2 non-constant columns, got {len(usable)}"
        )

    sample_size = len(next(iter(data.values())))
    mined: list[MinedHypothesis] = []
    for i in range(len(usable)):
        name_a, values_a = usable[i]
        for j in range(i + 1, len(usable)):
            name_b, values_b = usable[j]
            r = correlation(values_a, values_b)
            if abs(r) < min_abs_r:
                continue
            p_value = _pearson_p_value(r, sample_size)
            if p_value >= alpha:
                continue
            strength = _classify_strength(abs(r))
            mined.append(
                MinedHypothesis(
                    feature_a=name_a,
                    feature_b=name_b,
                    correlation=r,
                    p_value=p_value,
                    strength=strength,
                    hypothesis=_build_hypothesis(
                        feature_a=name_a,
                        feature_b=name_b,
                        r=r,
                        p_value=p_value,
                        n=sample_size,
                        strength=strength,
                        alpha=alpha,
                        min_abs_r=min_abs_r,
                    ),
                )
            )

    mined.sort(key=lambda hit: (-abs(hit.correlation), hit.feature_a, hit.feature_b))
    return mined
