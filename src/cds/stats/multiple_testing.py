"""Dependency-free multiple-testing corrections for confirmatory analyses."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum


class MultipleTestingMethod(str, Enum):
    """Supported family-wise error and false-discovery-rate procedures."""

    BONFERRONI = "bonferroni"
    HOLM = "holm"
    BENJAMINI_HOCHBERG = "benjamini-hochberg"
    BENJAMINI_YEKUTIELI = "benjamini-yekutieli"


@dataclass(frozen=True)
class MultipleTestingResult:
    """Adjusted p-values and decisions in the caller's original order."""

    method: MultipleTestingMethod
    alpha: float
    p_values: tuple[float, ...]
    adjusted_p_values: tuple[float, ...]
    rejected: tuple[bool, ...]

    @property
    def rejection_count(self) -> int:
        """Number of hypotheses rejected after correction."""
        return sum(self.rejected)


def _validate(p_values: Sequence[float], alpha: float) -> tuple[float, ...]:
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and strictly between 0 and 1")
    values = tuple(float(value) for value in p_values)
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
        raise ValueError("p-values must be finite and between 0 and 1")
    return values


def _restore_order(sorted_values: Sequence[float], order: Sequence[int]) -> tuple[float, ...]:
    restored = [0.0] * len(sorted_values)
    for sorted_index, original_index in enumerate(order):
        restored[original_index] = min(1.0, max(0.0, float(sorted_values[sorted_index])))
    return tuple(restored)


def _ranked(p_values: Sequence[float]) -> tuple[list[int], list[float]]:
    order = sorted(range(len(p_values)), key=lambda index: (p_values[index], index))
    return order, [p_values[index] for index in order]


def bonferroni(
    p_values: Sequence[float],
    *,
    alpha: float = 0.05,
) -> MultipleTestingResult:
    """Control family-wise error by multiplying every p-value by test count."""
    values = _validate(p_values, alpha)
    count = len(values)
    adjusted = tuple(min(1.0, value * count) for value in values)
    return MultipleTestingResult(
        method=MultipleTestingMethod.BONFERRONI,
        alpha=alpha,
        p_values=values,
        adjusted_p_values=adjusted,
        rejected=tuple(value <= alpha for value in adjusted),
    )


def holm(
    p_values: Sequence[float],
    *,
    alpha: float = 0.05,
) -> MultipleTestingResult:
    """Holm step-down family-wise error correction."""
    values = _validate(p_values, alpha)
    count = len(values)
    if count == 0:
        return MultipleTestingResult(MultipleTestingMethod.HOLM, alpha, (), (), ())
    order, ranked = _ranked(values)
    adjusted_ranked: list[float] = []
    running = 0.0
    for rank, value in enumerate(ranked, start=1):
        candidate = (count - rank + 1) * value
        running = max(running, candidate)
        adjusted_ranked.append(min(1.0, running))
    adjusted = _restore_order(adjusted_ranked, order)
    return MultipleTestingResult(
        method=MultipleTestingMethod.HOLM,
        alpha=alpha,
        p_values=values,
        adjusted_p_values=adjusted,
        rejected=tuple(value <= alpha for value in adjusted),
    )


def _fdr_adjusted(
    values: tuple[float, ...],
    *,
    dependency_factor: float,
) -> tuple[float, ...]:
    count = len(values)
    if count == 0:
        return ()
    order, ranked = _ranked(values)
    adjusted_ranked = [1.0] * count
    running = 1.0
    for index in range(count - 1, -1, -1):
        rank = index + 1
        candidate = ranked[index] * count * dependency_factor / rank
        running = min(running, candidate)
        adjusted_ranked[index] = min(1.0, running)
    return _restore_order(adjusted_ranked, order)


def benjamini_hochberg(
    p_values: Sequence[float],
    *,
    alpha: float = 0.05,
) -> MultipleTestingResult:
    """Benjamini-Hochberg false-discovery-rate correction.

    This procedure assumes independent or positively dependent tests. Use
    :func:`benjamini_yekutieli` when arbitrary dependence must be tolerated.
    """
    values = _validate(p_values, alpha)
    adjusted = _fdr_adjusted(values, dependency_factor=1.0)
    return MultipleTestingResult(
        method=MultipleTestingMethod.BENJAMINI_HOCHBERG,
        alpha=alpha,
        p_values=values,
        adjusted_p_values=adjusted,
        rejected=tuple(value <= alpha for value in adjusted),
    )


def benjamini_yekutieli(
    p_values: Sequence[float],
    *,
    alpha: float = 0.05,
) -> MultipleTestingResult:
    """Benjamini-Yekutieli FDR correction for arbitrary test dependence."""
    values = _validate(p_values, alpha)
    count = len(values)
    factor = sum(1.0 / rank for rank in range(1, count + 1)) if count else 1.0
    adjusted = _fdr_adjusted(values, dependency_factor=factor)
    return MultipleTestingResult(
        method=MultipleTestingMethod.BENJAMINI_YEKUTIELI,
        alpha=alpha,
        p_values=values,
        adjusted_p_values=adjusted,
        rejected=tuple(value <= alpha for value in adjusted),
    )


def adjust_p_values(
    p_values: Sequence[float],
    *,
    method: MultipleTestingMethod | str = MultipleTestingMethod.BENJAMINI_HOCHBERG,
    alpha: float = 0.05,
) -> MultipleTestingResult:
    """Dispatch to a named multiple-testing correction."""
    try:
        selected = MultipleTestingMethod(method)
    except ValueError as exc:
        raise ValueError(f"unsupported multiple-testing method: {method!r}") from exc
    procedures = {
        MultipleTestingMethod.BONFERRONI: bonferroni,
        MultipleTestingMethod.HOLM: holm,
        MultipleTestingMethod.BENJAMINI_HOCHBERG: benjamini_hochberg,
        MultipleTestingMethod.BENJAMINI_YEKUTIELI: benjamini_yekutieli,
    }
    return procedures[selected](p_values, alpha=alpha)


__all__ = [
    "MultipleTestingMethod",
    "MultipleTestingResult",
    "adjust_p_values",
    "benjamini_hochberg",
    "benjamini_yekutieli",
    "bonferroni",
    "holm",
]
