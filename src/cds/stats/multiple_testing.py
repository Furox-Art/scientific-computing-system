"""Multiple-testing corrections for families of statistical hypotheses.

The helpers in this module operate on raw p-values and return adjusted p-values
in the caller's original order.  They intentionally separate multiplicity
control from effect-size filtering so a family is corrected over every test
that was actually considered, not only the discoveries that survived an
unrelated practical-significance threshold.
"""

from __future__ import annotations

import math
from typing import Literal

CorrectionMethod = Literal["none", "bonferroni", "holm", "fdr_bh"]


def _validated_p_values(p_values: list[float] | tuple[float, ...]) -> tuple[float, ...]:
    values = tuple(float(value) for value in p_values)
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
        raise ValueError("p-values must be finite and lie in [0, 1]")
    return values


def adjust_p_values(
    p_values: list[float] | tuple[float, ...],
    *,
    method: CorrectionMethod = "fdr_bh",
) -> tuple[float, ...]:
    """Return multiplicity-adjusted p-values in original input order.

    Supported methods are:

    ``"none"``
        No correction. Useful when a caller has already controlled the family.
    ``"bonferroni"``
        Family-wise error-rate control using ``min(1, m * p)``.
    ``"holm"``
        Holm's step-down family-wise error-rate procedure.
    ``"fdr_bh"``
        Benjamini-Hochberg false-discovery-rate control.

    Args:
        p_values: Raw p-values from one pre-declared family of tests.
        method: Correction procedure.

    Returns:
        Adjusted p-values aligned with ``p_values``.

    Raises:
        ValueError: If a p-value is non-finite/outside ``[0, 1]`` or the
            correction method is unsupported.
    """
    values = _validated_p_values(p_values)
    if method not in ("none", "bonferroni", "holm", "fdr_bh"):
        raise ValueError(f"unsupported multiple-testing correction: {method!r}")
    count = len(values)
    if count == 0 or method == "none":
        return values
    if method == "bonferroni":
        return tuple(min(1.0, count * value) for value in values)

    order = sorted(range(count), key=lambda index: (values[index], index))
    sorted_values = [values[index] for index in order]
    adjusted_sorted = [0.0] * count

    if method == "holm":
        running = 0.0
        for rank, value in enumerate(sorted_values):
            scaled = min(1.0, (count - rank) * value)
            running = max(running, scaled)
            adjusted_sorted[rank] = running
    else:
        running = 1.0
        for reverse_rank in range(count - 1, -1, -1):
            rank = reverse_rank + 1
            scaled = min(1.0, sorted_values[reverse_rank] * count / rank)
            running = min(running, scaled)
            adjusted_sorted[reverse_rank] = running

    adjusted = [0.0] * count
    for sorted_index, original_index in enumerate(order):
        adjusted[original_index] = adjusted_sorted[sorted_index]
    return tuple(adjusted)


def rejected(
    p_values: list[float] | tuple[float, ...],
    *,
    alpha: float = 0.05,
    method: CorrectionMethod = "fdr_bh",
) -> tuple[bool, ...]:
    """Return which hypotheses survive multiplicity correction at ``alpha``."""
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and strictly between 0 and 1")
    return tuple(value < alpha for value in adjust_p_values(p_values, method=method))
