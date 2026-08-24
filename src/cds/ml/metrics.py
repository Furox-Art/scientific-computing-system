"""Classification and regression metrics — pure Python, zero dependencies.

Sklearn-flavored helpers operating on plain lists. Label-based metrics accept
strings or ints (matching :mod:`cds.ml` estimators) and use *first-appearance*
label ordering, which keeps results deterministic even for mixed label types.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

Label = str | int


def _check_pair(y_true: Sequence[object], y_pred: Sequence[object]) -> None:
    """Validate that ground truth and predictions align and are non-empty.

    Raises:
        ValueError: if either sequence is empty or their lengths differ.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("y_true and y_pred must be non-empty")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")


def accuracy(y_true: Sequence[Label], y_pred: Sequence[Label]) -> float:
    """Fraction of predictions equal to the ground truth.

    Args:
        y_true: ground-truth labels
        y_pred: predicted labels, aligned with ``y_true``

    Returns:
        ``correct / n`` in ``[0.0, 1.0]``.

    Raises:
        ValueError: if the sequences are empty or misaligned.
    """
    _check_pair(y_true, y_pred)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def confusion_matrix(y_true: Sequence[Label], y_pred: Sequence[Label]) -> ConfusionMatrixResult:
    """Confusion matrix over labels in first-appearance order.

    Rows index true labels, columns index predicted labels; entry ``(i, j)``
    counts samples whose truth is ``labels[i]`` and prediction ``labels[j]``.

    Raises:
        ValueError: if the sequences are empty or misaligned.
    """
    _check_pair(y_true, y_pred)
    labels: list[Label] = []
    for label in [*y_true, *y_pred]:
        if label not in labels:
            labels.append(label)
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred):
        matrix[index[t]][index[p]] += 1
    return ConfusionMatrixResult(labels=labels, matrix=matrix)


@dataclass
class ConfusionMatrixResult:
    """Labels and count grid produced by :func:`confusion_matrix`.

    Attributes:
        labels: distinct labels in first-appearance order.
        matrix: square count grid; rows are true labels, columns predictions.
    """

    labels: list[Label]
    matrix: list[list[int]]


@dataclass
class Prf:
    """Precision / recall / F1 triple for a single label.

    Attributes:
        precision: ``tp / (tp + fp)``, ``0.0`` when the denominator is zero.
        recall: ``tp / (tp + fn)``, ``0.0`` when the denominator is zero.
        f1: harmonic mean of precision and recall, ``0.0`` when either is zero.
    """

    precision: float
    recall: float
    f1: float


def _per_label_prf(
    y_true: Sequence[Label],
    y_pred: Sequence[Label],
    labels: Sequence[Label],
) -> dict[Label, Prf]:
    """Compute the P/R/F1 triple for each requested label."""
    result: dict[Label, Prf] = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        result[label] = Prf(precision=precision, recall=recall, f1=f1)
    return result


def precision_recall_f1(
    y_true: Sequence[Label],
    y_pred: Sequence[Label],
) -> dict[Label, Prf]:
    """Per-label precision, recall and F1.

    Labels appear as keys in first-appearance order. Zero-denominator cases
    (a label never predicted, never true, or never produced at all) score
    ``0.0`` rather than raising.

    Raises:
        ValueError: if the sequences are empty or misaligned.
    """
    _check_pair(y_true, y_pred)
    labels = list(confusion_matrix(y_true, y_pred).labels)
    return _per_label_prf(y_true, y_pred, labels)


def macro_prf(y_true: Sequence[Label], y_pred: Sequence[Label]) -> Prf:
    """Unweighted mean of the per-label precision / recall / F1 triples.

    Raises:
        ValueError: if the sequences are empty or misaligned.
    """
    _check_pair(y_true, y_pred)
    per_label = precision_recall_f1(y_true, y_pred)
    k = float(len(per_label))
    macro_p = sum(v.precision for v in per_label.values()) / k
    macro_r = sum(v.recall for v in per_label.values()) / k
    macro_f1 = sum(v.f1 for v in per_label.values()) / k
    return Prf(precision=macro_p, recall=macro_r, f1=macro_f1)


def mean_squared_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Mean of squared residuals.

    Raises:
        ValueError: if the sequences are empty or misaligned.
    """
    _check_pair(y_true, y_pred)
    return sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true)


def mean_absolute_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Mean of absolute residuals.

    Raises:
        ValueError: if the sequences are empty or misaligned.
    """
    _check_pair(y_true, y_pred)
    return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)


def r2_score(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Coefficient of determination ``1 - SS_res / SS_tot``.

    When ``SS_tot`` is zero (constant targets) the score is ``1.0`` for a
    perfect fit and ``0.0`` otherwise, mirroring the convention that no
    variance to explain yields no credit.

    Raises:
        ValueError: if the sequences are empty or misaligned.
    """
    _check_pair(y_true, y_pred)
    mean = sum(y_true) / len(y_true)
    ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
    ss_tot = sum((t - mean) ** 2 for t in y_true)
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def roc_auc(y_true: Sequence[Label], scores: Sequence[float]) -> float:
    """Binary ROC AUC via the rank statistic (tie-aware).

    Equivalent to the probability that a random positive sample outranks a
    random negative one, with ties counting half. The *second* distinct label
    encountered in ``y_true`` is treated as the positive class, so scores are
    read as "higher means more likely that later label". Computed with average
    ranks, running in ``O(n log n)`` without pairwise loops.

    Args:
        y_true: binary ground-truth labels (exactly two distinct values).
        scores: discriminant scores aligned with ``y_true``.

    Returns:
        AUC in ``[0.0, 1.0]``.

    Raises:
        ValueError: if inputs are empty, misaligned, or labels are not binary.
    """
    _check_pair(y_true, scores)
    first = y_true[0]
    if all(t == first for t in y_true):
        raise ValueError("y_true must contain exactly two distinct labels")
    positive = next(t for t in y_true if t != first)
    negatives = sum(1 for t in y_true if t == first)
    positives = len(y_true) - negatives

    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks: list[float] = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) - 1 and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        # average 1-based ranks for the tied block [i..j]
        tied_rank = (i + j + 2) / 2.0
        for pos in range(i, j + 1):
            ranks[order[pos]] = tied_rank
        i = j + 1

    rank_sum_pos = sum(ranks[k] for k in range(len(y_true)) if y_true[k] == positive)
    auc = (rank_sum_pos - positives * (positives + 1) / 2.0) / (positives * negatives)
    return auc
