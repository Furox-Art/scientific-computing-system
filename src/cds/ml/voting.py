"""Majority-voting ensemble classifier — pure Python, zero dependencies.

Combines heterogeneous :mod:`cds.ml` estimators under a single interface.
*Hard* voting takes a majority of the members' label predictions; *soft*
voting sums the members' ``predict_proba`` distributions over the union of
labels and returns the argmax.

Both modes are fully deterministic. Members are scanned in declaration order,
so ties resolve to the earliest-encountered label — which, whenever it is
among the leaders, is exactly the **first estimator's vote**.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from cds.ml.metrics import Label
from cds.ml.model_selection import SupervisedModel


@runtime_checkable
class SoftVotingModel(SupervisedModel, Protocol):
    """Structural interface for estimators usable in soft voting.

    Everything required by :class:`~cds.ml.model_selection.SupervisedModel`
    plus calibrated per-class probabilities.
    """

    def predict_proba(self, x: list[float]) -> dict[Label, float]:
        """Return class-membership probabilities for a single row."""
        ...


class VotingClassifier:
    """Majority-vote ensemble over cds.ml estimators.

    Every member is fitted on the same ``(X, y)`` pair; prediction combines
    their outputs:

    - ``voting="hard"`` — each member contributes one label vote and the most
      voted label wins. Ties are broken by the **first estimator's vote**:
      labels are tallied in declaration order with a strictly-greater rule,
      so among equally-voted leaders the label first encountered — i.e., the
      earliest member's choice — is returned deterministically.
    - ``voting="soft"`` — every member must also implement
      :meth:`SoftVotingModel.predict_proba`; the per-label probabilities are
      summed over the union of labels and the argmax wins, again resolving
      ties to the earliest-encountered label.

    Args:
        estimators: ``(name, model)`` pairs sharing the cds.ml estimator
            interface (``fit(X, y)`` / ``predict(x)``). Names must be unique
            and at least one pair is required.
        voting: ``"hard"`` (default) or ``"soft"``.

    Raises:
        ValueError: if ``voting`` is not ``"hard"``/``"soft"``, the estimator
            list is empty, or names are duplicated. In soft voting, members
            without ``predict_proba`` are rejected as well.
    """

    def __init__(
        self,
        estimators: list[tuple[str, SupervisedModel]],
        *,
        voting: str = "hard",
    ) -> None:
        """Validate configuration and store the still-unfitted members."""
        self.voting = voting
        self._estimators: list[tuple[str, SupervisedModel]] = list(estimators)
        self._prob_estimators: list[tuple[str, SoftVotingModel]] = []
        self._fitted = False
        self._validate()

    def fit(self, X: list[list[float]], y: Sequence[Label]) -> VotingClassifier:
        """Fit every member on the same ``(X, y)``.

        Args:
            X: Feature rows shared by all members.
            y: Class labels shared by all members.

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: if the configuration is invalid or a member rejects
                the training data (empty ``X``, length mismatch, ragged rows,
                feature-count problems).
        """
        self._validate()
        for _, model in self._estimators:
            model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, x: list[float]) -> Label:
        """Combined vote of all members for a single row.

        Args:
            x: One feature row matching the fitted feature count.

        Returns:
            The winning label under the configured voting mode.

        Raises:
            ValueError: if called before :meth:`fit`, or if a member surfaces
                a feature-count mismatch.
        """
        if not self._fitted:
            raise ValueError("model is not fitted")
        if self.voting == "hard":
            return self._majority_vote(x)
        return self._weighted_argmax(x)

    # ------------------------------------------------------------------ #
    # Validation helpers                                                  #
    # ------------------------------------------------------------------ #

    def _validate(self) -> None:
        """Re-check configuration and rebuild the soft-member view; idempotent.

        Raises:
            ValueError: on an unknown ``voting`` string, an empty estimator
                list, duplicate names, or (in soft voting) a member without
                ``predict_proba``.
        """
        if self.voting not in ("hard", "soft"):
            raise ValueError("voting must be 'hard' or 'soft'")
        if not self._estimators:
            raise ValueError("estimators must be non-empty")
        names = [name for name, _ in self._estimators]
        if len(set(names)) != len(names):
            raise ValueError("estimator names must be unique")
        members: list[tuple[str, SoftVotingModel]] = []
        if self.voting == "soft":
            for name, model in self._estimators:
                if not isinstance(model, SoftVotingModel):
                    raise ValueError(
                        f"estimator {name!r} must implement predict_proba() for soft voting"
                    )
                members.append((name, model))
        self._prob_estimators = members

    # ------------------------------------------------------------------ #
    # Vote combination                                                    #
    # ------------------------------------------------------------------ #

    def _majority_vote(self, x: list[float]) -> Label:
        """Most-voted label; ties keep the earliest-encountered leader."""
        counts: dict[Label, int] = {}
        best_label: Label = ""
        best_count = 0
        for _, model in self._estimators:
            label = model.predict(x)
            count = counts.get(label, 0) + 1
            counts[label] = count
            if count > best_count:
                best_count = count
                best_label = label
        return best_label

    def _weighted_argmax(self, x: list[float]) -> Label:
        """Argmax of summed member probabilities; ties → earliest label."""
        totals: dict[Label, float] = {}
        for _, model in self._prob_estimators:
            for label, prob in model.predict_proba(x).items():
                totals[label] = totals.get(label, 0.0) + prob
        best_label: Label = ""
        best_total = -1.0
        for label, total in totals.items():
            if total > best_total:
                best_total = total
                best_label = label
        return best_label
