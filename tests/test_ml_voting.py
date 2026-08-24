"""Tests for cds.ml.voting — majority-voting ensemble classifier."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from cds.ml.metrics import accuracy
from cds.ml.model_selection import SupervisedModel
from cds.ml.naive_bayes import GaussianNaiveBayes
from cds.ml.neighbors import KNeighborsClassifier
from cds.ml.tree import DecisionTreeClassifier
from cds.ml.voting import SoftVotingModel, VotingClassifier

XOR_X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
XOR_Y = ["a", "b", "b", "a"]


class ScriptedClassifier:
    """Hard-voting stub: always predicts one scripted label (no probas)."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.fitted_y: list[str | int] = []

    def fit(self, X: list[list[float]], y: Sequence[str | int]) -> ScriptedClassifier:
        self.fitted_y = list(y)
        return self

    def predict(self, x: list[float]) -> str | int:
        return self.label


class ScriptedProbaClassifier:
    """Soft-voting stub with a fixed, hand-set probability dictionary."""

    def __init__(self, probas: dict[str | int, float]) -> None:
        self.probas = probas
        self.fitted_y: list[str | int] = []

    def fit(self, X: list[list[float]], y: Sequence[str | int]) -> ScriptedProbaClassifier:
        self.fitted_y = list(y)
        return self

    def predict(self, x: list[float]) -> str | int:
        best_label: str | int = ""
        best_p = -1.0
        for label, p in self.predict_proba(x).items():
            if p > best_p:
                best_p = p
                best_label = label
        return best_label

    def predict_proba(self, x: list[float]) -> dict[str | int, float]:
        return dict(self.probas)


def hard_ensemble(*labels: str) -> VotingClassifier:
    return VotingClassifier(
        [(f"m{i}", ScriptedClassifier(label)) for i, label in enumerate(labels)]
    )


def soft_ensemble(*proba_dicts: dict[str | int, float]) -> VotingClassifier:
    return VotingClassifier(
        [(f"s{i}", ScriptedProbaClassifier(p)) for i, p in enumerate(proba_dicts)],
        voting="soft",
    )


class TestValidation:
    def test_empty_estimators_raise(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            VotingClassifier([])

    def test_duplicate_names_raise(self) -> None:
        stub = ScriptedClassifier("a")
        with pytest.raises(ValueError, match="unique"):
            VotingClassifier([("dup", stub), ("dup", stub)])

    def test_unknown_voting_raises(self) -> None:
        with pytest.raises(ValueError, match="'hard' or 'soft'"):
            VotingClassifier([("m", ScriptedClassifier("a"))], voting="rank")

    @pytest.mark.parametrize("voting", ["hard", "soft"])
    def test_predict_before_fit_raises(self, voting: str) -> None:
        models: list[tuple[str, SupervisedModel]]
        if voting == "hard":
            models = [("m", ScriptedClassifier("a"))]
        else:
            models = [("m", ScriptedProbaClassifier({"a": 1.0}))]
        ensemble = VotingClassifier(models, voting=voting)
        with pytest.raises(ValueError, match="not fitted"):
            ensemble.predict(XOR_X[0])

    def test_soft_rejects_member_without_predict_proba(self) -> None:
        with pytest.raises(ValueError, match="predict_proba"):
            VotingClassifier([("m", ScriptedClassifier("a"))], voting="soft")

    def test_feature_mismatch_propagates_from_member(self) -> None:
        ensemble = VotingClassifier([("tree", DecisionTreeClassifier(max_depth=2))])
        ensemble.fit(XOR_X * 4, XOR_Y * 4)
        with pytest.raises(ValueError, match="features"):
            ensemble.predict([1.0])


class TestFitContract:
    def test_fit_returns_self(self) -> None:
        ensemble = hard_ensemble("a")
        assert ensemble.fit(XOR_X, XOR_Y) is ensemble

    def test_members_fitted_on_shared_data(self) -> None:
        first = ScriptedClassifier("a")
        second = ScriptedClassifier("b")
        VotingClassifier([("m0", first), ("m1", second)]).fit(XOR_X, XOR_Y)
        assert first.fitted_y == XOR_Y
        assert second.fitted_y == XOR_Y


class TestHardVoting:
    def test_clear_majority_wins(self) -> None:
        ensemble = hard_ensemble("a", "a", "b").fit(XOR_X, XOR_Y)
        assert ensemble.predict([0.0]) == "a"
        assert hard_ensemble("b", "b", "a").fit(XOR_X, XOR_Y).predict([0.0]) == "b"

    def test_three_way_tie_resolves_to_first_estimator_vote(self) -> None:
        ensemble = hard_ensemble("a", "b", "c").fit(XOR_X, XOR_Y)
        assert ensemble.predict([0.0]) == "a"

    def test_three_way_tie_reordered_resolves_to_new_first_vote(self) -> None:
        ensemble = hard_ensemble("b", "c", "a").fit(XOR_X, XOR_Y)
        assert ensemble.predict([0.0]) == "b"

    def test_even_split_tie_beats_alphabetical_order(self) -> None:
        ensemble = hard_ensemble("z", "z", "a", "a").fit(XOR_X, XOR_Y)
        assert ensemble.predict([0.0]) == "z"

    def test_even_split_tie_reversed(self) -> None:
        ensemble = hard_ensemble("a", "a", "z", "z").fit(XOR_X, XOR_Y)
        assert ensemble.predict([0.0]) == "a"

    def test_hard_mode_needs_no_predict_proba(self) -> None:
        ensemble = VotingClassifier(
            [("m0", ScriptedClassifier("a")), ("m1", ScriptedClassifier("a"))]
        )
        assert isinstance(ensemble._estimators[0][1], ScriptedClassifier)

    def test_hard_predictions_deterministic_across_calls(self) -> None:
        ensemble = hard_ensemble("a", "a", "b").fit(XOR_X * 2, XOR_Y * 2)
        runs = [[ensemble.predict(row) for row in XOR_X] for _ in range(3)]
        assert runs[0] == runs[1] == runs[2]


class TestSoftVoting:
    def test_hand_computed_summed_probabilities(self) -> None:
        # m1: a=0.6, b=0.4 ; m2: a=0.1, b=0.9 → totals a=0.7, b=1.3.
        ensemble = soft_ensemble({"a": 0.6, "b": 0.4}, {"a": 0.1, "b": 0.9})
        ensemble.fit(XOR_X, XOR_Y)
        assert ensemble.predict([0.0]) == "b"

    def test_union_of_labels_across_members(self) -> None:
        # m1 only knows "a"; union gives a=0.8+0.3=1.1 vs b=0.5.
        ensemble = soft_ensemble({"a": 0.8}, {"a": 0.3, "b": 0.5})
        ensemble.fit(XOR_X, XOR_Y)
        assert ensemble.predict([0.0]) == "a"

    def test_exact_tie_prefers_earliest_encountered_label(self) -> None:
        ensemble = soft_ensemble({"a": 0.5}, {"b": 0.5}).fit(XOR_X, XOR_Y)
        assert ensemble.predict([0.0]) == "a"

    def test_exact_tie_reordered_prefers_new_first_label(self) -> None:
        ensemble = soft_ensemble({"b": 0.5}, {"a": 0.5}).fit(XOR_X, XOR_Y)
        assert ensemble.predict([0.0]) == "b"

    def test_soft_predictions_deterministic_across_calls(self) -> None:
        ensemble = soft_ensemble({"a": 0.6, "b": 0.4}, {"a": 0.1, "b": 0.9})
        ensemble.fit(XOR_X, XOR_Y)
        runs = [[ensemble.predict(row) for row in XOR_X] for _ in range(3)]
        assert runs[0] == runs[1] == runs[2]


BLOB_X = [
    [0.0, 0.0],
    [0.5, 0.0],
    [0.0, 0.5],
    [0.5, 0.5],
    [1.0, 0.0],
    [4.0, 4.0],
    [4.5, 4.0],
    [4.0, 4.5],
    [4.5, 4.5],
    [5.0, 4.0],
]
BLOB_Y = ["lo", "lo", "lo", "lo", "lo", "hi", "hi", "hi", "hi", "hi"]


def real_trio() -> list[tuple[str, SupervisedModel]]:
    return [
        ("tree", DecisionTreeClassifier(max_depth=3, seed=0)),
        ("knn", KNeighborsClassifier(k=1)),
        ("gnb", GaussianNaiveBayes()),
    ]


class TestRealModelIntegration:
    def test_hard_voting_perfect_accuracy_on_separable_blobs(self) -> None:
        ensemble = VotingClassifier(real_trio()).fit(BLOB_X, BLOB_Y)
        preds = [ensemble.predict(row) for row in BLOB_X]
        assert accuracy(BLOB_Y, preds) == 1.0

    def test_soft_voting_perfect_accuracy_on_separable_blobs(self) -> None:
        ensemble = VotingClassifier(real_trio(), voting="soft").fit(BLOB_X, BLOB_Y)
        preds = [ensemble.predict(row) for row in BLOB_X]
        assert accuracy(BLOB_Y, preds) == 1.0

    def test_soft_voting_model_protocol_accepts_real_members(self) -> None:
        for _, model in real_trio():
            assert isinstance(model, SoftVotingModel)
