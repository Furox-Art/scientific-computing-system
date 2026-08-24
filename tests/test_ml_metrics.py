"""Tests for cds.ml.metrics — classification and regression metrics."""

import pytest

from cds.ml.metrics import (
    accuracy,
    confusion_matrix,
    macro_prf,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_f1,
    r2_score,
    roc_auc,
)


class TestValidation:
    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            accuracy([], [])

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            precision_recall_f1(["a", "b"], ["a"])

    def test_regression_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            mean_squared_error([1.0], [1.0, 2.0])

    def test_r2_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            r2_score([1.0], [1.0, 2.0])

    def test_roc_auc_single_class_raises(self) -> None:
        with pytest.raises(ValueError, match="two distinct"):
            roc_auc(["a", "a", "a"], [0.9, 0.8, 0.7])


class TestAccuracy:
    def test_perfect_and_zero(self) -> None:
        y = ["a", "b", "a"]
        assert accuracy(y, y) == 1.0
        assert accuracy(y, ["b", "a", "b"]) == 0.0

    def test_partial(self) -> None:
        assert accuracy([1, 2, 3, 4], [1, 2, 0, 0]) == 0.5


class TestConfusionMatrix:
    def test_first_appearance_label_order(self) -> None:
        result = confusion_matrix(["b", "a", "b"], ["b", "a", "a"])
        assert result.labels == ["b", "a"]
        # rows = true (b, a), cols = predicted (b, a)
        assert result.matrix == [[1, 1], [0, 1]]

    def test_prediction_only_label_appended(self) -> None:
        result = confusion_matrix(["a"], ["z"])
        assert result.labels == ["a", "z"]
        assert result.matrix == [[0, 1], [0, 0]]


class TestPrf:
    def test_perfect_predictions(self) -> None:
        prf = precision_recall_f1(["x", "y"], ["x", "y"])
        assert prf["x"].precision == prf["x"].recall == prf["x"].f1 == 1.0

    def test_zero_denominators_score_zero(self) -> None:
        # label "never" is never predicted and never true
        prf = precision_recall_f1(["a", "b"], ["a", "b"])
        assert prf["a"] is not None
        # label never predicted: precision denominator zero
        mixed = precision_recall_f1(["a", "a", "b"], ["a", "a", "a"])
        assert mixed["b"].precision == 0.0
        assert mixed["b"].recall == 0.0
        assert mixed["b"].f1 == 0.0

    def test_partial_scores(self) -> None:
        prf = precision_recall_f1(["a", "a", "b", "b"], ["a", "b", "b", "b"])
        a = prf["a"]
        assert a.precision == pytest.approx(1.0)
        assert a.recall == pytest.approx(0.5)
        assert a.f1 == pytest.approx(2 * 1.0 * 0.5 / 1.5)

    def test_macro_averages(self) -> None:
        m = macro_prf(["a", "a", "b", "b"], ["a", "b", "b", "b"])
        assert m.precision == pytest.approx((1.0 + 2 / 3) / 2)
        assert m.recall == pytest.approx((0.5 + 1.0) / 2)


class TestRegressionMetrics:
    def test_mse_mae(self) -> None:
        assert mean_squared_error([0.0, 0.0], [1.0, -1.0]) == 1.0
        assert mean_absolute_error([0.0, 0.0], [1.0, -3.0]) == 2.0

    def test_r2_perfect_and_mean_predictor(self) -> None:
        assert r2_score([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
        mean_fit = [2.0, 2.0, 2.0]
        assert r2_score([1.0, 2.0, 3.0], mean_fit) == pytest.approx(0.0)

    def test_r2_constant_targets(self) -> None:
        assert r2_score([5.0, 5.0], [5.0, 5.0]) == 1.0
        assert r2_score([5.0, 5.0], [4.0, 4.0]) == 0.0


class TestRocAuc:
    """Semantics: the *second* distinct label seen is the positive class."""

    def test_perfect_separation(self) -> None:
        assert roc_auc(["neg", "neg", "pos", "pos"], [0.1, 0.2, 0.8, 0.9]) == 1.0

    def test_reversed_separation(self) -> None:
        # positive class ("neg", seen second) gets the LOW scores
        assert roc_auc(["pos", "pos", "neg", "neg"], [0.8, 0.9, 0.1, 0.2]) == 0.0

    def test_cross_class_tie_counts_half(self) -> None:
        # pos {0.5, 0.9} vs neg {0.1, 0.5}: one tie pair → AUC = 3.5/4
        auc = roc_auc(["neg", "pos", "neg", "pos"], [0.1, 0.5, 0.5, 0.9])
        assert auc == pytest.approx(0.875)

    def test_all_tied_scores(self) -> None:
        assert roc_auc(["pos", "neg", "pos", "neg"], [1.0, 1.0, 1.0, 1.0]) == 0.5

    def test_label_order_independence(self) -> None:
        # first-seen label being negative still works via complement logic
        assert roc_auc(["neg", "pos"], [0.3, 0.7]) == 1.0
