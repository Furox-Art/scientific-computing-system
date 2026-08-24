"""Tests for cds.ml.boosting — gradient-boosted trees."""

import pytest

from cds.ml.boosting import GradientBoostingClassifier

SEPARABLE_X = [[0.0], [1.0], [2.0], [3.0]]
SEPARABLE_Y = ["lo", "lo", "hi", "hi"]

XOR_X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
XOR_Y = ["a", "b", "b", "a"]


class TestHyperparameterValidation:
    def test_n_estimators_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="n_estimators"):
            GradientBoostingClassifier(n_estimators=0)

    def test_learning_rate_not_positive_raises(self) -> None:
        with pytest.raises(ValueError, match="learning_rate must be > 0"):
            GradientBoostingClassifier(learning_rate=0.0)

    def test_negative_max_depth_raises(self) -> None:
        with pytest.raises(ValueError, match="max_depth must be >= 0"):
            GradientBoostingClassifier(max_depth=-1)

    def test_min_samples_leaf_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="min_samples_leaf must be >= 1"):
            GradientBoostingClassifier(min_samples_leaf=0)


class TestFitValidation:
    def test_empty_x_raises(self) -> None:
        empty: list[list[float]] = []
        with pytest.raises(ValueError, match="non-empty"):
            GradientBoostingClassifier(n_estimators=1).fit(empty, [])

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            GradientBoostingClassifier(n_estimators=1).fit(SEPARABLE_X, SEPARABLE_Y[:-1])

    def test_ragged_rows_raise(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            GradientBoostingClassifier(n_estimators=1).fit([[1.0], [1.0, 2.0]], ["a", "b"])

    def test_single_label_raises(self) -> None:
        with pytest.raises(ValueError, match="labels must be binary"):
            GradientBoostingClassifier(n_estimators=1).fit(SEPARABLE_X, ["lo"] * 4)

    def test_three_labels_raise(self) -> None:
        with pytest.raises(ValueError, match="labels must be binary"):
            GradientBoostingClassifier(n_estimators=1).fit(SEPARABLE_X, ["a", "b", "c", "a"])


class TestFitPredict:
    def test_fit_returns_self(self) -> None:
        booster = GradientBoostingClassifier(n_estimators=5)
        assert booster.fit(SEPARABLE_X, SEPARABLE_Y) is booster

    def test_linearly_separable_data_predicted_perfectly(self) -> None:
        booster = GradientBoostingClassifier(n_estimators=30, learning_rate=0.5, max_depth=1).fit(
            SEPARABLE_X, SEPARABLE_Y
        )
        assert [booster.predict(row) for row in SEPARABLE_X] == SEPARABLE_Y

    def test_confident_probabilities_on_separable_data(self) -> None:
        booster = GradientBoostingClassifier(n_estimators=30, learning_rate=0.5, max_depth=1).fit(
            SEPARABLE_X, SEPARABLE_Y
        )
        assert booster.predict_proba([3.0])["hi"] > 0.99
        assert booster.predict_proba([0.0])["hi"] < 0.01

    def test_second_seen_label_is_positive_class(self) -> None:
        booster = GradientBoostingClassifier(n_estimators=25, learning_rate=0.5).fit(
            [[0.0], [0.5], [1.0], [1.5]], ["no", "no", "yes", "yes"]
        )
        assert booster.predict_proba([1.4])["yes"] > 0.5
        assert booster.predict([0.1]) == "no"

    def test_xor_learned_with_enough_stages(self) -> None:
        booster = GradientBoostingClassifier(n_estimators=60, learning_rate=0.4, max_depth=2).fit(
            XOR_X * 8, XOR_Y * 8
        )
        assert [booster.predict(row) for row in XOR_X] == XOR_Y

    def test_predict_proba_sums_to_one_across_both_labels(self) -> None:
        booster = GradientBoostingClassifier(n_estimators=10, learning_rate=0.3).fit(
            XOR_X * 2, XOR_Y * 2
        )
        probs = booster.predict_proba([0.9, 0.9])
        assert set(probs) == {"a", "b"}
        assert sum(probs.values()) == pytest.approx(1.0)

    def test_deterministic_repeated_fit_equality(self) -> None:
        a = GradientBoostingClassifier(n_estimators=15, learning_rate=0.3).fit(XOR_X * 2, XOR_Y * 2)
        b = GradientBoostingClassifier(n_estimators=15, learning_rate=0.3).fit(XOR_X * 2, XOR_Y * 2)
        for row in XOR_X:
            assert a.predict_proba(row) == b.predict_proba(row)
            assert a.predict(row) == b.predict(row)

    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(ValueError, match="not fitted"):
            GradientBoostingClassifier().predict(XOR_X[0])

    def test_feature_count_mismatch_raises(self) -> None:
        booster = GradientBoostingClassifier(n_estimators=5).fit(XOR_X, XOR_Y)
        with pytest.raises(ValueError, match="features"):
            booster.predict([1.0])

    def test_tied_probability_prefers_first_seen_label(self) -> None:
        # Balanced classes give F0 = 0 and depth-0 trees cannot move it.
        booster = GradientBoostingClassifier(max_depth=0).fit(XOR_X, XOR_Y)
        assert booster.predict([0.0, 0.0]) == "a"


class TestTreeStoppingRules:
    def test_max_depth_zero_fits_constant_base_rate(self) -> None:
        # Root leaves only: balanced Newton steps vanish, so p stays at 0.5.
        booster = GradientBoostingClassifier(max_depth=0, n_estimators=3).fit(XOR_X, XOR_Y)
        for row in XOR_X:
            assert booster.predict_proba(row)["b"] == pytest.approx(0.5)

    def test_min_samples_leaf_above_n_stops_at_root(self) -> None:
        booster = GradientBoostingClassifier(min_samples_leaf=10, n_estimators=3).fit(
            SEPARABLE_X, SEPARABLE_Y
        )
        base = booster.predict_proba([0.0])
        for row in SEPARABLE_X:
            assert booster.predict_proba(row) == base

    def test_deeper_model_respects_min_samples_leaf_mid_tree(self) -> None:
        # Children of size 3 are splittable; their own children (size < 3)
        # force leaves below max_depth — exercising the row-count stop rule.
        X = [[float(i)] for i in range(6)]
        y = ["lo", "lo", "lo", "hi", "hi", "hi"]
        booster = GradientBoostingClassifier(
            n_estimators=10, learning_rate=0.5, max_depth=3, min_samples_leaf=3
        ).fit(X, y)
        assert [booster.predict(row) for row in X] == y

    def test_no_varying_feature_fits_single_leaf(self) -> None:
        X = [[7.0], [7.0], [7.0]]
        booster = GradientBoostingClassifier(n_estimators=5, min_samples_leaf=1).fit(
            X, ["a", "b", "a"]
        )
        probs = booster.predict_proba([7.0])
        assert set(probs) == {"a", "b"}
        assert sum(probs.values()) == pytest.approx(1.0)
