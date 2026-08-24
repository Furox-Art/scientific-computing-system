"""Tests for cds.ml.ensemble — Random Forest classifier."""

import pytest

from cds.ml.ensemble import RandomForestClassifier
from cds.ml.tree import DecisionTreeClassifier

XOR_X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
XOR_Y = ["a", "b", "b", "a"]


class TestValidation:
    def test_n_trees_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="n_trees"):
            RandomForestClassifier(n_trees=0)

    def test_int_max_features_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="max_features must be >= 1"):
            RandomForestClassifier(max_features=0)

    def test_unknown_string_raises(self) -> None:
        with pytest.raises(ValueError, match="'sqrt', 'log2'"):
            RandomForestClassifier(max_features="banana")  # type: ignore[arg-type]

    def test_int_max_features_above_width_raises_on_fit(self) -> None:
        forest = RandomForestClassifier(n_trees=1, max_features=5)
        with pytest.raises(ValueError, match="<="):
            forest.fit(XOR_X, XOR_Y)

    def test_fit_validations_surface_from_tree_path(self) -> None:
        empty: list[list[float]] = []
        with pytest.raises(ValueError, match="non-empty"):
            RandomForestClassifier(n_trees=1).fit(empty, [])

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            RandomForestClassifier(n_trees=1).fit(XOR_X, XOR_Y[:-1])

    def test_ragged_rows_raise(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            RandomForestClassifier(n_trees=1).fit([[1.0], [1.0, 2.0]], ["a", "b"])


class TestFitPredict:
    def test_seeded_forest_is_deterministic(self) -> None:
        a = RandomForestClassifier(n_trees=10, seed=3).fit(XOR_X * 4, XOR_Y * 4)
        b = RandomForestClassifier(n_trees=10, seed=3).fit(XOR_X * 4, XOR_Y * 4)
        assert all(a.predict(row) == b.predict(row) for row in XOR_X)

    def test_xor_learned_with_enough_trees(self) -> None:
        forest = RandomForestClassifier(n_trees=25, max_depth=4, seed=11)
        assert forest.fit(XOR_X * 8, XOR_Y * 8) is forest
        preds = [forest.predict(row) for row in XOR_X]
        assert preds == XOR_Y

    def test_predict_proba_sums_to_one(self) -> None:
        forest = RandomForestClassifier(n_trees=5, seed=1).fit(XOR_X, XOR_Y)
        probs = forest.predict_proba([0.9, 0.9])
        assert set(probs) == {"a", "b"}
        assert sum(probs.values()) == pytest.approx(1.0)

    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(ValueError, match="not fitted"):
            RandomForestClassifier().predict(XOR_X[0])

    def test_feature_count_mismatch_raises(self) -> None:
        forest = RandomForestClassifier(n_trees=2, seed=0).fit(XOR_X, XOR_Y)
        with pytest.raises(ValueError, match="features"):
            forest.predict([1.0])

    def test_max_features_none_uses_all_features(self) -> None:
        forest = RandomForestClassifier(n_trees=3, max_features=None, seed=5).fit(
            XOR_X * 2, XOR_Y * 2
        )
        assert [forest.predict(r) for r in XOR_X] == XOR_Y

    def test_log2_and_explicit_counts(self) -> None:
        for spec in ("log2", 2):
            forest = RandomForestClassifier(n_trees=3, max_features=spec, seed=9).fit(
                XOR_X * 2, XOR_Y * 2
            )
            assert forest.predict([1.0, 0.0]) in ("a", "b")


class TestTreeMaxFeaturesIntegration:
    """The split-randomization hooks live on DecisionTreeClassifier itself."""

    def test_tree_max_features_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="max_features must be >= 1"):
            DecisionTreeClassifier(max_features=-1)

    def test_tree_max_features_above_width_raises_on_fit(self) -> None:
        tree = DecisionTreeClassifier(max_features=9)
        with pytest.raises(ValueError, match="<="):
            tree.fit([[1.0]], ["a"])

    def test_tree_with_subset_still_separates_linear_data(self) -> None:
        X = [[float(i), float(i % 2)] for i in range(6)]
        y = ["lo" if i < 3 else "hi" for i in range(6)]
        tree = DecisionTreeClassifier(max_depth=2, max_features=1, seed=42)
        assert tree.fit(X, y) is tree
        assert tree.predict([0.0, 0.0]) == "lo"
        assert tree.predict([5.0, 1.0]) == "hi"
