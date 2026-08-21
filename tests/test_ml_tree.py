"""Tests for :mod:`cds.ml.tree` (CART decision-tree classifier)."""

from __future__ import annotations

import pytest

from cds.ml import DecisionTreeClassifier
from cds.ml.tree import _gini


def test_gini_empty_label_list_is_zero() -> None:
    assert _gini([]) == 0.0


def test_separable_data_classified_perfectly() -> None:
    X = [[1.0], [2.0], [9.0], [10.0]]
    y = ["low", "low", "high", "high"]
    tree = DecisionTreeClassifier().fit(X, y)
    assert tree.predict([1.5]) == "low"  # routes left
    assert tree.predict([9.5]) == "high"  # routes right
    assert tree.predict_proba([1.5]) == {"low": 1.0}


def test_max_depth_zero_yields_empirical_leaf() -> None:
    tree = DecisionTreeClassifier(max_depth=0).fit([[1.0], [2.0]], ["a", "b"])
    assert tree.predict_proba([1.0]) == {"a": 0.5, "b": 0.5}
    # Tie in the distribution: first-seen label ("a") wins deterministically.
    assert tree.predict([1.0]) == "a"


def test_min_samples_split_forces_leaves() -> None:
    X = [[1.0], [2.0], [9.0], [10.0]]
    y = ["a", "a", "b", "b"]
    tree = DecisionTreeClassifier(min_samples_split=100).fit(X, y)
    assert tree.predict_proba([5.0]) == {"a": 0.5, "b": 0.5}


def test_constant_features_cannot_split() -> None:
    tree = DecisionTreeClassifier().fit([[5.0], [5.0], [5.0]], ["x", "y", "x"])
    assert tree.predict_proba([5.0]) == {"x": 2 / 3, "y": 1 / 3}


def test_duplicate_values_skip_branch_then_split() -> None:
    X = [[1.0], [1.0], [2.0], [2.0]]
    y = [0, 0, 1, 1]
    tree = DecisionTreeClassifier().fit(X, y)
    assert tree.predict([1.0]) == 0
    assert tree.predict([2.5]) == 1


def test_pure_node_stops_immediately() -> None:
    tree = DecisionTreeClassifier().fit([[1.0], [2.0], [3.0]], ["same", "same", "same"])
    assert tree.predict_proba([2.0]) == {"same": 1.0}


def test_two_feature_dataset_uses_best_feature() -> None:
    # Feature 0 separates perfectly; feature 1 is noise.
    X = [[0.0, 3.0], [0.1, 7.0], [9.0, 1.0], [9.1, 8.0]]
    y = ["neg", "neg", "pos", "pos"]
    tree = DecisionTreeClassifier().fit(X, y)
    assert tree.predict([0.05, 5.0]) == "neg"
    assert tree.predict([9.0, 5.0]) == "pos"


def test_init_validates_hyperparameters() -> None:
    with pytest.raises(ValueError, match="max_depth must be >= 0"):
        DecisionTreeClassifier(max_depth=-1)
    with pytest.raises(ValueError, match="min_samples_split must be >= 2"):
        DecisionTreeClassifier(min_samples_split=1)


def test_fit_validates_data() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        DecisionTreeClassifier().fit([], [])
    with pytest.raises(ValueError, match="same length"):
        DecisionTreeClassifier().fit([[1.0]], ["a", "b"])
    with pytest.raises(ValueError, match="same length"):
        DecisionTreeClassifier().fit([[1.0, 2.0], [3.0]], ["a", "b"])


def test_predict_before_fit_raises() -> None:
    with pytest.raises(ValueError, match="not fitted"):
        DecisionTreeClassifier().predict([1.0])


def test_predict_rejects_wrong_query_width() -> None:
    tree = DecisionTreeClassifier().fit([[1.0], [2.0]], ["a", "b"])
    with pytest.raises(ValueError, match="features"):
        tree.predict([])
