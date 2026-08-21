"""Tests for :mod:`cds.ml.neighbors` (k-NN classifier + regressor)."""

from __future__ import annotations

import pytest

from cds.ml import KNeighborsClassifier, KNeighborsRegressor


def _two_clusters() -> tuple[list[list[float]], list[str]]:
    X = [[0.0, 0.0], [0.5, 0.2], [4.0, 4.0], [4.5, 3.8]]
    y = ["a", "a", "b", "b"]
    return X, y


def test_classifier_predicts_own_cluster() -> None:
    X, y = _two_clusters()
    clf = KNeighborsClassifier(k=3).fit(X, y)
    assert clf.predict([0.1, 0.1]) == "a"
    assert clf.predict([4.2, 4.0]) == "b"


def test_predict_proba_sums_to_one() -> None:
    X, y = _two_clusters()
    clf = KNeighborsClassifier(k=4).fit(X, y)
    probs = clf.predict_proba([2.0, 2.0])
    assert probs == {"a": 0.5, "b": 0.5}


def test_vote_tie_resolves_to_earliest_neighbour_label() -> None:
    # Two equidistant neighbours with different labels: the label of the
    # neighbour with the lower training index must win.
    clf = KNeighborsClassifier(k=2).fit([[0.0], [2.0]], ["first", "second"])
    assert clf.predict([1.0]) == "first"


def test_k_equals_one_returns_exact_neighbour() -> None:
    reg = KNeighborsRegressor(k=1).fit([[0.0], [10.0]], [1.0, 3.0])
    assert reg.predict([9.9]) == 3.0


def test_regressor_means_neighbours() -> None:
    reg = KNeighborsRegressor(k=2).fit([[0.0], [1.0]], [2.0, 6.0])
    assert reg.predict([0.4]) == pytest.approx(4.0)


def test_int_labels_supported() -> None:
    clf = KNeighborsClassifier(k=1).fit([[0.0], [5.0]], [0, 1])
    assert clf.predict([4.9]) == 1


def test_init_rejects_k_below_one() -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        KNeighborsClassifier(k=0)
    with pytest.raises(ValueError, match="k must be >= 1"):
        KNeighborsRegressor(k=-1)


def test_fit_rejects_empty_x() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        KNeighborsClassifier().fit([], [])


def test_fit_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        KNeighborsClassifier().fit([[1.0]], ["a", "b"])


def test_fit_rejects_ragged_rows() -> None:
    with pytest.raises(ValueError, match="same length"):
        KNeighborsClassifier().fit([[1.0, 2.0], [1.0]], ["a", "b"])


def test_fit_rejects_k_above_sample_count() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        KNeighborsClassifier(k=5).fit([[1.0]], ["a"])
    with pytest.raises(ValueError, match="must not exceed"):
        KNeighborsRegressor(k=5).fit([[1.0]], [1.0])


def test_predict_before_fit_raises() -> None:
    with pytest.raises(ValueError, match="not fitted"):
        KNeighborsClassifier().predict([1.0])
    with pytest.raises(ValueError, match="not fitted"):
        KNeighborsRegressor().predict([1.0])


def test_predict_rejects_wrong_query_width() -> None:
    X, y = _two_clusters()
    clf = KNeighborsClassifier(k=1).fit(X, y)
    reg = KNeighborsRegressor(k=1).fit(X, [1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="features"):
        clf.predict([1.0])
    with pytest.raises(ValueError, match="features"):
        reg.predict([1.0])
