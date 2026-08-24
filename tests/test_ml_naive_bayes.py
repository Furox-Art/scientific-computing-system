"""Tests for :mod:`cds.ml.naive_bayes` (Gaussian Naive Bayes)."""

from __future__ import annotations

import math

import pytest

from cds.ml.metrics import Label
from cds.ml.naive_bayes import GaussianNaiveBayes


def _separable() -> tuple[list[list[float]], list[str]]:
    X = [
        [0.0, 0.0],
        [0.5, 0.2],
        [-0.2, 0.4],
        [10.0, 10.0],
        [10.5, 9.8],
        [9.8, 10.4],
    ]
    y = ["a", "a", "a", "b", "b", "b"]
    return X, y


def test_separate_gaussians_classified_perfectly() -> None:
    X, y = _separable()
    clf = GaussianNaiveBayes().fit(X, y)
    assert clf.predict([0.1, -0.1]) == "a"
    assert clf.predict([10.2, 10.1]) == "b"
    for row, label in zip(X, y):
        assert clf.predict(row) == label


def test_predict_matches_argmax_of_proba() -> None:
    X, y = _separable()
    clf = GaussianNaiveBayes().fit(X, y)
    probs = clf.predict_proba([4.0, 4.5])
    argmax = max(probs, key=lambda label: probs[label])
    assert clf.predict([4.0, 4.5]) == argmax


def test_probas_sum_to_one_everywhere() -> None:
    X, y = _separable()
    clf = GaussianNaiveBayes().fit(X, y)
    for query in ([0.0, 0.0], [5.0, 5.0], [100.0, -100.0], [-50.0, 50.0]):
        assert sum(clf.predict_proba(query).values()) == pytest.approx(1.0)


def test_single_feature_closed_form_with_unequal_priors() -> None:
    # Class "a" (prior 3/5): {0, 2, 4} -> N(2, 8/3).
    # Class "b" (prior 2/5): {10, 12} -> N(11, 1).
    X = [[0.0], [2.0], [4.0], [10.0], [12.0]]
    y = ["a", "a", "a", "b", "b"]
    clf = GaussianNaiveBayes().fit(X, y)
    query = 4.0
    log_joint_a = (
        math.log(3 / 5)
        - 0.5 * math.log(2 * math.pi * (8.0 / 3.0))
        - (query - 2.0) ** 2 / (2 * 8.0 / 3.0)
    )
    log_joint_b = (
        math.log(2 / 5) - 0.5 * math.log(2 * math.pi * 1.0) - (query - 11.0) ** 2 / (2 * 1.0)
    )
    expected_a = math.exp(log_joint_a) / (math.exp(log_joint_a) + math.exp(log_joint_b))
    expected_b = math.exp(log_joint_b) / (math.exp(log_joint_a) + math.exp(log_joint_b))
    probs = clf.predict_proba([query])
    assert probs["a"] == pytest.approx(expected_a)
    assert probs["b"] == pytest.approx(expected_b)
    assert sum(probs.values()) == pytest.approx(1.0)


def test_population_variance_used_in_closed_form() -> None:
    # Single class pair where ddof choice matters: class "a" on {0, 2} has
    # population variance 1 (sample variance would be 2).
    X = [[0.0], [2.0], [10.0], [12.0]]
    y = ["a", "a", "b", "b"]
    clf = GaussianNaiveBayes().fit(X, y)
    # Query x=11 sits exactly on b's mean (sigma^2 = 1); a's mean is 1,
    # giving a squared distance of 100 over 2 * sigma^2.
    log_joint_b = math.log(0.5) - 0.5 * math.log(2 * math.pi)
    log_joint_a = math.log(0.5) - 0.5 * math.log(2 * math.pi) - 50.0
    expected_b = math.exp(log_joint_b) / (math.exp(log_joint_a) + math.exp(log_joint_b))
    assert clf.predict_proba([11.0])["b"] == pytest.approx(expected_b)
    assert clf.predict([11.0]) == "b"


def test_zero_variance_feature_smoothed_and_harmless() -> None:
    # Feature 1 separates the classes; feature 2 is constant inside each
    # class (zero per-class variance) but varies globally.
    X = [[0.0, 1.0], [2.0, 1.0], [10.0, 5.0], [12.0, 5.0]]
    y = ["a", "a", "b", "b"]
    clf = GaussianNaiveBayes().fit(X, y)
    assert clf.epsilon_ == pytest.approx(26e-9)  # max global feature variance
    assert clf.predict([1.0, 1.0]) == "a"
    assert clf.predict([11.0, 5.0]) == "b"


def test_constant_matrix_falls_back_to_absolute_floor() -> None:
    X = [[7.0, 7.0], [7.0, 7.0]]
    clf = GaussianNaiveBayes().fit(X, ["a", "b"])
    assert clf.epsilon_ == pytest.approx(1e-12)
    # Identical likelihoods: posteriors collapse to the priors.
    assert clf.predict_proba([7.0, 7.0]) == {"a": 0.5, "b": 0.5}


def test_epsilon_matches_documented_rule() -> None:
    X = [[0.0, 0.0], [2.0, 1.0], [4.0, 2.0]]
    y = ["a", "b", "b"]
    clf = GaussianNaiveBayes().fit(X, y)
    # Global variances: feature 0 -> 8/3, feature 1 -> 2/3; max is 8/3.
    assert clf.epsilon_ == pytest.approx((8.0 / 3.0) * 1e-9)


def test_repeated_fits_are_deterministic() -> None:
    X, y = _separable()
    first = GaussianNaiveBayes().fit(X, y)
    second = GaussianNaiveBayes().fit(X, y)
    queries = ([0.0, 0.0], [5.0, 5.0], [10.0, 10.0])
    for query in queries:
        assert first.predict_proba(query) == second.predict_proba(query)
        assert first.predict(query) == second.predict(query)
    assert first.epsilon_ == second.epsilon_
    assert first._classes == second._classes


def test_tie_resolves_to_earliest_seen_label() -> None:
    clf = GaussianNaiveBayes().fit([[0.0], [2.0]], ["first", "second"])
    assert clf.predict_proba([1.0]) == {"first": 0.5, "second": 0.5}
    assert clf.predict([1.0]) == "first"


def test_int_labels_supported() -> None:
    clf = GaussianNaiveBayes().fit([[0.0], [10.0]], [0, 1])
    probs: dict[Label, float] = clf.predict_proba([9.9])
    assert probs[1] > probs[0]
    assert clf.predict([0.1]) == 0


def test_empty_width_rows_yield_prior_only_posteriors() -> None:
    clf = GaussianNaiveBayes().fit([[], [], []], ["a", "a", "b"])
    assert clf.predict_proba([]) == {"a": pytest.approx(2 / 3), "b": pytest.approx(1 / 3)}


def test_init_rejects_nonpositive_var_smoothing() -> None:
    with pytest.raises(ValueError, match="var_smoothing must be > 0"):
        GaussianNaiveBayes(var_smoothing=0.0)


def test_fit_rejects_empty_x() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        GaussianNaiveBayes().fit([], [])


def test_fit_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        GaussianNaiveBayes().fit([[1.0]], ["a", "b"])


def test_fit_rejects_ragged_rows() -> None:
    with pytest.raises(ValueError, match="same length"):
        GaussianNaiveBayes().fit([[1.0, 2.0], [1.0]], ["a", "b"])


def test_predict_before_fit_raises() -> None:
    clf = GaussianNaiveBayes()
    with pytest.raises(ValueError, match="not fitted"):
        clf.predict_proba([1.0])
    with pytest.raises(ValueError, match="not fitted"):
        clf.predict_proba_one([1.0])
    with pytest.raises(ValueError, match="not fitted"):
        clf.predict([1.0])


def test_predict_rejects_wrong_query_width() -> None:
    X, y = _separable()
    clf = GaussianNaiveBayes().fit(X, y)
    with pytest.raises(ValueError, match="features"):
        clf.predict_proba([1.0])
    with pytest.raises(ValueError, match="features"):
        clf.predict_proba_one([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="features"):
        clf.predict([1.0])
