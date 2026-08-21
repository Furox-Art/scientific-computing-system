"""Tests for :mod:`cds.ml.clustering` (k-means with k-means++ seeding)."""

from __future__ import annotations

import pytest

from cds.ml import KMeans


def _blobs() -> list[list[float]]:
    return [
        [0.0, 0.0],
        [0.2, 0.1],
        [0.1, 0.3],
        [5.0, 5.0],
        [5.2, 4.9],
        [4.8, 5.1],
    ]


def test_two_blobs_split_cleanly() -> None:
    res = KMeans(2, seed=0).fit(_blobs())
    assert set(res.labels) == {0, 1}
    assert sum(1 for lbl in res.labels if lbl == 0) == 3
    assert res.inertia < 0.5
    assert res.n_iter <= 10


def test_seed_gives_reproducible_result() -> None:
    a = KMeans(2, seed=7).fit(_blobs())
    b = KMeans(2, seed=7).fit(_blobs())
    assert a == b


def test_duplicate_points_hit_uniform_fallback_and_empty_cluster() -> None:
    # All points identical: k-means++ total distance is zero (uniform-draw
    # branch) and one cluster stays empty (keeps its centroid).
    res = KMeans(2, seed=1).fit([[0.0], [0.0], [0.0]])
    assert res.labels == [0, 0, 0]  # assignment ties go to the lowest index
    assert res.centroids[1] == [0.0]
    assert res.inertia == pytest.approx(0.0)
    assert res.n_iter == 1


def test_max_iter_bounds_iterations() -> None:
    res = KMeans(2, max_iter=1, seed=0).fit(_blobs())
    assert res.n_iter == 1


def test_k_equals_n_gives_zero_inertia() -> None:
    pts = [[0.0], [10.0], [20.0]]
    res = KMeans(3, seed=3).fit(pts)
    assert res.inertia == pytest.approx(0.0)
    assert len(set(res.labels)) == 3


def test_predict_routes_to_nearest_centroid() -> None:
    model = KMeans(2, seed=0)
    model.fit(_blobs())
    near_a = model.predict([0.1, 0.1])
    near_b = model.predict([5.1, 5.0])
    assert near_a != near_b


def test_predict_before_fit_raises() -> None:
    with pytest.raises(ValueError, match="not fitted"):
        KMeans(1).predict([0.0])


def test_predict_rejects_wrong_width() -> None:
    model = KMeans(1)
    model.fit([[1.0, 2.0]])
    with pytest.raises(ValueError, match="features"):
        model.predict([1.0])


def test_init_validates_hyperparameters() -> None:
    with pytest.raises(ValueError, match="n_clusters must be >= 1"):
        KMeans(0)
    with pytest.raises(ValueError, match="max_iter must be >= 1"):
        KMeans(2, max_iter=0)
    with pytest.raises(ValueError, match="tol must be >= 0"):
        KMeans(2, tol=-1e-9)


def test_fit_validates_data() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        KMeans(1).fit([])
    with pytest.raises(ValueError, match="same length"):
        KMeans(1).fit([[1.0], [2.0, 3.0]])
    with pytest.raises(ValueError, match="must not exceed"):
        KMeans(3).fit([[1.0], [2.0]])
