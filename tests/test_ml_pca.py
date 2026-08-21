"""Tests for :mod:`cds.ml.decomposition` (PCA via Jacobi eigen-solver)."""

from __future__ import annotations

import pytest

from cds.ml import PCA


def test_pca_finds_dominant_direction() -> None:
    # Data stretched along the (1,1) diagonal: PC1 must capture ~all variance.
    X = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]
    model = PCA(n_components=1).fit(X)
    assert model.explained_variance_ratio_[0] == pytest.approx(1.0, abs=1e-9)
    axis = model.components_[0]
    # Sign is arbitrary; the line spanned must be the diagonal.
    assert abs(axis[0]) == pytest.approx(abs(axis[1]), abs=1e-8)


def test_pca_projection_preserves_pairwise_distances() -> None:
    X = [[0.0, 0.0], [2.0, 0.5], [4.0, 1.0], [6.0, 1.5]]
    model, Z = PCA(n_components=1).fit_transform(X)
    assert len(Z) == 4
    assert all(len(z) == 1 for z in Z)
    # 2-D points are collinear → 1-D projection reconstructs them exactly.
    rebuilt = model.inverse_transform(Z)
    for original, row in zip(X, rebuilt):
        for a, b in zip(original, row):
            assert a == pytest.approx(b, abs=1e-8)


def test_pca_two_components_recover_full_space() -> None:
    X = [[1.0, 2.0], [3.0, 4.0], [5.0, 7.0], [7.0, 1.0]]
    model = PCA(n_components=2).fit(X)
    Z = model.transform(X)
    rebuilt = model.inverse_transform(Z)
    for original, row in zip(X, rebuilt):
        for a, b in zip(original, row):
            assert a == pytest.approx(b, abs=1e-8)
    assert sum(model.explained_variance_ratio_) == pytest.approx(1.0, abs=1e-9)


def test_pca_diagonal_covariance_hits_annihilation_skip() -> None:
    # Centered orthogonal columns give an already-diagonal covariance, so
    # Jacobi's inner loop skips every pair without rotating.
    # Var([-1, 1]) with ddof=1 is 2.0.
    X = [[-1.0, 0.0], [1.0, 0.0]]
    model = PCA(n_components=2).fit(X)
    assert model.explained_variance_[0] == pytest.approx(2.0)


def test_pca_partial_zero_pairs_skip_rotation() -> None:
    # Columns 0 and 1 are identical (correlated), column 2 is exactly
    # orthogonal to both: the sweep must rotate pair (0,1) while skipping
    # pairs (0,2) and (1,2) as already annihilated.
    c0 = [1.0, 1.0, -1.0, -1.0]
    c2 = [1.0, -1.0, 1.0, -1.0]
    X = [[c0[i], c0[i], c2[i]] for i in range(4)]
    model = PCA(n_components=3).fit(X)
    # Spectrum of [[4/3, 4/3, 0], [4/3, 4/3, 0], [0, 0, 4/3]]:
    # diagonalized to 8/3 (duplicated axis), 4/3 (orthogonal column), ~0.
    assert model.explained_variance_[0] == pytest.approx(8.0 / 3.0)
    assert model.explained_variance_[1] == pytest.approx(4.0 / 3.0)
    assert model.explained_variance_[2] == pytest.approx(0.0, abs=1e-9)
    assert sum(model.explained_variance_ratio_) == pytest.approx(1.0, abs=1e-9)


def test_pca_constant_data_zero_variance_guard() -> None:
    X = [[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]]
    model = PCA(n_components=1).fit(X)
    assert model.explained_variance_ == [pytest.approx(0.0)]
    assert model.explained_variance_ratio_ == [0.0]


def test_pca_init_validation() -> None:
    with pytest.raises(ValueError, match="n_components must be >= 1"):
        PCA(0)


def test_pca_fit_validations() -> None:
    with pytest.raises(ValueError, match="at least 2 samples"):
        PCA(1).fit([[1.0, 2.0]])
    with pytest.raises(ValueError, match="same length"):
        PCA(1).fit([[1.0, 2.0], [3.0]])
    with pytest.raises(ValueError, match="must not exceed"):
        PCA(3).fit([[1.0, 2.0], [3.0, 4.0]])


def test_pcar_result_transform_width_mismatch() -> None:
    model = PCA(1).fit([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="features"):
        model.transform([[1.0]])


def test_pca_inverse_width_mismatch() -> None:
    model = PCA(1).fit([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="components to invert"):
        model.inverse_transform([[1.0, 2.0]])
