"""Tests for :mod:`cds.ml.preprocessing` (StandardScaler + split)."""

from __future__ import annotations

import pytest

from cds.ml import StandardScaler, train_test_split


def test_scaler_centers_and_scales() -> None:
    X = [[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]]
    scaled = StandardScaler().fit_transform(X)
    cols = list(zip(*scaled))
    assert sum(cols[0]) == pytest.approx(0.0, abs=1e-12)
    assert sum(cols[1]) == pytest.approx(0.0, abs=1e-12)
    variances = [sum(v * v for v in col) / len(col) for col in cols]
    assert variances[0] == pytest.approx(1.0)
    assert variances[1] == pytest.approx(1.0)


def test_scaler_zero_variance_column_maps_to_zero() -> None:
    scaler = StandardScaler().fit([[2.0, 7.0], [2.0, 9.0]])
    scaled = scaler.transform([[2.0, 100.0], [2.0, -100.0]])
    assert [row[0] for row in scaled] == [0.0, 0.0]
    # Inverse restores the constant column to its fitted mean.
    restored = scaler.inverse_transform(scaled)
    assert restored[0][0] == 2.0
    assert restored[1][0] == 2.0


def test_scaler_round_trip() -> None:
    X = [[-4.0, 2.5], [0.0, 1.0], [4.0, 6.5]]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)
    restored = scaler.inverse_transform(scaled)
    for row_orig, row_rest in zip(X, restored):
        for a, b in zip(row_orig, row_rest):
            assert a == pytest.approx(b, abs=1e-12)


def test_scaler_fit_validations() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        StandardScaler().fit([])
    with pytest.raises(ValueError, match="same length"):
        StandardScaler().fit([[1.0, 2.0], [3.0]])


def test_scaler_transform_validations() -> None:
    scaler = StandardScaler()
    with pytest.raises(ValueError, match="not fitted"):
        scaler.transform([[1.0]])
    scaler.fit([[1.0, 2.0]])
    with pytest.raises(ValueError, match="non-empty"):
        scaler.transform([])
    with pytest.raises(ValueError, match="features"):
        scaler.transform([[1.0, 2.0, 3.0]])


def test_scaler_inverse_before_fit_raises() -> None:
    with pytest.raises(ValueError, match="not fitted"):
        StandardScaler().inverse_transform([[1.0]])


def test_split_partitions_and_sizes() -> None:
    X = [[float(i)] for i in range(20)]
    y = [i * 2.0 for i in range(20)]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, seed=42)
    assert len(X_te) == len(y_te) == 5
    assert len(X_tr) == len(y_tr) == 15
    # Alignment preserved within each partition.
    for xr, yr in zip(X_tr, y_tr):
        assert yr == xr[0] * 2.0
    # Disjoint partitions covering the original set.
    seen = sorted([row[0] for row in X_tr] + [row[0] for row in X_te])
    assert seen == [float(i) for i in range(20)]


def test_split_deterministic_given_seed() -> None:
    X = [[float(i)] for i in range(10)]
    y = [float(i) for i in range(10)]
    a = train_test_split(X, y, seed=7)
    b = train_test_split(X, y, seed=7)
    assert a == b


def test_split_shuffles_with_seed() -> None:
    X = [[float(i)] for i in range(10)]
    y = [float(i) for i in range(10)]
    X_tr, _, _, _ = train_test_split(X, y, seed=0)
    # The training rows must NOT be the identity prefix when shuffled.
    assert any(row[0] != i for i, row in zip(range(len(X_tr)), X_tr))


def test_split_validations() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        train_test_split([], [])
    with pytest.raises(ValueError, match="same length"):
        train_test_split([[1.0]], [1.0, 2.0])
    with pytest.raises(ValueError, match="same length"):
        train_test_split([[1.0, 2.0], [3.0]], [1.0, 2.0])
    with pytest.raises(ValueError, match=r"test_size must be in \(0, 1\)"):
        train_test_split([[1.0]], [1.0], test_size=0.0)
    with pytest.raises(ValueError, match=r"test_size must be in \(0, 1\)"):
        train_test_split([[1.0]], [1.0], test_size=1.0)
