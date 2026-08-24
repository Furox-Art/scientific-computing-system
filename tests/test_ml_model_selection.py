"""Tests for cds.ml.model_selection — k-fold cross-validation."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from cds.ml.metrics import Label, accuracy
from cds.ml.model_selection import CVResult, SupervisedModel, cross_val_score, k_fold_indices


class _ConstantModel:
    """Predicts the majority label of training data."""

    def fit(self, X: list[list[float]], y: Sequence[Label]) -> _ConstantModel:
        self._label: Label = max(set(y), key=list(y).count)
        return self

    def predict(self, x: list[float]) -> Label:
        return self._label


def _factory() -> SupervisedModel:
    return _ConstantModel()


class TestKFoldIndices:
    def test_contiguous_deterministic(self) -> None:
        folds = k_fold_indices(6, 3)
        assert len(folds) == 3
        for train, test in folds:
            assert sorted(train + test) == list(range(6))
        assert folds[0][1] == [0, 1]
        assert folds[2][1] == [4, 5]

    def test_uneven_sizes_early_folds_larger(self) -> None:
        folds = k_fold_indices(5, 2)
        sizes = [len(test) for _, test in folds]
        assert sizes == [3, 2]

    def test_shuffle_changes_folds_but_preserves_partition(self) -> None:
        plain = k_fold_indices(7, 7, shuffle=False)
        shuffled = k_fold_indices(7, 7, shuffle=True, seed=42)
        again = k_fold_indices(7, 7, shuffle=True, seed=42)
        assert all(sorted(t + v) == list(range(7)) for t, v in shuffled)
        assert shuffled != plain
        # same seed reproduces the exact partition
        assert shuffled == again

    def test_invalid_k_raises(self) -> None:
        with pytest.raises(ValueError, match="k must be >= 2"):
            k_fold_indices(5, 1)

    def test_n_smaller_than_k_raises(self) -> None:
        with pytest.raises(ValueError, match="n must be >= k"):
            k_fold_indices(2, 5)

    def test_invalid_n_raises(self) -> None:
        with pytest.raises(ValueError, match="n must be >= 1"):
            k_fold_indices(0, 2)


class TestCrossValScore:
    X = [[float(i)] for i in range(6)]
    y = ["a", "a", "a", "b", "b", "b"]

    def test_scores_and_mean(self) -> None:
        result = cross_val_score(_factory, self.X, self.y, k=3)
        assert isinstance(result, CVResult)
        assert len(result.scores) == 3
        assert result.mean_score == pytest.approx(sum(result.scores) / 3)

    def test_shuffled_seeded_is_reproducible(self) -> None:
        a = cross_val_score(_factory, self.X, self.y, k=2, shuffle=True, seed=7)
        b = cross_val_score(_factory, self.X, self.y, k=2, shuffle=True, seed=7)
        assert a == b

    def test_custom_scorer(self) -> None:
        result = cross_val_score(
            _factory,
            self.X,
            self.y,
            k=3,
            scorer=lambda yt, yp: 1.0 if list(yt) == list(yp) else 0.0,
        )
        assert all(s in (0.0, 1.0) for s in result.scores)

    def test_empty_X_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            cross_val_score(_factory, [], [], k=2)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            cross_val_score(_factory, self.X, self.y[:-1], k=2)

    def test_ragged_rows_raise(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            cross_val_score(_factory, [[1.0], [1.0, 2.0]], ["a", "b"], k=2)

    def test_k_exceeding_n_raises_via_fold_check(self) -> None:
        with pytest.raises(ValueError, match="n must be >= k"):
            cross_val_score(_factory, self.X, self.y, k=10)

    def test_accuracy_default_matches_explicit(self) -> None:
        explicit = cross_val_score(_factory, self.X, self.y, k=3, scorer=accuracy)
        default = cross_val_score(_factory, self.X, self.y, k=3)
        assert explicit == default
