"""Tests for the one-sided Jacobi SVD (cds.math_utils.svd)."""

from __future__ import annotations

import math

import pytest

from cds.math_utils.linalg import Matrix, identity, mat_mul, transpose
from cds.math_utils.svd import SVDResult, condition_number, rank, svd

SQUARE = [[12.0, -51.0, 4.0], [6.0, 167.0, -68.0], [-4.0, 24.0, -41.0]]
TALL = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
WIDE = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
RANK_DEFICIENT = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]
AXIS_ALIGNED_RANK_ONE = [[1.0, 0.0], [0.0, 0.0]]
RECON_CASES = [
    ("square", SQUARE),
    ("tall", TALL),
    ("wide", WIDE),
    ("rank-deficient", RANK_DEFICIENT),
    ("axis-aligned-rank-one", AXIS_ALIGNED_RANK_ONE),
    ("identity", [[1.0, 0.0], [0.0, 1.0]]),
    ("single-column", [[1.0], [2.0], [3.0]]),
    ("single-row", [[1.0, 2.0, 3.0]]),
]


def _embed_sigma(values: list[float], m: int, n: int) -> Matrix:
    k = len(values)
    return [[values[j] if i == j < k else 0.0 for j in range(n)] for i in range(m)]


def _assert_close(actual: Matrix, expected: Matrix, atol: float) -> None:
    assert len(actual) == len(expected)
    for row_a, row_e in zip(actual, expected):
        assert row_a == pytest.approx(row_e, abs=atol)


def _assert_orthogonal(q: Matrix, atol: float) -> None:
    _assert_close(mat_mul(transpose(q), q), identity(len(q)), atol)


class TestReconstruction:
    @pytest.mark.parametrize(
        ("name", "matrix"),
        RECON_CASES,
        ids=[case[0] for case in RECON_CASES],
    )
    def test_reconstruction(self, name: str, matrix: Matrix) -> None:
        result = svd(matrix)
        m, n = len(matrix), len(matrix[0])
        sigma = _embed_sigma(result.singular_values, m, n)
        reconstructed = mat_mul(mat_mul(result.U, sigma), result.Vt)
        _assert_close(reconstructed, matrix, atol=1e-8)

    def test_shapes(self) -> None:
        for matrix in [SQUARE, TALL, WIDE, RANK_DEFICIENT]:
            m, n = len(matrix), len(matrix[0])
            result = svd(matrix)
            assert len(result.U) == m
            assert all(len(row) == m for row in result.U)
            assert len(result.Vt) == n
            assert all(len(row) == n for row in result.Vt)
            assert len(result.singular_values) == min(m, n)


class TestProperties:
    def test_known_singular_values(self) -> None:
        result = svd([[3.0, 0.0], [0.0, 4.0]])
        assert result.singular_values == pytest.approx([4.0, 3.0])

    @pytest.mark.parametrize(
        "matrix",
        [SQUARE, TALL, RANK_DEFICIENT],
        ids=["square", "tall", "rank-deficient"],
    )
    def test_u_orthogonal(self, matrix: Matrix) -> None:
        _assert_orthogonal(svd(matrix).U, atol=1e-9)

    @pytest.mark.parametrize(
        "matrix",
        [SQUARE, TALL, WIDE],
        ids=["square", "tall", "wide"],
    )
    def test_vt_orthogonal(self, matrix: Matrix) -> None:
        _assert_orthogonal(svd(matrix).Vt, atol=1e-9)

    @pytest.mark.parametrize("matrix", [SQUARE, TALL, WIDE, RANK_DEFICIENT])
    def test_singular_values_descending(self, matrix: Matrix) -> None:
        values = svd(matrix).singular_values
        assert values == sorted(values, reverse=True)

    def test_deterministic(self) -> None:
        first = svd(SQUARE)
        second = svd(SQUARE)
        assert first == second

    def test_returns_dataclass(self) -> None:
        result = svd([[2.0]])
        assert isinstance(result, SVDResult)
        assert result.U == [[1.0]]
        assert result.singular_values == pytest.approx([2.0])
        assert result.Vt == [[1.0]]


class TestRankAndCondition:
    def test_rank_full_square(self) -> None:
        assert rank([[3.0, 0.0], [0.0, 4.0]]) == 2

    def test_rank_tall_full(self) -> None:
        assert rank(TALL) == 2

    def test_rank_deficient_with_tol(self) -> None:
        assert rank(RANK_DEFICIENT, tol=1e-6) == 1

    def test_rank_zero_matrix(self) -> None:
        assert rank([[0.0, 0.0], [0.0, 0.0]]) == 0

    def test_condition_diagonal(self) -> None:
        assert condition_number([[3.0, 0.0], [0.0, 4.0]]) == pytest.approx(4.0 / 3.0)

    def test_condition_zero_matrix_is_inf(self) -> None:
        assert condition_number([[0.0, 0.0], [0.0, 0.0]]) == math.inf

    def test_condition_rank_deficient_is_inf(self) -> None:
        assert condition_number(RANK_DEFICIENT, tol=1e-6) == math.inf

    def test_condition_wide_finite(self) -> None:
        value = condition_number(WIDE)
        assert math.isfinite(value) and value > 0


class TestSweepLimits:
    def test_max_sweeps_exhausted_still_valid(self) -> None:
        result = svd(SQUARE, max_sweeps=1)
        sigma = _embed_sigma(result.singular_values, 3, 3)
        reconstructed = mat_mul(mat_mul(result.U, sigma), result.Vt)
        _assert_close(reconstructed, SQUARE, atol=1.0)

    def test_axis_aligned_rank_one_values(self) -> None:
        result = svd(AXIS_ALIGNED_RANK_ONE)
        assert result.singular_values == pytest.approx([1.0, 0.0], abs=1e-9)


class TestValidation:
    @pytest.mark.parametrize("matrix", [[], [[]], [[1.0, 2.0], [3.0]]])
    def test_invalid_matrix_raises(self, matrix: Matrix) -> None:
        with pytest.raises(ValueError):
            svd(matrix)

    @pytest.mark.parametrize("tol", [0.0, -1e-3])
    def test_non_positive_tol_raises(self, tol: float) -> None:
        with pytest.raises(ValueError):
            svd(SQUARE, tol=tol)

    @pytest.mark.parametrize("max_sweeps", [0, -2])
    def test_invalid_max_sweeps_raises(self, max_sweeps: int) -> None:
        with pytest.raises(ValueError):
            svd(SQUARE, max_sweeps=max_sweeps)

    @pytest.mark.parametrize("tol", [0.0, -1.0])
    def test_rank_non_positive_tol_raises(self, tol: float) -> None:
        with pytest.raises(ValueError):
            rank(SQUARE, tol=tol)

    @pytest.mark.parametrize("tol", [0.0, -5.0])
    def test_condition_non_positive_tol_raises(self, tol: float) -> None:
        with pytest.raises(ValueError):
            condition_number(SQUARE, tol=tol)
