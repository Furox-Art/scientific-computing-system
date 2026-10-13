"""H-1: ``cholesky`` must reject every input for which "symmetric positive
definite" cannot be established.

The pivot test ``m[i][i] - Σ L[i][k]^2 <= 0`` is a complete decision
procedure *for symmetric input*: a real lower-triangular factor exists iff
the matrix is positive definite. That part already worked. What did not:

* a NaN entry slipped straight through — ``nan <= 0`` is ``False``, so
  ``math.sqrt(nan)`` produced an all-``nan`` ``L`` that the caller read as a
  decomposition;
* a non-square matrix silently dropped its extra columns and factored a
  *different* matrix, and a ragged one surfaced as a bare ``IndexError``;
* a non-symmetric matrix returned an ``L`` with ``L Lᵀ != A``.

Each of those now raises a documented ``ValueError``, and the tests below
assert the *values* a caller still gets for the inputs that are valid.
"""

from __future__ import annotations

import math

import pytest

from cds.math_utils.linalg import Matrix, cholesky, mat_mul, transpose

# Textbook example (Golub & Van Loan §4.2 style): A = L L^T with a known L.
_SPD: Matrix = [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]]
_SPD_LOWER: Matrix = [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]]


def _max_err(a: Matrix, b: Matrix) -> float:
    """Largest absolute entrywise difference between two equal-shaped matrices."""
    return max(
        abs(a[i][j] - b[i][j]) for i in range(len(a)) for j in range(len(a[0]))
    )


class TestCholeskyStillWorks:
    """Guards against the hardening rejecting valid input."""

    def test_known_decomposition_values(self) -> None:
        assert _max_err(cholesky(_SPD), _SPD_LOWER) < 1e-12

    def test_reconstruction_within_tolerance(self) -> None:
        lower = cholesky(_SPD)
        reconstruction = mat_mul(lower, transpose(lower))
        for i in range(len(_SPD)):
            for j in range(len(_SPD)):
                assert reconstruction[i][j] == pytest.approx(_SPD[i][j], abs=1e-9)

    def test_large_magnitude_matrix_is_accepted(self) -> None:
        """The symmetry tolerance is relative, so big entries are not flagged."""
        m: Matrix = [[1.0e8, 1.0e8], [1.0e8, 2.0e8]]
        lower = cholesky(m)
        assert lower[0][0] == pytest.approx(math.sqrt(1.0e8))
        assert lower[0][1] == 0.0
        assert mat_mul(lower, transpose(lower))[0][0] == pytest.approx(m[0][0], rel=1e-12)

    def test_scaled_identity_is_exact(self) -> None:
        assert _max_err(cholesky([[9.0, 0.0], [0.0, 4.0]]), [[3.0, 0.0], [0.0, 2.0]]) < 1e-12

    def test_single_entry_positive(self) -> None:
        assert cholesky([[9.0]]) == [[3.0]]


class TestCholeskyNonFiniteInput:
    """The NaN/Inf hole that made ``L`` all-``nan`` without an error."""

    @pytest.mark.parametrize(
        "matrix",
        [
            [[float("nan"), 0.0], [0.0, 1.0]],
            [[1.0, float("nan")], [float("nan"), 1.0]],
            [[float("inf"), 0.0], [0.0, 1.0]],
            [[1.0, float("-inf")], [float("-inf"), 1.0]],
            [[1.0, float("nan"), 2.0], [float("nan"), 3.0, 4.0], [2.0, 4.0, 5.0]],
        ],
    )
    def test_non_finite_entry_is_rejected(self, matrix: Matrix) -> None:
        """``nan <= 0.0`` is ``False``, so the old pivot test waved NaN through.

        The regression is the *presence of an error* combined with the fact
        that a valid call still returns values: the guard must not swallow
        legitimate input.
        """
        with pytest.raises(ValueError, match="non-finite entry"):
            cholesky(matrix)

    def test_non_finite_position_is_reported(self) -> None:
        with pytest.raises(ValueError, match=r"non-finite entry at \(1, 0\)"):
            cholesky([[4.0, 1.0], [float("nan"), 3.0]])


class TestCholeskyNonSquareInput:
    """A non-square matrix used to factor a different matrix, or IndexError."""

    def test_extra_column_is_rejected(self) -> None:
        """This returned ``[[2,0],[0.5,1.658…]]`` — a valid 2x2 factor of a
        matrix that is not the input. Garbage, not an exception.
        """
        with pytest.raises(ValueError, match="must be square"):
            cholesky([[4.0, 1.0, 0.0], [1.0, 3.0, 2.0]])

    def test_missing_row_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be square"):
            cholesky([[4.0, 1.0], [1.0, 3.0], [0.0, 2.0]])

    def test_ragged_input_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="ragged"):
            cholesky([[4.0, 1.0], [1.0]])

    def test_empty_input_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty square matrix"):
            cholesky([])


class TestCholeskyNonSymmetricInput:
    """A non-symmetric PD matrix yields an ``L`` with ``L Lᵀ != A``."""

    def test_asymmetric_positive_definite_is_rejected(self) -> None:
        """``[[4, 1], [3, 4]]`` has eigenvalues 5.73 and 2.27 — strongly PD —
        so every pivot is positive and the old code happily returned
        ``[[2,0],[1.5,1.3229]]`` whose product is ``[[4,3],[3,6.25]]``.
        """
        with pytest.raises(ValueError, match="not symmetric"):
            cholesky([[4.0, 1.0], [3.0, 4.0]])

    def test_symmetry_tolerance_admits_round_off(self) -> None:
        """A caller building the matrix through arithmetic still succeeds."""
        m: Matrix = [[4.0, 1.0], [1.0 + 1e-16, 3.0]]
        lower = cholesky(m)
        assert lower[0][1] == 0.0
        assert lower[0][0] == pytest.approx(2.0)
        assert lower[1][1] > 0.0


class TestCholeskyNonPositiveDefinite:
    """The documented non-PSD rejection, with the known non-PSD matrices."""

    @pytest.mark.parametrize(
        "matrix",
        [
            [[1.0, 2.0], [2.0, 1.0]],        # eigenvalues +3, -1
            [[-1.0, 0.0], [0.0, -1.0]],     # negative definite
            [[2.0, 4.0], [4.0, 2.0]],       # eigenvalues +6, -2
            [[1.0, 1.0], [1.0, 1.0]],       # positive *semi*-definite / singular
            [[0.0, 0.0], [0.0, 0.0]],       # zero matrix
            [[-9.0]],                        # 1x1 negative
            [
                [1.0, 2.0, 3.0],
                [2.0, 1.0, 4.0],
                [3.0, 4.0, 1.0],
            ],  # indefinite
        ],
    )
    def test_non_psd_matrices_raise(self, matrix: Matrix) -> None:
        with pytest.raises(ValueError, match="not positive definite"):
            cholesky(matrix)

    def test_error_message_names_the_gap(self) -> None:
        with pytest.raises(ValueError, match=r"eigenvalues > 0"):
            cholesky([[1.0, 2.0], [2.0, 1.0]])

    def test_eigenvalue_free_proof_of_recovery(self) -> None:
        """The rejection is a value claim: after the fix, a valid matrix works.

        This is the positive control for the parameterized rejections above.
        For ``[[2, 1], [1, 2]]`` the closed-form factor is
        ``[[√2, 0], [1/√2, √(3/2)]]``.
        """
        lower = cholesky([[2.0, 1.0], [1.0, 2.0]])
        expected: Matrix = [
            [math.sqrt(2.0), 0.0],
            [1.0 / math.sqrt(2.0), math.sqrt(1.5)],
        ]
        assert _max_err(lower, expected) < 1e-12
