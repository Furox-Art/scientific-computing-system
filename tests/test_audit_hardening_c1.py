"""Value-level regression tests for the audit-hardening findings.

Every numerical assertion here pins an actual value or bound; nothing in this
module is satisfied by "did not raise". Each class documents the bug it
guards, so a regression names the finding it re-introduced.
"""

from __future__ import annotations

import math

import pytest

from cds.math_utils.linalg import Matrix, mat_mul, transpose


def _reference_product(a: Matrix, b: Matrix) -> Matrix:
    """Independent index-based ``A * B`` used as the oracle."""
    rows, shared, cols = len(a), len(b), len(b[0])
    return [
        [math.fsum(a[i][k] * b[k][j] for k in range(shared)) for j in range(cols)]
        for i in range(rows)
    ]


class TestMatMulShapeHardening:
    """C-1: ``mat_mul`` must reject ragged operands instead of truncating."""

    def test_ragged_right_operand_is_rejected(self) -> None:
        """A ragged ``b`` used to yield ``[[7], [15]]`` — a silently short.

        ``b`` has three rows of unequal width, so ``cols_a == rows_b`` passes
        the old leading-row check and ``zip(*b)`` quietly dropped the tail of
        every column. The answer is a wrong number, not an exception.
        """
        ragged_b: Matrix = [[1, 2], [3]]
        with pytest.raises(ValueError, match="b is ragged"):
            mat_mul([[1.0, 2.0], [3.0, 4.0]], ragged_b)

    def test_ragged_left_operand_is_rejected(self) -> None:
        """A ragged ``a`` used to yield ``[[14], [4]]``; row 2 should be 24.

        The second row of ``a`` is shorter than the first, so the inner
        ``zip(row_a, col_b)`` truncated it to its first element only.
        """
        ragged_a: Matrix = [[1.0, 2.0, 3.0], [4.0]]
        with pytest.raises(ValueError, match="a is ragged"):
            mat_mul(ragged_a, [[1.0], [2.0], [3.0]])

    def test_ragged_operand_where_last_row_is_longest(self) -> None:
        """The ragged row need not be the short one for the bug to bite."""
        with pytest.raises(ValueError, match="is ragged"):
            mat_mul([[1.0, 2.0], [3.0, 4.0, 5.0]], [[1.0, 0.0], [0.0, 1.0]])

    def test_empty_operands_are_rejected(self) -> None:
        """``a[0]`` used to raise a bare ``IndexError``."""
        with pytest.raises(ValueError, match="a must be a non-empty matrix"):
            mat_mul([], [[1.0]])
        with pytest.raises(ValueError, match="b must be a non-empty matrix"):
            mat_mul([[1.0]], [])

    def test_empty_rows_are_rejected_as_ragged(self) -> None:
        """A row that vanished makes the operand ragged.

        ``[[], [1.0]]`` declares width 0 from row 0 and width 1 from row 1;
        the old code trusted row 0 and silently read a truncated matrix.
        """
        with pytest.raises(ValueError, match="is ragged"):
            mat_mul([[], [1.0]], [[1.0], [2.0]])

    def test_zero_width_operand_keeps_zero_width_result(self) -> None:
        """A 1x1 times a 1x0 operand is a 1x0 matrix, not ``[[0.0]]``.

        The old code derived the output width from ``zip(*b)``, which yields
        no tuples for a zero-column operand, so the fabricated ``[[0.0]]``
        carried a column that the algebra does not define.
        """
        assert mat_mul([[3.0]], [[]]) == [[]]

    def test_genuine_dimension_mismatch_still_raises(self) -> None:
        with pytest.raises(ValueError, match="incompatible shapes"):
            mat_mul([[1.0, 2.0]], [[1.0]])

    def test_reported_shape_uses_every_row(self) -> None:
        """The error message names the true operand shapes."""
        with pytest.raises(ValueError, match=r"2x4 and 3x2"):
            mat_mul(
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
            )


class TestMatMulTransposedProducts:
    """C-1: transposed products that the truncation bug could corrupt."""

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]),
            ([[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]),
            ([[1.0, 2.0, 3.0]], [[4.0], [5.0], [6.0]]),
            (
                [[2.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
                [[1.0, 4.0, 7.0], [2.0, 5.0, 8.0], [3.0, 6.0, 9.0]],
            ),
            (
                [[0.5, -1.5], [2.5, 0.0], [-0.25, 4.0]],
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            ),
        ],
    )
    def test_transposed_product_matches_reference(self, a: Matrix, b: Matrix) -> None:
        """``A @ B`` and ``Bᵀ @ Aᵀ`` must equal the index-based oracle.

        Both orders read the same entries through ``zip(*b)`` and
        ``zip(row_a, col_b)``, so the trailing-column truncation could
        corrupt either. The ``(A B)ᵀ = Bᵀ Aᵀ`` identity is asserted exactly,
        which fails if either product dropped entries.
        """
        product = mat_mul(a, b)
        expected = _reference_product(a, b)
        assert product == expected
        assert len(product) == len(a)
        assert all(len(row) == len(b[0]) for row in product)

        flipped = mat_mul(transpose(b), transpose(a))
        assert flipped == transpose(product)
        assert flipped == _reference_product(transpose(b), transpose(a))
        assert len(flipped) == len(b[0])
        assert all(len(row) == len(a) for row in flipped)

    def test_gram_product_equals_reference(self) -> None:
        """``Aᵀ @ A`` (used by the SVD driver) must keep every column."""
        a: Matrix = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        gram = mat_mul(transpose(a), a)
        assert gram == _reference_product(transpose(a), a)
        # Entry (0,0) = 1+9+25, (1,1) = 4+16+36, off-diagonal = 2+12+30
        assert gram[0][0] == pytest.approx(35.0)
        assert gram[1][1] == pytest.approx(56.0)
        assert gram[0][1] == gram[1][0] == pytest.approx(44.0)

    def test_result_width_equals_full_right_width(self) -> None:
        """Regression pin for the exact truncation failure mode.

        With a well-formed 2x3 ``b`` the product must have 3 columns. The
        pre-fix code derived the column count from ``zip(*b)``, which could
        only ever produce as many columns as the *shortest* row contributed.
        """
        a: Matrix = [[1.0, 2.0], [3.0, 4.0]]
        b: Matrix = [[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]
        product = mat_mul(a, b)
        # Row 0: 1*5+2*8, 1*6+2*9, 1*7+2*10
        assert product[0] == pytest.approx([21.0, 24.0, 27.0])
        # Row 1: 3*5+4*8, 3*6+4*9, 3*7+4*10
        assert product[1] == pytest.approx([47.0, 54.0, 61.0])
        assert len(product[0]) == len(product[1]) == 3

    def test_identity_products_are_exact(self) -> None:
        a = [[1.5, -2.25, 3.0], [0.5, 7.0, -1.0], [2.0, 2.0, 2.0]]
        eye = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        assert mat_mul(a, eye) == a
        assert mat_mul(eye, a) == a
        assert mat_mul(transpose(a), a) == _reference_product(transpose(a), a)
