"""Scale-relative singularity classification for ``determinant``.

The old implementation delegated to ``lu_decomposition``, whose pivot
threshold was relative to the *global* max entry: a well-conditioned
mixed-scale matrix like ``diag(1e-20, 1e20)`` (determinant 1) was rejected as
"nearly singular" and reported 0.0. The classification is now relative to
each column's own original scale, and the behavior is documented and pinned
here, including the documented nearly-singular -> 0.0 classification and the
sub-normal underflow guard.
"""

from __future__ import annotations

import pytest

from cds.math_utils.linalg import determinant, mat_mul


class TestScaleRelativeClassification:
    def test_mixed_scale_well_conditioned_not_singular(self) -> None:
        # kappa = 1, determinant exactly 1; the old global-scale threshold
        # (1e-15 * 1e20 = 1e5) rejected the 1e-20 pivot and returned 0.0.
        assert determinant([[1e-20, 0.0], [0.0, 1e20]]) == pytest.approx(1.0, rel=1e-12)

    def test_uniform_tiny_scale(self) -> None:
        assert determinant([[1e-20, 0.0], [0.0, 1e-20]]) == pytest.approx(1e-40, rel=1e-12)

    def test_uniform_huge_scale(self) -> None:
        assert determinant([[1e20, 0.0], [0.0, 1e20]]) == pytest.approx(1e40, rel=1e-12)

    def test_column_rescaling_scales_the_determinant(self) -> None:
        a = [[2.0, 1.0], [1.0, 3.0]]  # det = 5
        scaled = [[row[0] * 1e-18, row[1]] for row in a]
        assert determinant(scaled) == pytest.approx(5e-18, rel=1e-12)

    def test_global_rescaling_scales_the_determinant(self) -> None:
        a = [[2.0, 1.0], [1.0, 3.0]]
        scaled = [[x * 1e-10 for x in row] for row in a]
        assert determinant(scaled) == pytest.approx(5e-20, rel=1e-12)

    def test_subnormal_column_scale_underflow_guard(self) -> None:
        # NEAR_ZERO * 1e-320 underflows to 0.0; the non-zero pivot must still
        # pass (and a literal zero pivot must still be rejected, below).
        assert determinant([[1e-320, 0.0], [0.0, 1e100]]) == pytest.approx(1e-220, rel=1e-12)

    def test_zero_column_at_mixed_scale_is_singular(self) -> None:
        assert determinant([[1e-20, 0.0], [0.0, 0.0]]) == 0.0

    def test_exactly_singular_is_zero(self) -> None:
        assert determinant([[1.0, 2.0], [2.0, 4.0]]) == 0.0
        assert determinant([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 1.0, 1.0]]) == 0.0

    def test_nearly_singular_classified_zero_documented(self) -> None:
        # Pivot 1e-16 < 1e-15 * column scale (~4): classified numerically
        # zero by the documented scale-relative rule.
        assert determinant([[1.0, 2.0], [2.0, 4.0 + 1e-16]]) == 0.0

    def test_ill_conditioned_but_resolvable_is_nonzero_estimate(self) -> None:
        # Pivot 1e-12 is above the 1e-15 column-relative threshold: the
        # matrix is not classified singular; the product is an estimate.
        det = determinant([[1.0, 1.0], [1.0, 1.0 + 1e-12]])
        assert det == pytest.approx(1e-12, rel=1e-3)
        assert det > 0.0


class TestSignAndValues:
    def test_row_swap_sign(self) -> None:
        assert determinant([[0.0, 1.0], [1.0, 0.0]]) == pytest.approx(-1.0)

    def test_three_by_three_permutation_signs(self) -> None:
        even = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
        odd = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        assert determinant(even) == pytest.approx(1.0)
        assert determinant(odd) == pytest.approx(-1.0)

    def test_known_three_by_three(self) -> None:
        a = [[6.0, 1.0, 1.0], [4.0, -2.0, 5.0], [2.0, 8.0, 7.0]]
        assert determinant(a) == pytest.approx(-306.0, rel=1e-12)

    def test_identity_skips_zero_factors(self) -> None:
        assert determinant([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) == 1.0

    def test_product_law_under_uniform_rescaling(self) -> None:
        # Uniform scaling keeps the product well-conditioned (kappa ~ 2), so
        # the classification is unchanged and det(AB) = det(A)det(B) holds to
        # rounding. (A mixed-scale product like diag(1e-10,1e10) @ B would be
        # genuinely ill-conditioned, kappa ~ 1e20, and is classified zero by
        # the documented rule instead.)
        a = [[1e-10, 0.0], [0.0, 1e-10]]
        b = [[2.0, 1.0], [1.0, 3.0]]
        assert determinant(mat_mul(a, b)) == pytest.approx(
            determinant(a) * determinant(b), rel=1e-12
        )
