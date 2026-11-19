"""Closed-form and published-reference oracles for the one-sided Jacobi SVD.

The previous implementation diagonalized the Gram matrix ``Aᵀ A``, which
squares the condition number: small singular values surfaced with absolute
error ~sqrt(eps)·σ_max and were unusable. These oracles pin the accuracy the
one-sided (Hestenes) method must deliver — *relative* accuracy on tiny
singular values — using only closed forms and a published example, no
third-party numerical library.
"""

from __future__ import annotations

import math

import pytest

from cds.math_utils.linalg import Matrix, mat_mul, transpose
from cds.math_utils.svd import SVDResult, condition_number, rank, svd


def _givens(n: int, p: int, q: int, theta: float) -> Matrix:
    g = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    c, s = math.cos(theta), math.sin(theta)
    g[p][p] = c
    g[p][q] = -s
    g[q][p] = s
    g[q][q] = c
    return g


def _reconstruct(a: Matrix, result: SVDResult) -> Matrix:
    m, n = len(a), len(a[0])
    values = result.singular_values
    sigma = [[values[j] if i == j < len(values) else 0.0 for j in range(n)] for i in range(m)]
    return mat_mul(mat_mul(result.U, sigma), result.Vt)


class TestClosedFormOracles:
    def test_ill_conditioned_2x2_relative_accuracy(self) -> None:
        # A = [[1, 1], [0, 1e-10]]: sigma_max^2 + sigma_min^2 = ||A||_F^2 and
        # sigma_max * sigma_min = |det A|, so sigma_min = |det A| / sigma_max
        # is a stable closed form (the quadratic formula cancels here).
        a = [[1.0, 1.0], [0.0, 1e-10]]
        sigma_max = math.sqrt(2.0 + 1e-20)
        sigma_min = 1e-10 / sigma_max
        result = svd(a)
        assert result.singular_values[0] == pytest.approx(sigma_max, rel=1e-12)
        # A Gram-matrix method loses this value to ~1e-8 absolute noise
        # (>100x relative error); one-sided Jacobi must keep it relative.
        assert result.singular_values[1] == pytest.approx(sigma_min, rel=1e-12)
        rec = _reconstruct(a, result)
        assert rec[1][1] == pytest.approx(1e-10, rel=1e-9)

    def test_moderately_ill_conditioned_2x2(self) -> None:
        a = [[1.0, 1.0], [0.0, 1e-8]]
        sigma_max = math.sqrt(2.0 + 1e-16)
        sigma_min = 1e-8 / sigma_max
        result = svd(a)
        assert result.singular_values[0] == pytest.approx(sigma_max, rel=1e-12)
        assert result.singular_values[1] == pytest.approx(sigma_min, rel=1e-12)

    def test_extreme_column_scales_do_not_overflow(self) -> None:
        # det = 1 exactly; sigma_max = sqrt(2)*1e150, sigma_min = 1/sigma_max.
        # The rotation math must survive column norms spanning 300 orders.
        a = [[1e150, 1e150], [0.0, 1e-150]]
        sigma_max = math.sqrt(2.0) * 1e150
        sigma_min = 1.0 / sigma_max
        result = svd(a)
        assert result.singular_values[0] == pytest.approx(sigma_max, rel=1e-12)
        assert result.singular_values[1] == pytest.approx(sigma_min, rel=1e-12)

    def test_diagonal_with_huge_dynamic_range_is_exact(self) -> None:
        a = [[1e150, 0.0], [0.0, 1e-150]]
        result = svd(a)
        assert result.singular_values == [1e150, 1e-150]


class TestPublishedReference:
    def test_wikipedia_svd_example(self) -> None:
        # The worked example from the Wikipedia "Singular value decomposition"
        # article: AᵀA has eigenvalues 9, 5, 4, 0, 0, so the singular values
        # are 3, sqrt(5), 2, 0 (k = min(4, 5) = 4).
        a = [
            [1.0, 0.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, 0.0],
        ]
        result = svd(a)
        assert result.singular_values == pytest.approx([3.0, math.sqrt(5.0), 2.0, 0.0], abs=1e-12)
        rec = _reconstruct(a, result)
        for i in range(4):
            assert rec[i] == pytest.approx(a[i], abs=1e-12)


class TestConstructedSVD:
    def test_rotated_diagonal_benign_spectrum(self) -> None:
        # A = U0 diag(s) V0^T with known orthogonal U0/V0 built from Givens
        # rotations: the singular values are exactly `values` by construction.
        values = [3.0, 1.0, 0.5]
        n = len(values)
        u0 = mat_mul(_givens(n, 0, 1, 0.3), _givens(n, n - 2, n - 1, 0.7))
        v0 = mat_mul(_givens(n, 0, 1, -0.2), _givens(n, n - 2, n - 1, 1.1))
        sigma = [[values[j] if i == j else 0.0 for j in range(n)] for i in range(n)]
        a = mat_mul(mat_mul(u0, sigma), transpose(v0))
        result = svd(a)
        for computed, expected in zip(result.singular_values, values):
            assert computed == pytest.approx(expected, rel=1e-12)
        rec = _reconstruct(a, result)
        for i in range(n):
            assert rec[i] == pytest.approx(a[i], abs=1e-12)

    def test_graded_rotation_2x2_relative_accuracy(self) -> None:
        # Demmel-Veselic setting A = D1 * R * D2: every entry is a single
        # product (no cancellation when forming A), so the tiny singular
        # value lives in tiny entries and is resolvable. Closed form via
        # sigma_max*sigma_min = |det A| and sigma_max ~ ||A||_F.
        d1, d2, theta = [1.0, 1e-8], [1.0, 1e-4], 0.3
        r = _givens(2, 0, 1, theta)
        a = [[d1[i] * r[i][j] * d2[j] for j in range(2)] for i in range(2)]
        det = abs(a[0][0] * a[1][1] - a[0][1] * a[1][0])
        sigma_max = math.sqrt(sum(x * x for row in a for x in row))
        sigma_min = det / sigma_max
        result = svd(a)
        assert result.singular_values[0] == pytest.approx(sigma_max, rel=1e-12)
        # A Gram-matrix method would bury this ~1e-12 value under ~1e-8
        # absolute noise; one-sided Jacobi keeps it to machine relative.
        assert result.singular_values[1] == pytest.approx(sigma_min, rel=1e-12)

    def test_graded_rotation_3x3_closed_form_identities(self) -> None:
        # A = D1 * R * D2 with moderate grading (kappa ~ 1e12, still below
        # 1/eps). For 3x3 the stable closed forms are the identities
        #   sigma1*sigma2*sigma3 = |det A|,  sigma1^2+sigma2^2+sigma3^2 = ||A||_F^2
        # with det A = prod(D1)*prod(D2)*det(R) = prod(D1)*prod(D2).
        d1 = d2 = [1.0, 1e-2, 1e-4]
        r = mat_mul(_givens(3, 0, 1, 0.3), _givens(3, 1, 2, 0.7))
        a = [[d1[i] * r[i][j] * d2[j] for j in range(3)] for i in range(3)]
        det = math.prod(d1) * math.prod(d2)
        fro_sq = sum(x * x for row in a for x in row)
        result = svd(a)
        s = result.singular_values
        assert s[0] * s[1] * s[2] == pytest.approx(det, rel=1e-9)
        assert sum(x * x for x in s) == pytest.approx(fro_sq, rel=1e-12)
        assert s == sorted(s, reverse=True)
        rec = _reconstruct(a, result)
        for i in range(3):
            assert rec[i] == pytest.approx(a[i], abs=1e-12)

    def test_scaled_permutation_exact_spectrum(self) -> None:
        # A = D1 * P * D2 with P a permutation: one nonzero per row/column,
        # so the singular values are exactly the sorted nonzero magnitudes.
        d1, d2 = [1.0, 1e-6, 1e-12], [1.0, 1e-3, 1e-6]
        perm = [2, 0, 1]
        a = [[0.0] * 3 for _ in range(3)]
        expected = []
        for i in range(3):
            value = d1[i] * d2[perm[i]]
            a[i][perm[i]] = value
            expected.append(value)
        result = svd(a)
        assert result.singular_values == pytest.approx(sorted(expected, reverse=True), rel=1e-13)


class TestRankConditionOnGradedMatrices:
    def test_rank_sees_the_tiny_singular_value(self) -> None:
        a = [[1.0, 1.0], [0.0, 1e-10]]
        assert rank(a, tol=1e-14) == 2
        assert rank(a, tol=1e-6) == 1

    def test_condition_number_matches_closed_form(self) -> None:
        a = [[1.0, 1.0], [0.0, 1e-10]]
        sigma_max = math.sqrt(2.0 + 1e-20)
        expected = sigma_max / (1e-10 / sigma_max)
        assert condition_number(a, tol=1e-14) == pytest.approx(expected, rel=1e-10)
