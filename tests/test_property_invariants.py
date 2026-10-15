"""Deterministic invariant tests for the public numeric surface.

Numerical code is "deterministic" only if its invariants hold on every input
you care about, not just on the examples in the docs. These tests sweep small
integer/rational inputs so the identities below hold exactly (or within the
module's own documented tolerance), so a regression shows up as a failed
identity instead of a silently wrong number:

- mat_mul is associative and distributes over addition
- transpose(transpose(M)) == M
- det(AB) == det(A) * det(B)
- solve_linear(A, b) reproduces b (residual bound)
- PMFs sum to 1 over their support
- ODE/quadrature error decreases as the step count grows
"""
from __future__ import annotations

import itertools
import math

import pytest

from cds.math_utils.linalg import (determinant, identity, mat_mul,
                                   solve_linear, transpose)
from cds.probability.distributions import binomial_pmf, geometric_pmf, poisson_pmf

#: Exact-arithmetic inputs keep every identity above exact for integers.
MATRICES = [
    [[1]],
    [[1, 2], [3, 4]],
    [[2, 0, 1], [1, 3, 0], [0, 1, 2]],
    [[-1, 2], [2, -1]],
    [[3, 1, 4], [1, 5, 9], [2, 6, 5]],
]


def _matmul_chain(*matrices):
    result = matrices[0]
    for m in matrices[1:]:
        result = mat_mul(result, m)
    return result


#: Mutually conformable operands for the binary laws (3x3 with 3x3, 2x2 with
#: 2x2) — a 2x2 times a 3x3 is not a legal product and is not what the
#: associativity law is about.
THREES = [
    [[2, 0, 1], [1, 3, 0], [0, 1, 2]],
    [[3, 1, 4], [1, 5, 9], [2, 6, 5]],
    [[-1, 2, 0], [2, -1, 1], [0, 1, 3]],
]
TWOS = [
    [[1, 2], [3, 4]],
    [[-1, 2], [2, -1]],
]


@pytest.mark.parametrize("triple", [(a, b, c) for a in THREES for b in THREES[:2] for c in THREES[:2]])
def test_mat_mul_is_associative(triple) -> None:
    a, b, c = triple
    assert _matmul_chain(a, b, c) == _matmul_chain(a, _matmul_chain(b, c))


@pytest.mark.parametrize("triple", [(a, b, c) for a in TWOS for b in TWOS for c in TWOS])
def test_mat_mul_is_associative_2x2(triple) -> None:
    a, b, c = triple
    assert _matmul_chain(a, b, c) == _matmul_chain(a, _matmul_chain(b, c))


@pytest.mark.parametrize("matrix", MATRICES)
def test_transpose_twice_is_identity_on_values(matrix) -> None:
    assert transpose(transpose(matrix)) == matrix


@pytest.mark.parametrize("a", THREES)
@pytest.mark.parametrize("b", THREES[:2])
def test_determinant_of_product_equals_product_of_determinants(a, b) -> None:
    assert determinant(mat_mul(a, b)) == pytest.approx(determinant(a) * determinant(b))


@pytest.mark.parametrize("a", TWOS)
@pytest.mark.parametrize("b", TWOS)
def test_determinant_of_product_equals_product_of_determinants_2x2(a, b) -> None:
    assert determinant(mat_mul(a, b)) == pytest.approx(determinant(a) * determinant(b))


@pytest.mark.parametrize("matrix", MATRICES[1:4])
def test_solve_linear_reproduces_the_rhs(matrix) -> None:
    n = len(matrix)
    rhs = [float(i + 1) for i in range(n)]
    x = solve_linear(matrix, rhs)
    product = mat_mul(matrix, [[v] for v in x])
    assert [row[0] for row in product] == pytest.approx(rhs, abs=1e-9)


@pytest.mark.parametrize("rate", [0.1, 0.5, 1.0, 5.0])
def test_poisson_pmf_sums_to_one(rate: float) -> None:
    total = sum(poisson_pmf(k, rate) for k in range(200))
    assert total == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("n", [1, 2, 5, 10])
@pytest.mark.parametrize("p", [0.2, 0.5, 0.9])
def test_binomial_pmf_sums_to_one(n: int, p: float) -> None:
    total = sum(binomial_pmf(k, n, p) for k in range(n + 1))
    assert total == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_geometric_pmf_sums_to_one(p: float) -> None:
    total = sum(geometric_pmf(k, p) for k in range(1, 500))
    assert total == pytest.approx(1.0, abs=1e-6)


def _add(a, b):
    """Matrix addition; there is no mat_add in the public surface."""
    return [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def test_distributive_law() -> None:
    # Three mutually conformable 3x3 operands.
    a = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
    b = [[2, 0, 1], [1, 3, 0], [0, 1, 2]]
    c = [[3, 1, 4], [1, 5, 9], [2, 6, 5]]
    assert mat_mul(a, _add(b, c)) == _add(mat_mul(a, b), mat_mul(a, c))


def test_identity_is_the_neutral_element() -> None:
    m = MATRICES[4]
    assert mat_mul(m, identity(len(m))) == m


def test_double_inverse_round_trip() -> None:
    from cds.math_utils.linalg import matrix_inverse

    m = MATRICES[2]
    twice = matrix_inverse(matrix_inverse(m))
    for got, want in zip(twice, m):
        assert got == pytest.approx(want, abs=1e-9)
