"""Singular Value Decomposition — one-sided Jacobi via the Gram matrix.

Factorizes an m×n real matrix ``A`` into ``A = U · Σ · Vᵀ`` with full
orthogonal factors. The algorithm orthogonalizes the columns of ``A``
through two-sided Jacobi rotations applied to the Gram matrix
``B = Aᵀ A``: each rotation is a plane rotation chosen to annihilate one
symmetric pair of off-diagonal entries of ``B`` and is accumulated into
``V``. Sweeps repeat until the off-diagonal Frobenius mass of ``B``
drops below ``tol`` (scaled by the total mass of ``B``) or until
``max_sweeps`` is exhausted. Singular values are the column norms of the
rotated columns ``A·V``, sorted descending; ``U`` is obtained by
normalizing those columns (re-orthogonalized with Gram-Schmidt for
stability) and, when ``m > n`` or when ``A`` is rank-deficient,
completed to a full m×m orthogonal basis via Gram-Schmidt against the
existing columns.

References:
    - Golub, G.H. & Van Loan, C.F. Matrix Computations (4th ed.), sec. 8.6
    - Hestenes, M.R. (1958). "Inversion of Matrices by Biorthogonalization
      and Related Results." Journal of the SIAM 6(1), 51-90.
    - Rutishauser, H. (1966). "The Jacobi method for real symmetric
      matrices." Numerische Mathematik 9(1), 1-10.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from cds.core._numeric import NEAR_ZERO
from cds.math_utils.linalg import Matrix, Vector, identity, mat_mul, transpose

__all__ = ["SVDResult", "condition_number", "rank", "svd"]


@dataclass(frozen=True)
class SVDResult:
    """Full singular value decomposition ``A = U · diag(S) · Vt``.

    With ``k = min(m, n)`` the rectangular diagonal ``Σ`` (m×n) embeds
    ``singular_values`` on its leading diagonal, so ``A`` is reconstructed
    by ``U @ Sigma @ Vt``.

    Attributes:
        U: m×m orthogonal matrix; columns are left singular vectors.
        singular_values: k singular values in descending order.
        Vt: n×n orthogonal matrix, transposed; rows are right singular
            vectors.
    """

    U: Matrix
    singular_values: list[float]
    Vt: Matrix


def _validate_matrix(a: Matrix) -> tuple[int, int]:
    """Validate a rectangular matrix and return its ``(rows, cols)`` shape.

    Raises:
        ValueError: if the matrix is empty, has empty rows, or is ragged.
    """
    if not a:
        raise ValueError("matrix must be non-empty")
    cols = len(a[0])
    if cols == 0:
        raise ValueError("matrix must have at least one column")
    if any(len(row) != cols for row in a):
        raise ValueError("matrix rows must all have the same length")
    return len(a), cols


def _off_diagonal_norm(b: Matrix) -> float:
    """Frobenius norm of the off-diagonal part of a symmetric matrix."""
    n = len(b)
    return math.sqrt(sum(b[i][j] ** 2 for i in range(n) for j in range(n) if i != j))


def _rotate(b: Matrix, v: Matrix, p: int, q: int) -> None:
    """Apply one Jacobi rotation in-place to ``B`` and accumulate it into ``V``.

    Picks the plane rotation ``J`` acting in the ``(p, q)`` plane that
    zeroes ``B[p][q]`` and applies ``B <- Jᵀ B J`` and ``V <- V J``. The
    half-angle form ``theta = atan2(-2 B[p][q], B[p][p] - B[q][q]) / 2``
    is overflow-safe and needs no special casing of degenerate blocks
    (a zero ``B[p][q]`` yields a no-op or a harmless column swap).
    """
    theta = 0.5 * math.atan2(-2.0 * b[p][q], b[p][p] - b[q][q])
    c = math.cos(theta)
    s = math.sin(theta)
    n = len(b)
    for k in range(n):
        if k != p and k != q:
            bkp = b[k][p]
            bkq = b[k][q]
            b[k][p] = c * bkp - s * bkq
            b[p][k] = b[k][p]
            b[k][q] = s * bkp + c * bkq
            b[q][k] = b[k][q]
    app = b[p][p]
    aqq = b[q][q]
    apq = b[p][q]
    two_sc = 2.0 * s * c
    b[p][p] = c * c * app - two_sc * apq + s * s * aqq
    b[q][q] = s * s * app + two_sc * apq + c * c * aqq
    b[p][q] = 0.0
    b[q][p] = 0.0
    for i in range(n):
        vip = v[i][p]
        viq = v[i][q]
        v[i][p] = c * vip - s * viq
        v[i][q] = s * vip + c * viq


def _orthogonalize(col: Vector, basis: list[Vector]) -> Vector:
    """Subtract from ``col`` its projection onto every vector in ``basis``."""
    res = col[:]
    for u in basis:
        proj = sum(r * x for r, x in zip(res, u))
        res = [r - proj * x for r, x in zip(res, u)]
    return res


def _next_basis_column(partial: list[Vector], dim: int) -> Vector:
    """First canonical unit vector completing ``partial`` to an orthogonal set.

    Scans ``e_0 .. e_{dim-1}`` in order, orthogonalizes each candidate
    against the placed columns and returns the first whose residual is
    non-degenerate, normalized. Zero placeholders in ``partial`` are inert
    under projection, so not-yet-filled slots need no special handling.

    Raises:
        AssertionError: if the complement of the placed columns is empty
            (unreachable while fewer than ``dim`` independent columns are
            placed).
    """
    for j in range(dim):
        e = [0.0] * dim
        e[j] = 1.0
        res = _orthogonalize(e, partial)
        norm = math.sqrt(sum(x * x for x in res))
        if norm > NEAR_ZERO:
            return [x / norm for x in res]
    raise AssertionError("unreachable: complement of placed columns is empty")  # pragma: no cover


def svd(a: Matrix, *, tol: float = 1e-12, max_sweeps: int = 50) -> SVDResult:
    """Singular Value Decomposition via one-sided Jacobi rotations.

    Args:
        a: m×n real matrix (``m, n >= 1``) as a rectangular nested list.
        tol: convergence tolerance; sweeping stops once the off-diagonal
            Frobenius mass of the Gram matrix ``Aᵀ A`` falls below ``tol``
            scaled by the total mass of the Gram matrix.
        max_sweeps: maximum number of full Jacobi sweeps over all index
            pairs.

    Returns:
        SVDResult carrying full factors — ``U`` (m×m), ``singular_values``
        (descending, length ``min(m, n)``) and ``Vt`` (n×n) — such that
        ``A = U · diag(S) · Vt``.

    Raises:
        ValueError: if ``a`` is empty or ragged, ``tol <= 0`` or
            ``max_sweeps < 1``.
    """
    m, n = _validate_matrix(a)
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    if max_sweeps < 1:
        raise ValueError("max_sweeps must be at least 1")

    b = mat_mul(transpose(a), a)
    v = identity(n)
    gram_mass = math.sqrt(sum(x * x for row in b for x in row))
    threshold = tol * max(1.0, gram_mass)
    for _ in range(max_sweeps):
        if _off_diagonal_norm(b) <= threshold:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                _rotate(b, v, p, q)

    w = mat_mul(a, v)
    svals = [math.sqrt(sum(w[i][k] ** 2 for i in range(m))) for k in range(n)]
    order = sorted(range(n), key=lambda k: -svals[k])

    k_dim = min(m, n)
    floor = NEAR_ZERO * max(1.0, max(svals))
    cols: list[Vector] = [[0.0] * m for _ in range(k_dim)]
    pending: list[int] = []
    for pos in range(k_dim):
        k = order[pos]
        if svals[k] <= floor:
            pending.append(pos)
            continue
        col = _orthogonalize([w[i][k] for i in range(m)], cols)
        col_norm = math.sqrt(sum(x * x for x in col))
        cols[pos] = [x / col_norm for x in col]
    for pos in pending:
        cols[pos] = _next_basis_column(cols, m)
    for _ in range(m - k_dim):
        cols.append(_next_basis_column(cols, m))

    u = [[cols[j][i] for j in range(m)] for i in range(m)]
    values = [svals[order[pos]] for pos in range(k_dim)]
    # `order` permutes the Jacobi output into descending singular value, and it
    # must be applied to V as well as to U and the values. Returning
    # `transpose(v)` unpermuted left row j of Vt paired with the wrong singular
    # value, so `A = U @ diag(s) @ Vt` failed whenever the Jacobi sweep did not
    # happen to produce descending order — e.g. `diag(1, 5, 3)` reconstructed as
    # `[[0,0,1],[5,0,0],[0,3,0]]`. Random matrices masked it because the sweep
    # usually orders the columns already. `order` is a permutation of all n
    # indices, so this also fixes the trailing columns beyond `k_dim`.
    v_sorted = [[v[i][order[j]] for j in range(n)] for i in range(n)]
    return SVDResult(U=u, singular_values=values, Vt=transpose(v_sorted))


def rank(a: Matrix, *, tol: float = 1e-10) -> int:
    """Number of singular values greater than ``tol`` (the matrix rank).

    Args:
        a: m×n real matrix.
        tol: counting threshold for singular values. Because the Jacobi
            iteration works on the squared Gram matrix, numerically zero
            singular values can surface at the ``sqrt(eps)`` level of the
            largest one; pass a larger ``tol`` to classify such matrices
            as rank-deficient.

    Returns:
        An integer between ``0`` and ``min(m, n)``.

    Raises:
        ValueError: if ``a`` is empty or ragged, or ``tol <= 0``.
    """
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    return sum(1 for s in svd(a).singular_values if s > tol)


def condition_number(a: Matrix, *, tol: float = 1e-12) -> float:
    """Ratio ``s_max / s_min`` of the extreme singular values.

    Args:
        a: m×n real matrix.
        tol: smallest singular values at or below this threshold are
            treated as zero, yielding ``math.inf``.

    Returns:
        The 2-norm condition number; ``math.inf`` when ``a`` is
        rank-deficient (smallest singular value <= ``tol``).

    Raises:
        ValueError: if ``a`` is empty or ragged, or ``tol <= 0``.
    """
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    values = svd(a).singular_values
    smallest = values[-1]
    if smallest <= tol:
        return math.inf
    return values[0] / smallest
