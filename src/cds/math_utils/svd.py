"""Singular Value Decomposition — one-sided Jacobi on the columns of A.

Factorizes an m×n real matrix ``A`` into ``A = U · Σ · Vᵀ`` with full
orthogonal factors. The algorithm is Hestenes' one-sided Jacobi method: it
orthogonalizes the columns of ``A`` *directly*, choosing each plane rotation
from the inner products of the working columns ``W = A·V`` and accumulating
the rotations into ``V``. It never forms the Gram matrix ``Aᵀ A``: squaring
the matrix squares its condition number, which buries small singular values
in ``sqrt(eps)`` rounding noise. Working on the columns instead gives the
computed singular values (the column norms of ``A·V``) relative accuracy on
the order of machine epsilon, even when they are many orders of magnitude
below the largest one. Sweeps repeat until every column pair is orthogonal
to ``tol`` (relative to the column norms) or until ``max_sweeps`` is
exhausted. Singular values are sorted descending; ``U`` is obtained by
normalizing the rotated columns (re-orthogonalized with Gram-Schmidt for
stability) and, when ``m > n`` or when ``A`` is rank-deficient, completed to a
full m×m orthogonal basis via Gram-Schmidt against the existing columns.

References:
    - Golub, G.H. & Van Loan, C.F. Matrix Computations (4th ed.), sec. 8.6
    - Hestenes, M.R. (1958). "Inversion of Matrices by Biorthogonalization
      and Related Results." Journal of the SIAM 6(1), 51-90.
    - Rutishauser, H. (1966). "The Jacobi method for real symmetric
      matrices." Numerische Mathematik 9(1), 1-10.
    - Demmel, J. & Veselić, K. (1992). "Jacobi's Method is More Accurate
      than QR." SIAM J. Matrix Anal. Appl. 13(4), 1204-1245.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from cds.core._numeric import NEAR_ZERO
from cds.math_utils.linalg import Matrix, Vector, identity, transpose

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


def _column_rotation(w: Matrix, v: Matrix, p: int, q: int, tol: float) -> bool:
    """Orthogonalize columns ``p`` and ``q`` of ``W`` with one Jacobi rotation.

    This is the one-sided (Hestenes) step: the rotation is computed from the
    column inner products of ``W = A·V`` itself and applied to ``W`` and
    ``V`` in place. The Gram matrix ``Aᵀ A`` is never formed, so the condition
    number is not squared and small singular values keep relative accuracy.

    The rotation angle solves ``gamma·t² + (beta - alpha)·t - gamma = 0`` with
    ``alpha = ||W_p||²``, ``beta = ||W_q||²``, ``gamma = <W_p, W_q>``; the
    smaller root ``t = sign(zeta)/(|zeta| + sqrt(1 + zeta²))``,
    ``zeta = (beta - alpha)/(2·gamma)``, is the numerically stable choice.
    ``math.hypot`` keeps the square roots overflow-safe for extreme
    column-norm ratios.

    Returns:
        True if a rotation was applied; False if the pair is already
        orthogonal to ``tol`` (relative to the column norms) or a column is
        zero (a zero column is orthogonal to everything).
    """
    m = len(w)
    alpha = sum(w[i][p] * w[i][p] for i in range(m))
    beta = sum(w[i][q] * w[i][q] for i in range(m))
    gamma = sum(w[i][p] * w[i][q] for i in range(m))
    if gamma == 0.0:
        return False
    # gamma != 0 implies both columns are nonzero (Cauchy-Schwarz), so the
    # norm product below is positive and the relative test is well-defined.
    if abs(gamma) <= tol * math.sqrt(alpha) * math.sqrt(beta):
        return False
    zeta = (beta - alpha) / (2.0 * gamma)
    t = math.copysign(1.0, zeta) / (abs(zeta) + math.hypot(1.0, zeta))
    c = 1.0 / math.hypot(1.0, t)
    s = c * t
    for i in range(m):
        wp, wq = w[i][p], w[i][q]
        w[i][p] = c * wp - s * wq
        w[i][q] = s * wp + c * wq
    for i in range(len(v)):
        vp, vq = v[i][p], v[i][q]
        v[i][p] = c * vp - s * vq
        v[i][q] = s * vp + c * vq
    return True


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
        tol: convergence tolerance; a column pair counts as orthogonal once
            ``|<w_p, w_q>| <= tol * ||w_p|| * ||w_q||``. Sweeping stops when
            every pair passes or ``max_sweeps`` is exhausted.
        max_sweeps: maximum number of full Jacobi sweeps over all index
            pairs.

    Returns:
        SVDResult carrying full factors — ``U`` (m×m), ``singular_values``
        (descending, length ``min(m, n)``) and ``Vt`` (n×n) — such that
        ``A = U · diag(S) · Vt``.

    Raises:
        ValueError: if ``a`` is empty or ragged, ``tol <= 0`` or
            ``max_sweeps < 1``.

    Warning:
        Condition-number limit. One-sided Jacobi does not square the
        condition number (it never forms ``Aᵀ A``), so small singular values
        keep *relative* accuracy far better than a Gram-matrix approach.
        The limit is still floating point: the relative error of the
        smallest singular values grows roughly like ``eps * kappa_2(A)``.
        Once ``kappa_2(A)`` approaches ``1/eps`` (~4.5e15), trailing
        singular values sit at the rounding-noise floor and must be treated
        as zero — use :func:`rank` / :func:`condition_number` with an
        explicit ``tol`` for that classification instead of reading the raw
        trailing values.
    """
    m, n = _validate_matrix(a)
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    if max_sweeps < 1:
        raise ValueError("max_sweeps must be at least 1")

    # Hestenes one-sided Jacobi: rotate the columns of A itself (W = A·V),
    # accumulating the rotations into V. No Gram matrix is formed anywhere.
    w = [row[:] for row in a]
    v = identity(n)
    for _ in range(max_sweeps):
        rotated = False
        for p in range(n - 1):
            for q in range(p + 1, n):
                if _column_rotation(w, v, p, q, tol):
                    rotated = True
        if not rotated:
            break

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
        tol: counting threshold for singular values. The one-sided Jacobi
            iteration works on the columns of ``A`` directly (no squared
            Gram matrix), so numerically zero singular values surface near
            the ``eps`` level of the largest one rather than ``sqrt(eps)``;
            pass a larger ``tol`` to classify such matrices as
            rank-deficient. For condition numbers approaching ``1/eps``
            (~4.5e15) the trailing values are rounding noise and an explicit
            ``tol`` is the only reliable classifier.

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
        rank-deficient (smallest singular value <= ``tol``). Values beyond
        ~``1/eps`` (~4.5e15) are not meaningful in double precision: the
        underlying smallest singular value is then at the rounding-noise
        floor (see the condition-number warning in :func:`svd`).

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
