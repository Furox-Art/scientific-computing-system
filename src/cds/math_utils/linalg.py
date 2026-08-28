"""Linear algebra — pure Python, no dependencies.

References:
    - Golub, G.H. & Van Loan, C.F. Matrix Computations (4th ed.)
    - Trefethen, L.N. & Bau, D. Numerical Linear Algebra.
    - Von Mises, R. (1929). Praktische Verfahren der Gleichungsauflösung (power iteration).
    - Householder, A.S. (1958). Unitary triangularization of a nonsymmetric
      matrix. JACM 5(4), 339-342.
    - Cholesky, A.-L. / Benoît (1924). Méthode de résolution des équations
      normales (Bulletin Géodésique, 2, 67-77).
"""

from __future__ import annotations

import math

from cds.core._numeric import NEAR_ZERO, NEWTON_TOLERANCE

Matrix = list[list[float]]
Vector = list[float]


def dot(a: Vector, b: Vector) -> float:
    """Inner product of two equal-length vectors.

    Raises:
        ValueError: if `a` and `b` have different lengths.
    """
    if len(a) != len(b):
        raise ValueError(f"vectors a and b must have the same length (got {len(a)} and {len(b)})")
    return sum(x * y for x, y in zip(a, b))


def mat_mul(a: Matrix, b: Matrix) -> Matrix:
    """Matrix multiplication A * B.

    Pre-transposes B so that columns are read as contiguous rows, which
    keeps memory access row-major and lets the inner loops run over
    Python's C-implemented ``zip``/``sum`` rather than indexed lookups.
    """
    rows_a, cols_a = len(a), len(a[0])
    rows_b, cols_b = len(b), len(b[0])
    if cols_a != rows_b:
        raise ValueError(f"incompatible shapes: {rows_a}x{cols_a} and {rows_b}x{cols_b}")

    # Transpose B once up front: each output column becomes a row we can
    # iterate cheaply, instead of striding through B column-by-column.
    b_T = list(zip(*b))

    return [[sum(ai * bi for ai, bi in zip(row_a, col_b)) for col_b in b_T] for row_a in a]


def transpose(m: Matrix) -> Matrix:
    """Return the transpose of a 2-D matrix (rows <-> columns)."""
    if not m:
        return []
    return [[m[i][j] for i in range(len(m))] for j in range(len(m[0]))]


def determinant(m: Matrix) -> float:
    """Compute matrix determinant using PLU decomposition (O(N^3)).

    Avoids the O(N!) complexity of minor expansion.
    """
    n = len(m)
    if n == 0:
        return 1.0
    if n == 1:
        return m[0][0]

    try:
        P, L, U = lu_decomposition(m)
    except ValueError:
        # If matrix is singular, determinant is 0
        return 0.0

    # Determinant of LU is product of diag(U)
    # (diag(L) is all 1s).
    det = 1.0
    for i in range(n):
        det *= U[i][i]

    # Determinant of P is (-1)^s where s is number of row swaps.
    # We compute it using cycle decomposition: s = n - number_of_cycles.
    num_cycles = 0
    p_indices = [row.index(1.0) for row in P]
    visited = [False] * n
    for i in range(n):
        if not visited[i]:
            num_cycles += 1
            curr = i
            while not visited[curr]:
                visited[curr] = True
                curr = p_indices[curr]

    return float(det * ((-1) ** (n - num_cycles)))


def identity(n: int) -> Matrix:
    """Create n×n identity matrix."""
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def lu_decomposition(m: Matrix) -> tuple[Matrix, Matrix, Matrix]:
    """LU decomposition with partial pivoting (PA = LU).

    A = P_inv * L * U where P_inv is a permutation matrix,
    L is lower triangular (ones on diagonal) and U is upper triangular.

    Pivoting selects the largest-magnitude entry in each column as the pivot
    (Gaussian elimination with partial pivoting), and the singular-pivot test
    is *relative* to the matrix scale: a pivot is rejected only when it falls
    below ``NEAR_ZERO`` scaled by the largest-magnitude entry of the original
    matrix. A scale-invariant threshold keeps well-conditioned small-magnitude
    matrices (e.g. ``1e-20 * I``) from being misclassified as singular, while
    still flagging genuinely rank-deficient inputs.

    Returns:
        P, L, U matrices.

    Raises:
        ValueError: if matrix is singular
    """
    n = len(m)
    P = identity(n)
    L = [[0.0] * n for _ in range(n)]
    U = [row[:] for row in m]

    # Scale reference: the largest |entry| of A. The relative pivot threshold
    # is derived from this so the singularity test adapts to the matrix scale
    # rather than being a fixed absolute cutoff. When scale > 0 the threshold
    # tracks it (so a well-conditioned 1e-20 matrix is not wrongly rejected);
    # when the matrix is all zeros we fall back to the absolute NEAR_ZERO so
    # the zero matrix is still flagged as singular.
    #
    # The separate ``max_val == 0.0`` arm guards against sub-normal scales: for
    # a matrix like 2e-309 the product ``NEAR_ZERO * scale`` underflows to 0.0,
    # which would make the ``< pivot_tol`` test vacuously false and let an
    # exact 0.0 pivot slip through into a ZeroDivisionError during elimination.
    # A column whose largest remaining entry is literally 0.0 is always rank-
    # deficient, so rejecting it is correct at any scale.
    scale = max((abs(U[i][j]) for i in range(n) for j in range(n)), default=0.0)
    pivot_tol = NEAR_ZERO * scale if scale > 0 else NEAR_ZERO

    for k in range(n):
        # Partial pivoting
        pivot_idx = k
        max_val = abs(U[k][k])
        for i in range(k + 1, n):
            if abs(U[i][k]) > max_val:
                max_val = abs(U[i][k])
                pivot_idx = i

        # Reject a singular pivot. Two cases: an exact 0.0 column-max is
        # always rank-deficient and must be caught directly (when the matrix
        # scale is sub-normal the scaled ``pivot_tol`` can underflow to 0.0,
        # making the ``<`` test vacuous for a literal-zero pivot); otherwise
        # use the scale-relative threshold.
        if max_val == 0.0 or max_val < pivot_tol:
            raise ValueError(
                f"zero pivot at column {k} — the input matrix is singular or nearly singular; try regularizing or checking your data"
            )

        if pivot_idx != k:
            U[k], U[pivot_idx] = U[pivot_idx], U[k]
            P[k], P[pivot_idx] = P[pivot_idx], P[k]
            L[k], L[pivot_idx] = L[pivot_idx], L[k]

        L[k][k] = 1.0
        for i in range(k + 1, n):
            factor = U[i][k] / U[k][k]
            L[i][k] = factor
            for j in range(k, n):
                U[i][j] -= factor * U[k][j]

    return P, L, U


def _lu_solve(P: Matrix, L: Matrix, U: Matrix, b: Vector) -> Vector:
    """Solve ``LUx = Pb`` from a factored PLU, shared by solve/inverse.

    Forward-substitute ``Ly = Pb`` then back-substitute ``Ux = y``. ``P`` is
    applied to ``b`` first. Extracted so :func:`solve_linear` and
    :func:`matrix_inverse` share one substitution implementation instead of
    duplicating the two triangular solves.
    """
    n = len(L)
    # Apply permutation: Pb
    pb = [sum(P[i][j] * b[j] for j in range(n)) for i in range(n)]

    # forward: Ly = Pb
    y = [0.0] * n
    for i in range(n):
        y[i] = pb[i] - sum(L[i][j] * y[j] for j in range(i))

    # backward: Ux = y. The diagonal is guarded even though lu_decomposition
    # already rejects singular inputs: the relative-tolerance rejection there
    # can still leave a tiny pivot that underflows to 0.0 in U on the backward
    # pass, so we surface that as a clear singular-matrix error here. The guard
    # is itself scale-relative (mirroring lu_decomposition): a well-conditioned
    # matrix whose entries are ~1e-20 has U diagonals of the same magnitude,
    # which a fixed NEAR_ZERO would wrongly flag as zero. An exact 0.0 diagonal
    # is rejected directly so a sub-normal scale (where the scaled tolerance
    # underflows to 0.0) cannot slip a literal-zero pivot through.
    u_scale = max((abs(U[i][i]) for i in range(n)), default=0.0)
    u_tol = NEAR_ZERO * u_scale if u_scale > 0 else NEAR_ZERO
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if U[i][i] == 0.0 or abs(U[i][i]) < u_tol:
            raise ValueError(
                f"singular matrix — LU backward substitution failed at row {i}; matrix has no unique inverse"
            )
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]

    return x


def solve_linear(A: Matrix, b: Vector) -> Vector:
    """Solve Ax = b using PLU decomposition.

    Solves LUx = Pb.

    Raises:
        ValueError: if matrix is singular
    """
    P, L, U = lu_decomposition(A)
    return _lu_solve(P, L, U, b)


def matrix_inverse(m: Matrix) -> Matrix:
    """Compute matrix inverse via PLU decomposition.

    Reuses a single P, L, U factorization and solves A * x_i = e_i
    for each column of the identity matrix to build the inverse.

    Raises:
        ValueError: if matrix is singular
    """
    n = len(m)
    P, L, U = lu_decomposition(m)
    inv = [[0.0] * n for _ in range(n)]

    for col in range(n):
        # e is the standard basis vector
        b = [0.0] * n
        b[col] = 1.0
        x = _lu_solve(P, L, U, b)
        for row in range(n):
            inv[row][col] = x[row]

    return inv


def power_iteration(
    m: Matrix,
    max_iter: int = 1000,
    tol: float = NEWTON_TOLERANCE,
) -> tuple[float, Vector]:
    """Find the dominant eigenvalue and eigenvector by power iteration.

    Von Mises iteration (1929): repeatedly apply ``A``, renormalize, and read
    the eigenvalue off the Rayleigh quotient ``vᵗAv / vᵗv``.

    Convergence is decided on the **residual** ``‖Av − λv‖``, not on successive
    Rayleigh quotients agreeing. That distinction matters: the quotient can sit
    still while ``v`` is nowhere near an eigenvector. On ``[[3, 4], [4, -3]]``
    (eigenvalues +5 and −5) the old quotient test exited at iteration two and
    returned 4.0 — a number that is not an eigenvalue at all — because the
    iterate oscillates between two directions instead of converging.

    That oscillation is the method's fundamental limitation, not a bug: power
    iteration needs ``|λ₁| > |λ₂|`` strictly. When the residual test cannot be
    met this function now says so instead of returning a plausible-looking
    wrong answer.

    Args:
        m: square matrix.
        max_iter: iteration limit.
        tol: residual tolerance, scaled by ``max(1, |λ|)``.

    Returns:
        ``(eigenvalue, unit eigenvector)``. If the iterate collapses to zero —
        a nilpotent or zero matrix, where ``A^k v → 0`` — returns ``0.0`` with
        the last iterate, which is the correct dominant eigenvalue there.

    Raises:
        ValueError: if the residual never falls below the tolerance, which
            means ``|λ₁|`` is not strictly dominant (equal-magnitude or complex
            conjugate pair). Use a shifted or deflated iteration instead.
    """
    n = len(m)
    # A non-degenerate deterministic start. All-ones is orthogonal to any
    # eigenvector of the form (1, -1, ...), which silently costs the dominant
    # direction on symmetric matrices that have one.
    v = [1.0 / (i + 1) for i in range(n)]

    # Initial scaling
    max_val = max(abs(x) for x in v)
    v = [x / max_val for x in v]

    eigenvalue = 0.0
    for _ in range(max_iter):
        # w = A * v
        w = [sum(m[i][j] * v[j] for j in range(n)) for i in range(n)]

        # Scaling to prevent overflow in large systems
        # norm = sqrt(sum(w_i^2)). CPython floats overflow to inf rather
        # than raising OverflowError, so we detect both cases and fall back
        # to absolute-max scaling, which is safe for any magnitude.
        squared_sum = sum(x * x for x in w)
        if math.isinf(squared_sum):
            norm = max(abs(x) for x in w)
        else:
            try:
                norm = math.sqrt(squared_sum)
            except OverflowError:  # pragma: no cover - defensive for non-CPython libm
                # Defensive: still raised on some platforms for subnormal inputs
                norm = max(abs(x) for x in w)

        if norm < NEAR_ZERO:
            # A^k v collapsed to zero: the dominant eigenvalue is 0.
            return 0.0, v

        v_new = [x / norm for x in w]

        # Rayleigh quotient: (v^T * A * v) / (v^T * v)
        # Accurate for any normalization (L2 or L-inf)
        numerator = sum(v_new[i] * sum(m[i][j] * v_new[j] for j in range(n)) for i in range(n))
        denominator = sum(vi * vi for vi in v_new)
        new_eigenvalue = numerator / denominator if denominator > NEAR_ZERO else 0.0

        # Residual test: is v_new actually an eigenvector for new_eigenvalue?
        # Compared squared, so the tolerance means "‖Av − λv‖ <= tol·max(1,|λ|)"
        # without a second sqrt call on the hot path.
        av = [sum(m[i][j] * v_new[j] for j in range(n)) for i in range(n)]
        residual_sq = sum((av[i] - new_eigenvalue * v_new[i]) ** 2 for i in range(n))
        limit = tol * max(1.0, abs(new_eigenvalue))
        if residual_sq <= limit * limit:
            return new_eigenvalue, v_new
        eigenvalue = new_eigenvalue
        v = v_new

    raise ValueError(
        f"power iteration did not converge in {max_iter} iterations "
        f"(last Rayleigh quotient {eigenvalue!r}, residual {residual_sq**0.5:.3e}); "
        "|lambda_1| is probably not strictly dominant — an equal-magnitude or "
        "complex-conjugate pair needs a shifted or deflated iteration instead"
    )


def gram_schmidt(vectors: list[Vector]) -> list[Vector]:
    """Gram-Schmidt orthonormalization.

    Produces an orthonormal set from the input vectors.
    [Trefethen & Bau, Lecture 8]

    Args:
        vectors: list of linearly independent vectors

    Returns:
        orthonormal basis vectors
    """
    ortho: list[Vector] = []
    for v in vectors:
        u = v[:]
        for q in ortho:
            proj = sum(u[i] * q[i] for i in range(len(u)))
            u = [u[i] - proj * q[i] for i in range(len(u))]
        norm = math.sqrt(sum(x * x for x in u))
        if norm < NEAR_ZERO:
            continue
        ortho.append([x / norm for x in u])
    return ortho


def qr_decomposition(m: Matrix) -> tuple[Matrix, Matrix]:
    """QR decomposition via Householder reflections.

    Factorizes A (n×n) into an orthogonal matrix Q and upper-triangular R
    such that A = Q R. Householder triangularization is backward stable and
    preferred over classical Gram-Schmidt for numerical work.

    Reference:
        Householder, A. S. (1958). "Unitary triangularization of a
        nonsymmetric matrix." Journal of the ACM, 5(4), 339-342.
        See also Golub & Van Loan, §5.2; Trefethen & Bau, Lecture 10.

    Args:
        m: square matrix

    Returns:
        (Q, R) with Q orthogonal and R upper triangular
    """
    n = len(m)
    R = [row[:] for row in m]
    Q = identity(n)

    for k in range(n - 1):
        # column vector x = R[k:, k]
        x = [R[i][k] for i in range(k, n)]
        norm_x = math.sqrt(sum(xi * xi for xi in x))
        if norm_x < NEAR_ZERO:
            continue
        # Householder vector v
        alpha = -norm_x if x[0] >= 0 else norm_x
        v = x[:]
        v[0] -= alpha
        norm_v = math.sqrt(sum(vi * vi for vi in v))
        if norm_v < NEAR_ZERO:  # pragma: no cover - unreachable: norm_x>0 implies norm_v>0
            continue
        v = [vi / norm_v for vi in v]

        # apply H = I - 2 v v^T to R (rows k..n-1)
        for j in range(n):
            dot_vr = sum(v[i] * R[k + i][j] for i in range(n - k))
            for i in range(n - k):
                R[k + i][j] -= 2.0 * v[i] * dot_vr

        # accumulate Q = Q H (columns k..n-1)
        for i in range(n):
            dot_qv = sum(Q[i][k + j] * v[j] for j in range(n - k))
            for j in range(n - k):
                Q[i][k + j] -= 2.0 * dot_qv * v[j]

    return Q, R


def vector_norm(v: Vector, p: float = 2.0) -> float:
    """p-norm of a vector. ``p=2`` is Euclidean; ``p`` must be positive.

    For ``p == math.inf`` returns the max absolute entry.
    """
    if not v:
        raise ValueError("vector must be non-empty")
    if p == math.inf:
        return max(abs(x) for x in v)
    if p <= 0:
        raise ValueError("p must be positive (or math.inf)")
    if p == 1:
        return sum(abs(x) for x in v)
    if p == 2:
        return math.sqrt(sum(x * x for x in v))
    total = sum(abs(x) ** p for x in v)
    return float(total ** (1.0 / p))


def frobenius_norm(m: Matrix) -> float:
    """Frobenius norm ``√(Σ_ij a_ij²)``."""
    if not m or not m[0]:
        raise ValueError("matrix must be non-empty")
    return math.sqrt(sum(x * x for row in m for x in row))


def matrix_trace(m: Matrix) -> float:
    """Trace of a square matrix."""
    n = len(m)
    if n == 0 or any(len(row) != n for row in m):
        raise ValueError("matrix must be square and non-empty")
    return sum(m[i][i] for i in range(n))


def cholesky(m: Matrix) -> Matrix:
    """Cholesky decomposition of a symmetric positive-definite matrix.

    Returns the lower-triangular L such that A = L L^T. Roughly twice as
    efficient as LU for SPD systems and numerically stable.

    Reference:
        Benoît, C. (1924). "Note sur une méthode de résolution des équations
        normales... (Procédé du Commandant Cholesky)." Bulletin Géodésique,
        2, 67-77. See also Golub & Van Loan, §4.2.

    Args:
        m: symmetric positive-definite matrix

    Returns:
        lower-triangular matrix L with A = L L^T

    Raises:
        ValueError: if the matrix is not positive definite
    """
    n = len(m)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                diag = m[i][i] - s
                if diag <= 0.0:
                    raise ValueError(
                        "matrix is not positive definite — Cholesky decomposition requires symmetric positive definite input; check that the matrix is symmetric and all eigenvalues > 0"
                    )
                L[i][j] = math.sqrt(diag)
            else:
                L[i][j] = (m[i][j] - s) / L[j][j]
    return L
