"""Oracle-diff tier: pin *values and identities*, not just execution.

Every bug this file guards against was invisible to a 100% statement-and-branch
coverage gate, because coverage proves a line ran — not that the number it
produced was right. A test asserting a wrong expected value passes just as
green as a correct one, and a test that only checks ``0 <= p <= 1`` passes
whatever the value is.

So each test here does one of two things, and nothing else:

* compares against a reference computed **outside** this library — a closed
  form, a published table, or a hand derivation written into the test — or
* asserts a defining identity that must hold for arbitrary input, evaluated on
  inputs chosen to be adversarial rather than convenient.

The four sections below correspond to four real defects that shipped:

1. ``_normal_cdf`` hand-rolled Abramowitz & Stegun 7.1.26 with the wrong
   argument scaling and two missing coefficients. ``Phi(0)`` returned 0.304.
2. ``svd`` sorted ``U`` and the singular values into descending order but
   returned ``Vt`` unsorted, so ``A = U @ diag(s) @ Vt`` was false for any
   input whose Jacobi sweep did not already happen to be ordered.
3. ``wilcoxon_signed_rank`` divided the tie correction by 2 instead of 48,
   inflating it 24-fold and shrinking the variance.
4. ``power_iteration`` tested convergence on successive Rayleigh quotients
   rather than the residual, and started from all-ones. On ``[[3,4],[4,-3]]``
   it returned 4.0, which is not an eigenvalue of that matrix.

Deliberately dependency-free, like the package it tests.
"""

from __future__ import annotations

import math

import pytest

from cds.math_utils.linalg import power_iteration
from cds.math_utils.svd import svd
from cds.stats.nonparametric import wilcoxon_signed_rank
from cds.stats.time_series import _normal_cdf

# ---------------------------------------------------------------------------
# 1. Standard normal CDF against published values
# ---------------------------------------------------------------------------

# Phi(z) to 15 significant figures. Independent of this library: these are the
# standard tabulated values, reproducible from erfc(-z/sqrt(2))/2 in any
# arbitrary-precision system.
NORMAL_CDF_TABLE: list[tuple[float, float]] = [
    (-4.0, 0.0000316712418331),
    (-3.0, 0.0013498980316301),
    (-1.959963984540054, 0.0250000000000000),
    (-1.0, 0.1586552539314571),
    (-0.5, 0.3085375387259869),
    (0.0, 0.5000000000000000),
    (0.5, 0.6914624612740131),
    (1.0, 0.8413447460685429),
    (1.959963984540054, 0.9750000000000000),
    (2.5758293035489004, 0.9950000000000000),
    (3.0, 0.9986501019683699),
    (4.0, 0.9999683287581669),
]


@pytest.mark.parametrize(("z", "expected"), NORMAL_CDF_TABLE)
def test_normal_cdf_matches_published_values(z: float, expected: float) -> None:
    assert _normal_cdf(z) == pytest.approx(expected, abs=1e-14)


def test_normal_cdf_is_symmetric() -> None:
    """Phi(-z) + Phi(z) == 1 exactly, for every z."""
    for z in (0.1, 0.5, 1.0, 2.0, 3.5, 6.0):
        assert _normal_cdf(-z) + _normal_cdf(z) == pytest.approx(1.0, abs=1e-15)


def test_normal_cdf_at_zero_is_one_half() -> None:
    """The single value that the broken approximation got most wrong."""
    assert _normal_cdf(0.0) == 0.5


# ---------------------------------------------------------------------------
# 2. SVD: assert the defining identity, on adversarial inputs
# ---------------------------------------------------------------------------

# Inputs picked because their singular values do NOT come out in descending
# order from the Jacobi sweep. A random dense matrix hides the bug; a diagonal
# or permutation-like one exposes it immediately.
SVD_IDENTITY_CASES: list[list[list[float]]] = [
    [[1.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 3.0]],
    [[3.0, 0.0], [0.0, 4.0]],
    [[0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [1.0, 0.0, 0.0]],
    [[1.0e-9, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0e9]],
    [[2.0, 0.0], [0.0, 7.0], [0.0, 0.0]],
    [[0.0, 0.0, 6.0], [1.0, 0.0, 0.0]],
    [[4.0, 1.0, 2.0], [1.0, 3.0, 0.0], [2.0, 0.0, 5.0]],
]


@pytest.mark.parametrize("a", SVD_IDENTITY_CASES)
def test_svd_reconstructs_the_input(a: list[list[float]]) -> None:
    """``A == U @ diag(s) @ Vt`` — the guarantee the docstring makes."""
    result = svd(a)
    rows, cols = len(a), len(a[0])
    k = len(result.singular_values)

    sigma = [[0.0] * cols for _ in range(rows)]
    for i in range(k):
        sigma[i][i] = result.singular_values[i]

    us = [
        [sum(result.U[i][t] * sigma[t][j] for t in range(rows)) for j in range(cols)]
        for i in range(rows)
    ]
    reconstructed = [
        [sum(us[i][t] * result.Vt[t][j] for t in range(cols)) for j in range(cols)]
        for i in range(rows)
    ]

    scale = max(abs(x) for row in a for x in row)
    for i in range(rows):
        for j in range(cols):
            assert reconstructed[i][j] == pytest.approx(a[i][j], abs=1e-9 * max(1.0, scale))


@pytest.mark.parametrize("a", SVD_IDENTITY_CASES)
def test_svd_singular_values_are_descending(a: list[list[float]]) -> None:
    values = svd(a).singular_values
    assert values == sorted(values, reverse=True)


def test_svd_singular_values_of_a_diagonal_are_its_absolute_entries() -> None:
    """Closed form: sigma(diag(d)) == sorted(|d|) descending."""
    values = svd([[1.0, 0.0, 0.0], [0.0, -5.0, 0.0], [0.0, 0.0, 3.0]]).singular_values
    assert values == pytest.approx([5.0, 3.0, 1.0], abs=1e-12)


# ---------------------------------------------------------------------------
# 3. Wilcoxon signed-rank: hand-derived tie-corrected reference
# ---------------------------------------------------------------------------


def test_wilcoxon_tie_correction_uses_the_textbook_divisor() -> None:
    """Twelve observations forming six tied pairs of |d|, all positive.

    Derivation, entirely outside the library:

        midranks   1.5 1.5 3.5 3.5 5.5 5.5 7.5 7.5 9.5 9.5 11.5 11.5
        W+         = sum of all midranks              = 78
        mu         = n(n+1)/4 = 12*13/4               = 39
        tie sum    = 6 groups of t=2, each t^3 - t    = 36
        var        = n(n+1)(2n+1)/24 - 36/48
                   = 162.5 - 0.75                     = 161.75
        sigma      = sqrt(161.75)                      = 12.718097341976904
        z          = (78 - 39) / sigma                 = 3.066496422485931
        p          = erfc(|z| / sqrt(2))               = 0.002165834001025

    Dividing the tie sum by 2 instead of 48 gives var = 144.5, z = 3.24437,
    p = 0.00117710 — which is what this function used to return.
    """
    differences = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0]
    result = wilcoxon_signed_rank(differences)

    assert result.statistic == pytest.approx(78.0, abs=1e-12)
    assert result.n_effective == 12
    assert result.z == pytest.approx(3.066496422485931, abs=1e-12)
    assert result.p_value == pytest.approx(0.002165834001025, abs=1e-12)


def test_wilcoxon_without_ties_matches_the_untied_variance() -> None:
    """No ties, so the correction term must vanish entirely.

    n = 5, W+ = 1 + 3 + 5 = 9, mu = 5*6/4 = 7.5
    var = 5*6*11/24 = 13.75, sigma = 3.7080992435478315
    z = 1.5 / sigma = 0.404517
    """
    result = wilcoxon_signed_rank([1.0, -2.0, 3.0, -4.0, 5.0])
    assert result.statistic == pytest.approx(9.0, abs=1e-12)
    assert result.z == pytest.approx(0.4045199174779452, abs=1e-9)


# ---------------------------------------------------------------------------
# 4. power_iteration: residual, not a stalled quotient
# ---------------------------------------------------------------------------

# Strictly dominant spectra, |lambda_1| > |lambda_2|, eigenvalue in closed form.
POWER_CASES: list[tuple[list[list[float]], float]] = [
    ([[2.0, 1.0], [1.0, 2.0]], 3.0),  # 2 +- 1
    ([[5.0, 0.0], [0.0, 2.0]], 5.0),  # diagonal
    ([[2.0, 0.0], [0.0, -7.0]], 7.0),  # dominant eigenvalue is negative
    ([[4.0, 1.0], [1.0, 4.0]], 5.0),  # 4 +- 1
]

# |lambda_1| == |lambda_2|, where power iteration has no convergence guarantee.
# Whether it converges depends on where the start vector happens to fall, so
# the contract is not "it works" — it is "it never lies".
EQUAL_MAGNITUDE_CASES: list[list[list[float]]] = [
    [[3.0, 4.0], [4.0, -3.0]],  # +5 and -5
    [[0.0, 1.0], [1.0, 0.0]],  # +1 and -1
    [[0.0, -1.0], [1.0, 0.0]],  # +i and -i: no real eigenvector exists
]


def _residual(a: list[list[float]], eigenvalue: float, v: list[float]) -> float:
    n = len(a)
    av = [sum(a[i][j] * v[j] for j in range(n)) for i in range(n)]
    return math.sqrt(sum((av[i] - eigenvalue * v[i]) ** 2 for i in range(n)))


@pytest.mark.parametrize(("a", "expected_magnitude"), POWER_CASES)
def test_power_iteration_returns_a_true_eigenpair(
    a: list[list[float]], expected_magnitude: float
) -> None:
    """The returned pair must satisfy ``Av = lambda v``, not merely look plausible."""
    eigenvalue, v = power_iteration(a)

    assert abs(eigenvalue) == pytest.approx(expected_magnitude, abs=1e-8)
    assert math.sqrt(sum(x * x for x in v)) == pytest.approx(1.0, abs=1e-12)
    assert _residual(a, eigenvalue, v) < 1e-8


@pytest.mark.parametrize("a", EQUAL_MAGNITUDE_CASES)
def test_power_iteration_never_returns_a_non_eigenvalue(a: list[list[float]]) -> None:
    """Either refuse, or return something that really is an eigenpair.

    The old implementation did neither: on ``[[3,4],[4,-3]]`` it stopped as soon
    as two successive Rayleigh quotients agreed and handed back 4.0, which is
    not an eigenvalue of that matrix (they are +5 and -5).
    """
    try:
        eigenvalue, v = power_iteration(a)
    except ValueError as exc:
        assert "converge" in str(exc)
        return
    assert _residual(a, eigenvalue, v) < 1e-8


def test_power_iteration_rejects_a_complex_pair() -> None:
    """A rotation has eigenvalues +i and -i and no real eigenvector at all."""
    with pytest.raises(ValueError, match="strictly dominant"):
        power_iteration([[0.0, -1.0], [1.0, 0.0]], max_iter=500)
