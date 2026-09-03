"""Hypothesis-driven invariants for :mod:`cds.math_utils.special`.

Companion to ``tests/test_property_invariants.py`` (which covers the older
numerics): this module pins the v1.6.1 incomplete gamma/beta kernels to
their mathematical identities. Hypothesis *searches* the parameter space
for a counterexample and *shrinks* any failure to the smallest input that
still reproduces it.

This suite is optional: it only runs when Hypothesis is installed
(``pip install ".[property]"`` or ``[all]``). CI gates it behind the
dedicated ``property-tests`` job so a missing dependency never breaks the
default ``[test]`` matrix. The module-level guard below makes that safe:
without Hypothesis, every test skips rather than erroring on import.

Properties covered (gamma + beta + chi-square link):

1. Complementarity — ``gammp(a, x) + gammq(a, x) == 1`` for ``a > 0``,
   ``x >= 0``; both values stay inside ``[0, 1]``.
2. Boundary values — ``P(a, 0) == 0`` and ``Q(a, 0) == 1``.
3. Monotonicity — ``P(a, x)`` is non-decreasing in ``x``.
4. Closed form — ``P(1, x) == 1 - exp(-x)`` and ``exp(gammln(n)) == (n-1)!``.
5. Beta range — ``0 <= betai(a, b, x) <= 1`` for ``x`` in ``[0, 1]``.
6. Beta endpoints — ``I(a, b, 0) == 0`` and ``I(a, b, 1) == 1``.
7. Beta symmetry — ``I_x(a, b) == 1 - I_{1-x}(b, a)``.
8. Beta monotonicity — ``I_x(a, b)`` is non-decreasing in ``x``; and the
   uniform case ``I_x(1, 1) == x``.
9. Chi-square link — ``chi2_sf(x, df) == 1 - gammp(df / 2, x / 2)``.
"""

from __future__ import annotations

import math

import pytest

# Every import below runs after the ``importorskip`` guard, so module-level
# placement triggers E402. That's intentional and correct here: we *cannot*
# import Hypothesis before confirming it's installed. Each line carries its
# own ``noqa: E402`` so ruff's import sorter (I001) and the E402 rule both
# stay satisfied.
hypothesis = pytest.importorskip("hypothesis")  # noqa: E402
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from cds.math_utils.special import betai, gammln, gammp, gammq  # noqa: E402
from cds.stats._distributions import chi2_sf  # noqa: E402

# Same example budget as tests/test_property_invariants.py: fast in CI while
# still exercising Hypothesis's search + shrink loop.
MAX_EXAMPLES = 50

# The kernels hold ~1e-12 accuracy; 1e-9 keeps the properties about the
# *mathematics* rather than the last ulp of the continued fraction.
TOL = 1e-9


# --------------------------------------------------------------------------- #
# Shared strategies (ranges per issue #80: a in [0.5, 20], x scaled to a)
# --------------------------------------------------------------------------- #
@st.composite
def gamma_params(draw: st.DrawFn) -> tuple[float, float]:
    """A shape ``a`` in [0.5, 20] with ``x`` scaled to it (``x`` in [0, 3a]).

    Scaling ``x`` to ``a`` keeps the search on the transition frontier
    (``x < a + 1`` picks the series branch, larger ``x`` the continued
    fraction) instead of wasting examples deep in a saturated tail.
    """
    a = draw(st.floats(min_value=0.5, max_value=20.0, allow_nan=False, allow_infinity=False))
    x = draw(st.floats(min_value=0.0, max_value=3.0 * a, allow_nan=False, allow_infinity=False))
    return (a, x)


@st.composite
def gamma_pair(draw: st.DrawFn) -> tuple[float, float, float]:
    """A shape ``a`` plus an ordered pair ``x1 <= x2`` (both scaled to ``a``)."""
    a, x1 = draw(gamma_params())
    x2 = draw(
        st.floats(min_value=x1, max_value=3.0 * a + 1.0, allow_nan=False, allow_infinity=False)
    )
    return (a, x1, x2)


@st.composite
def beta_params(draw: st.DrawFn) -> tuple[float, float, float]:
    """Shapes ``a``, ``b`` in [0.5, 10] with ``x`` in [0, 1]."""
    a = draw(st.floats(min_value=0.5, max_value=10.0, allow_nan=False, allow_infinity=False))
    b = draw(st.floats(min_value=0.5, max_value=10.0, allow_nan=False, allow_infinity=False))
    x = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    return (a, b, x)


@st.composite
def beta_pair(draw: st.DrawFn) -> tuple[float, float, float, float]:
    """Shapes ``a``, ``b`` with an ordered pair ``x1 <= x2`` in [0, 1]."""
    a, b, x1 = draw(beta_params())
    x2 = draw(st.floats(min_value=x1, max_value=1.0, allow_nan=False, allow_infinity=False))
    return (a, b, x1, x2)


@st.composite
def chi2_params(draw: st.DrawFn) -> tuple[float, float]:
    """Degrees of freedom in [0.5, 30] with ``x`` scaled to it."""
    df = draw(st.floats(min_value=0.5, max_value=30.0, allow_nan=False, allow_infinity=False))
    x = draw(st.floats(min_value=0.0, max_value=3.0 * df, allow_nan=False, allow_infinity=False))
    return (x, df)


# --------------------------------------------------------------------------- #
# 1-3. Incomplete gamma: complementarity, range, boundary, monotonicity
# --------------------------------------------------------------------------- #
class TestGammaInvariants:
    @given(params=gamma_params())
    @settings(max_examples=MAX_EXAMPLES)
    def test_complementarity(self, params: tuple[float, float]) -> None:
        # P(a, x) + Q(a, x) == 1: the lower and upper tails partition unity.
        a, x = params
        assert math.isclose(gammp(a, x) + gammq(a, x), 1.0, abs_tol=TOL)

    @given(params=gamma_params())
    @settings(max_examples=MAX_EXAMPLES)
    def test_values_in_unit_interval(self, params: tuple[float, float]) -> None:
        # Regularized tails are probabilities, hence confined to [0, 1].
        a, x = params
        assert -TOL <= gammp(a, x) <= 1.0 + TOL
        assert -TOL <= gammq(a, x) <= 1.0 + TOL

    @given(a=st.floats(min_value=0.5, max_value=20.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=MAX_EXAMPLES)
    def test_boundary_at_zero(self, a: float) -> None:
        # No mass below zero: P(a, 0) == 0 and Q(a, 0) == 1.
        assert gammp(a, 0.0) == 0.0
        assert gammq(a, 0.0) == 1.0

    @given(params=gamma_pair())
    @settings(max_examples=MAX_EXAMPLES)
    def test_monotone_in_x(self, params: tuple[float, float, float]) -> None:
        # P(a, x) accumulates mass, so x1 <= x2 implies P(a, x1) <= P(a, x2)
        # (and dually Q is non-increasing). The TOL slack absorbs float
        # noise where both tails saturate at the same plateau.
        a, x1, x2 = params
        assert gammp(a, x2) >= gammp(a, x1) - TOL
        assert gammq(a, x2) <= gammq(a, x1) + TOL


# --------------------------------------------------------------------------- #
# 4. Incomplete gamma: closed forms
# --------------------------------------------------------------------------- #
class TestGammaClosedForms:
    @given(x=st.floats(min_value=0.0, max_value=60.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=MAX_EXAMPLES)
    def test_unit_shape_is_exponential_cdf(self, x: float) -> None:
        # P(1, x) == 1 - exp(-x): the Gamma(1) law is the unit exponential.
        assert math.isclose(gammp(1.0, x), 1.0 - math.exp(-x), rel_tol=TOL, abs_tol=TOL)

    @given(n=st.integers(min_value=1, max_value=10))
    @settings(max_examples=MAX_EXAMPLES)
    def test_log_gamma_matches_factorial(self, n: int) -> None:
        # exp(gammln(n)) == (n - 1)! for positive integers.
        assert math.isclose(math.exp(gammln(float(n))), float(math.factorial(n - 1)), rel_tol=TOL)


# --------------------------------------------------------------------------- #
# 5-8. Incomplete beta: range, endpoints, symmetry, monotonicity
# --------------------------------------------------------------------------- #
class TestBetaInvariants:
    @given(params=beta_params())
    @settings(max_examples=MAX_EXAMPLES)
    def test_values_in_unit_interval(self, params: tuple[float, float, float]) -> None:
        # The regularized beta is a CDF on [0, 1], hence confined there.
        a, b, x = params
        assert -TOL <= betai(a, b, x) <= 1.0 + TOL

    @given(
        a=st.floats(min_value=0.5, max_value=10.0, allow_nan=False, allow_infinity=False),
        b=st.floats(min_value=0.5, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_endpoints(self, a: float, b: float) -> None:
        # Empty interval has no mass; the full interval has all of it.
        assert betai(a, b, 0.0) == 0.0
        assert betai(a, b, 1.0) == 1.0

    @given(params=beta_params())
    @settings(max_examples=MAX_EXAMPLES)
    def test_reflection_symmetry(self, params: tuple[float, float, float]) -> None:
        # I_x(a, b) == 1 - I_{1-x}(b, a): swapping the shapes mirrors x.
        a, b, x = params
        assert math.isclose(betai(a, b, x), 1.0 - betai(b, a, 1.0 - x), abs_tol=TOL)

    @given(params=beta_pair())
    @settings(max_examples=MAX_EXAMPLES)
    def test_monotone_in_x(self, params: tuple[float, float, float, float]) -> None:
        # A CDF never decreases: I_{x1} <= I_{x2} for x1 <= x2. The TOL
        # slack absorbs float noise where both ends saturate at one edge.
        a, b, x1, x2 = params
        assert betai(a, b, x2) >= betai(a, b, x1) - TOL

    @given(x=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=MAX_EXAMPLES)
    def test_uniform_shape_is_identity(self, x: float) -> None:
        # I_x(1, 1) == x: the Beta(1, 1) law is uniform on [0, 1].
        assert math.isclose(betai(1.0, 1.0, x), x, abs_tol=TOL)


# --------------------------------------------------------------------------- #
# 9. Chi-square link
# --------------------------------------------------------------------------- #
class TestChiSquareLink:
    @given(params=chi2_params())
    @settings(max_examples=MAX_EXAMPLES)
    def test_sf_matches_upper_gamma(self, params: tuple[float, float]) -> None:
        # chi2_sf(x, df) == 1 - gammp(df / 2, x / 2): the chi-square tail is
        # the upper regularized gamma tail at half scale.
        x, df = params
        assert math.isclose(chi2_sf(x, df), 1.0 - gammp(df / 2.0, x / 2.0), abs_tol=TOL)

    @given(params=chi2_params())
    @settings(max_examples=MAX_EXAMPLES)
    def test_sf_in_unit_interval(self, params: tuple[float, float]) -> None:
        # A survival probability lives in [0, 1].
        x, df = params
        assert -TOL <= chi2_sf(x, df) <= 1.0 + TOL
