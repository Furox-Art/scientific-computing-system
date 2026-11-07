"""Tests for deterministic numerical quadrature."""

import math

import pytest

from cds.numerical_integration import (
    QuadratureResult,
    adaptive_simpson,
    gaussian_quadrature,
    gaussian_quadrature_2d,
    romberg,
    simpson,
    simpson_2d,
    simpson_38,
    trapezoid,
)


class TestTrapezoid:
    def test_constant(self) -> None:
        # ∫_0^1 5 dx = 5
        assert abs(trapezoid(lambda x: 5.0, 0, 1, 10) - 5.0) < 1e-12

    def test_linear_is_exact(self) -> None:
        # Trapezoid is exact for linear integrands.
        assert abs(trapezoid(lambda x: 3 * x + 1, 0, 2, 1) - 8.0) < 1e-12

    def test_quadratic_error_order(self) -> None:
        # ∫_0^1 x^2 dx = 1/3, error O(h^2): halving h quarters the error.
        coarse = abs(trapezoid(lambda x: x * x, 0, 1, 10) - 1 / 3)
        fine = abs(trapezoid(lambda x: x * x, 0, 1, 20) - 1 / 3)
        assert fine < coarse / 3  # roughly 4x reduction

    def test_reversed_limits(self) -> None:
        # Sign flips with reversed limits; trapezoid O(h^2) sets the tolerance.
        assert abs(trapezoid(lambda x: x * x, 1, 0, 1000) - (-1 / 3)) < 1e-5

    def test_sin_over_pi(self) -> None:
        # ∫_0^π sin(x) dx = 2
        assert abs(trapezoid(math.sin, 0, math.pi, 1000) - 2.0) < 1e-4

    def test_invalid_n(self) -> None:
        with pytest.raises(ValueError):
            trapezoid(lambda x: x, 0, 1, 0)


class TestSimpson:
    def test_constant(self) -> None:
        assert abs(simpson(lambda x: 5.0, 0, 1, 10) - 5.0) < 1e-12

    def test_cubic_is_exact(self) -> None:
        # Simpson 1/3 is exact for cubics (degree <= 3).
        assert abs(simpson(lambda x: x**3, 0, 1, 2) - 0.25) < 1e-12

    def test_high_accuracy_sin(self) -> None:
        assert abs(simpson(math.sin, 0, math.pi, 100) - 2.0) < 1e-7

    def test_reversed_limits(self) -> None:
        assert abs(simpson(lambda x: x * x, 1, 0, 10) - (-1 / 3)) < 1e-12

    def test_invalid_odd_n(self) -> None:
        with pytest.raises(ValueError):
            simpson(lambda x: x, 0, 1, 3)

    def test_invalid_n_below_two(self) -> None:
        with pytest.raises(ValueError):
            simpson(lambda x: x, 0, 1, 1)


class TestSimpson38:
    def test_constant(self) -> None:
        assert abs(simpson_38(lambda x: 5.0, 0, 1, 3) - 5.0) < 1e-12

    def test_cubic_is_exact(self) -> None:
        # 3/8 rule is exact for cubics.
        assert abs(simpson_38(lambda x: x**3, 0, 1, 3) - 0.25) < 1e-12

    def test_invalid_n_not_multiple_of_3(self) -> None:
        with pytest.raises(ValueError):
            simpson_38(lambda x: x, 0, 1, 4)

    def test_invalid_n_below_3(self) -> None:
        with pytest.raises(ValueError):
            simpson_38(lambda x: x, 0, 1, 2)


class TestRomberg:
    def test_polynomial(self) -> None:
        result = romberg(lambda x: x**2, 0, 1)
        assert abs(result.value - 1 / 3) < 1e-12
        assert result.method == "romberg"
        assert result.n_eval >= 1

    def test_converges_to_high_precision(self) -> None:
        # ∫_0^1 e^x dx = e - 1
        result = romberg(math.exp, 0, 1, tol=1e-12)
        assert abs(result.value - (math.e - 1)) < 1e-10

    def test_error_estimate_recorded(self) -> None:
        result = romberg(lambda x: x**4, 0, 1)
        assert result.error_estimate >= 0

    def test_beats_trapezoid_accuracy(self) -> None:
        # Same smooth integrand: Romberg extrapolation should beat a plain
        # trapezoid with comparable node count.
        rom = romberg(math.sin, 0, math.pi, tol=1e-10).value
        trap = trapezoid(math.sin, 0, math.pi, 4)
        assert abs(rom - 2.0) < abs(trap - 2.0)

    def test_invalid_max_iter(self) -> None:
        with pytest.raises(ValueError):
            romberg(lambda x: x, 0, 1, max_iter=0)

    def test_max_iter_one_skips_refinement_loop(self) -> None:
        # max_iter=1 → range(1, 1) is empty, so the refinement loop body never
        # executes and the single trapezoid R[0][0] is returned as-is (the
        # `break` at line 197 is never reached because the loop never starts).
        result = romberg(lambda x: x * x, 0, 1, max_iter=1)
        # Single trapezoid of x² over [0,1] = 0.5*(0+1) = 0.5 (exact is 1/3).
        assert abs(result.value - 0.5) < 1e-12
        assert result.n_eval == 1


class TestGaussianQuadrature:
    def test_constant(self) -> None:
        assert abs(gaussian_quadrature(lambda x: 5.0, 0, 1, 2) - 5.0) < 1e-12

    def test_degree_2n_minus_1_exact(self) -> None:
        # With n nodes, Gauss-Legendre is exact for degree <= 2n-1.
        # n=4 => exact up to degree 7.
        assert abs(gaussian_quadrature(lambda x: x**7, 0, 1, 4) - 0.125) < 1e-12

    def test_not_exact_above_degree(self) -> None:
        # Degree 8 with n=4 should NOT be exact.
        assert abs(gaussian_quadrature(lambda x: x**8, 0, 1, 4) - 1 / 9) > 1e-6

    def test_n_equals_one(self) -> None:
        # 1-point rule: midpoint, exact for linear integrands.
        # ∫_0^2 (2x+1) dx = 6.
        assert abs(gaussian_quadrature(lambda x: 2 * x + 1, 0, 2, 1) - 6.0) < 1e-12

    def test_reversed_limits(self) -> None:
        assert abs(gaussian_quadrature(lambda x: x * x, 1, 0, 3) - (-1 / 3)) < 1e-6

    def test_sin_over_pi(self) -> None:
        assert abs(gaussian_quadrature(math.sin, 0, math.pi, 5) - 2.0) < 1e-4

    def test_invalid_n(self) -> None:
        with pytest.raises(ValueError):
            gaussian_quadrature(lambda x: x, 0, 1, 0)

    def test_invalid_negative_n(self) -> None:
        with pytest.raises(ValueError):
            gaussian_quadrature(lambda x: x, 0, 1, -1)

    def test_internal_nodes_rejects_n0(self) -> None:
        # The wrapper catches n<1 before reaching _gauss_legendre_nodes.
        # Directly call the cached helper to cover its own n<1 guard.
        from cds.numerical_integration.quadrature import _gauss_legendre_nodes

        with pytest.raises(ValueError):
            _gauss_legendre_nodes(0)

    def test_node_cache_reused(self) -> None:
        # Calling twice with the same n must reuse cached nodes/weights and
        # give identical results.
        f = lambda x: x**6  # noqa: E731
        a = gaussian_quadrature(f, 0, 1, 6)
        b = gaussian_quadrature(f, 0, 1, 6)
        assert a == b


class TestAdaptiveSimpson:
    def test_polynomial(self) -> None:
        result = adaptive_simpson(lambda x: x**2, 0, 1)
        assert abs(result.value - 1 / 3) < 1e-10
        assert result.method == "adaptive_simpson"
        assert result.n_eval >= 1

    def test_high_accuracy_sin(self) -> None:
        result = adaptive_simpson(math.sin, 0, math.pi, tol=1e-12)
        assert abs(result.value - 2.0) < 1e-10

    def test_error_estimate_is_reported(self) -> None:
        # The accumulated Lyness estimate is exposed instead of a bare NaN.
        result = adaptive_simpson(math.sin, 0, math.pi, tol=1e-12)
        assert math.isfinite(result.error_estimate)
        assert result.error_estimate >= 0.0
        assert result.error_estimate < 1e-9

    def test_error_estimate_on_kink(self) -> None:
        # A C^0 kink forces refinement, which the error estimate should reflect.
        result = adaptive_simpson(lambda x: abs(x - 0.5), 0, 1, tol=1e-8)
        assert math.isfinite(result.error_estimate)
        assert result.error_estimate >= 0.0

    def test_gaussian_bell(self) -> None:
        # ∫_{-∞}^{∞} e^{-x^2} dx = sqrt(pi); over [-3,3] the truncated tails leave
        # a small systematic offset (~4e-5), so relax against the full-line exact.
        val = adaptive_simpson(lambda x: math.exp(-x * x), -3, 3, tol=1e-10).value
        assert abs(val - math.sqrt(math.pi)) < 1e-3

    def test_concentrates_work_on_hard_region(self) -> None:
        # |x-0.5| is C^0 (kink) near the middle; adaptive should still hit tol.
        val = adaptive_simpson(lambda x: abs(x - 0.5), 0, 1, tol=1e-8).value
        assert abs(val - 0.25) < 1e-4

    def test_invalid_max_depth(self) -> None:
        with pytest.raises(ValueError):
            adaptive_simpson(lambda x: x, 0, 1, max_depth=0)

    def test_divergent_integrand_raises(self) -> None:
        # An integrand that feeds NaN into Simpson (e.g. NaN propagation from
        # an intermediate computation) should trigger the NaN guard.
        with pytest.raises(RuntimeError, match="NaN"):
            adaptive_simpson(lambda x: float("nan"), 0.0, 1.0)


class TestQuadratureResult:
    def test_fields(self) -> None:
        r = QuadratureResult(value=1.5, method="romberg", n_eval=10, error_estimate=0.01)
        assert r.value == 1.5
        assert r.method == "romberg"
        assert r.n_eval == 10
        assert r.error_estimate == 0.01


class TestCrossMethodAgreement:
    def test_all_methods_agree_on_exp(self) -> None:
        # ∫_0^1 e^x dx = e - 1. Every rule should land near the exact value.
        exact = math.e - 1
        f = math.exp
        assert abs(trapezoid(f, 0, 1, 1000) - exact) < 1e-6
        assert abs(simpson(f, 0, 1, 100) - exact) < 1e-10
        assert abs(romberg(f, 0, 1).value - exact) < 1e-10
        assert abs(gaussian_quadrature(f, 0, 1, 8) - exact) < 1e-8
        assert abs(adaptive_simpson(f, 0, 1).value - exact) < 1e-10


class TestLegendreEdgeCases:
    """Cover the n=0 and n=1 branches of the internal _legendre helper."""

    @staticmethod
    def _legendre(n: int, x: float) -> tuple[float, float]:
        from cds.numerical_integration.quadrature import _legendre

        return _legendre(n, x)

    def test_legendre_n0(self) -> None:
        # P_0(x) = 1, P_0'(x) = 0
        val, deriv = self._legendre(0, 0.5)
        assert val == 1.0
        assert deriv == 0.0

    def test_legendre_n1(self) -> None:
        # P_1(x) = x, P_1'(x) = 1
        val, deriv = self._legendre(1, 0.7)
        assert val == 0.7
        assert deriv == 1.0


# ---------------------------------------------------------------------------
# 2-D tensor-product quadrature
# ---------------------------------------------------------------------------


class TestSimpson2d:
    def test_constant(self) -> None:
        # ∬_0^1×0^1 5 dx dy = 5
        assert abs(simpson_2d(lambda x, y: 5.0, 0, 1, 0, 1, 2, 2) - 5.0) < 1e-12

    def test_separable_cubic_is_exact(self) -> None:
        # Simpson 1/3 is exact per axis for cubics, so x^3*y^3 integrates exactly.
        assert abs(simpson_2d(lambda x, y: x**3 * y**3, 0, 1, 0, 1, 2, 2) - 0.0625) < 1e-12

    def test_sin_sin_over_pi_square(self) -> None:
        # ∬_0^π × _0^π sin(x) sin(y) dx dy = 2 * 2 = 4
        f = lambda x, y: math.sin(x) * math.sin(y)  # noqa: E731
        assert abs(simpson_2d(f, 0, math.pi, 0, math.pi, 50, 50) - 4.0) < 1e-5

    def test_error_order_quartic(self) -> None:
        # O(h^4): doubling the panel count in each axis should cut the error
        # by roughly 16x for a smooth non-polynomial integrand.
        f = lambda x, y: math.exp(x + y)  # noqa: E731
        exact = (math.e - 1) ** 2  # ∬ e^{x+y} = (e-1)^2
        coarse = abs(simpson_2d(f, 0, 1, 0, 1, 10, 10) - exact)
        fine = abs(simpson_2d(f, 0, 1, 0, 1, 20, 20) - exact)
        assert fine < coarse / 10  # well above noise, below the 16x ideal

    def test_reversed_limits(self) -> None:
        # Reversing both limits flips the sign of the 2-D integral.
        assert abs(simpson_2d(lambda x, y: x * y, 1, 0, 1, 0, 10, 10) - 0.25) < 1e-12

    def test_one_reversed_axis(self) -> None:
        # Reversing only the y-axis flips the sign once.
        assert abs(simpson_2d(lambda x, y: x * y, 0, 1, 1, 0, 10, 10) - (-0.25)) < 1e-12

    def test_non_square_domain(self) -> None:
        # ∬_0^2 × _1^3 x*y dx dy = (∫_0^2 x)(∫_1^3 y) = 2 * 4 = 8
        assert abs(simpson_2d(lambda x, y: x * y, 0, 2, 1, 3, 20, 20) - 8.0) < 1e-10

    def test_invalid_nx_odd(self) -> None:
        with pytest.raises(ValueError, match="nx"):
            simpson_2d(lambda x, y: 1.0, 0, 1, 0, 1, 3, 2)

    def test_invalid_ny_odd(self) -> None:
        with pytest.raises(ValueError, match="ny"):
            simpson_2d(lambda x, y: 1.0, 0, 1, 0, 1, 2, 3)

    def test_invalid_nx_below_two(self) -> None:
        with pytest.raises(ValueError, match="nx"):
            simpson_2d(lambda x, y: 1.0, 0, 1, 0, 1, 1, 2)

    def test_invalid_ny_below_two(self) -> None:
        with pytest.raises(ValueError, match="ny"):
            simpson_2d(lambda x, y: 1.0, 0, 1, 0, 1, 2, 1)


class TestGaussianQuadrature2d:
    def test_constant(self) -> None:
        assert abs(gaussian_quadrature_2d(lambda x, y: 5.0, 0, 1, 0, 1, 2) - 5.0) < 1e-12

    def test_exact_for_degree_2n_minus_1_each_axis(self) -> None:
        # n=3 nodes => exact up to degree 5 in each variable.
        # x^5 * y^5 over [0,1]^2 = (1/6)^2 = 1/36.
        assert abs(gaussian_quadrature_2d(lambda x, y: x**5 * y**5, 0, 1, 0, 1, 3) - 1 / 36) < 1e-12

    def test_not_exact_above_degree(self) -> None:
        # Degree 6 with n=3 (exact only to 5) should NOT be exact.
        val = gaussian_quadrature_2d(lambda x, y: x**6 * y**6, 0, 1, 0, 1, 3)
        assert abs(val - 1 / 49) > 1e-6

    def test_separable_matches_1d_product(self) -> None:
        # A separable integrand g(x)h(y) must equal gaussian_quadrature(g) * (h).
        g = math.exp
        h = math.cos
        prod = gaussian_quadrature(g, 0, 1, 5) * gaussian_quadrature(h, 0, 1, 5)
        two_d = gaussian_quadrature_2d(lambda x, y: g(x) * h(y), 0, 1, 0, 1, 5)
        assert abs(two_d - prod) < 1e-12

    def test_gaussian_bell_product(self) -> None:
        # ∬_{[-2,2]^2} e^{-(x^2+y^2)} dx dy. The full-line value is π, but the
        # truncated domain and Gauss-Legendre's fixed nodes leave a small
        # systematic offset. Anchor to the converged Simpson-2D value instead,
        # and require Gauss to agree with it — this is what we can actually
        # guarantee without an analytic closed form for the truncated domain.
        f = lambda x, y: math.exp(-(x * x + y * y))  # noqa: E731
        val = gaussian_quadrature_2d(f, -2, 2, -2, 2, 8)
        simpson_ref = simpson_2d(f, -2, 2, -2, 2, 100, 100)
        assert abs(val - simpson_ref) < 1e-4
        # Sanity: both rules should still be in the right neighbourhood of π.
        assert 3.0 < val < 3.2

    def test_reversed_limits(self) -> None:
        # Reversing both axes flips sign once per axis -> net positive (two flips).
        assert abs(gaussian_quadrature_2d(lambda x, y: x * y, 1, 0, 1, 0, 3) - 0.25) < 1e-12

    def test_one_reversed_axis(self) -> None:
        assert abs(gaussian_quadrature_2d(lambda x, y: x * y, 0, 1, 1, 0, 3) - (-0.25)) < 1e-12

    def test_n_equals_one(self) -> None:
        # 1-point rule evaluates the integrand at the domain midpoint.
        # ∬_0^2×0^4 (x + y + 1) dx dy = area * f(midpoint) = 8 * (1+2+1) = 32,
        # and the exact double integral is also 32 (linear in each variable).
        assert abs(gaussian_quadrature_2d(lambda x, y: x + y + 1, 0, 2, 0, 4, 1) - 32.0) < 1e-12

    def test_non_square_domain(self) -> None:
        # ∬_0^2 × _1^3 x*y dx dy = 8 (same as the Simpson 2-D test).
        assert abs(gaussian_quadrature_2d(lambda x, y: x * y, 0, 2, 1, 3, 5) - 8.0) < 1e-10

    def test_invalid_n(self) -> None:
        with pytest.raises(ValueError):
            gaussian_quadrature_2d(lambda x, y: 1.0, 0, 1, 0, 1, 0)

    def test_invalid_negative_n(self) -> None:
        with pytest.raises(ValueError):
            gaussian_quadrature_2d(lambda x, y: 1.0, 0, 1, 0, 1, -1)


class Test2dCrossMethodAgreement:
    def test_simpson_and_gauss_agree_on_exp(self) -> None:
        # Both 2-D rules should land near the exact (e-1)^2 on a smooth integrand.
        exact = (math.e - 1) ** 2
        f = lambda x, y: math.exp(x + y)  # noqa: E731
        assert abs(simpson_2d(f, 0, 1, 0, 1, 100, 100) - exact) < 1e-8
        assert abs(gaussian_quadrature_2d(f, 0, 1, 0, 1, 10) - exact) < 1e-8
