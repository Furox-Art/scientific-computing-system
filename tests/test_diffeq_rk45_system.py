"""Closed-form reference tests for the adaptive system solver ``rk45_system``.

Every accuracy claim is checked against an analytic solution (harmonic
oscillator, exponential decay, damped oscillator, Gaussian integral of a
linear exponent), not against another numerical method. Controller behavior
(adaptivity, rejection, backward integration, endpoint snap, precision floor)
is pinned separately.
"""

from __future__ import annotations

import math

import pytest

from cds.diffeq import ODESystemSolution, rk45, rk45_system, solve_system


def _harmonic(t: float, y: list[float]) -> list[float]:
    return [y[1], -y[0]]


def _decay(t: float, y: list[float]) -> list[float]:
    return [-y[0], -2.0 * y[1]]


class TestClosedFormReferences:
    def test_harmonic_oscillator_one_period(self) -> None:
        # x'' = -x, x(0)=1, x'(0)=0  =>  x(t)=cos t, x'(t)=-sin t.
        sol = rk45_system(_harmonic, 0.0, [1.0, 0.0], 2.0 * math.pi, atol=1e-10, rtol=1e-10)
        assert sol.y[-1][0] == pytest.approx(1.0, abs=1e-7)
        assert sol.y[-1][1] == pytest.approx(0.0, abs=1e-7)
        # Every grid point must track the closed form, not just the endpoint.
        for t, y in zip(sol.t, sol.y):
            assert y[0] == pytest.approx(math.cos(t), abs=1e-7)
            assert y[1] == pytest.approx(-math.sin(t), abs=1e-7)

    def test_coupled_decay(self) -> None:
        sol = rk45_system(_decay, 0.0, [1.0, 1.0], 1.0, atol=1e-10, rtol=1e-10)
        assert sol.y[-1][0] == pytest.approx(math.exp(-1.0), rel=1e-8)
        assert sol.y[-1][1] == pytest.approx(math.exp(-2.0), rel=1e-8)

    def test_damped_oscillator_closed_form(self) -> None:
        # x'' = -x - 0.1 x' (the README quick-start system). Underdamped:
        # a=-0.05, omega=sqrt(1-0.0025);
        # x(t) = e^{at}(cos wt - (a/w) sin wt), x'(t) = -e^{at}(a^2/w + w) sin wt.
        a = -0.05
        w = math.sqrt(1.0 - 0.0025)

        def rhs(t: float, y: list[float]) -> list[float]:
            return [y[1], -y[0] - 0.1 * y[1]]

        t_end = 50.0
        sol = rk45_system(rhs, 0.0, [1.0, 0.0], t_end, atol=1e-11, rtol=1e-11)
        env = math.exp(a * t_end)
        x_exact = env * (math.cos(w * t_end) - (a / w) * math.sin(w * t_end))
        v_exact = -env * (a * a / w + w) * math.sin(w * t_end)
        assert sol.y[-1][0] == pytest.approx(x_exact, rel=1e-7, abs=1e-12)
        assert sol.y[-1][1] == pytest.approx(v_exact, rel=1e-7, abs=1e-12)

    def test_non_autonomous_gaussian(self) -> None:
        # y' = t*y, y(0)=1  =>  y(t) = exp(t^2/2).
        sol = rk45_system(lambda t, y: [t * y[0]], 0.0, [1.0], 2.0, atol=1e-11, rtol=1e-11)
        assert sol.y[-1][0] == pytest.approx(math.exp(2.0), rel=1e-8)

    def test_polynomial_rhs_is_integrated_exactly(self) -> None:
        # y' = [t, 2t] => y = [t^2/2, t^2]; the embedded pair's error estimate
        # is zero here, exercising the aggressive step-growth branch.
        sol = rk45_system(lambda t, y: [t, 2.0 * t], 0.0, [0.0, 0.0], 3.0)
        assert sol.y[-1][0] == pytest.approx(4.5, abs=1e-12)
        assert sol.y[-1][1] == pytest.approx(9.0, abs=1e-12)


class TestControllerBehavior:
    def test_adaptive_beats_fixed_step_at_same_accuracy(self) -> None:
        t_end = 50.0
        adaptive = rk45_system(_harmonic, 0.0, [1.0, 0.0], t_end, atol=1e-9, rtol=1e-9)
        # Measured: adaptive ~739 steps, endpoint error ~1.9e-8. Fixed-step
        # RK4 needs >1000 steps (dt=0.05) and still lands ~40x less accurate.
        fixed_t, fixed_y = solve_system(_harmonic, 0.0, [1.0, 0.0], t_end, dt=0.05)
        err_adaptive = abs(adaptive.y[-1][0] - math.cos(t_end))
        err_fixed = abs(fixed_y[-1][0] - math.cos(t_end))
        assert err_adaptive < err_fixed
        assert adaptive.steps < len(fixed_t)

    def test_tighter_tolerance_reduces_error(self) -> None:
        t_end = math.pi / 2
        loose = rk45_system(_harmonic, 0.0, [1.0, 0.0], t_end, atol=1e-4, rtol=1e-4)
        tight = rk45_system(_harmonic, 0.0, [1.0, 0.0], t_end, atol=1e-11, rtol=1e-11)
        exact = math.cos(t_end)
        assert abs(tight.y[-1][0] - exact) < abs(loose.y[-1][0] - exact)

    def test_backward_integration(self) -> None:
        sol = rk45_system(_decay, 0.0, [1.0, 1.0], -1.0, atol=1e-10, rtol=1e-10)
        assert sol.y[-1][0] == pytest.approx(math.exp(1.0), rel=1e-8)
        assert sol.steps > 0
        assert all(sol.t[i] > sol.t[i + 1] for i in range(len(sol.t) - 1))

    def test_zero_span_returns_initial_state(self) -> None:
        sol = rk45_system(_decay, 0.0, [3.0, 7.0], 0.0)
        assert sol.steps == 0
        assert sol.t == [0.0]
        assert sol.y == [[3.0, 7.0]]

    def test_endpoint_snap(self) -> None:
        sol = rk45_system(_decay, 0.0, [1.0, 1.0], 1.0, dt=0.3)
        assert sol.t[-1] == 1.0

    def test_initial_state_is_copied(self) -> None:
        y0 = [1.0, 0.0]
        sol = rk45_system(_harmonic, 0.0, y0, 0.5)
        sol.y[0][0] = 999.0
        assert y0 == [1.0, 0.0]

    def test_method_and_type(self) -> None:
        sol = rk45_system(_decay, 0.0, [1.0, 1.0], 0.5)
        assert isinstance(sol, ODESystemSolution)
        assert sol.method == "rk45"

    def test_scalar_rk45_unchanged(self) -> None:
        # The shared tableau must not have altered the scalar solver.
        sol = rk45(lambda t, y: -y, 0.0, 1.0, 2.0, dt=0.1, atol=1e-8, rtol=1e-8)
        assert abs(sol.y[-1] - math.exp(-2.0)) < 1e-6


class TestValidation:
    def test_empty_state_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one component"):
            rk45_system(lambda t, y: [], 0.0, [], 1.0)

    def test_wrong_derivative_count_raises(self) -> None:
        with pytest.raises(ValueError, match="derivative"):
            rk45_system(lambda t, y: [1.0], 0.0, [1.0, 2.0], 1.0)

    def test_precision_floor_raises(self) -> None:
        # Mirrors the scalar rk45 floor test: over a huge span the growing
        # step eventually satisfies t + h == t while t is enormous, tripping
        # the machine-precision floor.
        with pytest.raises(RuntimeError, match="machine precision floor"):
            rk45_system(
                lambda t, y: [0.0, 0.0],
                0.0,
                [1.0, 1.0],
                1e308,
                atol=1e-30,
                rtol=1e-30,
            )

    def test_zero_tolerance_zero_component_fails_closed(self) -> None:
        # atol=rtol=0 on a zero component: the per-component scale falls back
        # to NEAR_MIN_FLOAT instead of dividing by zero; tiny spans still
        # integrate (the error estimate rounds to zero at that scale).
        sol = rk45_system(lambda t, y: [y[0] + 1.0], 0.0, [0.0], 1e-5, atol=0.0, rtol=0.0)
        assert sol.steps >= 1
        assert sol.y[-1][0] == pytest.approx(math.exp(1e-5) - 1.0, abs=1e-9)
