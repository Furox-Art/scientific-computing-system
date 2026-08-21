"""Tests for :mod:`cds.diffeq._implicit` (backward Euler + trapezoid)."""

from __future__ import annotations

import math

import pytest

from cds.diffeq import (
    ODESolution,
    backward_euler,
    backward_euler_system,
    trapezoid_method,
    trapezoid_method_system,
)

_STIFF_K = 1000.0


def _stiff_exact(t: float) -> float:
    """Exact solution of dy/dt = -k(y-1), y(0)=0: y = 1 - e^(-kt)."""
    return 1.0 - math.exp(-_STIFF_K * t)


def test_backward_euler_handles_stiff_decay() -> None:
    # Explicit Euler diverges at dt=0.001 (k*dt=1); implicit stays stable.
    sol = backward_euler(lambda t, y: -_STIFF_K * (y - 1.0), 0.0, 0.0, 0.05, dt=0.001)
    assert sol.method == "backward-euler"
    assert abs(sol.y[-1] - _stiff_exact(0.05)) < 1e-3


def test_trapezoid_more_accurate_than_backward_euler_on_smooth_problem() -> None:
    def f(t: float, y: float) -> float:
        return -y

    be = backward_euler(f, 0.0, 1.0, 2.0, dt=0.05)
    cn = trapezoid_method(f, 0.0, 1.0, 2.0, dt=0.05)
    exact = math.exp(-2.0)
    assert abs(cn.y[-1] - exact) < abs(be.y[-1] - exact)
    assert cn.method == "trapezoid"


def test_trapezoid_is_second_order() -> None:
    def f(t: float, y: float) -> float:
        return -y

    err_coarse = abs(trapezoid_method(f, 0.0, 1.0, 1.0, dt=0.1).y[-1] - math.exp(-1.0))
    err_fine = abs(trapezoid_method(f, 0.0, 1.0, 1.0, dt=0.05).y[-1] - math.exp(-1.0))
    assert err_fine < err_coarse / 3.0  # halving dt cuts error ~4x


def test_analytic_jacobian_path_matches_fd_path() -> None:
    def f(t: float, y: float) -> float:
        return -_STIFF_K * (y - 1.0)

    def jac(t: float, y: float) -> float:
        return -_STIFF_K

    with_jac = backward_euler(f, 0.0, 0.0, 0.02, dt=0.002, jac=jac)
    fd = backward_euler(f, 0.0, 0.0, 0.02, dt=0.002)
    assert with_jac.y[-1] == pytest.approx(fd.y[-1], rel=1e-6)


def test_backward_integration_in_time() -> None:
    # Integrate the harmonic decay from t=0 back to t=-1.
    sol = backward_euler(lambda t, y: -y, 0.0, 1.0, -1.0, dt=0.01)
    exact = math.exp(1.0)
    assert sol.t[-1] == pytest.approx(-1.0)
    assert abs(sol.y[-1] - exact) < 0.05
    assert sol.steps > 50


def test_zero_length_span_returns_initial_condition() -> None:
    sol = backward_euler(lambda t, y: -y, 1.0, 7.0, 1.0, dt=0.1)
    assert sol.t == [1.0]
    assert sol.y == [7.0]
    assert sol.steps == 0


def test_result_type_matches_explicit_solvers() -> None:
    sol = backward_euler(lambda t, y: -y, 0.0, 1.0, 0.5, dt=0.1)
    assert isinstance(sol, ODESolution)
    assert len(sol.t) == len(sol.y) == sol.steps + 1


def test_newton_budget_exhaustion_raises() -> None:
    # Linear stage equation converges in ONE Newton update, so a budget of
    # exactly 1 exhausts before the convergence re-check can fire.
    with pytest.raises(ValueError, match="did not converge"):
        backward_euler(lambda t, y: -y, 0.0, 1.0, 1.0, dt=0.5, max_iter=1)


def test_singular_jacobian_raises() -> None:
    # Residual Jacobian is I - h·df/dy. With analytic df/dy = 2 and h = 0.5
    # it is exactly [[0]] — a deterministic singular LU solve. (The finite-
    # difference Jacobian carries ~1e-11 rounding, so it only diverges.)
    with pytest.raises(ValueError, match="singular Jacobian"):
        backward_euler(
            lambda t, y: 2.0 * y,
            0.0,
            1.0,
            1.0,
            dt=0.5,
            jac=lambda t, y: 2.0,
        )


def test_huge_tolerance_short_circuits_newton() -> None:
    # First residual already within tol → returns without touching the Jacobian.
    sol = backward_euler(lambda t, y: -_STIFF_K * (y - 1.0), 0.0, 0.0, 0.01, dt=0.005, tol=1e6)
    assert sol.steps == 2


def test_parameter_validation_scalar() -> None:
    def f(t: float, y: float) -> float:
        return -y

    with pytest.raises(ValueError, match="dt must be positive"):
        backward_euler(f, 0.0, 1.0, 1.0, dt=0.0)
    with pytest.raises(ValueError, match="tol must be positive"):
        trapezoid_method(f, 0.0, 1.0, 1.0, dt=0.1, tol=0.0)
    with pytest.raises(ValueError, match="max_iter must be >= 1"):
        backward_euler(f, 0.0, 1.0, 1.0, dt=0.1, max_iter=0)


# --------------------------------------------------------------------- #
# System variants                                                        #
# --------------------------------------------------------------------- #


def _harmonic(t: float, y: list[float]) -> list[float]:
    return [y[1], -y[0]]


def test_backward_euler_system_damps_energy() -> None:
    ts, ys = backward_euler_system(_harmonic, 0.0, [1.0, 0.0], math.pi, dt=0.001)
    assert ts[-1] == pytest.approx(math.pi)
    # Backward Euler is dissipative: amplitude must shrink below 1 and stay
    # bounded (no blow-up like explicit Euler on this conservative system).
    amplitudes = [math.hypot(s[0], s[1]) for s in ys]
    assert amplitudes[-1] < 1.0
    assert max(amplitudes) <= 1.0 + 1e-9


def test_trapezoid_system_conserves_energy_closely() -> None:
    ts, ys = trapezoid_method_system(_harmonic, 0.0, [1.0, 0.0], 4 * math.pi, dt=0.01)
    amplitudes = [math.hypot(s[0], s[1]) for s in ys]
    # CN nearly conserves the oscillator invariant over 2 full periods.
    assert abs(amplitudes[-1] - 1.0) < 0.01


def test_system_with_analytic_jacobian() -> None:
    def jac(t: float, y: list[float]) -> list[list[float]]:
        return [[0.0, 1.0], [-1.0, 0.0]]

    ts, ys = trapezoid_method_system(_harmonic, 0.0, [1.0, 0.0], math.pi, dt=0.01, jac=jac)
    assert ys[-1][0] == pytest.approx(-1.0, abs=0.01)


def test_system_backward_time_direction() -> None:
    ts, ys = backward_euler_system(lambda t, y: [-y[0]], 0.0, [1.0], -0.1, dt=0.005)
    assert ts[-1] == pytest.approx(-0.1)
    assert ys[-1][0] == pytest.approx(math.exp(0.1), abs=0.01)


def test_system_validation() -> None:
    def f(t: float, y: list[float]) -> list[float]:
        return [-y[0]]

    with pytest.raises(ValueError, match="non-empty"):
        backward_euler_system(f, 0.0, [], 1.0, dt=0.1)
    with pytest.raises(ValueError, match="dt must be positive"):
        trapezoid_method_system(f, 0.0, [1.0], 1.0, dt=-1.0)
