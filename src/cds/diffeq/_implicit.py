"""Implicit / A-stable ODE solvers for stiff problems.

Implements backward Euler (θ = 1) and the trapezoidal / Crank–Nicolson method
(θ = 1/2) as one unified θ-method core. Every implicit step solves the stage
equation ``x = y + h·[(1−θ)·f(t,y) + θ·f(t+h,x)]`` with damped-free Newton
iteration, using :func:`cds.math_utils.linalg.solve_linear` for the linear
subsystems (partial-pivoted LU) and an analytic or central-difference
Jacobian.

References:
    - Hairer, E. & Wanner, G. (1996). Solving ODEs II: Stiff and
      Differential-Algebraic Problems (2nd ed.), §IV.3.
    - Crank, J. & Nicolson, P. (1947). Proc. Camb. Phil. Soc. 43(1).
"""

from __future__ import annotations

import math
from collections.abc import Callable

from cds.core._numeric import LOOP_EPSILON, RK45_DEFAULT_DT
from cds.diffeq.solvers import ODESolution
from cds.math_utils.linalg import solve_linear

SystemRHS = Callable[[float, list[float]], list[float]]
Jacobian = Callable[[float, list[float]], list[list[float]]]

DEFAULT_NEWTON_TOL = 1e-10
DEFAULT_NEWTON_MAX_ITER = 50
_FD_RELATIVE_H = 1e-7


def _finite_difference_jacobian(
    f: SystemRHS,
    t: float,
    y: list[float],
) -> list[list[float]]:
    """Central-difference approximation of ∂f/∂y at ``(t, y)``."""
    n = len(y)
    jac = [[0.0] * n for _ in range(n)]
    for j in range(n):
        h = _FD_RELATIVE_H * max(1.0, abs(y[j]))
        y_plus = list(y)
        y_minus = list(y)
        y_plus[j] += h
        y_minus[j] -= h
        f_plus = f(t, y_plus)
        f_minus = f(t, y_minus)
        for i in range(n):
            jac[i][j] = (f_plus[i] - f_minus[i]) / (2.0 * h)
    return jac


def _newton_solve(
    residual: Callable[[list[float]], list[float]],
    jac: Callable[[list[float]], list[list[float]]],
    x0: list[float],
    *,
    tol: float,
    max_iter: int,
) -> list[float]:
    """Solve ``residual(x) = 0`` with plain Newton iteration.

    Raises:
        ValueError: when the linearized Jacobian is singular or the iteration
            budget is exhausted without meeting ``tol``.
    """
    x = list(x0)
    for _ in range(max_iter):
        r = residual(x)
        if max(abs(v) for v in r) <= tol:
            return x
        try:
            delta = solve_linear(jac(x), [-v for v in r])
        except ValueError as exc:
            raise ValueError("implicit step failed: singular Jacobian") from exc
        x = [xi + d for xi, d in zip(x, delta)]
    raise ValueError("implicit solve did not converge within max_iter")


def _integrate_system(
    f: SystemRHS,
    t0: float,
    y0: list[float],
    t_end: float,
    dt: float,
    *,
    theta: float,
    jac: Jacobian | None,
    tol: float,
    max_iter: int,
) -> tuple[list[float], list[list[float]]]:
    """Fixed-step θ-method driver shared by every public implicit solver."""
    if not y0:
        raise ValueError("y0 must be non-empty")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if tol <= 0:
        raise ValueError("tol must be positive")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")

    # Same direction rule as the explicit solvers (see euler_method): the sign
    # of ``t_end - t0`` decides forward/backward integration; ``dt`` is a
    # magnitude. copysign(1, 0) is +1, so t_end == t0 skips the loop cleanly.
    direction = math.copysign(1.0, t_end - t0)
    t_vals = [t0]
    y_vals = [list(y0)]
    t = t0
    y = list(y0)

    while (t_end - t) * direction > LOOP_EPSILON:
        h = direction * min(dt, abs(t_end - t))
        t_new = t + h
        f_old = f(t, y)

        def residual(x: list[float], t_new: float = t_new) -> list[float]:
            f_new = f(t_new, x)
            return [
                xv - yv - h * ((1.0 - theta) * fv + theta * fnv)
                for xv, yv, fv, fnv in zip(x, y, f_old, f_new)
            ]

        def system_jac(x: list[float], t_new: float = t_new) -> list[list[float]]:
            # Newton needs the Jacobian of the *residual*, not of f:
            # d/dx [x - y - h((1-θ)f(t,y) + θ f(t+h,x))] = I - hθ·∂f/∂y.
            jf = jac(t_new, x) if jac is not None else _finite_difference_jacobian(f, t_new, x)
            n = len(x)
            return [
                [(1.0 if i == j else 0.0) - h * theta * jf[i][j] for j in range(n)]
                for i in range(n)
            ]

        y = _newton_solve(residual, system_jac, y, tol=tol, max_iter=max_iter)
        t = t_new
        t_vals.append(t)
        y_vals.append(list(y))

    return t_vals, y_vals


def _wrap_scalar_jac(jac: Callable[[float, float], float] | None) -> Jacobian | None:
    """Lift a scalar Jacobian ``df/dy`` into 1×1-matrix form."""
    if jac is None:
        return None

    def wrapped(t: float, y: list[float]) -> list[list[float]]:
        return [[jac(t, y[0])]]

    return wrapped


def backward_euler(
    f: Callable[[float, float], float],
    t0: float,
    y0: float,
    t_end: float,
    dt: float = RK45_DEFAULT_DT,
    *,
    jac: Callable[[float, float], float] | None = None,
    tol: float = DEFAULT_NEWTON_TOL,
    max_iter: int = DEFAULT_NEWTON_MAX_ITER,
) -> ODESolution:
    """Backward Euler for dy/dt = f(t, y): unconditionally stable, O(dt) accurate.

    The method of choice for stiff decay problems where explicit methods
    require absurdly small steps. [Hairer & Wanner 1996, §IV.3]

    Args:
        f: right-hand side function f(t, y)
        t0: initial time
        y0: initial value y(t0)
        t_end: end time (may be less than ``t0`` for backward integration)
        dt: time step magnitude (direction follows sign of ``t_end - t0``)
        jac: optional analytic df/dy(t, y); central differences otherwise
        tol: Newton convergence tolerance on the residual (∞-norm)
        max_iter: Newton iteration budget per step

    Returns:
        An :class:`ODESolution` with ``method == "backward-euler"``.

    Raises:
        ValueError: on invalid parameters or Newton failure (singular
            Jacobian / exhausted budget).
    """

    def f_sys(t: float, y: list[float]) -> list[float]:
        return [f(t, y[0])]

    t_vals, y_vals = _integrate_system(
        f_sys,
        t0,
        [y0],
        t_end,
        dt,
        theta=1.0,
        jac=_wrap_scalar_jac(jac),
        tol=tol,
        max_iter=max_iter,
    )
    return ODESolution(
        t=t_vals,
        y=[v[0] for v in y_vals],
        method="backward-euler",
        steps=len(t_vals) - 1,
    )


def trapezoid_method(
    f: Callable[[float, float], float],
    t0: float,
    y0: float,
    t_end: float,
    dt: float = RK45_DEFAULT_DT,
    *,
    jac: Callable[[float, float], float] | None = None,
    tol: float = DEFAULT_NEWTON_TOL,
    max_iter: int = DEFAULT_NEWTON_MAX_ITER,
) -> ODESolution:
    """Trapezoidal (Crank–Nicolson) method: A-stable, second order.

    Averages the explicit and implicit slopes, giving O(dt²) global error —
    strictly more accurate than backward Euler on smooth problems while
    remaining stable on stiff ones. [Crank & Nicolson 1947]

    Args:
        f: right-hand side function f(t, y)
        t0: initial time
        y0: initial value y(t0)
        t_end: end time (may be less than ``t0`` for backward integration)
        dt: time step magnitude (direction follows sign of ``t_end - t0``)
        jac: optional analytic df/dy(t, y); central differences otherwise
        tol: Newton convergence tolerance on the residual (∞-norm)
        max_iter: Newton iteration budget per step

    Returns:
        An :class:`ODESolution` with ``method == "trapezoid"``.

    Raises:
        ValueError: on invalid parameters or Newton failure (singular
            Jacobian / exhausted budget).
    """

    def f_sys(t: float, y: list[float]) -> list[float]:
        return [f(t, y[0])]

    t_vals, y_vals = _integrate_system(
        f_sys,
        t0,
        [y0],
        t_end,
        dt,
        theta=0.5,
        jac=_wrap_scalar_jac(jac),
        tol=tol,
        max_iter=max_iter,
    )
    return ODESolution(
        t=t_vals,
        y=[v[0] for v in y_vals],
        method="trapezoid",
        steps=len(t_vals) - 1,
    )


def backward_euler_system(
    f: SystemRHS,
    t0: float,
    y0: list[float],
    t_end: float,
    dt: float = RK45_DEFAULT_DT,
    *,
    jac: Jacobian | None = None,
    tol: float = DEFAULT_NEWTON_TOL,
    max_iter: int = DEFAULT_NEWTON_MAX_ITER,
) -> tuple[list[float], list[list[float]]]:
    """Backward Euler for coupled systems dy/dt = f(t, y).

    Args:
        f: right-hand side f(t, y) returning derivative vectors
        t0: initial time
        y0: initial state vector (non-empty)
        t_end: end time (may be less than ``t0`` for backward integration)
        dt: time step magnitude (direction follows sign of ``t_end - t0``)
        jac: optional analytic Jacobian matrix df_i/dy_j(t, y)
        tol: Newton convergence tolerance on the residual (∞-norm)
        max_iter: Newton iteration budget per step

    Returns:
        ``(t_values, y_values)`` with ``y_values[i]`` the state at
        ``t_values[i]`` — the same shape as :func:`cds.diffeq.solve_system`.

    Raises:
        ValueError: on invalid parameters or Newton failure.
    """
    return _integrate_system(
        f,
        t0,
        y0,
        t_end,
        dt,
        theta=1.0,
        jac=jac,
        tol=tol,
        max_iter=max_iter,
    )


def trapezoid_method_system(
    f: SystemRHS,
    t0: float,
    y0: list[float],
    t_end: float,
    dt: float = RK45_DEFAULT_DT,
    *,
    jac: Jacobian | None = None,
    tol: float = DEFAULT_NEWTON_TOL,
    max_iter: int = DEFAULT_NEWTON_MAX_ITER,
) -> tuple[list[float], list[list[float]]]:
    """Trapezoidal (Crank–Nicolson) method for coupled systems dy/dt = f(t, y).

    Args:
        f: right-hand side f(t, y) returning derivative vectors
        t0: initial time
        y0: initial state vector (non-empty)
        t_end: end time (may be less than ``t0`` for backward integration)
        dt: time step magnitude (direction follows sign of ``t_end - t0``)
        jac: optional analytic Jacobian matrix df_i/dy_j(t, y)
        tol: Newton convergence tolerance on the residual (∞-norm)
        max_iter: Newton iteration budget per step

    Returns:
        ``(t_values, y_values)`` with ``y_values[i]`` the state at
        ``t_values[i]`` — the same shape as :func:`cds.diffeq.solve_system`.

    Raises:
        ValueError: on invalid parameters or Newton failure.
    """
    return _integrate_system(
        f,
        t0,
        y0,
        t_end,
        dt,
        theta=0.5,
        jac=jac,
        tol=tol,
        max_iter=max_iter,
    )
