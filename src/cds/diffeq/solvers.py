"""Numerical ODE solvers.

References:
    - Euler, L. (1768). Institutionum calculi integralis.
    - Runge, C. (1895). Über die numerische Auflösung von Differentialgleichungen.
    - Kutta, M.W. (1901). Beitrag zur näherungsweisen Integration totaler DGL.
    - Butcher, J.C. Numerical Methods for Ordinary Differential Equations (3rd ed.)
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from dataclasses import dataclass

from cds.core._numeric import (
    LOOP_EPSILON,
    RK45_DEFAULT_ATOL,
    RK45_DEFAULT_DT,
    RK45_DEFAULT_RTOL,
    RK45_STEP_GROW,
    RK45_STEP_SAFETY,
    RK45_STEP_SHRINK,
)


@dataclass
class ODESolution:
    """Result of an ODE integration."""

    t: list[float]
    y: list[float]
    method: str
    steps: int


def euler_method(
    f: Callable[[float, float], float],
    t0: float,
    y0: float,
    t_end: float,
    dt: float = RK45_DEFAULT_DT,
) -> ODESolution:
    """Euler's method for dy/dt = f(t, y).

    First-order explicit method. Local truncation error O(dt²),
    global error O(dt). [Euler 1768]

    Args:
        f: right-hand side function f(t, y)
        t0: initial time
        y0: initial value y(t0)
        t_end: end time (may be less than ``t0`` for backward integration)
        dt: time step (always positive; direction follows the sign of ``t_end - t0``)
    """
    # Direction of integration: +1 forward, -1 backward. ``copysign(1, 0)``
    # is +1, so a zero-length span (``t_end == t0``) still skips the loop and
    # returns just the initial condition. Without this, ``t_end < t0`` made the
    # ``while t < t_end`` guard immediately False and the solver silently
    # returned the initial value instead of integrating backward.
    direction = math.copysign(1.0, t_end - t0)
    t_vals = [t0]
    y_vals = [y0]
    t, y = t0, y0
    steps = 0

    while (t_end - t) * direction > LOOP_EPSILON:
        h = direction * min(dt, abs(t_end - t))
        y = y + h * f(t, y)
        t = t + h
        t_vals.append(t)
        y_vals.append(y)
        steps += 1

    return ODESolution(t=t_vals, y=y_vals, method="euler", steps=steps)


def rk4(
    f: Callable[[float, float], float],
    t0: float,
    y0: float,
    t_end: float,
    dt: float = RK45_DEFAULT_DT,
) -> ODESolution:
    """Classical 4th-order Runge-Kutta method.

    Local truncation error O(dt⁵), global error O(dt⁴). [Runge 1895, Kutta 1901]

    The standard Butcher tableau:
        0   |
        1/2 | 1/2
        1/2 | 0   1/2
        1   | 0   0   1
        ----|----------------
            | 1/6 1/3 1/3 1/6

    Args:
        f: right-hand side function f(t, y)
        t0: initial time
        y0: initial value y(t0)
        t_end: end time (may be less than ``t0`` for backward integration)
        dt: time step (always positive; direction follows the sign of ``t_end - t0``)
    """
    # See ``euler_method`` for why direction is derived from the sign of
    # ``t_end - t0`` rather than reading ``dt``'s sign: it keeps ``dt`` an
    # unambiguous magnitude and makes backward integration actually progress.
    direction = math.copysign(1.0, t_end - t0)
    t_vals = [t0]
    y_vals = [y0]
    t, y = t0, y0
    steps = 0

    while (t_end - t) * direction > LOOP_EPSILON:
        h = direction * min(dt, abs(t_end - t))
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h * k1 / 2)
        k3 = f(t + h / 2, y + h * k2 / 2)
        k4 = f(t + h, y + h * k3)
        y = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t + h
        t_vals.append(t)
        y_vals.append(y)
        steps += 1

    return ODESolution(t=t_vals, y=y_vals, method="rk4", steps=steps)


def midpoint_method(
    f: Callable[[float, float], float],
    t0: float,
    y0: float,
    t_end: float,
    dt: float = RK45_DEFAULT_DT,
) -> ODESolution:
    """Explicit midpoint method (2nd-order Runge-Kutta).

    Local truncation error O(dt³), global error O(dt²).

    Args:
        f: right-hand side function f(t, y)
        t0: initial time
        y0: initial value y(t0)
        t_end: end time (may be less than ``t0`` for backward integration)
        dt: time step (always positive; direction follows the sign of ``t_end - t0``)
    """
    direction = math.copysign(1.0, t_end - t0)
    t_vals = [t0]
    y_vals = [y0]
    t, y = t0, y0
    steps = 0

    while (t_end - t) * direction > LOOP_EPSILON:
        h = direction * min(dt, abs(t_end - t))
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h * k1 / 2)
        y = y + h * k2
        t = t + h
        t_vals.append(t)
        y_vals.append(y)
        steps += 1

    return ODESolution(t=t_vals, y=y_vals, method="midpoint", steps=steps)


def rk45(
    f: Callable[[float, float], float],
    t0: float,
    y0: float,
    t_end: float,
    dt: float = RK45_DEFAULT_DT,
    atol: float = RK45_DEFAULT_ATOL,
    rtol: float = RK45_DEFAULT_RTOL,
) -> ODESolution:
    """Dormand-Prince (RK45) adaptive step-size method.

    Computes 4th and 5th order estimates to approximate local error
    and adjust the step size automatically. [Dormand & Prince 1980]

    Args:
        f: right-hand side f(t, y)
        t0: initial time
        y0: initial value
        t_end: end time (may be less than ``t0`` for backward integration)
        dt: initial time step (always positive; direction follows the sign of ``t_end - t0``)
        atol: absolute tolerance
        rtol: relative tolerance
    """
    # Dormand-Prince Butcher Tableau coefficients
    a = [0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1]
    b = [
        [],
        [1 / 5],
        [3 / 40, 9 / 40],
        [44 / 45, -56 / 15, 32 / 9],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
        [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
    ]
    c5 = [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]
    c4 = [5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40]

    # Direction of integration (+1 forward, -1 backward). The step size is
    # tracked as an always-positive magnitude ``h_mag`` and applied as
    # ``direction * h_mag`` so the min/max shrink/grow logic below keeps its
    # forward semantics regardless of integration direction.
    direction = math.copysign(1.0, t_end - t0)
    t, y = t0, y0
    t_vals = [t]
    y_vals = [y]
    h_mag = dt
    steps = 0

    # Absolute step-size floor, scaled to the integration span, below which no
    # further progress can be made (a "machine precision floor"). Prevents the
    # adaptive loop from spinning forever on stiff/diverging problems.
    span = abs(t_end - t0) if t_end != t0 else 1.0
    eps_floor = 16 * sys.float_info.epsilon * max(abs(t), span)

    while (t_end - t) * direction > LOOP_EPSILON:
        # Signed step for the RK stages and time advancement. Snap the last
        # step to land exactly on t_end when the remaining span is smaller
        # than the proposed magnitude (avoids overshooting the endpoint).
        if abs(t_end - t) < h_mag:
            h = t_end - t
            h_mag = abs(h)
        else:
            h = direction * h_mag

        k = [0.0] * 7
        k[0] = f(t, y)
        for i in range(1, 7):
            y_next = y + h * sum(b[i][j] * k[j] for j in range(i))
            k[i] = f(t + a[i] * h, y_next)

        # Estimate 5th and 4th order solutions
        y5 = y + h * sum(c5[i] * k[i] for i in range(7))
        y4 = y + h * sum(c4[i] * k[i] for i in range(7))

        # Local error estimate
        error = abs(y5 - y4)
        tolerance = atol + rtol * abs(y)

        if error <= tolerance:
            # Step accepted
            t += h
            y = y5
            t_vals.append(t)
            y_vals.append(y)
            steps += 1

        # Adjust step size (operates on the magnitude only, preserving direction)
        if error > 0:
            h_opt = h_mag * (tolerance / error) ** 0.2
            h_mag = min(
                max(RK45_STEP_SHRINK * h_mag, RK45_STEP_SAFETY * h_opt), RK45_STEP_GROW * h_mag
            )
        else:
            h_mag *= 10.0  # Error is zero, aggressively increase step up to max scale

        # Precision floor to prevent infinite loop: either the step size has
        # shrunk below the span-scaled epsilon floor, or it has become so small
        # that adding it to t makes no progress (t + h == t).
        if h_mag < eps_floor or t + direction * h_mag == t:
            raise RuntimeError("Step size h reached machine precision floor.")

    return ODESolution(t=t_vals, y=y_vals, method="rk45", steps=steps)


def _checked_derivative(
    f: Callable[[float, list[float]], list[float]], t: float, y: list[float], n: int
) -> list[float]:
    dy = f(t, y)
    if len(dy) != n:
        raise ValueError(
            f"ODE right-hand side returned {len(dy)} derivative(s) for a state of length {n}"
        )
    return dy


def solve_system(
    f: Callable[[float, list[float]], list[float]],
    t0: float,
    y0: list[float],
    t_end: float,
    dt: float = RK45_DEFAULT_DT,
) -> tuple[list[float], list[list[float]]]:
    """RK4 for systems of ODEs: dy/dt = f(t, y) where y is a vector.

    Args:
        f: right-hand side f(t, y) returning a list of derivatives
        t0: initial time
        y0: initial state vector
        t_end: end time (may be less than ``t0`` for backward integration)
        dt: time step (always positive; direction follows the sign of ``t_end - t0``)

    Returns:
        (t_values, y_values) where y_values[i] is the state vector at t_values[i]
    """
    direction = math.copysign(1.0, t_end - t0)
    n = len(y0)
    t_vals = [t0]
    y_vals = [list(y0)]
    t = t0
    y = list(y0)

    while (t_end - t) * direction > LOOP_EPSILON:
        h = direction * min(dt, abs(t_end - t))

        k1 = _checked_derivative(f, t, y, n)
        y_tmp = [y[i] + h * k1[i] / 2 for i in range(n)]
        k2 = _checked_derivative(f, t + h / 2, y_tmp, n)
        y_tmp = [y[i] + h * k2[i] / 2 for i in range(n)]
        k3 = _checked_derivative(f, t + h / 2, y_tmp, n)
        y_tmp = [y[i] + h * k3[i] for i in range(n)]
        k4 = _checked_derivative(f, t + h, y_tmp, n)

        y = [y[i] + (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(n)]
        t = t + h
        t_vals.append(t)
        y_vals.append(list(y))

    return t_vals, y_vals
