"""Finite-difference PDE solvers on uniform 1-D grids — zero dependencies.

Explicit time marching for the two classic model equations of mathematical
physics, discretised on a uniform grid of ``nx`` points spanning
``[0, length]`` with spacing ``dx = length / (nx - 1)``:

Heat equation ``u_t = alpha * u_xx`` (parabolic)
    Forward-Time Central-Space (FTCS). Every interior point is blended with
    its two neighbours each step::

        u[i] <- u[i] + r * (u[i-1] - 2*u[i] + u[i+1]),   r = alpha*dt/dx**2

    FTCS is stable if and only if ``r <= 0.5`` (von Neumann analysis). When
    ``dt`` is omitted the solver picks ``dt = 0.9 * dx**2 / (2 * alpha)``,
    i.e. the stability limit shrunk by a 10 % safety margin giving
    ``r = 0.45``; a caller-supplied ``dt`` with ``r > 0.5`` is rejected.

Wave equation ``u_tt = c**2 * u_xx`` (hyperbolic)
    Explicit central differences in space and time (three-level leapfrog)::

        u_new[i] <- 2*u[i] - u_old[i] + C**2 * (u[i-1] - 2*u[i] + u[i+1])

    with Courant number ``C = c*dt/dx``, subject to the CFL condition
    ``C <= 1``. The first step is seeded from the initial velocity ``v0``
    with the matching second-order Taylor expansion
    ``u(dt) = u0 + dt*v0 + dt**2/2 * c**2 * u_xx``. When ``dt`` is omitted
    the solver picks ``dt = 0.9 * dx / c`` (``C = 0.9``); a caller-supplied
    ``dt`` with ``C > 1`` is rejected.

Both solvers march a whole number of uniform steps,
``n_steps = ceil(t_final / dt)``, so the simulated horizon ``n_steps * dt``
matches ``t_final`` up to one step. Boundary handling: ``"dirichlet"`` pins
the two end values at their initial levels; ``"neumann"`` mirrors the
adjacent interior point onto each end after every step (zero-flux, insulated
ends). All failures surface as lowercase :class:`ValueError`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["HeatResult", "WaveResult", "solve_heat", "solve_wave"]

_ALLOWED_BOUNDARIES = ("dirichlet", "neumann")
_HEAT_SAFETY = 0.9
_WAVE_SAFETY = 0.9
_MAX_HEAT_COURANT = 0.5
_MAX_WAVE_COURANT = 1.0


@dataclass
class HeatResult:
    """Outcome of one heat-equation integration.

    Attributes:
        u_final: temperature profile after the last step, one entry per
            grid point, ordered left to right.
        dt: time step actually used (auto-chosen or caller-supplied).
        n_steps: number of uniform FTCS steps taken; the simulated horizon
            is ``n_steps * dt``.
    """

    u_final: list[float]
    dt: float
    n_steps: int


@dataclass
class WaveResult:
    """Outcome of one wave-equation integration.

    Attributes:
        u_final: displacement profile after the last step, one entry per
            grid point, ordered left to right.
        dt: time step actually used (auto-chosen or caller-supplied).
        n_steps: number of uniform leapfrog steps taken; the simulated
            horizon is ``n_steps * dt``.
    """

    u_final: list[float]
    dt: float
    n_steps: int


def _check_grid(u0: list[float], nx: int, length: float, t_final: float) -> None:
    """Validate the arguments shared by both solvers.

    Args:
        u0: initial profile candidate.
        nx: declared grid size.
        length: domain length.
        t_final: requested simulation horizon.

    Raises:
        ValueError: if ``u0`` does not hold exactly ``nx`` points, ``nx``
            is below 3, or ``length`` / ``t_final`` is not strictly
            positive.
    """
    if len(u0) != nx:
        raise ValueError(f"u0 must have exactly {nx} points, got {len(u0)}")
    if nx < 3:
        raise ValueError("nx must be at least 3")
    if length <= 0:
        raise ValueError("length must be positive")
    if t_final <= 0:
        raise ValueError("t_final must be positive")


def _pick_dt(dt: float | None, auto_dt: float) -> float:
    """Resolve the time step, rejecting non-positive caller-supplied values.

    Args:
        dt: caller-supplied step, or ``None`` to take ``auto_dt`` — the
            pre-computed automatic choice, which already carries the scheme
            safety margin and therefore always satisfies stability.
        auto_dt: automatic step used when ``dt`` is ``None``.

    Returns:
        The time step to march with.

    Raises:
        ValueError: if a caller-supplied ``dt`` is not strictly positive.
    """
    if dt is None:
        return auto_dt
    if dt <= 0:
        raise ValueError("dt must be positive")
    return dt


def solve_heat(
    u0: list[float],
    alpha: float,
    length: float,
    t_final: float,
    nx: int,
    *,
    boundary: str = "dirichlet",
    dt: float | None = None,
) -> HeatResult:
    """Integrate the 1-D heat equation with explicit FTCS time stepping.

    Marches ``n_steps = ceil(t_final / dt)`` uniform applications of the
    three-point stencil ``u += r * (left - 2*u + right)`` with
    ``r = alpha*dt/dx**2``. When ``dt`` is omitted it defaults to
    ``0.9 * dx**2 / (2 * alpha)`` — the FTCS stability limit
    ``alpha*dt/dx**2 <= 0.5`` shrunk by a 10 % safety margin (``r = 0.45``).

    Args:
        u0: initial temperatures, one per grid point, left to right.
        alpha: thermal diffusivity, must be positive.
        length: domain length; grid spacing is ``length / (nx - 1)``.
        t_final: target simulation horizon.
        nx: number of grid points, at least 3.
        boundary: ``"dirichlet"`` pins the end values at ``u0[0]`` /
            ``u0[-1]`` for all time; ``"neumann"`` mirrors the neighbouring
            interior value onto each end (zero flux, insulated ends).
        dt: optional forced time step. Must satisfy the stability condition
            ``alpha*dt/dx**2 <= 0.5`` or the call is rejected.

    Returns:
        A :class:`HeatResult` holding the final profile, the time step used
        and the number of steps taken.

    Raises:
        ValueError: if ``len(u0) != nx``, ``nx < 3``, ``alpha <= 0``,
            ``length <= 0``, ``t_final <= 0``, ``boundary`` is neither
            ``"dirichlet"`` nor ``"neumann"``, ``dt`` is not strictly
            positive, or ``dt`` breaks the stability condition.
    """
    _check_grid(u0, nx, length, t_final)
    if boundary not in _ALLOWED_BOUNDARIES:
        raise ValueError(f"boundary must be 'dirichlet' or 'neumann', got {boundary!r}")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    dx = length / (nx - 1)
    dt_step = _pick_dt(dt, _HEAT_SAFETY * dx * dx / (2.0 * alpha))
    r = alpha * dt_step / (dx * dx)
    if r > _MAX_HEAT_COURANT:
        raise ValueError(
            f"dt={dt} violates heat stability: alpha*dt/dx**2 = {r:.4g} exceeds {_MAX_HEAT_COURANT}"
        )
    n_steps = math.ceil(t_final / dt_step)
    u = [float(value) for value in u0]
    for _ in range(n_steps):
        new_u = list(u)
        for i in range(1, nx - 1):
            new_u[i] = u[i] + r * (u[i - 1] - 2.0 * u[i] + u[i + 1])
        if boundary == "neumann":
            new_u[0] = new_u[1]
            new_u[-1] = new_u[-2]
        u = new_u
    return HeatResult(u_final=u, dt=dt_step, n_steps=n_steps)


def solve_wave(
    u0: list[float],
    v0: list[float],
    c: float,
    length: float,
    t_final: float,
    nx: int,
    *,
    dt: float | None = None,
) -> WaveResult:
    """Integrate the 1-D wave equation with explicit central differences.

    Uses the three-level leapfrog stencil
    ``u_new = 2*u - u_old + C**2 * (left - 2*u + right)`` with
    ``C = c*dt/dx``, seeded by the second-order Taylor step
    ``u(dt) = u0 + dt*v0 + dt**2/2 * c**2 * u_xx``. When ``dt`` is omitted
    it defaults to ``0.9 * dx / c`` — the CFL limit ``c*dt/dx <= 1`` shrunk
    by a 10 % safety margin. Ends are Dirichlet-pinned at ``u0[0]`` /
    ``u0[-1]``.

    Args:
        u0: initial displacements, one per grid point, left to right.
        v0: initial velocities aligned with ``u0``.
        c: wave speed, must be positive.
        length: domain length; grid spacing is ``length / (nx - 1)``.
        t_final: target simulation horizon.
        nx: number of grid points, at least 3.
        dt: optional forced time step. Must satisfy the CFL condition
            ``c*dt/dx <= 1`` or the call is rejected.

    Returns:
        A :class:`WaveResult` holding the final displacement profile, the
        time step used and the number of steps taken.

    Raises:
        ValueError: if ``len(u0) != nx`` or ``len(v0) != nx``, ``nx < 3``,
            ``c <= 0``, ``length <= 0``, ``t_final <= 0``, ``dt`` is not
            strictly positive, or ``dt`` breaks the CFL condition.
    """
    _check_grid(u0, nx, length, t_final)
    if len(v0) != nx:
        raise ValueError(f"v0 must have exactly {nx} points, got {len(v0)}")
    if c <= 0:
        raise ValueError("c must be positive")
    dx = length / (nx - 1)
    dt_step = _pick_dt(dt, _WAVE_SAFETY * dx / c)
    courant = c * dt_step / dx
    if courant > _MAX_WAVE_COURANT:
        raise ValueError(
            f"dt={dt} violates wave CFL: c*dt/dx = {courant:.4g} exceeds {_MAX_WAVE_COURANT}"
        )
    lam = courant * courant
    n_steps = math.ceil(t_final / dt_step)
    u_prev = [float(value) for value in u0]
    vel = [float(value) for value in v0]

    # First step: Taylor seed built from the initial velocity.
    u_curr = list(u_prev)
    for i in range(1, nx - 1):
        curvature = u_prev[i - 1] - 2.0 * u_prev[i] + u_prev[i + 1]
        u_curr[i] = u_prev[i] + dt_step * vel[i] + 0.5 * lam * curvature

    # Remaining steps: three-level central difference in time.
    for _ in range(n_steps - 1):
        u_next = list(u_curr)
        for i in range(1, nx - 1):
            curvature = u_curr[i - 1] - 2.0 * u_curr[i] + u_curr[i + 1]
            u_next[i] = 2.0 * u_curr[i] - u_prev[i] + lam * curvature
        u_prev, u_curr = u_curr, u_next
    return WaveResult(u_final=u_curr, dt=dt_step, n_steps=n_steps)
