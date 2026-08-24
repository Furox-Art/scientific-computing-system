"""Symplectic integrators for separable Hamiltonian systems.

Given a Hamiltonian split as ``H(q, p, t) = K(p) + V(q, t)`` the caller
supplies the generalized force ``f(t, q, p) = p_dot = -dV/dq``; the canonical
equations ``q_dot = dK/dp`` and ``p_dot = f`` are then advanced with
structure-preserving fixed-step schemes. Unlike generic Runge-Kutta methods,
symplectic integrators conserve a nearby ("shadow") Hamiltonian exactly, so
the energy error stays bounded over arbitrarily long integrations instead of
drifting secularly.

Two schemes are provided:

- :func:`symplectic_euler` — first-order semi-implicit (Euler-Cromer) method.
- :func:`velocity_verlet` — second-order Störmer-Verlet / leapfrog method,
  the workhorse of molecular dynamics and celestial mechanics.

References:
    - Ruth, R.D. (1983). A canonical integrating technique. IEEE Trans. Nucl.
      Sci. 30(4), 2669-2671.
    - Verlet, L. (1967). Computer "experiments" on classical fluids. Phys.
      Rev. 159(1), 98-103.
    - Hairer, E., Lubich, C. & Wanner, G. (2006). Geometric Numerical
      Integration: Structure-Preserving Algorithms for Ordinary Differential
      Equations (2nd ed.), Springer, §VI.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from cds.core._numeric import LOOP_EPSILON

#: Right-hand side of the momentum equation: p_dot = f(t, q, p).
HamiltonianForce = Callable[[float, list[float], list[float]], list[float]]

#: One symplectic step: (t, q, p, h) -> (q_next, p_next).
StepFn = Callable[[float, list[float], list[float], float], tuple[list[float], list[float]]]


@dataclass(frozen=True)
class SymplecticSolution:
    """Result of a symplectic integration.

    Attributes:
        ts: time grid, starting at ``t0`` and ending exactly at ``t_end``.
        qs: coordinates; ``qs[i]`` is the state at ``ts[i]``. Row 0 repeats
            the initial condition.
        ps: momenta, parallel to ``qs``.
    """

    ts: list[float]
    qs: list[list[float]]
    ps: list[list[float]]


def _validate_state(q0: list[float], p0: list[float]) -> None:
    """Validate the initial-state arguments shared by both integrators.

    Raises:
        ValueError: when either state vector is empty or the two vectors
            disagree in length.
    """
    if not q0 or not p0:
        raise ValueError("q0 and p0 must be non-empty")
    if len(q0) != len(p0):
        raise ValueError("q0 and p0 must have the same length")


def _validate_span(t0: float, t_end: float, dt: float) -> None:
    """Validate the time-grid arguments shared by both integrators.

    Raises:
        ValueError: when ``dt`` is zero or ``t_end`` equals ``t0``.
    """
    if dt == 0:
        raise ValueError("dt must be non-zero")
    if t_end == t0:
        raise ValueError("t_end must be different from t0")


def _eval_force(force: HamiltonianForce, t: float, q: list[float], p: list[float]) -> list[float]:
    """Evaluate ``force`` and reject ragged output (dimension mismatch)."""
    p_dot = force(t, q, p)
    if len(p_dot) != len(q):
        raise ValueError("force must return p_dot with the same length as q")
    return p_dot


def _integrate(
    step: StepFn,
    q0: list[float],
    p0: list[float],
    t0: float,
    t_end: float,
    dt: float,
) -> SymplecticSolution:
    """Fixed-step driver shared by both symplectic schemes.

    Same direction rule as the explicit solvers (see ``euler_method``): the
    sign of ``t_end - t0`` decides forward/backward integration while ``dt``
    is an always-positive magnitude. The final step is truncated so the
    trajectory lands exactly on ``t_end``.
    """
    _validate_state(q0, p0)
    _validate_span(t0, t_end, dt)
    direction = math.copysign(1.0, t_end - t0)
    step_mag = abs(dt)
    t = t0
    q = list(q0)
    p = list(p0)
    ts = [t0]
    qs = [list(q0)]
    ps = [list(p0)]

    while (t_end - t) * direction > LOOP_EPSILON:
        remaining = abs(t_end - t)
        h = direction * min(step_mag, remaining)
        q, p = step(t, q, p, h)
        # Snap the final step onto the endpoint so ``ts[-1] == t_end`` holds
        # exactly instead of accumulating floating-point drift.
        t = t_end if remaining <= step_mag else t + h
        ts.append(t)
        qs.append(list(q))
        ps.append(list(p))

    return SymplecticSolution(ts=ts, qs=qs, ps=ps)


def symplectic_euler(
    force: HamiltonianForce,
    q0: list[float],
    p0: list[float],
    t0: float,
    t_end: float,
    dt: float,
) -> SymplecticSolution:
    """Semi-implicit (symplectic) Euler: first-order, bounded energy error.

    One step of size ``h`` reads::

        p <- p + h * f(t, q, p)
        q <- q + h * p_new

    The momenta are updated first and the positions use the *refreshed*
    momenta — the flip that makes Euler symplectic. Local error O(h²),
    global error O(h); for separable Hamiltonians the energy error remains
    bounded instead of growing secularly. [Ruth 1983]

    Args:
        force: generalized force f(t, q, p) returning p_dot = -dV/dq
        q0: initial coordinates (non-empty)
        p0: initial momenta (same length as ``q0``)
        t0: initial time
        t_end: end time (may be less than ``t0`` for backward integration)
        dt: step magnitude (non-zero; direction follows sign of ``t_end - t0``)

    Returns:
        A :class:`SymplecticSolution` whose first row is the initial state
        and whose last time equals ``t_end`` exactly.

    Raises:
        ValueError: on empty or mismatched state vectors, zero ``dt``,
            ``t_end == t0``, or a ``force`` whose output length disagrees
            with ``q0``.
    """

    def step(t: float, q: list[float], p: list[float], h: float) -> tuple[list[float], list[float]]:
        p_next = [pi + h * fi for pi, fi in zip(p, _eval_force(force, t, q, p))]
        q_next = [qi + h * pi for qi, pi in zip(q, p_next)]
        return q_next, p_next

    return _integrate(step, q0, p0, t0, t_end, dt)


def velocity_verlet(
    force: HamiltonianForce,
    q0: list[float],
    p0: list[float],
    t0: float,
    t_end: float,
    dt: float,
) -> SymplecticSolution:
    """Velocity Verlet (Störmer-Verlet): second-order symplectic scheme.

    One step of size ``h`` reads::

        p_half = p + f(t, q, p) * h / 2
        q      = q + p_half * h
        p      = p_half + f(t + h, q, p_half) * h / 2

    Kick-drift-kick form of the leapfrog. Local error O(h³), global error
    O(h²), and — being symplectic and time-reversible — an energy error that
    oscillates around the true value without secular drift, which makes it
    the standard integrator for long molecular-dynamics and orbital runs.
    [Verlet 1967; Hairer, Lubich & Wanner 2006, §VI]

    Args:
        force: generalized force f(t, q, p) returning p_dot = -dV/dq
        q0: initial coordinates (non-empty)
        p0: initial momenta (same length as ``q0``)
        t0: initial time
        t_end: end time (may be less than ``t0`` for backward integration)
        dt: step magnitude (non-zero; direction follows sign of ``t_end - t0``)

    Returns:
        A :class:`SymplecticSolution` whose first row is the initial state
        and whose last time equals ``t_end`` exactly.

    Raises:
        ValueError: on empty or mismatched state vectors, zero ``dt``,
            ``t_end == t0``, or a ``force`` whose output length disagrees
            with ``q0``.
    """

    def step(t: float, q: list[float], p: list[float], h: float) -> tuple[list[float], list[float]]:
        p_half = [pi + 0.5 * h * fi for pi, fi in zip(p, _eval_force(force, t, q, p))]
        q_next = [qi + h * phi for qi, phi in zip(q, p_half)]
        p_next = [
            phi + 0.5 * h * fi for phi, fi in zip(p_half, _eval_force(force, t + h, q_next, p_half))
        ]
        return q_next, p_next

    return _integrate(step, q0, p0, t0, t_end, dt)
