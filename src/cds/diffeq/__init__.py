"""Ordinary differential equation solvers — Euler, RK4, RK45, implicit methods."""

from cds.diffeq._implicit import (
    backward_euler,
    backward_euler_system,
    trapezoid_method,
    trapezoid_method_system,
)
from cds.diffeq.solvers import (
    ODESolution,
    euler_method,
    midpoint_method,
    rk4,
    rk45,
    solve_system,
)
from cds.diffeq.symplectic import (
    SymplecticSolution,
    symplectic_euler,
    velocity_verlet,
)

__all__ = [
    "ODESolution",
    "backward_euler",
    "backward_euler_system",
    "euler_method",
    "midpoint_method",
    "rk4",
    "rk45",
    "solve_system",
    "trapezoid_method",
    "trapezoid_method_system",
    "SymplecticSolution",
    "symplectic_euler",
    "velocity_verlet",
]
