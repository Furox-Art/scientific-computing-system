"""Numerical optimization algorithms."""

from cds.optimization._metaheuristics import nelder_mead, simulated_annealing
from cds.optimization.constrained import (
    ProjectedGradientResult,
    projected_gradient_descent,
    quadratic_penalty,
)
from cds.optimization.minimize import (
    adam,
    gradient_descent,
    line_search,
    newton_method,
)

__all__ = [
    "gradient_descent",
    "line_search",
    "nelder_mead",
    "newton_method",
    "simulated_annealing",
    "adam",
    "ProjectedGradientResult",
    "projected_gradient_descent",
    "quadratic_penalty",
]
