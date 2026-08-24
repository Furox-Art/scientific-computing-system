"""Constrained optimization: projected gradient descent and quadratic penalties.

Two complementary wrappers around the unconstrained machinery in
:mod:`cds.optimization.minimize`:

- :func:`projected_gradient_descent` minimizes over an axis-aligned box by
  clamping every gradient step componentwise back into ``[lower, upper]``.
- :func:`quadratic_penalty` converts inequality constraints ``c(x) <= 0``
  into a sequence of unconstrained solves of
  ``fun(x) + 0.5 * penalty * sum(max(0, c(x))^2)`` with a growing penalty.

Both reuse :func:`cds.optimization.minimize._compute_gradient` (central
differences, ``f(list)`` calling convention) so numerical-gradient behavior is
identical to the existing solvers; :func:`quadratic_penalty` additionally
delegates its inner solves to :func:`cds.optimization.minimize.gradient_descent`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from cds.optimization.minimize import _compute_gradient, gradient_descent

# Tolerance for the inner gradient-descent solves of the penalized objective.
# Chosen above the central-difference noise floor (~1e-8 for O(1) objectives)
# so inner convergence is deterministic across penalty scales.
_INNER_TOLERANCE: float = 1e-6

# Cap on the inner learning-rate fraction: the inner step is
# ``_INNER_LR_CAP / (1 + penalty)``, which keeps plain gradient descent stable
# as the penalized-objective curvature grows proportionally to ``penalty``.
_INNER_LR_CAP: float = 0.5


@dataclass
class ProjectedGradientResult:
    """Result of a constrained optimization run.

    Attributes:
        x: final iterate (feasible for the box in the projected case; the
            penalty method only approximately enforces its constraints).
        fun: objective value ``fun(x)`` at the returned point.
        iterations: number of optimizer steps performed (total inner steps
            summed over penalty rounds for :func:`quadratic_penalty`).
        converged: whether the stopping test fired before ``max_iter``.
    """

    x: list[float]
    fun: float
    iterations: int
    converged: bool


def projected_gradient_descent(
    fun: Callable[[list[float]], float],
    x0: list[float],
    lower: list[float],
    upper: list[float],
    *,
    lr: float = 0.1,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> ProjectedGradientResult:
    """Minimize ``fun`` over the axis-aligned box ``lower[i] <= x[i] <= upper[i]``.

    Each iteration computes a central-difference gradient, takes the plain
    step ``x - lr * grad``, and projects componentwise onto the box. The run
    stops when the projected update changes no coordinate by more than
    ``tol``, i.e. when ``max_i |clip(x_i - lr * g_i) - x_i| < tol`` (the
    infinity norm of the projected step), or after ``max_iter`` iterations;
    ``converged`` reports which event occurred.

    Args:
        fun: scalar objective taking a feature vector.
        x0: starting point (length >= 1); out-of-box components are pulled
            inside by the first projection.
        lower: per-axis lower bounds, same length as ``x0``.
        upper: per-axis upper bounds, same length as ``x0``.
        lr: gradient-step size (> 0).
        max_iter: maximum number of projected steps (>= 1).
        tol: convergence threshold on the infinity norm of the projected
            update (> 0).

    Returns:
        :class:`ProjectedGradientResult` with the final iterate (always inside
        the box), its objective value, iterations performed, and the
        convergence flag.

    Raises:
        ValueError: if ``x0`` is empty, ``lower`` or ``upper`` does not match
            the length of ``x0``, ``lr`` <= 0, ``max_iter`` < 1, ``tol`` <= 0,
            or any ``lower[i]`` exceeds ``upper[i]``.
    """
    if not x0:
        raise ValueError("x0 must be non-empty")
    if len(lower) != len(x0):
        raise ValueError("lower must have the same length as x0")
    if len(upper) != len(x0):
        raise ValueError("upper must have the same length as x0")
    if lr <= 0:
        raise ValueError("lr must be positive")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    if tol <= 0:
        raise ValueError("tol must be positive")
    if any(lo > hi for lo, hi in zip(lower, upper)):
        raise ValueError("each bound must satisfy lower <= upper")

    x_vec: list[float] = list(x0)
    for i in range(max_iter):
        grad = _compute_gradient(fun, x_vec)
        projected = [
            min(max(xi - lr * gi, lo), hi) for xi, gi, lo, hi in zip(x_vec, grad, lower, upper)
        ]
        step_inf = max(abs(new - old) for new, old in zip(projected, x_vec))
        x_vec = projected
        if step_inf < tol:
            return ProjectedGradientResult(
                x=x_vec, fun=fun(x_vec), iterations=i + 1, converged=True
            )
    return ProjectedGradientResult(x=x_vec, fun=fun(x_vec), iterations=max_iter, converged=False)


def _penalized_objective(
    fun: Callable[[list[float]], float],
    constraints: Sequence[Callable[[list[float]], float]],
    penalty: float,
) -> Callable[[list[float]], float]:
    """Build ``fun(z) + 0.5 * penalty * sum(max(0, c(z))^2)`` over ``constraints``.

    Constraints are expressed as ``c(x) <= 0`` when satisfied; only violated
    constraints (positive ``c(x)``) contribute to the penalty term.
    """

    def objective(z: list[float]) -> float:
        violation = sum(max(0.0, c(z)) ** 2 for c in constraints)
        return fun(z) + 0.5 * penalty * violation

    return objective


def quadratic_penalty(
    fun: Callable[[list[float]], float],
    constraints: Sequence[Callable[[list[float]], float]],
    x0: list[float],
    *,
    penalty0: float = 1.0,
    growth: float = 10.0,
    rounds: int = 6,
    max_iter: int = 200,
) -> ProjectedGradientResult:
    """Minimize ``fun`` subject to ``c(x) <= 0`` via a growing quadratic penalty.

    Each of ``rounds`` rounds solves the unconstrained problem
    ``fun(x) + 0.5 * penalty * sum(max(0, c(x))^2)`` with plain gradient
    descent (:func:`cds.optimization.minimize.gradient_descent`, central-
    difference gradients), warm-starting every round from the previous
    solution. The penalty starts at ``penalty0`` and is multiplied by
    ``growth`` after each round, so the final iterate trades a residual
    constraint violation of order ``O(1 / penalty)`` against conditioning.
    Inner learning rates use ``_INNER_LR_CAP / (1 + penalty)`` to remain
    stable as the penalized curvature grows with the penalty.

    Args:
        fun: scalar objective taking a feature vector.
        constraints: callables ``c(x)`` satisfied when ``c(x) <= 0``.
        x0: starting point (length >= 1).
        penalty0: initial penalty weight (> 0).
        growth: multiplicative penalty schedule factor (> 1).
        rounds: number of penalty rounds (>= 1).
        max_iter: gradient-descent iteration budget per round (>= 1).

    Returns:
        :class:`ProjectedGradientResult` where ``fun`` is the original
        objective at the final iterate, ``iterations`` sums the inner steps
        over all rounds, and ``converged`` reflects the final inner solve.

    Raises:
        ValueError: if ``x0`` is empty, ``penalty0`` <= 0, ``growth`` <= 1,
            ``rounds`` < 1, or ``max_iter`` < 1.
    """
    if not x0:
        raise ValueError("x0 must be non-empty")
    if penalty0 <= 0:
        raise ValueError("penalty0 must be positive")
    if growth <= 1:
        raise ValueError("growth must be greater than 1")
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")

    x_vec: list[float] = list(x0)
    penalty = penalty0
    iterations_total = 0
    converged = False
    for _ in range(rounds):
        inner = gradient_descent(
            _penalized_objective(fun, constraints, penalty),
            x_vec,
            lr=_INNER_LR_CAP / (1.0 + penalty),
            tol=_INNER_TOLERANCE,
            max_iter=max_iter,
        )
        x_vec = list(inner.x)
        iterations_total += inner.iterations
        converged = inner.converged
        penalty *= growth
    return ProjectedGradientResult(
        x=x_vec, fun=fun(x_vec), iterations=iterations_total, converged=converged
    )
