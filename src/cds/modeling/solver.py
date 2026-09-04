"""Numeric solvers built on the symbolic engine and :mod:`cds.optimization`.

This module is the bridge between *symbolic* expressions
(:mod:`cds.modeling.expression`) and the *numeric* solvers CDS already ships
(:mod:`cds.optimization`). It reuses those tested routines rather than
re-implementing root finding or least-squares fitting, so the symbolic and
numeric tracks stay consistent.

Two entry points:

* :func:`solve_equation` — find a root of ``f(x) = expr`` for a single
  variable, via Newton-Raphson (:func:`cds.optimization.newton_method`).
* :func:`fit_parameters` — fit a :class:`cds.modeling.MathModel`'s parameters
  to observed data by minimizing the residual sum of squares, via gradient
  descent (:func:`cds.optimization.gradient_descent`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from cds.core._numeric import DEFAULT_TOLERANCE, GD_DEFAULT_LR, NEWTON_TOLERANCE
from cds.modeling.expression import Expression
from cds.modeling.model import MathModel
from cds.optimization import gradient_descent, newton_method


@dataclass
class SolveResult:
    """Outcome of :func:`solve_equation` (root finding).

    Attributes:
        x: the root found.
        residual: ``|f(x)|`` at the root (should be near zero on convergence).
        iterations: number of Newton steps taken.
        converged: whether the residual dropped below tolerance.
    """

    x: float
    residual: float
    iterations: int
    converged: bool


@dataclass
class FitResult:
    """Outcome of :func:`fit_parameters` (least-squares fitting).

    Attributes:
        parameters: fitted values, keyed by parameter name.
        residual: final sum-of-squared-residuals objective value.
        iterations: number of gradient-descent steps taken.
        converged: whether the gradient magnitude dropped below tolerance.
    """

    parameters: dict[str, float]
    residual: float
    iterations: int
    converged: bool


def solve_equation(
    expr: Expression,
    variable: str,
    x0: float = 1.0,
    tol: float = NEWTON_TOLERANCE,
    max_iter: int = 1000,
) -> SolveResult:
    """Find a root of ``expr`` (i.e. solve ``expr = 0``) for one variable.

    Compiles ``expr`` to a callable and hands it to Newton-Raphson.

    Args:
        expr: the symbolic expression whose root to find.
        variable: the single free variable to solve for.
        x0: starting guess.
        tol: convergence tolerance on ``|expr(x)|``.
        max_iter: iteration cap.

    Returns:
        a :class:`SolveResult` describing the root found.

    Raises:
        ValueError: if ``variable`` is not free in ``expr`` (propagated from
            :meth:`Expression.to_func`).
    """
    f = expr.to_func(variable)
    opt = newton_method(f, x0=x0, tol=tol, max_iter=max_iter)
    # newton_method returns OptResult[float] (scalar root-finder), so opt.x is
    # statically a float — no runtime narrowing needed.
    return SolveResult(
        x=opt.x,
        residual=abs(opt.value),
        iterations=opt.iterations,
        converged=opt.converged,
    )


def fit_parameters(
    model: MathModel,
    observed: Sequence[tuple[dict[str, float], float]],
    parameter_names: Sequence[str],
    x0: Sequence[float] | None = None,
    *,
    target_label: str | None = None,
    lr: float = GD_DEFAULT_LR,
    tol: float = DEFAULT_TOLERANCE,
    max_iter: int = 10000,
) -> FitResult:
    """Fit a model's parameters to observed data via least squares.

    The objective minimised is the residual sum of squares between the
    model's prediction and the observed values, summed over all observations::

        L(p) = Σ_i (model.evaluate(obs_env_i)[target] - observed_value_i) ** 2

    where the ``target`` equation is either ``target_label`` or, if omitted,
    the model's first equation.

    Args:
        model: the :class:`MathModel` whose parameters are tuned.
        observed: a sequence of ``(env, value)`` pairs; each ``env`` provides
            the free-variable values for one observation and ``value`` the
            measured outcome to fit.
        parameter_names: unique parameter names to fit (order matches ``x0``
            and the returned :attr:`FitResult.parameters`).
        x0: starting guesses, positionally aligned with ``parameter_names``.
            Defaults to all-zeros. When provided, its length must exactly
            match ``parameter_names``.
        target_label: which equation's output to fit. If ``None``, the first
            equation in ``model.equations`` is used.
        lr: gradient-descent learning rate.
        tol: convergence tolerance on gradient magnitude.
        max_iter: iteration cap.

    Returns:
        a :class:`FitResult` with the fitted parameters.

    Raises:
        ValueError: if ``parameter_names`` is empty or contains duplicates,
            if the model has no equations, if there are no observations, if
            ``x0`` has the wrong length, or if ``target_label`` is unknown.
    """
    names = list(parameter_names)
    if not names:
        raise ValueError("parameter_names must list at least one parameter to fit")
    if len(set(names)) != len(names):
        raise ValueError("parameter_names must contain unique names")
    if not model.equations:
        raise ValueError("model must contain at least one equation to fit")

    observations = list(observed)
    if not observations:
        raise ValueError("observed must contain at least one (env, value) pair")

    start = list(x0) if x0 is not None else [0.0] * len(names)
    if len(start) != len(names):
        raise ValueError("x0 length must exactly match parameter_names")

    # Resolve the target equation once, after validating that the model has at
    # least one equation so the default-target path cannot leak an IndexError.
    if target_label is None:
        target_label = model.equations[0][0]
    target_expr = model.equation(target_label)
    base_params = dict(model.parameters)

    def objective(params: list[float]) -> float:
        env_overrides = dict(zip(names, params))
        params_full = {**base_params, **env_overrides}
        total = 0.0
        for env, observed_value in observations:
            merged = {**env, **params_full}
            predicted = target_expr.evaluate(merged)
            residual = predicted - observed_value
            total += residual * residual
        return total

    opt = gradient_descent(objective, x0=start, lr=lr, tol=tol, max_iter=max_iter)
    # gradient_descent's list-input overload returns OptResult[list[float]], so
    # opt.x is statically a list[float] — no runtime narrowing needed.
    return FitResult(
        parameters=dict(zip(names, opt.x)),
        residual=opt.value,
        iterations=opt.iterations,
        converged=opt.converged,
    )
