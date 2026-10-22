"""Minimization algorithms for single and multi-variable functions."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypedDict, TypeVar, cast, overload

from cds.core._numeric import (
    ADAM_DEFAULT_BETAS,
    ADAM_DEFAULT_EPS,
    ADAM_DEFAULT_LR,
    DEFAULT_FD_STEP,
    DEFAULT_TOLERANCE,
    GD_DEFAULT_LR,
    NEAR_ZERO,
    NEWTON_DERIVATIVE_STEP,
    NEWTON_TOLERANCE,
)
from cds.math_utils.calculus import derivative as _central_difference

# Type parameter for the minimizer payload: a single scalar (``float``) or a
# vector of scalars (``list[float]``). Constraining ``T`` to these two keeps
# :class:`OptResult` honest about what the minimizers actually return and lets
# callers read ``OptResult[float].x`` as a ``float`` without narrowing the union.
_T = TypeVar("_T", float, list[float])


# Adam optimizer state shape: {"m": float|[float], "v": float|[float], "t": int}.
# The scalar and vector variants are tracked separately so callers that resume a
# scalar run get a scalar back, and vice versa.
#
# Deliberately *not* generic: ``TypedDict`` + ``Generic`` inheritance raises
# ``TypeError: metaclass conflict`` on Python 3.10 (fixed upstream only in
# 3.11+, see CPython issue #89026), and CDS supports ``>=3.10``. The
# ``m``/``v`` union is narrowed inside :func:`adam` via ``cast`` instead.
class AdamState(TypedDict, total=False):
    """State checkpoint for resuming an Adam run.

    ``m`` / ``v`` are floats for scalar optimization and lists for vector
    optimization; ``t`` is the last completed iteration (0-based).
    """

    m: float | list[float]
    v: float | list[float]
    t: int


@dataclass
class OptResult(Generic[_T]):
    """Result of an optimization run.

    Parameterized over the ``x`` payload so the scalar and vector minimizers
    each advertise the concrete type they return: :func:`newton_method` and
    :func:`line_search` always return :class:`OptResult`\\ ``[float]``, while
    :func:`gradient_descent` and :func:`adam` return ``OptResult[float]`` or
    ``OptResult[list[float]]`` depending on whether ``x0`` was a scalar or a
    list (see their overloads).
    """

    x: _T
    value: float
    iterations: int
    converged: bool
    state: AdamState | None = None


@overload
def _compute_gradient(f: Callable[..., float], x: float, h_base: float = DEFAULT_FD_STEP) -> float:
    """Scalar overload: central-difference derivative at a single point."""


@overload
def _compute_gradient(
    f: Callable[..., float], x: list[float], h_base: float = DEFAULT_FD_STEP
) -> list[float]:
    """Vector overload: coordinate-wise gradient via finite differences."""


def _compute_gradient(
    f: Callable[..., float], x: float | list[float], h_base: float = DEFAULT_FD_STEP
) -> float | list[float]:
    """Compute numerical gradient with adaptive step size for precision.

    The scalar branch delegates to :func:`cds.math_utils.calculus.derivative`
    so the central-difference formula ``(f(x+h) - f(x-h)) / (2h)`` with an
    ``x``-scaled step lives in exactly one place. The vector branch is kept
    here because it uses the ``f(list)`` calling convention (positional index
    access), whereas :func:`cds.math_utils.calculus.gradient` expects
    ``f(*coords)`` — they are genuinely different APIs, not duplicates.
    """
    if isinstance(x, (int, float)):
        return _central_difference(f, x, h_base)

    grad = [0.0] * len(x)
    for i in range(len(x)):
        h = h_base * max(1.0, abs(x[i]))
        x_plus = x[:]
        x_plus[i] += h
        x_minus = x[:]
        x_minus[i] -= h
        grad[i] = float((f(x_plus) - f(x_minus)) / (2 * h))
    return grad


@overload
def _update_x(x: float, grad: float, step: float) -> float:
    """Scalar overload: return the next iterate ``x - step * grad``."""


@overload
def _update_x(x: list[float], grad: list[float], step: float) -> list[float]:
    """Vector overload: element-wise gradient step applied to each coordinate."""


def _update_x(
    x: float | list[float], grad: float | list[float], step: float
) -> float | list[float]:
    """Apply gradient update step for scalar or vector inputs."""
    if isinstance(x, (int, float)):
        # grad is the scalar branch here (see _compute_gradient return type)
        return x - step * (grad if isinstance(grad, float) else grad[0])

    grad_list: list[float] = grad if isinstance(grad, list) else [grad]
    return [xi - step * gi for xi, gi in zip(x, grad_list)]


def _magnitude(vec: float | list[float]) -> float:
    """Compute magnitude of a scalar or vector."""
    if isinstance(vec, (int, float)):
        return abs(vec)
    return math.sqrt(sum(vi * vi for vi in vec))


@overload
def gradient_descent(
    f: Callable[..., float],
    x0: float,
    lr: float = GD_DEFAULT_LR,
    tol: float = DEFAULT_TOLERANCE,
    max_iter: int = 10000,
    h: float = DEFAULT_FD_STEP,
) -> OptResult[float]:
    """Scalar overload: minimize a 1-D function from a scalar start."""


@overload
def gradient_descent(
    f: Callable[..., float],
    x0: list[float],
    lr: float = GD_DEFAULT_LR,
    tol: float = DEFAULT_TOLERANCE,
    max_iter: int = 10000,
    h: float = DEFAULT_FD_STEP,
) -> OptResult[list[float]]:
    """Vector overload: minimize a multivariate function from a vector start."""


def gradient_descent(
    f: Callable[..., float],
    x0: float | list[float],
    lr: float = GD_DEFAULT_LR,
    tol: float = DEFAULT_TOLERANCE,
    max_iter: int = 10000,
    h: float = DEFAULT_FD_STEP,
) -> OptResult[float] | OptResult[list[float]]:
    """Minimize a scalar or vector function using gradient descent.

    Args:
        f: objective function
        x0: starting point (scalar or list of floats)
        lr: learning rate
        tol: convergence tolerance on gradient magnitude
        max_iter: iteration limit
        h: step size for numerical gradient
    """
    if isinstance(x0, (int, float)):
        # Scalar branch — typed so OptResult[float] is returned without a cast.
        x: float = x0
        for i in range(max_iter):
            grad = _compute_gradient(f, x, h)
            if _magnitude(grad) < tol:
                return OptResult(x=x, value=f(x), iterations=i, converged=True)
            x = _update_x(x, grad, lr)
        return OptResult(x=x, value=f(x), iterations=max_iter, converged=False)

    # Vector branch — typed so OptResult[list[float]] is returned without a cast.
    # ``grad_vec`` (not ``grad``) so mypy keeps the scalar/vector types separate:
    # the scalar branch above binds the name ``grad`` to ``float``, and a single
    # function scope can't hold both ``float`` and ``list[float]`` for one name.
    x_vec: list[float] = list(x0)
    for i in range(max_iter):
        grad_vec = _compute_gradient(f, x_vec, h)
        if _magnitude(grad_vec) < tol:
            return OptResult(x=x_vec, value=f(x_vec), iterations=i, converged=True)
        x_vec = _update_x(x_vec, grad_vec, lr)
    return OptResult(x=x_vec, value=f(x_vec), iterations=max_iter, converged=False)


def newton_method(
    f: Callable[[float], float],
    x0: float,
    tol: float = NEWTON_TOLERANCE,
    max_iter: int = 1000,
    h_base: float = NEWTON_DERIVATIVE_STEP,
) -> OptResult[float]:
    """Find a root of f using Newton-Raphson method with adaptive step size.

    Args:
        f: function whose root to find
        x0: starting point
        tol: convergence tolerance
        max_iter: iteration limit
        h_base: base step for numerical derivative
    """
    x = float(x0)
    for i in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return OptResult(x=x, value=fx, iterations=i, converged=True)

        # Newton's derivative comes from the same central-difference kernel as
        # the gradient methods (``h_base`` defaults to NEWTON_DERIVATIVE_STEP,
        # finer than the gradient DEFAULT_FD_STEP).
        dfx = _compute_gradient(f, x, h_base)

        if abs(dfx) < NEAR_ZERO:
            return OptResult(
                x=x,
                value=f(x),
                iterations=i,
                converged=False,
            )
        x -= fx / dfx
    return OptResult(
        x=x,
        value=f(x),
        iterations=max_iter,
        converged=False,
    )


@overload
def adam(
    f: Callable[..., float],
    x0: float,
    lr: float = ADAM_DEFAULT_LR,
    beta1: float = ADAM_DEFAULT_BETAS[0],
    beta2: float = ADAM_DEFAULT_BETAS[1],
    eps: float = ADAM_DEFAULT_EPS,
    tol: float = DEFAULT_TOLERANCE,
    max_iter: int = 10000,
    h: float = DEFAULT_FD_STEP,
    state: AdamState | None = None,
    grad_f: Callable[..., float | list[float]] | None = None,
) -> OptResult[float]:
    """Scalar overload: adapt the learning rate per-iteration for a scalar x0."""


@overload
def adam(
    f: Callable[..., float],
    x0: list[float],
    lr: float = ADAM_DEFAULT_LR,
    beta1: float = ADAM_DEFAULT_BETAS[0],
    beta2: float = ADAM_DEFAULT_BETAS[1],
    eps: float = ADAM_DEFAULT_EPS,
    tol: float = DEFAULT_TOLERANCE,
    max_iter: int = 10000,
    h: float = DEFAULT_FD_STEP,
    state: AdamState | None = None,
    grad_f: Callable[..., float | list[float]] | None = None,
) -> OptResult[list[float]]:
    """Vector overload: coordinate-wise adaptive moments for a vector x0."""


def adam(
    f: Callable[..., float],
    x0: float | list[float],
    lr: float = ADAM_DEFAULT_LR,
    beta1: float = ADAM_DEFAULT_BETAS[0],
    beta2: float = ADAM_DEFAULT_BETAS[1],
    eps: float = ADAM_DEFAULT_EPS,
    tol: float = DEFAULT_TOLERANCE,
    max_iter: int = 10000,
    h: float = DEFAULT_FD_STEP,
    state: AdamState | None = None,
    grad_f: Callable[..., float | list[float]] | None = None,
) -> OptResult[float] | OptResult[list[float]]:
    """Minimize using Adam optimizer (adaptive learning rate) for scalars or vectors.

    Args:
        f: objective function
        x0: starting point
        lr: learning rate
        beta1: first moment decay
        beta2: second moment decay
        eps: numerical stability constant
        tol: convergence tolerance
        max_iter: iteration limit
        h: step for numerical gradient
        state: optional dictionary to resume optimization (contains m, v, t)
        grad_f: optional gradient function. If None, numerical gradient is used.
    """
    if isinstance(x0, (int, float)):
        x_scalar: float = float(x0)

        if state is None:
            m_s = 0.0
            v_s = 0.0
            t_start = 1
        else:
            m_s = float(cast(float, state["m"]))
            v_s = float(cast(float, state["v"]))
            t_start = int(state["t"]) + 1

        last_t = t_start - 1
        for i in range(t_start, t_start + max_iter):
            last_t = i
            if grad_f:
                grad_s: float = float(cast(float, grad_f(x_scalar)))
            else:
                grad_s = _compute_gradient(f, x_scalar, h)

            if abs(grad_s) < tol:
                return OptResult(
                    x=x_scalar,
                    value=f(x_scalar),
                    iterations=i - t_start + 1,
                    converged=True,
                    state={"m": m_s, "v": v_s, "t": i},
                )
            m_s = beta1 * m_s + (1 - beta1) * grad_s
            v_s = beta2 * v_s + (1 - beta2) * grad_s**2
            m_hat = m_s / (1 - beta1**i)
            v_hat = v_s / (1 - beta2**i)
            x_scalar -= lr * m_hat / (math.sqrt(v_hat) + eps)

        return OptResult(
            x=x_scalar,
            value=f(x_scalar),
            iterations=max_iter,
            converged=False,
            state={"m": m_s, "v": v_s, "t": last_t},
        )
    else:
        x_list: list[float] = list(x0)

        if state is None:
            m_l = [0.0] * len(x_list)
            v_l = [0.0] * len(x_list)
            t_start = 1
        else:
            m_l = list(cast(list[float], state["m"]))
            v_l = list(cast(list[float], state["v"]))
            t_start = int(state["t"]) + 1

        last_t = t_start - 1
        for i in range(t_start, t_start + max_iter):
            last_t = i
            if grad_f:
                grad_l: list[float] = list(cast(list[float], grad_f(x_list)))
            else:
                grad_l = _compute_gradient(f, x_list, h)

            if _magnitude(grad_l) < tol:
                return OptResult(
                    x=x_list,
                    value=f(x_list),
                    iterations=i - t_start + 1,
                    converged=True,
                    state={"m": m_l, "v": v_l, "t": i},
                )

            for j in range(len(x_list)):
                m_l[j] = beta1 * m_l[j] + (1 - beta1) * grad_l[j]
                v_l[j] = beta2 * v_l[j] + (1 - beta2) * grad_l[j] ** 2
                m_hat = m_l[j] / (1 - beta1**i)
                v_hat = v_l[j] / (1 - beta2**i)
                x_list[j] -= lr * m_hat / (math.sqrt(v_hat) + eps)

        return OptResult(
            x=x_list,
            value=f(x_list),
            iterations=max_iter,
            converged=False,
            state={"m": m_l, "v": v_l, "t": last_t},
        )


def line_search(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = DEFAULT_TOLERANCE,
    max_iter: int = 100,
) -> OptResult[float]:
    """Golden section search for minimum in [a, b].

    Args:
        f: unimodal function to minimize
        a: left bound
        b: right bound
        tol: convergence tolerance on interval width
        max_iter: iteration limit
    """
    phi = (math.sqrt(5) - 1) / 2
    if a > b:
        raise ValueError(f"line_search requires a <= b (got a={a}, b={b})")
    for i in range(max_iter):
        if abs(b - a) < tol:
            mid = (a + b) / 2
            return OptResult(
                x=mid,
                value=f(mid),
                iterations=i,
                converged=True,
            )
        x1 = b - phi * (b - a)
        x2 = a + phi * (b - a)
        if f(x1) < f(x2):
            b = x2
        else:
            a = x1
    mid = (a + b) / 2
    return OptResult(
        x=mid,
        value=f(mid),
        iterations=max_iter,
        converged=False,
    )
