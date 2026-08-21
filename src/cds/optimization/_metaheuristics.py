"""Derivative-free optimizers: Nelder–Mead simplex and simulated annealing.

Both algorithms need no gradients, making them the tools of choice for
noisy / non-smooth / black-box objectives that defeat :func:`gradient_descent`
and :func:`newton_method`. Decision logic is factored into tiny pure helpers
(:func:`_nm_action`, :func:`_metropolis_accept`, :func:`_clamp`) so every
branch is directly unit-testable without depending on RNG luck.

References:
    - Nelder, J.A. & Mead, R. (1965). Computer Journal 7(4), 308-313.
    - Kirkpatrick, S., Gelatt, C.D. & Vecchi, M.P. (1983). Science 220(4598),
      671-680.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence

from cds.optimization.minimize import OptResult

Bound = tuple[float, float]


# --------------------------------------------------------------------- #
# Pure decision helpers                                                  #
# --------------------------------------------------------------------- #


def _nm_action(f_r: float, f_best: float, f_second_worst: float, f_worst: float) -> str:
    """Classify a reflected point into the classic Nelder–Mead moves."""
    if f_r < f_best:
        return "expand"
    if f_r < f_second_worst:
        return "reflect"
    if f_r < f_worst:
        return "contract-outside"
    return "contract-inside"


def _points_coincide(a: Sequence[float], b: Sequence[float]) -> bool:
    """True when two vertices are exactly equal component-wise.

    A contracted point that lands exactly on the centroid duplicates an
    existing vertex and collapses the simplex (the classic 1-D stagnation);
    callers respond by shrinking instead.
    """
    return len(a) == len(b) and all(xa == xb for xa, xb in zip(a, b))


def _metropolis_accept(delta: float, temperature: float, u: float) -> bool:
    """Metropolis rule: always take improvements, uphill with probability e^(-Δ/t)."""
    if delta <= 0:
        return True
    return u < math.exp(-delta / temperature)


def _clamp_point(x: list[float], bounds: Sequence[Bound] | None) -> list[float]:
    """Clamp a candidate point into the optional box constraints."""
    if bounds is None:
        return list(x)
    return [min(max(xi, lo), hi) for xi, (lo, hi) in zip(x, bounds)]


# --------------------------------------------------------------------- #
# Nelder–Mead                                                            #
# --------------------------------------------------------------------- #


def nelder_mead(
    f: Callable[[list[float]], float],
    x0: list[float],
    *,
    step: float = 0.05,
    tol_fx: float = 1e-12,
    tol_x: float = 1e-8,
    max_iter: int = 1000,
) -> OptResult[list[float]]:
    """Minimize ``f`` over R^n with the Nelder–Mead simplex method.

    Standard coefficients: reflection α=1, expansion γ=2, contraction ρ=½,
    shrink σ=½. Convergence requires BOTH the objective spread
    ``f_worst − f_best`` and the simplex diameter to fall below ``tol_fx`` /
    ``tol_x`` — the objective test alone falsely fires when the simplex
    becomes symmetric around an off-optimum point (e.g. ``x²`` straddling
    the origin at ±ε).

    Args:
        f: scalar objective taking a feature vector
        x0: starting point (length >= 1)
        step: per-axis offset used to build the initial simplex (> 0)
        tol_fx: convergence threshold on the simplex's objective spread (> 0)
        tol_x: convergence threshold on the simplex diameter (> 0)
        max_iter: maximum number of simplex updates

    Returns:
        :class:`OptResult` whose ``x`` is the best point found and
        ``converged`` reports whether both spread tests fired.

    Raises:
        ValueError: if ``x0`` is empty, any tolerance/step <= 0, or
            ``max_iter < 1``.
    """
    if not x0:
        raise ValueError("x0 must be non-empty")
    if step <= 0:
        raise ValueError("step must be positive")
    if tol_fx <= 0:
        raise ValueError("tol_fx must be positive")
    if tol_x <= 0:
        raise ValueError("tol_x must be positive")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")

    n = len(x0)
    simplex = [list(x0)]
    for d in range(n):
        vertex = list(x0)
        vertex[d] += step
        simplex.append(vertex)
    scores = [f(v) for v in simplex]

    def sort_simplex() -> None:
        order = sorted(range(n + 1), key=lambda i: scores[i])
        simplex[:] = [simplex[i] for i in order]
        scores[:] = [scores[i] for i in order]

    iterations = 0
    converged = False
    for iterations in range(1, max_iter + 1):
        sort_simplex()
        diameter = max(abs(v[d] - simplex[0][d]) for v in simplex[1:] for d in range(n))
        if scores[-1] - scores[0] < tol_fx and diameter < tol_x:
            converged = True
            break

        centroid = [sum(v[d] for v in simplex[:-1]) / n for d in range(n)]
        worst = simplex[-1]
        reflected = [centroid[d] + (centroid[d] - worst[d]) for d in range(n)]
        f_r = f(reflected)

        action = _nm_action(f_r, scores[0], scores[-2], scores[-1])
        if action == "expand":
            expanded = [centroid[d] + 2.0 * (reflected[d] - centroid[d]) for d in range(n)]
            f_e = f(expanded)
            if f_e < f_r:
                simplex[-1], scores[-1] = expanded, f_e
            else:
                simplex[-1], scores[-1] = reflected, f_r
        elif action == "reflect":
            simplex[-1], scores[-1] = reflected, f_r
        elif action == "contract-outside":
            contracted = [centroid[d] + 0.5 * (reflected[d] - centroid[d]) for d in range(n)]
            f_c = f(contracted)
            if f_c <= f_r and not _points_coincide(contracted, centroid):
                simplex[-1], scores[-1] = contracted, f_c
            else:
                _shrink(simplex, scores, f, n)
        else:  # contract-inside
            contracted = [centroid[d] + 0.5 * (worst[d] - centroid[d]) for d in range(n)]
            f_c = f(contracted)
            if f_c < scores[-1] and not _points_coincide(contracted, centroid):
                simplex[-1], scores[-1] = contracted, f_c
            else:
                _shrink(simplex, scores, f, n)

    sort_simplex()
    return OptResult(
        x=list(simplex[0]),
        value=scores[0],
        iterations=iterations,
        converged=converged,
    )


def _shrink(
    simplex: list[list[float]],
    scores: list[float],
    f: Callable[[list[float]], float],
    n: int,
) -> None:
    """Shrink the simplex toward its best vertex (in place, keeping scores aligned)."""
    for i in range(1, n + 1):
        simplex[i] = [simplex[0][d] + 0.5 * (simplex[i][d] - simplex[0][d]) for d in range(n)]
        scores[i] = f(simplex[i])


# --------------------------------------------------------------------- #
# Simulated annealing                                                    #
# --------------------------------------------------------------------- #


def simulated_annealing(
    f: Callable[[list[float]], float],
    x0: list[float],
    *,
    t_init: float = 1.0,
    t_min: float = 1e-8,
    cooling: float = 0.95,
    sigma: float = 0.1,
    max_iter: int = 10_000,
    seed: int | None = None,
    bounds: Sequence[Bound] | None = None,
) -> OptResult[list[float]]:
    """Minimize ``f`` via simulated annealing with Gaussian proposals.

    Proposal scale shrinks with temperature (``sigma·t/t_init``); the
    Metropolis rule accepts improvements always and uphill moves with
    probability ``e^(−Δ/t)``, driven by a caller-seeded
    :class:`random.Random`. The best-so-far point is tracked separately from
    the wandering current point, so cooling can never lose the optimum seen.

    Args:
        f: scalar objective taking a feature vector
        x0: starting point (length >= 1)
        t_init: initial temperature (> 0)
        t_min: cooling floor; run stops early when ``t <= t_min``
        cooling: multiplicative schedule factor in ``(0, 1]``
        sigma: base standard deviation of Gaussian proposals (> 0)
        max_iter: maximum proposal evaluations
        seed: RNG seed for reproducibility (``None`` → OS entropy)
        bounds: optional box constraints as ``(low, high)`` per axis;
            proposals are clamped into the box

    Returns:
        :class:`OptResult` carrying the best-seen point; ``converged`` means
        the cooling floor was reached within ``max_iter``.

    Raises:
        ValueError: on empty ``x0``, malformed temperatures/schedule/sigma,
            ``max_iter < 1``, or a bound with ``low >= high``.
    """
    if not x0:
        raise ValueError("x0 must be non-empty")
    if t_init <= 0:
        raise ValueError("t_init must be positive")
    if not 0 < cooling <= 1:
        raise ValueError("cooling must be in (0, 1]")
    if t_min <= 0:
        raise ValueError("t_min must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    box: list[Bound] | None = None
    if bounds is not None:
        box = [(lo, hi) for lo, hi in bounds]
        if any(lo >= hi for lo, hi in box):
            raise ValueError("each bound must satisfy low < high")
        if len(box) != len(x0):
            raise ValueError("bounds must have one (low, high) pair per axis")

    rng = random.Random(seed)
    current = _clamp_point(list(x0), box)
    f_current = f(current)
    best_x = list(current)
    f_best = f_current
    temperature = t_init
    iterations = 0
    converged = False

    for iterations in range(1, max_iter + 1):
        if temperature <= t_min:
            converged = True
            break

        scale = sigma * (temperature / t_init)
        proposal = [xi + rng.gauss(0.0, scale) for xi in current]
        proposal = _clamp_point(proposal, box)
        f_proposal = f(proposal)

        if _metropolis_accept(f_proposal - f_current, temperature, rng.random()):
            current = proposal
            f_current = f_proposal
            if f_current < f_best:
                f_best = f_current
                best_x = list(current)

        temperature *= cooling

    return OptResult(
        x=best_x,
        value=f_best,
        iterations=iterations,
        converged=converged,
    )
