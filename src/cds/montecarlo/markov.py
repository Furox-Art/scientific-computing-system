"""Markov chain Monte Carlo — Metropolis-Hastings sampling.

References:
    - Metropolis, N. et al. (1953). Equation of State Calculations by Fast
      Computing Machines.
    - Hastings, W.K. (1970). Monte Carlo Sampling Methods Using Markov Chains
      and Their Applications.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass

__all__ = [
    "MHResult",
    "metropolis_hastings",
]


@dataclass
class MHResult:
    """Result of a Metropolis-Hastings run."""

    samples: list[float]
    acceptance_rate: float


def _validate_integer(name: str, value: int, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _log_density(log_target: Callable[[float], float], value: float) -> float:
    """Evaluate a log-density, allowing finite values and ``-inf`` only."""
    result = float(log_target(value))
    if math.isnan(result) or result == math.inf:
        raise ValueError("log_target must return a finite value or -inf")
    return result


def metropolis_hastings(
    log_target: Callable[[float], float],
    x0: float,
    *,
    n_samples: int = 5000,
    burn_in: int = 500,
    thin: int = 1,
    proposal_scale: float = 1.0,
    seed: int | None = None,
) -> MHResult:
    """Sample a target distribution with random-walk Metropolis-Hastings.

    Proposals are Gaussian, ``x' = x + N(0, proposal_scale)``, hence symmetric
    and the acceptance rule reduces to accepting when
    ``log(U) < log_target(x') - log_target(x)`` for ``U ~ Uniform(0, 1)``.
    ``-inf`` log-densities represent zero target density: such proposals are
    rejected, while a finite-density proposal can move a chain out of a
    zero-density starting point. ``NaN`` and ``+inf`` log-density values are
    rejected because they make the acceptance ratio undefined.

    Args:
        log_target: log of the (possibly unnormalized) target density.
        x0: finite initial state of the chain.
        n_samples: number of samples to keep (after burn-in and thinning).
        burn_in: initial iterations to discard.
        thin: keep only every ``thin``-th post-burn-in iterate.
        proposal_scale: finite positive standard deviation of the Gaussian proposal.
        seed: optional random seed for reproducibility.

    Returns:
        MHResult with the retained samples and the acceptance rate over all
        ``burn_in + n_samples * thin`` proposals.

    Raises:
        ValueError: if counts are invalid, ``x0`` or ``proposal_scale`` is
            non-finite, or ``log_target`` returns ``NaN``/``+inf``.
    """
    _validate_integer("n_samples", n_samples, minimum=1)
    _validate_integer("burn_in", burn_in, minimum=0)
    _validate_integer("thin", thin, minimum=1)
    if not math.isfinite(proposal_scale) or proposal_scale <= 0:
        raise ValueError("proposal_scale must be finite and > 0")
    if not math.isfinite(x0):
        raise ValueError("x0 must be finite")

    rng = random.Random(seed)
    current = x0
    log_current = _log_density(log_target, x0)
    samples: list[float] = []
    accepted = 0
    n_iters = burn_in + n_samples * thin
    for step in range(n_iters):
        proposal = current + rng.gauss(0.0, proposal_scale)
        if not math.isfinite(proposal):
            raise ArithmeticError("proposal became non-finite")
        log_proposal = _log_density(log_target, proposal)
        if log_proposal == -math.inf:
            move = False
        elif log_current == -math.inf:
            move = True
        else:
            move = math.log(1.0 - rng.random()) < log_proposal - log_current
        if move:
            accepted += 1
            current = proposal
            log_current = log_proposal
        if step >= burn_in and (step - burn_in) % thin == thin - 1:
            samples.append(current)
    return MHResult(samples=samples, acceptance_rate=accepted / n_iters)
