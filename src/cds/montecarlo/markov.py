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
    ``-inf`` log-densities are treated safely: a proposal with ``-inf``
    density is always rejected, and a current state with ``-inf`` density
    always accepts a finite-density proposal.

    Args:
        log_target: log of the (possibly unnormalized) target density.
        x0: initial state of the chain.
        n_samples: number of samples to keep (after burn-in and thinning).
        burn_in: initial iterations to discard.
        thin: keep only every ``thin``-th post-burn-in iterate.
        proposal_scale: standard deviation of the Gaussian proposal.
        seed: optional random seed for reproducibility.

    Returns:
        MHResult with the retained samples and the acceptance rate over all
        ``burn_in + n_samples * thin`` proposals.

    Raises:
        ValueError: if ``n_samples < 1``, ``burn_in < 0``, ``thin < 1`` or
            ``proposal_scale <= 0``.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    if burn_in < 0:
        raise ValueError("burn_in must be >= 0")
    if thin < 1:
        raise ValueError("thin must be >= 1")
    if proposal_scale <= 0:
        raise ValueError("proposal_scale must be > 0")

    rng = random.Random(seed)
    current = x0
    log_current = log_target(x0)
    samples: list[float] = []
    accepted = 0
    n_iters = burn_in + n_samples * thin
    for step in range(n_iters):
        proposal = current + rng.gauss(0.0, proposal_scale)
        log_proposal = log_target(proposal)
        if math.isinf(log_proposal):
            move = False
        elif math.isinf(log_current):
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
