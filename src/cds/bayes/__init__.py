"""Bayesian conjugate analysis — pure Python."""

from __future__ import annotations

from cds.bayes.conjugate import (
    bayes_factor,
    beta_binomial_update,
    beta_credible_interval,
    credible_interval,
    gamma_credible_interval,
    gamma_poisson_update,
    normal_credible_interval,
    normal_normal_update,
)

__all__ = [
    "bayes_factor",
    "beta_binomial_update",
    "beta_credible_interval",
    "credible_interval",
    "gamma_credible_interval",
    "gamma_poisson_update",
    "normal_credible_interval",
    "normal_normal_update",
]
