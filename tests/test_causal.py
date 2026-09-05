from __future__ import annotations

import math

import pytest

from cds.causal import (
    CausalAssumptions,
    audit_causal_assumptions,
    linear_backdoor_effect,
    randomized_mean_effect,
)
from cds.validation import CheckStatus


IDENTIFIED = CausalAssumptions(
    temporal_order=True,
    no_unmeasured_confounding=True,
    positivity=True,
    consistency=True,
)


def test_causal_assumption_audit_distinguishes_pass_warning_and_failure() -> None:
    assumptions = CausalAssumptions(
        temporal_order=True,
        no_unmeasured_confounding=None,
        positivity=False,
        consistency=True,
    )
    report = audit_causal_assumptions(assumptions)
    assert [check.status for check in report.checks] == [
        CheckStatus.PASS,
        CheckStatus.WARNING,
        CheckStatus.FAIL,
        CheckStatus.PASS,
    ]
    assert not assumptions.identified
    assert IDENTIFIED.identified


def test_linear_backdoor_recovers_adjusted_effect() -> None:
    treatment = [0.0, 1.0, 0.0, 1.0, 2.0, 2.0]
    confounder = [0.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    outcome = [1.0 + 2.0 * t + 3.0 * x for t, x in zip(treatment, confounder, strict=True)]
    covariates = [[x] for x in confounder]

    estimate = linear_backdoor_effect(
        outcome,
        treatment,
        covariates,
        assumptions=IDENTIFIED,
    )
    assert estimate.effect == pytest.approx(2.0)
    assert estimate.method == "linear-backdoor-adjustment"
    assert estimate.n_observations == 6
    assert estimate.adjustment_count == 1
    assert estimate.assumptions is IDENTIFIED


def test_linear_backdoor_without_covariates_and_pivoting() -> None:
    treatment = [0.0, 10.0, 20.0]
    outcome = [5.0 + 4.0 * value for value in treatment]
    estimate = linear_backdoor_effect(
        outcome,
        treatment,
        [[], [], []],
        assumptions=IDENTIFIED,
    )
    assert estimate.effect == pytest.approx(4.0)
    assert estimate.adjustment_count == 0


def test_observational_effect_is_fail_closed_when_assumptions_are_unresolved() -> None:
    assumptions = CausalAssumptions(True, None, False, True)
    with pytest.raises(ValueError, match="no_unmeasured_confounding, positivity"):
        linear_backdoor_effect([1.0, 2.0], [0.0, 1.0], [[], []], assumptions=assumptions)


def test_linear_backdoor_input_contracts() -> None:
    with pytest.raises(ValueError, match="outcome must contain only finite"):
        linear_backdoor_effect([1.0, math.nan], [0.0, 1.0], [[], []], assumptions=IDENTIFIED)
    with pytest.raises(ValueError, match="treatment must contain only finite"):
        linear_backdoor_effect([1.0, 2.0], [0.0, math.inf], [[], []], assumptions=IDENTIFIED)
    with pytest.raises(ValueError, match="equal length"):
        linear_backdoor_effect([1.0], [0.0, 1.0], [[]], assumptions=IDENTIFIED)
    with pytest.raises(ValueError, match="at least one observation"):
        linear_backdoor_effect([], [], [], assumptions=IDENTIFIED)
    with pytest.raises(ValueError, match="treatment must vary"):
        linear_backdoor_effect([1.0, 2.0], [1.0, 1.0], [[], []], assumptions=IDENTIFIED)
    with pytest.raises(ValueError, match="one row per observation"):
        linear_backdoor_effect([1.0, 2.0], [0.0, 1.0], [[]], assumptions=IDENTIFIED)
    with pytest.raises(ValueError, match="equal width"):
        linear_backdoor_effect(
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0],
            [[1.0], [2.0, 3.0], [4.0]],
            assumptions=IDENTIFIED,
        )
    with pytest.raises(ValueError, match="covariates row 1"):
        linear_backdoor_effect(
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0],
            [[1.0], [math.nan], [3.0]],
            assumptions=IDENTIFIED,
        )


def test_linear_backdoor_rejects_underdetermined_and_singular_designs() -> None:
    with pytest.raises(ValueError, match="at least as many observations"):
        linear_backdoor_effect(
            [1.0, 2.0],
            [0.0, 1.0],
            [[1.0, 0.0], [0.0, 1.0]],
            assumptions=IDENTIFIED,
        )

    treatment = [0.0, 1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="design matrix is singular"):
        linear_backdoor_effect(
            [1.0, 3.0, 5.0, 7.0],
            treatment,
            [[value] for value in treatment],
            assumptions=IDENTIFIED,
        )


def test_randomized_mean_effect_requires_declared_randomization() -> None:
    with pytest.raises(ValueError, match="explicitly randomized"):
        randomized_mean_effect([3.0], [1.0], randomized=False)

    estimate = randomized_mean_effect([5.0, 7.0], [1.0, 3.0], randomized=True)
    assert estimate.effect == pytest.approx(4.0)
    assert estimate.method == "randomized-difference-in-means"
    assert estimate.n_observations == 4
    assert estimate.adjustment_count == 0
    assert estimate.assumptions is None


def test_randomized_mean_effect_validates_groups() -> None:
    with pytest.raises(ValueError, match="both be non-empty"):
        randomized_mean_effect([], [1.0], randomized=True)
    with pytest.raises(ValueError, match="both be non-empty"):
        randomized_mean_effect([1.0], [], randomized=True)
    with pytest.raises(ValueError, match="treated_outcomes"):
        randomized_mean_effect([math.inf], [1.0], randomized=True)
    with pytest.raises(ValueError, match="control_outcomes"):
        randomized_mean_effect([1.0], [math.nan], randomized=True)
