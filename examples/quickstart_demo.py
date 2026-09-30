"""CDS 5-minute quickstart: every claim printed below is verified by this script.

Runs on the standard library alone -- no NumPy, no SciPy, no extras:

    python examples/quickstart_demo.py

Every check compares a computed quantity against a closed-form or
analytically known answer, and prints the comparison as a pass/fail assertion
rather than a raw last-digit float. That matters for reproducibility: floating
point summation order differs slightly between CPython versions and
platforms, so printing a 1e-15 residual verbatim would produce a transcript
that only matches on the machine that generated it. Asserting a bound instead
is both deterministic and stronger evidence of correctness.

The transcript is mirrored in the README ("Quickstart") and enforced by
``tests/test_docs_and_metadata.py``, which re-runs this module and compares the
captured stdout against the README code block.
"""

from __future__ import annotations

import math

import cds
from cds import diffeq, math_utils, montecarlo, stats, units
from cds.uncertainty import propagate_linear, propagate_monte_carlo

SEED = 20260930


def _close(actual: float, expected: float, tol: float) -> bool:
    return abs(actual - expected) <= tol


def linear_algebra() -> None:
    """Decompose a matrix and verify the factors reconstruct it exactly."""
    a = [
        [4.0, 1.0, 0.0],
        [1.0, 3.0, 1.0],
        [0.0, 1.0, 2.0],
    ]
    svd = math_utils.svd(a)
    expected = [4.732050807568878, 3.0, 1.2679491924311226]
    matched = all(_close(got, want, 1e-12) for got, want in zip(svd.singular_values, expected))
    print("[1] SVD of a symmetric 3x3 matrix")
    print(f"    singular values match closed form : {matched}")

    # The defining property of an SVD is A = U diag(sigma) Vt. Rebuild A from
    # the factors and measure the residual: this is the correctness check.
    rebuilt = [
        [
            sum(svd.U[i][k] * svd.singular_values[k] * svd.Vt[k][j] for k in range(3))
            for j in range(3)
        ]
        for i in range(3)
    ]
    residual = math_utils.frobenius_norm(
        [[rebuilt[i][j] - a[i][j] for j in range(3)] for i in range(3)]
    )
    print(f"    reconstruction residual below 1e-12: {residual < 1e-12}")

    # Solve a small linear system and check the answer by substitution.
    coeff = [[2.0, 1.0], [4.0, 3.0]]
    rhs = [5.0, 11.0]
    x = math_utils.solve_linear(coeff, rhs)
    solved = _close(x[0], 2.0, 1e-12) and _close(x[1], 1.0, 1e-12)
    substitution = math.sqrt(
        sum((sum(coeff[i][j] * x[j] for j in range(2)) - rhs[i]) ** 2 for i in range(2))
    )
    print(f"    solve_linear yields exact [2, 1]    : {solved}")
    print(f"    substitution residual below 1e-12  : {substitution < 1e-12}")


def differential_equations() -> None:
    """RK4 on y' = y, whose exact solution is exp(t): a real accuracy check."""
    sol = diffeq.rk4(lambda t, y: y, 0.0, 1.0, 1.0, dt=0.01)
    error = abs(sol.y[-1] - math.exp(1.0))
    print("[2] RK4 on y' = y, y(0) = 1, exact solution exp(t)")
    print(f"    steps taken                    : {sol.steps}")
    print(f"    agrees with exp(1) within 1e-9 : {error < 1e-9}")


def statistics() -> None:
    """Least-squares fit with a goodness-of-fit statistic."""
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.10, 3.95, 6.10, 7.85, 10.20]
    fit = stats.linear_regression(x, y)
    slope_ok = _close(fit.slope, 2.01, 1e-12)
    intercept_ok = _close(fit.intercept, 0.01, 1e-12)
    print("[3] Ordinary least squares on data with slope 2.01, intercept 0.01")
    print(f"    recovered slope within 1e-12   : {slope_ok}")
    print(f"    recovered intercept within 1e-12: {intercept_ok}")
    print(f"    R^2 above 0.99                  : {fit.r_squared > 0.99}")
    test = stats.one_sample_ttest(y, popmean=6.0)
    print(f"    t-test vs mu=6.0 has p above 0.01: {test.p_value > 0.01}")


def monte_carlo() -> None:
    """Seeded Monte Carlo integration: E[X^2] over [0,1] is exactly 1/3."""
    mc = montecarlo.mc_expectation(lambda x: x**2, n_samples=200_000, a=0.0, b=1.0, seed=SEED)
    within_3se = abs(mc.estimate - 1.0 / 3.0) <= 3.0 * mc.std_error
    print("[4] Monte Carlo: E[X^2] for X ~ Uniform(0,1), exact value 1/3")
    print(f"    samples used              : {mc.samples}")
    print(f"    estimate rounds to 0.333985: {round(mc.estimate, 6) == 0.333985}")
    print(f"    within 3 standard errors  : {within_3se}")


def uncertainty_propagation() -> None:
    """Propagate input standard uncertainties into an output quantity.

    ``f(a, b) = sqrt(a^2 + b^2)`` is the RSS combination of two independent
    standard uncertainties -- a real GUM-style uncertainty budget.
    """
    result = propagate_linear(
        lambda a, b: math.hypot(a, b),
        [3.0, 4.0],
        standard_uncertainties=[0.3, 0.4],
    )
    print("[5] Uncertainty propagation for f(a,b) = sqrt(a^2 + b^2)")
    print(f"    output value is 5 to 1e-12  : {_close(result.value, 5.0, 1e-12)}")
    print(f"    combined sigma ~ 0.3672     : {_close(result.standard_uncertainty, 0.3672, 1e-3)}")
    print(f"    method is linearized        : {result.method == 'linearized'}")

    # Same inputs through the Monte Carlo propagator, which makes no linearity
    # assumption. Two independent methods agreeing is stronger evidence than
    # either alone.
    mc = propagate_monte_carlo(
        lambda a, b: math.hypot(a, b),
        [3.0, 4.0],
        standard_uncertainties=[0.3, 0.4],
        samples=100_000,
        seed=SEED,
    )
    agree = _close(mc.standard_deviation, result.standard_uncertainty, 1e-2)
    print(f"    Monte Carlo sigma agrees   : {agree}")
    print(f"    interval brackets 5.0      : {mc.lower < 5.0 < mc.upper}")


def dimensional_analysis() -> None:
    """Dimensional consistency check -- a unit mistake becomes an exception."""
    energy = units.NEWTON * units.METER
    print("[6] Dimensional analysis")
    print(f"    N*m symbol                    : {energy.symbol}")
    print(f"    compatible with joule         : {units.dimensions_compatible(energy, units.JOULE)}")
    print(
        f"    not compatible with newton    : {not units.dimensions_compatible(energy, units.NEWTON)}"
    )


def main() -> int:
    print(f"CDS {cds.__version__} -- quickstart (seed={SEED})")
    print()
    linear_algebra()
    print()
    differential_equations()
    print()
    statistics()
    print()
    monte_carlo()
    print()
    uncertainty_propagation()
    print()
    dimensional_analysis()
    print()
    print("All six checks ran on the Python standard library alone.")
    print("Next: `cds modules` lists every scientific module with its capabilities.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
