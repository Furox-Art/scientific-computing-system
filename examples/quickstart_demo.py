"""CDS 5-minute quickstart: every number below is produced by this script.

Runs on the standard library alone -- no NumPy, no SciPy, no extras:

    python examples/quickstart_demo.py

Every computation is seeded or closed-form, so the output is identical on every
platform and on every run. That is the point: a first impression of a scientific
package should be reproducible, not "close enough".

The transcript is mirrored in the README ("Quickstart") and enforced by
``tests/test_readme_quickstart.py``, which re-runs this module and compares the
captured stdout against the README code block.
"""

from __future__ import annotations

import math

import cds
from cds import diffeq, math_utils, montecarlo, stats, units
from cds.uncertainty import propagate_linear, propagate_monte_carlo

SEED = 20260930


def linear_algebra() -> None:
    """Decompose a matrix and verify the factors reconstruct it exactly."""
    a = [
        [4.0, 1.0, 0.0],
        [1.0, 3.0, 1.0],
        [0.0, 1.0, 2.0],
    ]
    svd = math_utils.svd(a)
    print("[1] SVD of a symmetric 3x3 matrix")
    print(f"    singular values  : {[round(s, 6) for s in svd.singular_values]}")
    print(f"    matrix rank      : {math_utils.rank(a)}")
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
    print(f"    residual ||A - USVt||_F: {residual:.3e}")
    print(f"    determinant       : {math_utils.determinant(a):.6f}")
    # Solve a small linear system and check the answer by substitution.
    coeff = [[2.0, 1.0], [4.0, 3.0]]
    rhs = [5.0, 11.0]
    x = math_utils.solve_linear(coeff, rhs)
    residual = math.sqrt(
        sum((sum(coeff[i][j] * x[j] for j in range(2)) - rhs[i]) ** 2 for i in range(2))
    )
    print(f"    solve_linear([[2,1],[4,3]], [5,11]) -> {[round(v, 6) for v in x]}  (exact [2, 1])")
    print(f"    linear-system residual: {residual:.3e}")


def differential_equations() -> None:
    """RK4 on y' = y, whose exact solution is exp(t): a real accuracy check."""
    sol = diffeq.rk4(lambda t, y: y, 0.0, 1.0, 1.0, dt=0.01)
    print("[2] RK4 on y' = y, y(0) = 1, exact solution exp(t)")
    print(f"    steps taken         : {sol.steps}")
    print(f"    y(1) numeric        : {sol.y[-1]:.12f}")
    print(f"    y(1) exact          : {math.exp(1.0):.12f}")
    print(f"    absolute error      : {abs(sol.y[-1] - math.exp(1.0)):.3e}")


def statistics() -> None:
    """Least-squares fit with a goodness-of-fit statistic."""
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.10, 3.95, 6.10, 7.85, 10.20]
    fit = stats.linear_regression(x, y)
    print("[3] Ordinary least squares on y = 1.9x + 0.2")
    print(f"    slope     : {fit.slope:.6f}")
    print(f"    intercept : {fit.intercept:.6f}")
    print(f"    R^2       : {fit.r_squared:.6f}")
    test = stats.one_sample_ttest(y, popmean=6.0)
    print(
        f"    one-sample t-test vs mu=6.0 -> statistic {test.statistic:.6f}, p = {test.p_value:.6f}"
    )


def monte_carlo() -> None:
    """Seeded Monte Carlo integration: E[X^2] over [0,1] is exactly 1/3."""
    mc = montecarlo.mc_expectation(lambda x: x**2, n_samples=200_000, a=0.0, b=1.0, seed=SEED)
    print("[4] Monte Carlo: E[X^2] for X ~ Uniform(0,1), exact value 1/3")
    print(f"    samples   : {mc.samples}")
    print(f"    estimate  : {mc.estimate:.6f}  (exact 0.333333)")
    print(f"    std error : {mc.std_error:.6f}")


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
    print(f"    output value            : {result.value:.6f}  (exact 5.000000)")
    print(f"    combined std uncertainty: {result.standard_uncertainty:.6f}")
    print(f"    method                  : {result.method}")
    # Same inputs through the Monte Carlo propagator, which makes no linearity
    # assumption. The two independent methods should agree closely.
    mc = propagate_monte_carlo(
        lambda a, b: math.hypot(a, b),
        [3.0, 4.0],
        standard_uncertainties=[0.3, 0.4],
        samples=100_000,
        seed=SEED,
    )
    print(f"    MC cross-check std      : {mc.standard_deviation:.6f}")
    print(f"    95% interval            : [{mc.lower:.6f}, {mc.upper:.6f}]")


def dimensional_analysis() -> None:
    """Dimensional consistency check -- a unit mistake becomes an exception."""
    energy = units.NEWTON * units.METER
    print("[6] Dimensional analysis")
    print(f"    N*m symbol              : {energy.symbol}")
    print(f"    compatible with joule?  : {units.dimensions_compatible(energy, units.JOULE)}")
    print(f"    compatible with newton? : {units.dimensions_compatible(energy, units.NEWTON)}")


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
