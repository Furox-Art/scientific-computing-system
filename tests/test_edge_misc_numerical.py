"""Final coverage push: 97% → 100%.

Targets the last 61 uncovered lines across 14 files:
- stats/hypothesis_tests.py  (15 lines — _gser, _gcf, _gammp, _gammq, _betacf, _betai internals)
- math_utils/linalg.py      (13 lines — error guards, degenerate matrices, overflow)
- ml/neural.py              (5 lines  — sigmoid overflow, derivative branches)
- quantum/multi_qubit.py    (6 lines  — measure fallback, is_entangled validation)
- data_analysis/dataset.py (4 lines  — missing-column errors, empty repr)
- graph/algorithms.py       (3 lines  — dijkstra stale heap, union-find edge cases)
- montecarlo/methods.py     (3 lines  — autoseed when seed=None)
- stats/descriptive.py     (2 lines  — median empty, correlation too few)
- stats/regression.py      (2 lines  — length/identical-x errors)
- hypothesis/generator.py  (2 lines  — invalid domain string fallback)
- cli.py                    (2 lines  — PYTHONPATH-existing, __main__ guard)
- diffeq/solvers.py         (1 line   — rk45 precision floor)
- math_utils/calculus.py    (1 line   — odd n auto-correct)
- optimization/minimize.py (1 line   — gradient_descent non-converge)
- __main__.py               (1 line   — __main__ guard)
"""

import math
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

from cds.core.models import Domain
from cds.data_analysis.dataset import DataSet
from cds.diffeq.solvers import rk45
from cds.graph.algorithms import (
    Graph,
    _find,
    _union,
    dijkstra,
)
from cds.hypothesis.generator import SimpleOfflineGenerator
from cds.math_utils.calculus import integral
from cds.math_utils.linalg import (
    determinant,
    dot,
    mat_mul,
    matrix_inverse,
    power_iteration,
    qr_decomposition,
    solve_linear,
    transpose,
)
from cds.ml.neural import Layer
from cds.optimization.minimize import gradient_descent
from cds.quantum.multi_qubit import QuantumRegister, is_entangled
from cds.stats.descriptive import correlation, median
from cds.stats.hypothesis_tests import (
    _betacf,
    _betai,
    _gammp,
    _gammq,
    _gcf,
    _gser,
)
from cds.stats.regression import linear_regression

# ---------------------------------------------------------------------------
# 1. stats/hypothesis_tests.py — 15 missing lines
# ---------------------------------------------------------------------------


class TestGserInternals:
    """Cover _gser edge: x <= 0 returns 0.0 (line 60)."""

    def test_gser_zero_x_returns_zero(self) -> None:
        assert _gser(1.0, 0.0) == 0.0

    def test_gser_negative_x_returns_zero(self) -> None:
        assert _gser(2.0, -5.0) == 0.0

    def test_gser_small_positive_works(self) -> None:
        val = _gser(1.0, 0.5)
        assert 0.0 < val < 1.0


class TestGcfInternals:
    """Cover _gcf underflow guards (lines 87, 90)."""

    def test_gcf_small_a_large_x(self) -> None:
        # For small a and large x, the Lentz loop may hit underflow guards.
        val = _gcf(0.5, 1e20)
        assert 0.0 <= val <= 1.0

    def test_gcf_boundary_values(self) -> None:
        val = _gcf(1.0, 100.0)
        assert 0.0 <= val <= 1.0


class TestGammpInternals:
    """Cover _gammp ValueError + both branches (lines 101-105)."""

    def test_gammp_negative_x_raises(self) -> None:
        with pytest.raises(ValueError):
            _gammp(1.0, -1.0)

    def test_gammp_zero_a_raises(self) -> None:
        with pytest.raises(ValueError):
            _gammp(0.0, 1.0)

    def test_gammp_negative_a_raises(self) -> None:
        with pytest.raises(ValueError):
            _gammp(-1.0, 1.0)

    def test_gammp_series_branch(self) -> None:
        # x < a+1 → uses _gser
        val = _gammp(5.0, 1.0)
        assert 0.0 < val < 1.0

    def test_gammp_cf_branch(self) -> None:
        # x >= a+1 → uses 1 - _gcf
        val = _gammp(1.0, 100.0)
        assert 0.0 < val <= 1.0


class TestGammqInternals:
    """Cover _gammq ValueError (line 111)."""

    def test_gammq_negative_x_raises(self) -> None:
        with pytest.raises(ValueError):
            _gammq(1.0, -1.0)

    def test_gammq_zero_a_raises(self) -> None:
        with pytest.raises(ValueError):
            _gammq(0.0, 1.0)


class TestBetacfInternals:
    """Cover _betacf underflow guards (lines 128, 136, 139, 145, 148)."""

    def test_betacf_extreme_values(self) -> None:
        # Extreme parameters push Lentz loop toward underflow guards.
        val = _betacf(0.5, 0.5, 0.99999)
        assert val > 0.0

    def test_betacf_small_x(self) -> None:
        val = _betacf(1.0, 5.0, 0.001)
        assert val > 0.0


class TestBetaiInternals:
    """Cover _betai ValueError (line 163)."""

    def test_betai_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            _betai(1.0, 1.0, 1.5)

    def test_betai_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            _betai(1.0, 1.0, -0.1)

    def test_betai_boundary_zero(self) -> None:
        assert _betai(1.0, 1.0, 0.0) == 0.0

    def test_betai_boundary_one(self) -> None:
        assert _betai(1.0, 1.0, 1.0) == 1.0


# ---------------------------------------------------------------------------
# 2. math_utils/linalg.py — 13 missing lines
# ---------------------------------------------------------------------------


class TestLinalgErrorGuards:
    """Cover ValueError branches in dot, mat_mul (lines 22, 35)."""

    def test_dot_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            dot([1.0, 2.0], [1.0])

    def test_mat_mul_incompatible_raises(self) -> None:
        with pytest.raises(ValueError, match="incompatible"):
            mat_mul([[1.0, 2.0]], [[1.0]])


class TestLinalgEmptyDegenerate:
    """Cover empty/det edge cases (lines 48, 59, 61)."""

    def test_transpose_empty(self) -> None:
        assert transpose([]) == []

    def test_determinant_zero_by_zero(self) -> None:
        assert determinant([]) == 1.0

    def test_determinant_one_by_one(self) -> None:
        assert determinant([[7.0]]) == 7.0


class TestLinalgSingular:
    """Cover singular matrix errors (lines 163, 199)."""

    def test_solve_linear_singular_raises(self) -> None:
        singular = [[1.0, 2.0], [2.0, 4.0]]
        with pytest.raises(ValueError, match="singular"):
            solve_linear(singular, [1.0, 2.0])

    def test_matrix_inverse_singular_raises(self) -> None:
        singular = [[1.0, 2.0], [2.0, 4.0]]
        with pytest.raises(ValueError, match="singular"):
            matrix_inverse(singular)


class TestPowerIterationEdgeCases:
    """Cover overflow, zero-norm, non-convergence (lines 239-241, 244, 259)."""

    def test_overflow_fallback(self) -> None:
        # Massive matrix values trigger OverflowError in sqrt(sum(x*x)),
        # falling back to max(abs(x)) scaling.
        huge = [[1e308, 0.0], [0.0, 1e308]]
        eigenvalue, v = power_iteration(huge, max_iter=100)
        # eigenvalue should be very large (1e308)
        assert eigenvalue > 1e300
        assert len(v) == 2

    def test_zero_norm_break(self) -> None:
        # Nilpotent matrix: A² = 0, so first iteration yields zero vector.
        nilpotent = [[0.0, 1.0], [0.0, 0.0]]
        eigenvalue, v = power_iteration(nilpotent, max_iter=100)
        # eigenvalue should be near zero since nilpotent has no eigenvalue > 0
        assert abs(eigenvalue) < 1e-10

    def test_non_convergence(self) -> None:
        # Convergence is decided on the residual, so an iteration budget too
        # small to drive ‖Av − λv‖ under the tolerance must report failure
        # rather than hand back the last iterate as if it were an answer.
        with pytest.raises(ValueError, match="did not converge"):
            power_iteration([[2.0, 1.0], [1.0, 2.0]], max_iter=1)

    def test_equal_magnitude_eigenvalues_raise(self) -> None:
        # Power iteration requires |lambda_1| > |lambda_2| strictly. A real
        # rotation has the conjugate pair +i / -i and no real eigenvector, so
        # no iterate can ever satisfy the residual test.
        with pytest.raises(ValueError, match="strictly dominant"):
            power_iteration([[0.0, -1.0], [1.0, 0.0]], max_iter=200)

    def test_symmetric_indefinite_pair_is_found(self) -> None:
        # [[3, 4], [4, -3]] has eigenvalues +5 and -5. Equal magnitude, so the
        # old all-ones start oscillated and the old quotient test exited early
        # with 4.0 — not an eigenvalue at all. Verify both that the eigenvalue
        # is right and that the vector really is an eigenvector.
        eigenvalue, v = power_iteration([[3.0, 4.0], [4.0, -3.0]])
        assert abs(abs(eigenvalue) - 5.0) < 1e-9
        av = [3.0 * v[0] + 4.0 * v[1], 4.0 * v[0] - 3.0 * v[1]]
        residual = math.sqrt(sum((av[i] - eigenvalue * v[i]) ** 2 for i in range(2)))
        assert residual < 1e-9


class TestQRDegenerateColumns:
    """Cover zero-column and degenerate Householder (lines 314, 321)."""

    def test_zero_column_continues(self) -> None:
        m = [[1.0, 0.0], [0.0, 0.0]]
        Q, R = qr_decomposition(m)
        assert len(Q) == 2
        assert len(R) == 2

    def test_degenerate_householder(self) -> None:
        # First column is already axis-aligned so Householder vector
        # may have near-zero norm.
        m = [[1.0, 2.0], [0.0, 1.0]]
        Q, R = qr_decomposition(m)
        assert len(Q) == 2


# ---------------------------------------------------------------------------
# 3. ml/neural.py — 5 missing lines
# ---------------------------------------------------------------------------


class TestNeuralSigmoidOverflow:
    """Cover sigmoid OverflowError handler (lines 72-74)."""

    def test_sigmoid_large_positive(self) -> None:
        layer = Layer(1, 1, activation="sigmoid")
        result = layer._activate(1000.0)
        assert result == 1.0

    def test_sigmoid_large_negative(self) -> None:
        layer = Layer(1, 1, activation="sigmoid")
        result = layer._activate(-1000.0)
        assert result == 0.0


class TestNeuralDerivatives:
    """Cover derivative branches (lines 78, 81)."""

    def test_relu_derivative_zero(self) -> None:
        layer = Layer(1, 1, activation="relu")
        assert layer._activate_derivative(-1.0, 0.0) == 0.0

    def test_sigmoid_derivative(self) -> None:
        layer = Layer(1, 1, activation="sigmoid")
        a = 0.7
        result = layer._activate_derivative(0.5, a)
        assert abs(result - a * (1.0 - a)) < 1e-15


# ---------------------------------------------------------------------------
# 4. quantum/multi_qubit.py — 6 missing lines
# ---------------------------------------------------------------------------


class TestMeasureFallback:
    """Cover measure() floating-point fallback (lines 59-63)."""

    def test_measure_fallback_on_last_qubit(self) -> None:
        # Create a register where last amplitude is ~1.0 so cumulative
        # barely triggers the fallback path.
        reg = QuantumRegister(2, [0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 1.0 + 0j])
        reg.normalize()
        # r=1.0 always hits fallback since r <= cumulative is False until last
        result = reg.measure(seed=999)
        assert result == 3

    def test_measure_fallback_with_seed_edge(self) -> None:
        # Seed that generates r very close to 1.0
        reg = QuantumRegister.from_bits(2, 0)
        # Apply H to create superposition, then measure many times to
        # exercise all code paths
        from cds.quantum.multi_qubit import h_gate

        h_gate(reg, 0)
        # This exercises the main loop; the fallback is for rounding only
        result = reg.measure(seed=42)
        assert result in (0, 1)


class TestIsEntangledValidation:
    """Cover is_entangled ValueError for non-2-qubit (line 232)."""

    def test_is_entangled_three_qubits_raises(self) -> None:
        reg = QuantumRegister(3, [1.0 + 0j] * 8)
        with pytest.raises(ValueError, match="2-qubit"):
            is_entangled(reg)

    def test_is_entangled_one_qubit_raises(self) -> None:
        reg = QuantumRegister(1, [1.0 + 0j, 0.0 + 0j])
        with pytest.raises(ValueError, match="2-qubit"):
            is_entangled(reg)


# ---------------------------------------------------------------------------
# 5. data_analysis/dataset.py — 4 missing lines
# ---------------------------------------------------------------------------


class TestDatasetMissingColumns:
    """Cover column/select/group_by ValueError (lines 37, 57, 65)."""

    def test_column_not_found_raises(self) -> None:
        ds = DataSet([{"x": 1}])
        with pytest.raises(ValueError, match="not found"):
            ds.column("y")

    def test_select_not_found_raises(self) -> None:
        ds = DataSet([{"x": 1}])
        with pytest.raises(ValueError, match="not found"):
            ds.select("y")

    def test_group_by_not_found_raises(self) -> None:
        ds = DataSet([{"x": 1}])
        with pytest.raises(ValueError, match="not found"):
            ds.group_by("y")


class TestDatasetEmptyRepr:
    """Cover empty DataSet __repr__ (line 83)."""

    def test_empty_repr(self) -> None:
        ds = DataSet([])
        assert repr(ds) == "DataSet(empty)"

    def test_nonempty_repr(self) -> None:
        ds = DataSet([{"x": 1}])
        assert "rows=1" in repr(ds)


# ---------------------------------------------------------------------------
# 6. graph/algorithms.py — 3 missing lines
# ---------------------------------------------------------------------------


class TestDijkstraStaleHeap:
    """Cover stale heap-entry continue (line 114)."""

    def test_dijkstra_with_duplicate_entries(self) -> None:
        # Build graph and run dijkstra — the algorithm pushes
        # duplicate entries when a shorter path is found, which
        # leaves stale entries in the heap (covered by line 114).
        g = Graph(n_vertices=4)
        g.add_edge(0, 1, 1.0)
        g.add_edge(0, 2, 4.0)
        g.add_edge(1, 2, 2.0)
        g.add_edge(1, 3, 6.0)
        g.add_edge(2, 3, 3.0)
        dist, _ = dijkstra(g, 0)
        assert dist[3] == 6.0  # 0→1→2→3 is the shortest path


class TestUnionFindEdgeCases:
    """Cover equal-roots return False + rank-swap (lines 143, 145)."""

    def test_union_same_root_returns_false(self) -> None:
        parent = {0: 0, 1: 1}
        rank = {0: 0, 1: 0}
        # First union merges 0 and 1
        assert _union(parent, rank, 0, 1) is True
        # Second union: same set → no merge
        assert _union(parent, rank, 0, 1) is False

    def test_union_rank_swap(self) -> None:
        parent = {0: 0, 1: 1}
        rank = {0: 2, 1: 0}
        # rank[0] > rank[1], so 1's root becomes 0's root (no swap needed).
        result = _union(parent, rank, 0, 1)
        assert result is True
        assert _find(parent, 1) == 0

    def test_union_rank_swap_reverse(self) -> None:
        # rank[ra] < rank[rb] triggers the swap: ra, rb = rb, ra
        parent = {0: 0, 1: 1}
        rank = {0: 0, 1: 2}
        result = _union(parent, rank, 0, 1)
        assert result is True
        assert _find(parent, 0) == 1


# ---------------------------------------------------------------------------
# 7. montecarlo/methods.py — 3 missing lines (seed=None autoseed)
# ---------------------------------------------------------------------------


class TestEstimatePiAutoSeed:
    """Cover seed=None branch that imports os/sys (lines 58-60)."""

    def test_seed_none_uses_autoseed(self) -> None:
        # Mock os.urandom to return a predictable seed.
        fake_bytes = b"\x00\x00\x00\x2a"  # 42 in little-endian
        with mock.patch("os.urandom", return_value=fake_bytes):
            from cds.montecarlo.methods import estimate_pi

            result = estimate_pi(n_samples=100, seed=None)
        assert result.estimate > 0.0
        assert result.samples == 100


# ---------------------------------------------------------------------------
# 8. stats/descriptive.py — 2 missing lines
# ---------------------------------------------------------------------------


class TestDescriptiveEdgeCases:
    """Cover median empty (line 34) and correlation too few (line 91)."""

    def test_median_empty_returns_zero(self) -> None:
        assert median([]) == 0.0

    def test_correlation_single_point_raises(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            correlation([1.0], [2.0])


# ---------------------------------------------------------------------------
# 9. stats/regression.py — 2 missing lines
# ---------------------------------------------------------------------------


class TestRegressionEdgeCases:
    """Cover length mismatch (line 21) and identical-x (line 28)."""

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="matching lists"):
            linear_regression([1.0], [1.0, 2.0])

    def test_too_few_points_raises(self) -> None:
        with pytest.raises(ValueError, match="matching lists"):
            linear_regression([1.0], [2.0])

    def test_identical_x_raises(self) -> None:
        with pytest.raises(ValueError, match="identical"):
            linear_regression([5.0, 5.0, 5.0], [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# 10. hypothesis/generator.py — 2 missing lines
# ---------------------------------------------------------------------------


class TestGeneratorInvalidDomainString:
    """Cover invalid domain string fallback (lines 134-135)."""

    def test_invalid_domain_string_fallback(self) -> None:
        gen = SimpleOfflineGenerator()
        hypos = gen.generate("test question", domain="nonexistent_domain_xyz", n=1)
        assert len(hypos) == 1
        assert hypos[0].domain == Domain.GENERAL_SCIENCE


# ---------------------------------------------------------------------------
# 11. cli.py — 2 missing lines (PYTHONPATH existing + __main__ guard)
# ---------------------------------------------------------------------------


class TestCLIEnvVars:
    """Cover the PYTHONPATH-already-set branch in the dashboard command."""

    def test_dashboard_with_existing_pythonpath(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Drive the dashboard command with PYTHONPATH already in the
        # environment, so the "if PYTHONPATH in env" branch in _cmd_dashboard
        # fires. subprocess.run is mocked so streamlit is never launched; a
        # KeyboardInterrupt surfaces as the "Dashboard stopped" message.
        import argparse

        from cds.cli import _cmd_dashboard

        captured_env: dict[str, str] = {}

        def fake_run(
            cmd: list[str], *, env: dict[str, str] | None = None, **kwargs: object
        ) -> None:
            # Snapshot the environment passed to subprocess so we can assert
            # that the existing PYTHONPATH was preserved and prepended-to.
            if env is not None:
                captured_env.update(env)
            raise KeyboardInterrupt()

        monkeypatch.setattr(subprocess, "run", fake_run)
        monkeypatch.setattr(Path, "exists", lambda self: True)
        monkeypatch.setattr(os, "environ", {"PYTHONPATH": "/fake/path"})

        ns = argparse.Namespace()
        rc = _cmd_dashboard(ns)
        # The existing PYTHONPATH value must still be present (prepended).
        assert "/fake/path" in captured_env["PYTHONPATH"]
        assert rc == 0


class TestCLIMainGuard:
    """Cover `if __name__ == "__main__"` guard in cli.py."""

    def test_cli_main_module_run(self) -> None:
        # Run the cli package as __main__ via subprocess.
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        src_path = os.path.join(root, "src")
        env = os.environ.copy()
        env["PYTHONPATH"] = (
            f"{src_path}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else src_path
        )
        result = subprocess.run(
            [sys.executable, "-m", "cds.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=root,
            env=env,
        )
        assert result.returncode == 0
        assert "Scientific Computing" in result.stdout or "scientific" in result.stdout.lower()


# ---------------------------------------------------------------------------
# 12. __main__.py — 1 missing line
# ---------------------------------------------------------------------------


class TestMainModuleRun:
    """Cover if __name__ == "__main__": app() in __main__.py (line 6)."""

    def test_main_module_via_subprocess(self) -> None:
        main_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "src",
                "cds",
                "__main__.py",
            )
        )
        assert os.path.isfile(main_path), f"missing entry point: {main_path}"
        # The test env resolves `cds` from the source tree via
        # pyproject's `pythonpath = ["src"]`, but a fresh subprocess does
        # not inherit that — add src/ to PYTHONPATH explicitly so
        # `python -m cds` finds the local package, not a stale install.
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        env = os.environ.copy()
        src_path = os.path.join(root, "src")
        env["PYTHONPATH"] = (
            f"{src_path}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else src_path
        )
        result = subprocess.run(
            [sys.executable, "-m", "cds", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=root,
            env=env,
        )
        assert result.returncode == 0
        assert "usage: cds" in result.stdout


# ---------------------------------------------------------------------------
# 13. diffeq/solvers.py — 1 missing line (rk45 precision floor)
# ---------------------------------------------------------------------------


class TestRK45PrecisionFloor:
    """Cover RuntimeError when step size hits machine precision (line 220)."""

    def test_rk45_precision_floor_raises(self) -> None:
        # Use a function that forces extremely tiny step sizes
        # by integrating over a huge range with tight tolerance.
        def constant_zero(t: float, y: float) -> float:
            return 0.0

        with pytest.raises(RuntimeError, match="precision"):
            rk45(
                constant_zero,
                t0=0.0,
                y0=1.0,
                t_end=1e308,
                atol=1e-30,
                rtol=1e-30,
            )


# ---------------------------------------------------------------------------
# 14. math_utils/calculus.py — 1 missing line (odd n auto-correct)
# ---------------------------------------------------------------------------


class TestIntegralOddN:
    """Cover n += 1 when odd n is passed (line 16)."""

    def test_integral_odd_n_auto_corrects(self) -> None:
        # Pass odd n; internally it becomes n+1 (even).
        # ∫₀¹ x² dx = 1/3
        result = integral(lambda x: x * x, 0.0, 1.0, n=101)
        assert abs(result - 1.0 / 3.0) < 1e-6


# ---------------------------------------------------------------------------
# 15. optimization/minimize.py — 1 missing line (non-converge)
# ---------------------------------------------------------------------------


class TestGradientDescentNonConverge:
    """Cover final return with converged=False (line 77)."""

    def test_gd_non_converge(self) -> None:
        # Use a very small learning rate and few iterations so it won't converge.
        result = gradient_descent(
            lambda x: x**2,
            x0=10.0,
            lr=1e-15,
            max_iter=2,
        )
        assert result.converged is False
        assert result.iterations == 2
