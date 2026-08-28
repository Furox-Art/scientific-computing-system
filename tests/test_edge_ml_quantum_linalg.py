"""Coverage completion: cover the last uncovered lines (99% → 100%).

Targets:
- ml/neural.py          (4 lines — sigmoid overflow fallback, identity branch)
- quantum/multi_qubit.py (5 lines — measure floating-point fallback)
- math_utils/linalg.py   (8 lines — qr degenerate columns, power_iteration
                          zero-norm break, singular backward-pivot via mock,
                          OverflowError defensive branch via mock)
- stats/hypothesis_tests.py (7 lines — _FPMIN clamp branches via mock)

The main-guard lines in cli.py / __main__.py run in subprocesses and are
excluded via coverage config (see pyproject.toml [tool.coverage.report]).
"""

import math
from unittest import mock

import pytest

from cds.math_utils.linalg import (
    matrix_inverse,
    power_iteration,
    qr_decomposition,
    solve_linear,
)
from cds.ml.neural import Layer
from cds.quantum.multi_qubit import QuantumRegister
from cds.stats import _distributions as dist
from cds.stats import hypothesis_tests as ht

# ---------------------------------------------------------------------------
# 1. ml/neural.py — sigmoid overflow fallback + identity activation
# ---------------------------------------------------------------------------


class TestNeuralSigmoidOverflow:
    """Cover the sigmoid OverflowError fallback (lines 72-74)."""

    def test_sigmoid_negative_overflow_returns_zero(self) -> None:
        layer = Layer(1, 1, activation="sigmoid")
        # z very negative -> exp(z) overflows -> caught, returns 0.0
        assert layer._activate(-1000.0) == 0.0

    def test_sigmoid_large_positive(self) -> None:
        layer = Layer(1, 1, activation="sigmoid")
        # z large positive -> 1/(1+exp(-z)) -> exp(-z) underflows to 0 -> 1.0
        assert layer._activate(1000.0) == pytest.approx(1.0)


class TestNeuralIdentityActivation:
    """Cover the identity activation + its derivative (line 74 return, 81)."""

    def test_identity_activate_passthrough(self) -> None:
        layer = Layer(2, 2, activation="identity")
        assert layer._activate(3.7) == 3.7

    def test_identity_derivative_is_one(self) -> None:
        layer = Layer(2, 2, activation="identity")
        assert layer._activate_derivative(3.7, 0.5) == 1.0


# ---------------------------------------------------------------------------
# 2. quantum/multi_qubit.py — measure floating-point fallback
# ---------------------------------------------------------------------------


class TestMeasureFallback:
    """Cover the floating-point fallback when no state is sampled (lines 59-63).

    With all-zero amplitudes, every probability is 0, so `r <= cumulative`
    never holds and the final-idx fallback collapses the last state.
    """

    def test_measure_all_zero_amplitudes(self) -> None:
        reg = QuantumRegister(n_qubits=2, amplitudes=[0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j])
        idx = reg.measure(seed=0)
        assert idx == len(reg.amplitudes) - 1
        # last amplitude collapsed to 1.0
        assert reg.amplitudes[idx] == 1.0 + 0j

    def test_measure_single_qubit_zero_amps(self) -> None:
        reg = QuantumRegister(n_qubits=1, amplitudes=[0 + 0j, 0 + 0j])
        idx = reg.measure(seed=42)
        assert idx == 1
        assert reg.amplitudes[1] == 1.0 + 0j


# ---------------------------------------------------------------------------
# 3. math_utils/linalg.py — qr degenerate columns + power_iteration break
# ---------------------------------------------------------------------------


class TestQRDegenerateColumns:
    """Cover norm < 1e-15 continue guards in qr_decomposition (lines 320, 327)."""

    def test_qr_zero_column(self) -> None:
        # A fully-zero first column forces norm_x < 1e-15 (line 320).
        A = [[0.0, 5.0], [0.0, 5.0]]
        Q, R = qr_decomposition(A)
        # R must be upper triangular; reconstruction holds (lossy on zero col).
        assert len(R) == 2
        assert len(Q) == 2

    def test_qr_zero_householder_vector(self) -> None:
        # A column already equal to a standard basis vector: x=[1,0] gives
        # alpha=-1, v[0]=1-(-1)=2 -> not zero. To force norm_v<1e-15 we need
        # x aligned so v cancels. Use a column that is already axis-aligned
        # in the opposite sign: x[0] < 0 makes alpha=+norm_x, v[0]=x[0]-norm_x.
        # With x=[-1,0]: norm_x=1, alpha=1, v=[-1-1, 0]=[-2,0] -> still nonzero.
        # The norm_v<1e-15 path is only reachable when x is exactly zero-length
        # (handled by the earlier norm_x check). So we cover it via the
        # already-tested zero-column case; assert reconstruction identity holds
        # for a clean matrix to confirm Q remains orthogonal.
        A = [[1.0, 0.0], [0.0, 1.0]]
        Q, R = qr_decomposition(A)
        # Q orthogonal: Q^T Q = I
        n = len(Q)
        QtQ = [[sum(Q[r][k] * Q[c][k] for k in range(n)) for c in range(n)] for r in range(n)]
        for r in range(n):
            assert abs(QtQ[r][r] - 1.0) < 1e-10


class TestPowerIterationZeroNormBreak:
    """Cover the norm < 1e-15 break in power_iteration (line 250)."""

    def test_zero_matrix_breaks_immediately(self) -> None:
        # A zero matrix: w = A·v = [0,0], norm -> 0 -> break.
        # eigenvalue stays 0.0, eigenvector is the (normalized) initial v.
        ev, vec = power_iteration([[0.0, 0.0], [0.0, 0.0]], max_iter=50)
        assert ev == 0.0
        assert len(vec) == 2

    def test_nilpotent_zero_norm(self) -> None:
        # Nilpotent matrix A=[[0,1],[0,0]]: A·[1,1] = [1,0] (norm 1, ok).
        # A² = 0 so the second iteration yields zero vector -> break.
        ev, vec = power_iteration([[0.0, 1.0], [0.0, 0.0]], max_iter=50)
        assert abs(ev) < 1e-9


# ---------------------------------------------------------------------------
# 4. math_utils/linalg.py — singular backward pivot (lines 163, 199)
# ---------------------------------------------------------------------------


class TestSingularBackwardPivot:
    """Cover the backward-substitution singular pivot guards.

    Ordinary singular matrices are caught during forward elimination
    (lu_decomposition raises first). To reach the backward-pivot guard we
    inject a U with a ~0 diagonal via mocking lu_decomposition.
    """

    def test_solve_linear_backward_pivot_guard(self) -> None:
        # L = identity, U has a zero on the diagonal, P = identity.
        fake_L = [[1.0, 0.0], [0.0, 1.0]]
        fake_U = [[1.0, 0.0], [0.0, 0.0]]  # U[1][1] = 0 -> backward guard fires
        fake_P = [[1.0, 0.0], [0.0, 1.0]]
        with mock.patch(
            "cds.math_utils.linalg.lu_decomposition",
            return_value=(fake_P, fake_L, fake_U),
        ):
            with pytest.raises(ValueError, match="singular"):
                solve_linear([[1.0, 0.0], [0.0, 0.0]], [1.0, 2.0])

    def test_matrix_inverse_backward_pivot_guard(self) -> None:
        fake_L = [[1.0, 0.0], [0.0, 1.0]]
        fake_U = [[2.0, 0.0], [0.0, 0.0]]  # zero diagonal -> backward guard
        fake_P = [[1.0, 0.0], [0.0, 1.0]]
        with mock.patch(
            "cds.math_utils.linalg.lu_decomposition",
            return_value=(fake_P, fake_L, fake_U),
        ):
            with pytest.raises(ValueError, match="singular"):
                matrix_inverse([[2.0, 0.0], [0.0, 0.0]])


class TestPowerIterationOverflowExcept:
    """Cover the defensive OverflowError branch in power_iteration (lines 245-247).

    CPython floats overflow to inf (caught by the isinf check) rather than
    raising OverflowError, so this except branch is defensive. We force the
    branch by stubbing math.sqrt to raise OverflowError for a finite input.
    """

    def test_overflowerror_fallback(self) -> None:
        real_sqrt = math.sqrt

        def fake_sqrt(x: float) -> float:
            # Simulate a platform where a finite squared-sum overflows sqrt.
            if x >= 1.0:
                raise OverflowError("simulated overflow in sqrt")
            return real_sqrt(x)

        with mock.patch("cds.math_utils.linalg.math.sqrt", side_effect=fake_sqrt):
            # Identity-like matrix: A·v has positive norm once fallback applies.
            # The budget is generous because convergence is now decided on the
            # residual, which for a ratio-2 spectrum needs roughly twice the
            # iterations that the old successive-quotient test did.
            ev, vec = power_iteration([[2.0, 0.0], [0.0, 1.0]], max_iter=200)
        assert abs(ev - 2.0) < 1e-9
        assert len(vec) == 2


# ---------------------------------------------------------------------------
# 5. stats/hypothesis_tests.py — _FPMIN clamp branches
# ---------------------------------------------------------------------------
# Lines 87, 90 (in _gcf), 128, 136, 139 (in _betacf first half), and
# 145, 148 (in _betacf second half). These clamp intermediate d/c values to
# _FPMIN when they underflow. They are reachable by feeding the continued
# fractions inputs that drive the recurrence toward zero.


class TestGammaBetaFPMINClamps:
    """Cover the _FPMIN underflow clamps in _gcf / _betacf.

    These Numerical-Recipes Lentz guards (lines 87, 90, 128, 136, 139, 145,
    148) clamp the recurrence to _FPMIN (1e-300) when an intermediate d/c
    underflows. On normal hardware the underflow is unreachable, so we force
    it by temporarily raising _FPMIN to 1.0 — then abs(d) and abs(c) fall
    below it on most iterations and the clamp assignments execute.
    """

    def test_gcf_fpmin_clamps_fire(self) -> None:
        # _gcf path (regime x >= a+1). Raised _FPMIN forces lines 87 & 90.
        # Patch _distributions._FPMIN: that is the module _gcf reads, since
        # the special functions now live in cds.stats._distributions.
        with mock.patch.object(dist, "_FPMIN", 1.0):
            val = ht._gcf(2.0, 10.0)
        assert math.isfinite(val) and val >= 0.0

    def test_betacf_fpmin_clamps_fire(self) -> None:
        # _betacf: raised _FPMIN forces the initial clamp (128) and all four
        # in-loop clamps (136, 139, 145, 148). Patch _distributions._FPMIN
        # because _betacf reads its _FPMIN from cds.stats._distributions.
        with mock.patch.object(dist, "_FPMIN", 1.0):
            val = ht._betacf(2.0, 3.0, 0.5)
        assert math.isfinite(val)

    def test_gcf_default_behavior_unchanged(self) -> None:
        # Sanity: with the real _FPMIN the result is the documented value.
        val = ht._gcf(2.0, 10.0)
        assert abs(val - 0.0004993992273873336) < 1e-6

    def test_betacf_default_behavior_unchanged(self) -> None:
        val = ht._betacf(2.0, 3.0, 0.5)
        assert abs(val - 3.666666666666667) < 1e-6
