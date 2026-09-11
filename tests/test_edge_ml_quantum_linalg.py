"""Coverage completion: cover defensive numerical branches.

Targets:
- ml/neural.py           (sigmoid overflow fallback, identity branch)
- quantum/multi_qubit.py (measurement floating-point fallback)
- math_utils/linalg.py   (QR/power-iteration/singular defensive branches)
- stats/hypothesis_tests.py (_FPMIN clamp branches)

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
    """Cover the sigmoid OverflowError fallback."""

    def test_sigmoid_negative_overflow_returns_zero(self) -> None:
        layer = Layer(1, 1, activation="sigmoid")
        assert layer._activate(-1000.0) == 0.0

    def test_sigmoid_large_positive(self) -> None:
        layer = Layer(1, 1, activation="sigmoid")
        assert layer._activate(1000.0) == pytest.approx(1.0)


class TestNeuralIdentityActivation:
    """Cover the identity activation and its derivative."""

    def test_identity_activate_passthrough(self) -> None:
        layer = Layer(2, 2, activation="identity")
        assert layer._activate(3.7) == 3.7

    def test_identity_derivative_is_one(self) -> None:
        layer = Layer(2, 2, activation="identity")
        assert layer._activate_derivative(3.7, 0.5) == 1.0


# ---------------------------------------------------------------------------
# 2. quantum/multi_qubit.py — measurement floating-point fallback
# ---------------------------------------------------------------------------


class TestMeasureFallback:
    """Cover fallback only for a valid state at the r == 1.0 rounding edge."""

    def test_measure_valid_state_rounding_fallback(self) -> None:
        reg = QuantumRegister.zeros(2)
        # random.Random.random() normally returns [0, 1), so 1.0 is used only
        # to exercise the defensive rounding fallback without constructing an
        # invalid zero-norm quantum state.
        with mock.patch("cds.quantum.multi_qubit.random.Random.random", return_value=1.0):
            idx = reg.measure(seed=0)
        assert idx == len(reg.amplitudes) - 1
        assert reg.amplitudes[idx] == 1.0 + 0j


# ---------------------------------------------------------------------------
# 3. math_utils/linalg.py — QR degenerate columns + power_iteration break
# ---------------------------------------------------------------------------


class TestQRDegenerateColumns:
    """Cover degenerate-column guards in qr_decomposition."""

    def test_qr_zero_column(self) -> None:
        matrix = [[0.0, 5.0], [0.0, 5.0]]
        q, r = qr_decomposition(matrix)
        assert len(r) == 2
        assert len(q) == 2

    def test_qr_zero_householder_vector(self) -> None:
        matrix = [[1.0, 0.0], [0.0, 1.0]]
        q, _r = qr_decomposition(matrix)
        n = len(q)
        qtq = [
            [sum(q[row][k] * q[column][k] for k in range(n)) for column in range(n)]
            for row in range(n)
        ]
        for row in range(n):
            assert abs(qtq[row][row] - 1.0) < 1e-10


class TestPowerIterationZeroNormBreak:
    """Cover the zero-norm break in power_iteration."""

    def test_zero_matrix_breaks_immediately(self) -> None:
        eigenvalue, vector = power_iteration([[0.0, 0.0], [0.0, 0.0]], max_iter=50)
        assert eigenvalue == 0.0
        assert len(vector) == 2

    def test_nilpotent_zero_norm(self) -> None:
        eigenvalue, _vector = power_iteration([[0.0, 1.0], [0.0, 0.0]], max_iter=50)
        assert abs(eigenvalue) < 1e-9


# ---------------------------------------------------------------------------
# 4. math_utils/linalg.py — singular backward pivot + overflow fallback
# ---------------------------------------------------------------------------


class TestSingularBackwardPivot:
    """Cover backward-substitution singular pivot guards via controlled mocks."""

    def test_solve_linear_backward_pivot_guard(self) -> None:
        fake_l = [[1.0, 0.0], [0.0, 1.0]]
        fake_u = [[1.0, 0.0], [0.0, 0.0]]
        fake_p = [[1.0, 0.0], [0.0, 1.0]]
        with mock.patch(
            "cds.math_utils.linalg.lu_decomposition",
            return_value=(fake_p, fake_l, fake_u),
        ):
            with pytest.raises(ValueError, match="singular"):
                solve_linear([[1.0, 0.0], [0.0, 0.0]], [1.0, 2.0])

    def test_matrix_inverse_backward_pivot_guard(self) -> None:
        fake_l = [[1.0, 0.0], [0.0, 1.0]]
        fake_u = [[2.0, 0.0], [0.0, 0.0]]
        fake_p = [[1.0, 0.0], [0.0, 1.0]]
        with mock.patch(
            "cds.math_utils.linalg.lu_decomposition",
            return_value=(fake_p, fake_l, fake_u),
        ):
            with pytest.raises(ValueError, match="singular"):
                matrix_inverse([[2.0, 0.0], [0.0, 0.0]])


class TestPowerIterationOverflowExcept:
    """Cover the defensive OverflowError branch in power_iteration."""

    def test_overflowerror_fallback(self) -> None:
        real_sqrt = math.sqrt

        def fake_sqrt(x: float) -> float:
            if x >= 1.0:
                raise OverflowError("simulated overflow in sqrt")
            return real_sqrt(x)

        with mock.patch("cds.math_utils.linalg.math.sqrt", side_effect=fake_sqrt):
            eigenvalue, vector = power_iteration(
                [[2.0, 0.0], [0.0, 1.0]],
                max_iter=200,
            )
        assert abs(eigenvalue - 2.0) < 1e-9
        assert len(vector) == 2


# ---------------------------------------------------------------------------
# 5. stats/hypothesis_tests.py — _FPMIN clamp branches
# ---------------------------------------------------------------------------


class TestGammaBetaFPMINClamps:
    """Cover Numerical-Recipes _FPMIN underflow clamps."""

    def test_gcf_fpmin_clamps_fire(self) -> None:
        with mock.patch.object(dist, "_FPMIN", 1.0):
            value = ht._gcf(2.0, 10.0)
        assert math.isfinite(value) and value >= 0.0

    def test_betacf_fpmin_clamps_fire(self) -> None:
        with mock.patch.object(dist, "_FPMIN", 1.0):
            value = ht._betacf(2.0, 3.0, 0.5)
        assert math.isfinite(value)

    def test_gcf_default_behavior_unchanged(self) -> None:
        value = ht._gcf(2.0, 10.0)
        assert abs(value - 0.0004993992273873336) < 1e-6

    def test_betacf_default_behavior_unchanged(self) -> None:
        value = ht._betacf(2.0, 3.0, 0.5)
        assert abs(value - 3.666666666666667) < 1e-6
