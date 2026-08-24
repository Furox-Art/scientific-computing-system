"""Tests for cds.pde — explicit finite-difference heat and wave solvers."""

import math
from collections.abc import Callable

import pytest

from cds.pde import HeatResult, WaveResult, solve_heat, solve_wave


def _linspace_values(
    nx: int,
    length: float,
    fn: Callable[[float], float] | None = None,
) -> list[float]:
    """Sample ``fn`` (default: identity) on a uniform ``nx``-point grid."""
    xs = [length * i / (nx - 1) for i in range(nx)]
    return [xs[i] if fn is None else fn(xs[i]) for i in range(nx)]


def _gaussian_profile(nx: int, length: float, center: float, width: float) -> list[float]:
    return _linspace_values(nx, length, lambda x: math.exp(-(((x - center) / width) ** 2)))


def _trapezoid_mass(values: list[float], dx: float) -> float:
    return dx * (0.5 * values[0] + sum(values[1:-1]) + 0.5 * values[-1])


class TestSolveHeatValidation:
    def test_u0_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="exactly 5 points"):
            solve_heat([0.0] * 4, 1.0, 1.0, 0.1, 5)

    def test_nx_below_three_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 3"):
            solve_heat([0.0, 1.0], 1.0, 1.0, 0.1, 2)

    def test_zero_alpha_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha must be positive"):
            solve_heat([0.0] * 5, 0.0, 1.0, 0.1, 5)

    def test_negative_alpha_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha must be positive"):
            solve_heat([0.0] * 5, -1.0, 1.0, 0.1, 5)

    def test_non_positive_length_raises(self) -> None:
        with pytest.raises(ValueError, match="length must be positive"):
            solve_heat([0.0] * 5, 1.0, -1.0, 0.1, 5)

    def test_non_positive_t_final_raises(self) -> None:
        with pytest.raises(ValueError, match="t_final must be positive"):
            solve_heat([0.0] * 5, 1.0, 1.0, 0.0, 5)

    @pytest.mark.parametrize("bad", ["fixed", "", "DIRICHLET", "periodic"])
    def test_unknown_boundary_raises(self, bad: str) -> None:
        with pytest.raises(ValueError, match="boundary must be 'dirichlet' or 'neumann'"):
            solve_heat([0.0] * 5, 1.0, 1.0, 0.1, 5, boundary=bad)

    @pytest.mark.parametrize("bad_dt", [0.0, -1e-3])
    def test_non_positive_user_dt_raises(self, bad_dt: float) -> None:
        with pytest.raises(ValueError, match="dt must be positive"):
            solve_heat([0.0] * 11, 1.0, 1.0, 0.1, 11, dt=bad_dt)

    def test_unstable_user_dt_raises(self) -> None:
        # dx = 0.1, alpha = 1 -> r = dt/dx^2 = 6 >> 0.5.
        with pytest.raises(ValueError, match="violates heat stability"):
            solve_heat([0.0] * 11, 1.0, 1.0, 0.05, 11, dt=0.06)


class TestSolveHeatStepping:
    def test_auto_dt_matches_documented_formula(self) -> None:
        nx, length, alpha = 21, 2.0, 0.7
        res = solve_heat([0.0] * nx, alpha, length, 0.01, nx)
        dx = length / (nx - 1)
        assert res.dt == pytest.approx(0.9 * dx**2 / (2.0 * alpha))
        assert res.n_steps == math.ceil(0.01 / res.dt)

    def test_user_dt_is_respected_when_stable(self) -> None:
        res = solve_heat([0.0] * 11, 1.0, 1.0, 0.02, 11, dt=0.004)
        assert res.dt == 0.004
        assert res.n_steps == 5

    def test_courant_number_exactly_at_limit_is_accepted(self) -> None:
        # dx = 0.1, alpha = 1, dt = 0.005 -> r = 0.5 exactly: stable boundary.
        res = solve_heat([0.0] * 11, 1.0, 1.0, 0.005, 11, dt=0.005)
        assert res.n_steps == 1

    def test_single_step_when_t_final_below_dt(self) -> None:
        u0 = [float(i) for i in range(11)]
        res = solve_heat(u0, 1.0, 1.0, 1e-4, 11)
        assert res.n_steps == 1
        assert len(res.u_final) == 11

    def test_returns_heat_result_dataclass(self) -> None:
        res = solve_heat([0.0] * 5, 1.0, 1.0, 0.001, 5)
        assert isinstance(res, HeatResult)
        assert isinstance(res.u_final, list)
        assert isinstance(res.dt, float)
        assert isinstance(res.n_steps, int)


class TestSolveHeatPhysics:
    @pytest.mark.parametrize("boundary", ["dirichlet", "neumann"])
    def test_flat_profile_is_stationary(self, boundary: str) -> None:
        res = solve_heat([3.25] * 41, 1.3, 1.0, 0.5, 41, boundary=boundary)
        assert res.u_final == pytest.approx([3.25] * 41)

    def test_dirichlet_ends_stay_pinned(self) -> None:
        nx = 51
        u0 = [0.0 if i < nx // 2 else 1.0 for i in range(nx)]
        res = solve_heat(u0, 1.0, 1.0, 0.1, nx)
        assert res.u_final[0] == 0.0
        assert res.u_final[-1] == 1.0

    def test_neumann_mirrors_neighbour_after_final_step(self) -> None:
        nx = 31
        u0 = _gaussian_profile(nx, 1.0, 0.5, 0.15)
        res = solve_heat(u0, 1.0, 1.0, 0.05, nx, boundary="neumann")
        assert res.u_final[0] == res.u_final[1]
        assert res.u_final[-1] == res.u_final[-2]

    def test_gaussian_blob_peak_decays(self) -> None:
        nx, length = 201, 1.0
        u0 = _gaussian_profile(nx, length, center=0.5, width=0.08)
        res = solve_heat(u0, 0.5, length, 0.05, nx)
        assert res.u_final[nx // 2] < 0.75 * u0[nx // 2]

    def test_gaussian_blob_spreads_and_conserves_mass(self) -> None:
        nx, length, dx = 201, 1.0, 1.0 / 200
        u0 = _gaussian_profile(nx, length, center=0.5, width=0.08)
        res = solve_heat(u0, 0.2, length, 0.02, nx)
        m0 = _trapezoid_mass(u0, dx)
        m1 = _trapezoid_mass(res.u_final, dx)
        assert m1 == pytest.approx(m0, rel=1e-2)

    def test_gaussian_blob_stays_symmetric_under_neumann(self) -> None:
        nx = 101
        u0 = _gaussian_profile(nx, 1.0, center=0.5, width=0.12)
        res = solve_heat(u0, 1.0, 1.0, 0.08, nx, boundary="neumann")
        for i in range(nx):
            assert res.u_final[i] == pytest.approx(res.u_final[nx - 1 - i])

    def test_sine_ic_matches_analytic_decay_within_five_percent(self) -> None:
        alpha, nx, length, t_final = 1.0, 101, 1.0, 0.1
        u0 = _linspace_values(nx, length, lambda x: math.sin(math.pi * x))
        res = solve_heat(u0, alpha, length, t_final, nx)
        t_sim = res.n_steps * res.dt
        expected = math.exp(-alpha * math.pi**2 * t_sim)
        assert res.u_final[nx // 2] == pytest.approx(expected, rel=0.05)


class TestSolveWaveValidation:
    def make_args(self) -> tuple[list[float], list[float], float]:
        return [0.0] * 7, [0.0] * 7, 1.0

    def test_u0_length_mismatch_raises(self) -> None:
        u0, v0, c = self.make_args()
        with pytest.raises(ValueError, match="u0 must have exactly 7 points"):
            solve_wave([0.0] * 6, v0, c, 1.0, 0.1, 7)

    def test_v0_length_mismatch_raises(self) -> None:
        u0, v0, c = self.make_args()
        with pytest.raises(ValueError, match="v0 must have exactly 7 points"):
            solve_wave(u0, [0.0] * 8, c, 1.0, 0.1, 7)

    def test_nx_below_three_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 3"):
            solve_wave([0.0, 0.0], [0.0, 0.0], 1.0, 1.0, 0.1, 2)

    @pytest.mark.parametrize("bad_c", [0.0, -2.0])
    def test_non_positive_c_raises(self, bad_c: float) -> None:
        with pytest.raises(ValueError, match="c must be positive"):
            solve_wave([0.0] * 5, [0.0] * 5, bad_c, 1.0, 0.1, 5)

    def test_non_positive_length_raises(self) -> None:
        with pytest.raises(ValueError, match="length must be positive"):
            solve_wave([0.0] * 5, [0.0] * 5, 1.0, 0.0, 0.1, 5)

    def test_non_positive_t_final_raises(self) -> None:
        with pytest.raises(ValueError, match="t_final must be positive"):
            solve_wave([0.0] * 5, [0.0] * 5, 1.0, 1.0, -0.1, 5)

    @pytest.mark.parametrize("bad_dt", [0.0, -0.01])
    def test_non_positive_user_dt_raises(self, bad_dt: float) -> None:
        with pytest.raises(ValueError, match="dt must be positive"):
            solve_wave([0.0] * 11, [0.0] * 11, 1.0, 1.0, 0.1, 11, dt=bad_dt)

    def test_user_dt_breaking_cfl_raises(self) -> None:
        # dx = 0.1, c = 1 -> CFL = dt/dx = 1.5 > 1.
        with pytest.raises(ValueError, match="violates wave CFL"):
            solve_wave([0.0] * 11, [0.0] * 11, 1.0, 1.0, 0.05, 11, dt=0.15)


class TestSolveWaveStepping:
    def test_auto_dt_matches_documented_formula(self) -> None:
        nx, length, c = 21, 2.0, 1.4
        res = solve_wave([0.0] * nx, [0.0] * nx, c, length, 0.1, nx)
        dx = length / (nx - 1)
        assert res.dt == pytest.approx(0.9 * dx / c)
        assert res.n_steps == math.ceil(0.1 / res.dt)

    def test_user_dt_is_respected_when_within_cfl(self) -> None:
        res = solve_wave([0.0] * 11, [0.0] * 11, 1.0, 1.0, 0.03, 11, dt=0.01)
        assert res.dt == 0.01
        assert res.n_steps == 3

    def test_courant_number_exactly_at_limit_is_accepted(self) -> None:
        # dx = 0.1, c = 1, dt = 0.1 -> CFL = 1 exactly.
        res = solve_wave([0.0] * 11, [0.0] * 11, 1.0, 1.0, 0.1, 11, dt=0.1)
        assert res.n_steps == 1

    def test_single_step_skips_leapfrog_recurrence(self) -> None:
        nx = 11
        u0 = [float(i) for i in range(nx)]
        res = solve_wave(u0, [0.0] * nx, 1.0, 1.0, 0.05, nx)
        assert res.n_steps == 1
        assert len(res.u_final) == nx

    def test_returns_wave_result_dataclass(self) -> None:
        res = solve_wave([0.0] * 5, [0.0] * 5, 1.0, 1.0, 0.001, 5)
        assert isinstance(res, WaveResult)
        assert isinstance(res.u_final, list)
        assert isinstance(res.dt, float)
        assert isinstance(res.n_steps, int)


class TestSolveWavePhysics:
    def _standing_wave_setup(self, nx: int) -> tuple[list[float], list[float]]:
        u0 = _linspace_values(nx, 1.0, lambda x: math.sin(math.pi * x))
        return u0, [0.0] * nx

    def test_pinned_ends_hold_initial_displacement(self) -> None:
        nx = 41
        u0 = [i / (nx - 1) for i in range(nx)]
        res = solve_wave(u0, [0.0] * nx, 1.0, 1.0, 0.25, nx)
        assert res.u_final[0] == 0.0
        assert res.u_final[-1] == 1.0

    def test_standing_wave_reconstructs_after_one_period(self) -> None:
        nx, c, length = 201, 1.0, 1.0
        u0, v0 = self._standing_wave_setup(nx)
        res = solve_wave(u0, v0, c, length, 2.0, nx)
        phase = math.pi * c * (res.n_steps * res.dt) / length
        expected = [amp * math.cos(phase) for amp in u0]
        assert res.u_final == pytest.approx(expected, abs=0.02)

    def test_standing_wave_inverts_after_half_period(self) -> None:
        nx, c, length = 201, 1.0, 1.0
        u0, v0 = self._standing_wave_setup(nx)
        res = solve_wave(u0, v0, c, length, 1.0, nx)
        phase = math.pi * c * (res.n_steps * res.dt) / length
        expected = [amp * math.cos(phase) for amp in u0]
        assert res.u_final == pytest.approx(expected, abs=0.02)
        assert res.u_final[nx // 2] < 0.0

    def test_velocity_kick_starts_linear_motion(self) -> None:
        nx, c, length = 101, 1.0, 1.0
        v0 = _linspace_values(nx, length, lambda x: math.sin(math.pi * x))
        res = solve_wave([0.0] * nx, v0, c, length, 0.01, nx)
        t_sim = res.n_steps * res.dt
        omega = math.pi * c / length
        scale = math.sin(omega * t_sim) / omega
        expected = [v * scale for v in v0]
        assert res.u_final == pytest.approx(expected, abs=5e-4)

    def test_energy_of_standing_wave_stays_bounded(self) -> None:
        nx, c = 101, 1.0
        u0, v0 = self._standing_wave_setup(nx)
        res = solve_wave(u0, v0, c, 1.0, 5.0, nx)
        assert max(abs(value) for value in res.u_final) <= 1.0 + 1e-6


class TestPurityAndContracts:
    def test_solve_heat_does_not_mutate_inputs(self) -> None:
        u0 = [1.0, 2.0, 3.0, 2.0, 1.0]
        snapshot = list(u0)
        solve_heat(u0, 1.0, 1.0, 0.01, 5)
        assert u0 == snapshot

    def test_solve_wave_does_not_mutate_inputs(self) -> None:
        u0 = [0.0, 1.0, 0.0, 1.0, 0.0]
        v0 = [1.0, 0.0, -1.0, 0.0, 1.0]
        snap_u, snap_v = list(u0), list(v0)
        solve_wave(u0, v0, 1.0, 1.0, 0.01, 5)
        assert u0 == snap_u
        assert v0 == snap_v

    def test_results_are_independent_objects(self) -> None:
        u0 = _gaussian_profile(21, 1.0, 0.5, 0.2)
        first = solve_heat(u0, 1.0, 1.0, 0.01, 21)
        second = solve_heat(u0, 1.0, 1.0, 0.01, 21)
        assert first is not second
        assert first.u_final is not second.u_final
        assert first.u_final == second.u_final
