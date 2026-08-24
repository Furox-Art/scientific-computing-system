"""Tests for cds.interpolate — 1-D and 2-D interpolation factories."""

import pytest

from cds.interpolate import interp1d, interp2d


class TestInterp1dValidation:
    def test_unknown_kind_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown kind"):
            interp1d([0.0, 1.0], [0.0, 1.0], "cubic")

    def test_empty_x_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            interp1d([], [])

    def test_empty_y_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            interp1d([0.0, 1.0], [])

    def test_single_point_axis_raises(self) -> None:
        with pytest.raises(ValueError, match="at least two points"):
            interp1d([1.0], [2.0])

    def test_duplicate_x_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            interp1d([0.0, 1.0, 1.0], [1.0, 2.0, 3.0])

    def test_decreasing_x_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            interp1d([1.0, 0.0], [1.0, 2.0])

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            interp1d([0.0, 1.0], [1.0])


class TestInterp1dLinear:
    def test_two_point_midpoint(self) -> None:
        f = interp1d([2.0, 4.0], [10.0, 20.0])
        assert f(3.0) == pytest.approx(15.0)
        assert f(2.5) == pytest.approx(12.5)

    def test_multi_segment_hand_computed(self) -> None:
        f = interp1d([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 4.0, 9.0])
        assert f(0.5) == pytest.approx(0.5)
        assert f(1.5) == pytest.approx(2.5)
        assert f(2.75) == pytest.approx(7.75)

    def test_reproduces_knots(self) -> None:
        xs = [0.0, 1.0, 2.0, 3.0]
        ys = [3.0, -1.0, 4.0, 2.0]
        f = interp1d(xs, ys)
        for knot, value in zip(xs, ys):
            assert f(knot) == pytest.approx(value)

    def test_identity_data_returns_query(self) -> None:
        f = interp1d([0.0, 0.5, 1.0], [0.0, 0.5, 1.0])
        assert f(0.25) == pytest.approx(0.25)
        assert f(0.9) == pytest.approx(0.9)

    def test_inclusive_endpoints_accepted(self) -> None:
        f = interp1d([0.0, 1.0, 2.0], [5.0, 6.0, 8.0])
        assert f(0.0) == pytest.approx(5.0)
        assert f(2.0) == pytest.approx(8.0)


class TestInterp1dNearest:
    def test_rounds_to_closer_knot(self) -> None:
        f = interp1d([0.0, 1.0, 2.0, 3.0], [10.0, 20.0, 30.0, 40.0], "nearest")
        assert f(0.49) == 10.0
        assert f(0.51) == 20.0
        assert f(2.99) == 40.0

    def test_exact_midpoint_takes_lower_knot(self) -> None:
        f = interp1d([0.0, 1.0], [10.0, 20.0], "nearest")
        assert f(0.5) == 10.0

    def test_upper_boundary_takes_last_knot(self) -> None:
        f = interp1d([0.0, 1.0, 2.0, 3.0], [10.0, 20.0, 30.0, 40.0], "nearest")
        assert f(3.0) == 40.0

    def test_two_point_grid(self) -> None:
        f = interp1d([5.0, 6.0], [1.0, 2.0], "nearest")
        assert f(5.9) == 2.0
        assert f(5.1) == 1.0


class TestInterp1dDomain:
    def test_below_domain_raises(self) -> None:
        f = interp1d([0.0, 1.0], [0.0, 1.0])
        with pytest.raises(ValueError, match="outside"):
            f(-0.1)

    def test_above_domain_raises(self) -> None:
        f = interp1d([0.0, 1.0], [0.0, 1.0])
        with pytest.raises(ValueError, match="outside"):
            f(1.1)


class TestInterp2dValidation:
    def test_empty_x_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            interp2d([], [0.0, 1.0], [])

    def test_empty_y_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            interp2d([0.0, 1.0], [], [[]])

    def test_single_point_x_raises(self) -> None:
        with pytest.raises(ValueError, match="at least two points"):
            interp2d([1.0], [0.0, 1.0], [[0.0, 0.0]])

    def test_non_increasing_x_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            interp2d([0.0, 0.0], [0.0, 1.0], [[0.0, 0.0], [0.0, 0.0]])

    def test_non_increasing_y_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            interp2d([0.0, 1.0], [2.0, 1.0], [[0.0, 0.0], [0.0, 0.0]])

    def test_row_count_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="one row per point"):
            interp2d([0.0, 1.0], [0.0, 1.0], [[0.0, 0.0]])

    def test_ragged_z_raises(self) -> None:
        with pytest.raises(ValueError, match="same length as y"):
            interp2d([0.0, 1.0], [0.0, 1.0], [[0.0, 0.0], [0.0]])


class TestInterp2dBilinear:
    def test_unit_square_center_is_corner_average(self) -> None:
        f = interp2d([0.0, 1.0], [0.0, 1.0], [[0.0, 2.0], [4.0, 6.0]])
        assert f(0.5, 0.5) == pytest.approx(3.0)

    def test_rectangular_cell_hand_computed(self) -> None:
        f = interp2d([0.0, 2.0], [0.0, 4.0], [[0.0, 4.0], [8.0, 16.0]])
        assert f(1.0, 2.0) == pytest.approx(7.0)

    def test_edge_midpoints(self) -> None:
        f = interp2d([0.0, 2.0], [0.0, 4.0], [[0.0, 4.0], [8.0, 16.0]])
        assert f(1.0, 4.0) == pytest.approx(10.0)
        assert f(0.0, 2.0) == pytest.approx(2.0)

    def test_three_by_three_interior_cell(self) -> None:
        z = [[0.0, 1.0, 2.0], [10.0, 11.0, 12.0], [20.0, 21.0, 22.0]]
        f = interp2d([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], z)
        assert f(0.5, 0.5) == pytest.approx(5.5)
        assert f(1.25, 1.9) == pytest.approx(14.4)

    def test_reproduces_grid_corners(self) -> None:
        f = interp2d([0.0, 2.0], [0.0, 4.0], [[0.0, 4.0], [8.0, 16.0]])
        assert f(0.0, 0.0) == pytest.approx(0.0)
        assert f(2.0, 4.0) == pytest.approx(16.0)


class TestInterp2dDomain:
    def test_x_out_of_bounds_raises(self) -> None:
        f = interp2d([0.0, 2.0], [0.0, 4.0], [[0.0, 4.0], [8.0, 16.0]])
        with pytest.raises(ValueError, match="x query"):
            f(-0.1, 1.0)

    def test_y_out_of_bounds_raises(self) -> None:
        f = interp2d([0.0, 2.0], [0.0, 4.0], [[0.0, 4.0], [8.0, 16.0]])
        with pytest.raises(ValueError, match="y query"):
            f(1.0, 4.1)


class TestPurityAndDeterminism:
    def test_interp1d_calls_are_repeatable_and_inputs_unmodified(self) -> None:
        x = [0.0, 1.0]
        y = [0.0, 2.0]
        f = interp1d(x, y)
        assert f(0.3) == pytest.approx(0.6)
        assert f(0.3) == pytest.approx(0.6)
        assert x == [0.0, 1.0]
        assert y == [0.0, 2.0]

    def test_interp2d_calls_are_repeatable_and_inputs_unmodified(self) -> None:
        z = [[0.0, 4.0], [8.0, 16.0]]
        f = interp2d([0.0, 2.0], [0.0, 4.0], z)
        assert f(0.5, 0.5) == pytest.approx(2.625)
        assert f(0.5, 0.5) == pytest.approx(2.625)
        assert z == [[0.0, 4.0], [8.0, 16.0]]

    def test_independent_factories_are_deterministic(self) -> None:
        f1 = interp1d([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
        f2 = interp1d([0.0, 1.0, 2.0], [0.0, 1.0, 4.0], "linear")
        assert f1(1.5) == f2(1.5) == pytest.approx(2.5)
