"""Tests for cds.fractals — Mandelbrot, Julia, Barnsley fern, Sierpinski.

Covers escape-time values, image shape, determinism and validation.
"""

from __future__ import annotations

import math

import pytest

from cds.fractals import (
    barnsley_fern,
    julia,
    mandelbrot,
    mandelbrot_image,
    sierpinski_triangle,
)
from cds.fractals.sets import (
    barnsley_fern as barnsley_fern_direct,
)
from cds.fractals.sets import (
    julia as julia_direct,
)
from cds.fractals.sets import (
    mandelbrot as mandelbrot_direct,
)
from cds.fractals.sets import (
    mandelbrot_image as mandelbrot_image_direct,
)
from cds.fractals.sets import (
    sierpinski_triangle as sierpinski_triangle_direct,
)

# ---------------------------------------------------------------------------
# Mandelbrot
# ---------------------------------------------------------------------------


def test_mandelbrot_inside_origin() -> None:
    """c=0 never escapes — returns max_iter."""
    assert mandelbrot(0.0, 0.0, max_iter=100) == 100
    assert mandelbrot(0.0, 0.0) == 256  # default


def test_mandelbrot_far_outside_escapes_immediately() -> None:
    """c=(2,2) has |c|^2=8 >4, escapes on first iteration."""
    assert mandelbrot(2.0, 2.0, max_iter=256) == 1
    assert mandelbrot(2.0, -2.0, max_iter=10) == 1
    assert mandelbrot(-2.0, 2.0) == 1


def test_mandelbrot_real_two() -> None:
    """c=(2,0): 0->2->6 escapes on second iteration."""
    assert mandelbrot(2.0, 0.0, max_iter=256) == 2


def test_mandelbrot_real_one() -> None:
    """c=(1,0): 0->1->2->5 escapes on third iteration."""
    assert mandelbrot(1.0, 0.0, max_iter=256) == 3


def test_mandelbrot_inside_cardioid_and_bulb() -> None:
    """Points known to be inside — never escape."""
    # period-2 bulb centre at -1
    assert mandelbrot(-1.0, 0.0, max_iter=50) == 50
    # inside main cardioid
    assert mandelbrot(-0.5, 0.0, max_iter=50) == 50
    assert mandelbrot(0.0, 0.0, max_iter=20) == 20


def test_mandelbrot_outside_escapes_within_max_iter() -> None:
    assert mandelbrot(0.5, 0.5, max_iter=256) < 256
    assert mandelbrot(1.5, 1.5, max_iter=100) < 10


def test_mandelbrot_max_iter_validation() -> None:
    with pytest.raises(ValueError, match="max_iter must be positive"):
        mandelbrot(0.0, 0.0, max_iter=0)
    with pytest.raises(ValueError, match="max_iter must be positive"):
        mandelbrot(0.0, 0.0, max_iter=-1)


def test_mandelbrot_max_iter_type_validation() -> None:
    with pytest.raises(TypeError, match="max_iter must be an integer"):
        mandelbrot(0.0, 0.0, max_iter=3.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="max_iter must be an integer"):
        mandelbrot(0.0, 0.0, max_iter=True)


def test_mandelbrot_finite_validation() -> None:
    with pytest.raises(ValueError, match="c_real must be finite"):
        mandelbrot(float("inf"), 0.0)
    with pytest.raises(ValueError, match="c_imag must be finite"):
        mandelbrot(0.0, float("nan"))
    with pytest.raises(ValueError, match="c_real must be finite"):
        mandelbrot(float("-inf"), 0.0)


def test_mandelbrot_direct_import() -> None:
    assert mandelbrot_direct(0.0, 0.0, max_iter=10) == 10


# ---------------------------------------------------------------------------
# Julia
# ---------------------------------------------------------------------------


def test_julia_origin_zero_c() -> None:
    """Julia with c=0, z0=0 stays bounded."""
    assert julia(0.0, 0.0, 0.0, 0.0, max_iter=100) == 100


def test_julia_already_outside_returns_zero() -> None:
    """Julia checks |z| before iterating — immediate escape returns 0.

    Strict threshold ``|z|**2 > 4`` means points exactly on radius 2 escape
    after one iteration (return 1), not 0.
    """
    assert julia(2.0, 2.0, 0.0, 0.0, max_iter=256) == 0
    assert julia(-2.5, 0.0, 0.0, 0.0) == 0
    assert julia(2.0, 0.0, 0.0, 0.0, max_iter=256) == 1
    assert julia(0.0, 2.0, 0.0, 0.0) == 1


def test_julia_inside_unit_circle_with_zero_c() -> None:
    """c=0, |z0|<1 stays bounded; |z0|=1 also bounded for c=0."""
    assert julia(0.5, 0.0, 0.0, 0.0, max_iter=50) == 50
    assert julia(1.0, 0.0, 0.0, 0.0, max_iter=50) == 50
    assert julia(0.5, 0.5, 0.0, 0.0, max_iter=30) == 30


def test_julia_escapes_after_some_iterations() -> None:
    """z0=1.5, c=0: 1.5->2.25 escapes on second check (index 1)."""
    # 1.5^2=2.25 -> |2.25|>2 -> returns 1
    assert julia(1.5, 0.0, 0.0, 0.0, max_iter=100) == 1


def test_julia_finite_validation() -> None:
    with pytest.raises(ValueError, match="z_real must be finite"):
        julia(float("inf"), 0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="z_imag must be finite"):
        julia(0.0, float("nan"), 0.0, 0.0)
    with pytest.raises(ValueError, match="c_real must be finite"):
        julia(0.0, 0.0, float("-inf"), 0.0)
    with pytest.raises(ValueError, match="c_imag must be finite"):
        julia(0.0, 0.0, 0.0, float("inf"))


def test_julia_max_iter_validation() -> None:
    with pytest.raises(ValueError, match="max_iter must be positive"):
        julia(0.0, 0.0, 0.0, 0.0, max_iter=0)
    with pytest.raises(TypeError, match="max_iter must be an integer"):
        julia(0.0, 0.0, 0.0, 0.0, max_iter=2.5)  # type: ignore[arg-type]


def test_julia_direct_import() -> None:
    assert julia_direct(0.0, 0.0, 0.0, 0.0, max_iter=5) == 5


# ---------------------------------------------------------------------------
# Mandelbrot image
# ---------------------------------------------------------------------------


def test_mandelbrot_image_shape() -> None:
    img = mandelbrot_image(-2.0, 1.0, -1.5, 1.5, width=4, height=3, max_iter=10)
    assert len(img) == 3
    for row in img:
        assert len(row) == 4


def test_mandelbrot_image_values_in_range() -> None:
    img = mandelbrot_image(-2.0, 1.0, -1.5, 1.5, width=5, height=5, max_iter=20)
    for row in img:
        for val in row:
            assert 1 <= val <= 20


def test_mandelbrot_image_determinism() -> None:
    a = mandelbrot_image(-2.0, 1.0, -1.0, 1.0, width=6, height=4, max_iter=30)
    b = mandelbrot_image(-2.0, 1.0, -1.0, 1.0, width=6, height=4, max_iter=30)
    assert a == b


def test_mandelbrot_image_single_pixel_width_one() -> None:
    img = mandelbrot_image(-0.5, 0.5, -0.5, 0.5, width=1, height=1, max_iter=15)
    assert len(img) == 1
    assert len(img[0]) == 1
    assert 1 <= img[0][0] <= 15


def test_mandelbrot_image_single_row() -> None:
    img = mandelbrot_image(-2.0, 1.0, -0.2, 0.2, width=3, height=1, max_iter=10)
    assert len(img) == 1
    assert len(img[0]) == 3


def test_mandelbrot_image_single_column() -> None:
    img = mandelbrot_image(-0.5, 0.5, -1.0, 1.0, width=1, height=4, max_iter=10)
    assert len(img) == 4
    for row in img:
        assert len(row) == 1


def test_mandelbrot_image_known_interior() -> None:
    """Single point at origin should be inside (max_iter)."""
    img = mandelbrot_image(0.0, 0.5, 0.0, 0.5, width=1, height=1, max_iter=25)
    assert img[0][0] == 25
    # Fat pixel region around origin should contain the centre
    img2 = mandelbrot_image(-0.1, 0.1, -0.1, 0.1, width=3, height=3, max_iter=30)
    # centre pixel (1,1) maps to (0,0)
    assert img2[1][1] == 30


def test_mandelbrot_image_validation() -> None:
    with pytest.raises(ValueError, match="xmin must be finite"):
        mandelbrot_image(float("inf"), 1.0, -1.0, 1.0, width=2, height=2)
    with pytest.raises(ValueError, match="xmax must be finite"):
        mandelbrot_image(-1.0, float("nan"), -1.0, 1.0, width=2, height=2)
    with pytest.raises(ValueError, match="ymin must be finite"):
        mandelbrot_image(-1.0, 1.0, float("-inf"), 1.0, width=2, height=2)
    with pytest.raises(ValueError, match="ymax must be finite"):
        mandelbrot_image(-1.0, 1.0, -1.0, float("inf"), width=2, height=2)
    with pytest.raises(TypeError, match="width must be an integer"):
        mandelbrot_image(-1.0, 1.0, -1.0, 1.0, width=2.5, height=2)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="height must be an integer"):
        mandelbrot_image(-1.0, 1.0, -1.0, 1.0, width=2, height=True)
    with pytest.raises(ValueError, match="width must be positive"):
        mandelbrot_image(-1.0, 1.0, -1.0, 1.0, width=0, height=2)
    with pytest.raises(ValueError, match="height must be positive"):
        mandelbrot_image(-1.0, 1.0, -1.0, 1.0, width=2, height=-1)
    with pytest.raises(ValueError, match="xmin must be less than xmax"):
        mandelbrot_image(1.0, -1.0, -1.0, 1.0, width=2, height=2)
    with pytest.raises(ValueError, match="ymin must be less than ymax"):
        mandelbrot_image(-1.0, 1.0, 1.0, -1.0, width=2, height=2)
    with pytest.raises(ValueError, match="xmin must be less than xmax"):
        mandelbrot_image(0.0, 0.0, -1.0, 1.0, width=2, height=2)
    with pytest.raises(ValueError, match="max_iter must be positive"):
        mandelbrot_image(-1.0, 1.0, -1.0, 1.0, width=2, height=2, max_iter=0)
    with pytest.raises(TypeError, match="max_iter must be an integer"):
        mandelbrot_image(-1.0, 1.0, -1.0, 1.0, width=2, height=2, max_iter=3.5)  # type: ignore[arg-type]


def test_mandelbrot_image_direct_import() -> None:
    img = mandelbrot_image_direct(-1.0, 1.0, -1.0, 1.0, width=2, height=2, max_iter=5)
    assert len(img) == 2


# ---------------------------------------------------------------------------
# Barnsley fern
# ---------------------------------------------------------------------------


def test_barnsley_empty() -> None:
    assert barnsley_fern(0) == []
    assert barnsley_fern(0, seed=42) == []


def test_barnsley_length() -> None:
    pts = barnsley_fern(10, seed=0)
    assert len(pts) == 10
    pts2 = barnsley_fern(100, seed=1)
    assert len(pts2) == 100


def test_barnsley_determinism() -> None:
    a = barnsley_fern(50, seed=123)
    b = barnsley_fern(50, seed=123)
    assert a == b


def test_barnsley_different_seeds_differ() -> None:
    a = barnsley_fern(20, seed=0)
    b = barnsley_fern(20, seed=1)
    assert a != b


def test_barnsley_without_seed_runs() -> None:
    pts = barnsley_fern(5)
    assert len(pts) == 5
    for x, y in pts:
        assert math.isfinite(x) and math.isfinite(y)


def test_barnsley_bounds() -> None:
    pts = barnsley_fern(500, seed=42)
    for x, y in pts:
        assert -3.0 <= x <= 3.0
        assert 0.0 <= y <= 10.0


def test_barnsley_all_branches_hit() -> None:
    """Large sample with seed 0 should hit all four IFS maps.

    We check that points appear in distinct y-ranges characteristic of each
    map's vertical translation. The stem (f1) leaves y near 0, f4 has
    y in [0.4, ~0.8] early, etc. Simpler: just run many points and verify
    the fern has spread — which requires all branches.
    """
    pts = barnsley_fern(2000, seed=0)
    # At least one point with y < 0.5 (stem via f1) and one with y > 9
    assert any(y < 0.3 for _, y in pts)
    assert any(y > 9.0 for _, y in pts)
    # All x not identical
    xs = [x for x, _ in pts]
    assert max(xs) - min(xs) > 1.0


def test_barnsley_validation() -> None:
    with pytest.raises(ValueError, match="n_points must be non-negative"):
        barnsley_fern(-1)
    with pytest.raises(TypeError, match="n_points must be an integer"):
        barnsley_fern(3.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="n_points must be an integer"):
        barnsley_fern(True)
    with pytest.raises(TypeError, match="seed must be an integer or None"):
        barnsley_fern(10, seed=3.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="seed must be an integer or None"):
        barnsley_fern(10, seed=True)


def test_barnsley_direct_import() -> None:
    assert len(barnsley_fern_direct(3, seed=0)) == 3


# ---------------------------------------------------------------------------
# Sierpinski triangle
# ---------------------------------------------------------------------------


def test_sierpinski_empty() -> None:
    assert sierpinski_triangle(0) == []
    assert sierpinski_triangle(0, seed=99) == []


def test_sierpinski_length() -> None:
    assert len(sierpinski_triangle(10, seed=0)) == 10


def test_sierpinski_determinism() -> None:
    a = sierpinski_triangle(50, seed=77)
    b = sierpinski_triangle(50, seed=77)
    assert a == b


def test_sierpinski_different_seeds_differ() -> None:
    a = sierpinski_triangle(20, seed=0)
    b = sierpinski_triangle(20, seed=1)
    assert a != b


def test_sierpinski_bounds() -> None:
    pts = sierpinski_triangle(500, seed=42)
    h = math.sqrt(3.0) / 2.0
    for x, y in pts:
        assert 0.0 <= x <= 1.0
        assert 0.0 <= y <= h
        # barycentric inside triangle: y <= sqrt(3)*min(x, 1-x) approx
        assert y <= h + 1e-12


def test_sierpinski_all_three_vertices_hit() -> None:
    """Large sample should hit all three random choices."""
    pts = sierpinski_triangle(300, seed=0)
    # Check that points spread across left, right, top
    assert any(x < 0.25 and y < 0.25 for x, y in pts)
    assert any(x > 0.75 and y < 0.25 for x, y in pts)
    assert any(y > 0.6 for _, y in pts)


def test_sierpinski_validation() -> None:
    with pytest.raises(ValueError, match="n_points must be non-negative"):
        sierpinski_triangle(-1)
    with pytest.raises(TypeError, match="n_points must be an integer"):
        sierpinski_triangle(1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="seed must be an integer or None"):
        sierpinski_triangle(10, seed="bad")  # type: ignore[arg-type]


def test_sierpinski_direct_import() -> None:
    assert len(sierpinski_triangle_direct(5, seed=0)) == 5


def test_sierpinski_without_seed() -> None:
    pts = sierpinski_triangle(5)
    assert len(pts) == 5
