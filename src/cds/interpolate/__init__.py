"""Interpolation on 1-D samples and regular rectangular grids — zero dependencies.

Factory functions returning plain, pure callables. Every axis must be strictly
increasing, and queries are confined to the sampled domain: no extrapolation,
no hidden state, fully deterministic results.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

__all__ = ["interp1d", "interp2d"]


def _check_axis(values: Sequence[float], name: str) -> None:
    """Validate one interpolation axis.

    Args:
        values: candidate axis coordinates.
        name: axis label used in error messages.

    Raises:
        ValueError: if the axis is empty, holds a single point, or is not
            strictly increasing.
    """
    if len(values) == 0:
        raise ValueError(f"{name} must be non-empty")
    if len(values) == 1:
        raise ValueError(f"{name} must contain at least two points")
    if any(values[i] >= values[i + 1] for i in range(len(values) - 1)):
        raise ValueError(f"{name} must be strictly increasing")


def _require_in_bounds(grid: tuple[float, ...], query: float, name: str) -> None:
    """Reject queries falling outside the closed interval spanned by ``grid``.

    Args:
        grid: strictly increasing axis coordinates.
        query: point requested by the caller.
        name: axis label used in error messages.

    Raises:
        ValueError: if ``query`` lies below ``grid[0]`` or above ``grid[-1]``.
    """
    if query < grid[0] or query > grid[-1]:
        raise ValueError(f"{name} query {query} outside [{grid[0]}, {grid[-1]}]")


def _locate(grid: tuple[float, ...], query: float) -> int:
    """Find the segment holding ``query``.

    Args:
        grid: strictly increasing axis coordinates.
        query: in-bounds point to place.

    Returns:
        Index ``i`` in ``[0, len(grid) - 2]`` such that
        ``grid[i] <= query <= grid[i + 1]``.
    """
    i = 0
    while i < len(grid) - 2 and grid[i + 1] < query:
        i += 1
    return i


def interp1d(x: list[float], y: list[float], kind: str = "linear") -> Callable[[float], float]:
    """Build a piecewise interpolator over paired samples.

    Args:
        x: sample abscissae, strictly increasing.
        y: sample ordinates aligned with ``x``.
        kind: ``"linear"`` for piecewise-linear segments or ``"nearest"``
            for nearest-neighbour steps; exact midpoints resolve to the
            lower knot.

    Returns:
        A pure callable mapping any query in ``[x[0], x[-1]]`` (endpoints
        included) to the interpolated value.

    Raises:
        ValueError: if ``kind`` is unknown, either sequence is empty, the
            sequences differ in length, ``x`` is not strictly increasing,
            or a query falls outside the sampled domain.
    """
    if kind not in ("linear", "nearest"):
        raise ValueError(f"unknown kind {kind!r}; expected 'linear' or 'nearest'")
    if len(x) == 0 or len(y) == 0:
        raise ValueError("x and y must be non-empty")
    _check_axis(x, "x")
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    xs = tuple(x)
    ys = tuple(y)

    def linear_at(query: float) -> float:
        _require_in_bounds(xs, query, "x")
        i = _locate(xs, query)
        weight = (query - xs[i]) / (xs[i + 1] - xs[i])
        return ys[i] + weight * (ys[i + 1] - ys[i])

    def nearest_at(query: float) -> float:
        _require_in_bounds(xs, query, "x")
        i = _locate(xs, query)
        return ys[i] if query - xs[i] <= xs[i + 1] - query else ys[i + 1]

    return linear_at if kind == "linear" else nearest_at


def interp2d(
    x: list[float],
    y: list[float],
    z: list[list[float]],
) -> Callable[[float, float], float]:
    """Build a bilinear interpolator over a regular rectangular grid.

    ``z[i][j]`` must hold the value at ``(x[i], y[j])``; the result blends
    the four corners of the surrounding cell weighted by fractional area.

    Args:
        x: grid abscissae, strictly increasing.
        y: grid ordinates, strictly increasing.
        z: rectangular value table with one row per point of ``x`` and one
            column per point of ``y``.

    Returns:
        A pure callable mapping ``(qx, qy)`` inside the grid box (corners
        included) to the bilinearly interpolated value.

    Raises:
        ValueError: if an axis is empty, holds a single point, or is not
            strictly increasing; if ``z`` lacks exactly one row per point
            of ``x``, or any row length differs from ``len(y)``; or if a
            query falls outside the grid.
    """
    _check_axis(x, "x")
    _check_axis(y, "y")
    if len(z) != len(x):
        raise ValueError("z must have exactly one row per point in x")
    if any(len(row) != len(y) for row in z):
        raise ValueError("every row of z must have the same length as y")
    xs = tuple(x)
    ys = tuple(y)
    table = tuple(tuple(row) for row in z)

    def bilinear_at(qx: float, qy: float) -> float:
        _require_in_bounds(xs, qx, "x")
        _require_in_bounds(ys, qy, "y")
        i = _locate(xs, qx)
        j = _locate(ys, qy)
        wx = (qx - xs[i]) / (xs[i + 1] - xs[i])
        wy = (qy - ys[j]) / (ys[j + 1] - ys[j])
        low = table[i][j] * (1.0 - wx) + table[i + 1][j] * wx
        high = table[i][j + 1] * (1.0 - wx) + table[i + 1][j + 1] * wx
        return low * (1.0 - wy) + high * wy

    return bilinear_at
