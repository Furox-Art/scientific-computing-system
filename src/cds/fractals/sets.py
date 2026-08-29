"""Fractal sets — Mandelbrot, Julia, Sierpinski and Barnsley fern.

Pure-Python, zero-dependency (``math`` + ``random`` only) implementations of
classic fractal escape-time and iterated-function-system (IFS) constructions.
All inputs are plain ``float`` / ``int`` and outputs are ``list``-based so the
module stays dependency-free and ``mypy --strict`` clean.

Escape-time sets
    * :func:`mandelbrot` — iteration ``z <- z**2 + c`` from ``z0 = 0``.
    * :func:`julia` — same iteration from an arbitrary ``z0``.
    * :func:`mandelbrot_image` — dense sampling of the Mandelbrot set on a
      rectangular region of the complex plane.

IFS attractors
    * :func:`barnsley_fern` — Barnsley fern (1988) affine IFS, four maps with
      probabilities 0.01 / 0.85 / 0.07 / 0.07.
    * :func:`sierpinski_triangle` — Sierpinski triangle chaos game (1915),
      midpoint IFS over three vertices.

References:
    Mandelbrot, B. B. (1980). Fractal aspects of the iteration of
        ``z -> lambda z (1 - z)`` for complex ``lambda`` and ``z``.
    Barnsley, M. F. (1988). Fractals Everywhere. Academic Press.
    Sierpinski, W. (1915). Sur une courbe dont tout point est un point de
        ramification.
"""

from __future__ import annotations

import math
import random

__all__ = [
    "barnsley_fern",
    "julia",
    "mandelbrot",
    "mandelbrot_image",
    "sierpinski_triangle",
]


def _validate_max_iter(max_iter: int) -> None:
    """Validate that ``max_iter`` is a positive integer.

    Args:
        max_iter: candidate iteration limit.

    Raises:
        TypeError: if ``max_iter`` is not an ``int`` (``bool`` is rejected).
        ValueError: if ``max_iter`` is not strictly positive.
    """
    if not isinstance(max_iter, int) or isinstance(max_iter, bool):
        raise TypeError("max_iter must be an integer")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")


def _validate_finite(value: float, name: str) -> None:
    """Validate that ``value`` is a finite real number.

    Args:
        value: candidate coordinate.
        name: parameter name used in the error message.

    Raises:
        ValueError: if ``value`` is not finite (``inf`` or ``nan``).
    """
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _validate_non_negative_int(value: int, name: str) -> None:
    """Validate that ``value`` is a non-negative integer.

    Args:
        value: candidate count.
        name: parameter name used in the error message.

    Raises:
        TypeError: if ``value`` is not an ``int`` (``bool`` rejected).
        ValueError: if ``value`` is negative.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def mandelbrot(c_real: float, c_imag: float, max_iter: int = 256) -> int:
    """Mandelbrot escape iteration count for ``c = c_real + i c_imag``.

    Iterates ``z_{n+1} = z_n**2 + c`` from ``z_0 = 0`` and returns the
    1-based iteration index where ``|z|**2 > 4``. If the orbit never exceeds
    radius 2 within ``max_iter`` steps, ``max_iter`` is returned (point is
    considered inside the set).

    Args:
        c_real: real part of ``c``.
        c_imag: imaginary part of ``c``.
        max_iter: positive iteration limit.

    Returns:
        Escape iteration in ``[1, max_iter]`` if the orbit escapes, otherwise
        ``max_iter``. ``max_iter`` is also returned for points that never
        escape.

    Raises:
        TypeError: if ``max_iter`` is not an ``int``.
        ValueError: if ``max_iter`` is not positive or ``c_real`` / ``c_imag``
            is not finite.

    References:
        Mandelbrot, B. B. (1980). Fractal aspects of the iteration of
        ``z -> lambda z (1 - z)``.
    """
    _validate_finite(c_real, "c_real")
    _validate_finite(c_imag, "c_imag")
    _validate_max_iter(max_iter)
    z_re = 0.0
    z_im = 0.0
    for i in range(max_iter):
        z_re2 = z_re * z_re - z_im * z_im + c_real
        z_im2 = 2.0 * z_re * z_im + c_imag
        z_re, z_im = z_re2, z_im2
        if z_re * z_re + z_im * z_im > 4.0:
            return i + 1
    return max_iter


def julia(
    z_real: float,
    z_imag: float,
    c_real: float,
    c_imag: float,
    max_iter: int = 256,
) -> int:
    """Julia escape iteration count for ``z0 = z_real + i z_imag``.

    Iterates ``z_{n+1} = z_n**2 + c`` from ``z_0`` and returns the 0-based
    iteration index where ``|z|**2 > 4`` is first observed *before* the
    update. Points already outside radius 2 return ``0``; points that never
    escape within ``max_iter`` return ``max_iter``.

    Args:
        z_real: real part of the initial ``z``.
        z_imag: imaginary part of the initial ``z``.
        c_real: real part of the constant ``c``.
        c_imag: imaginary part of the constant ``c``.
        max_iter: positive iteration limit.

    Returns:
        Escape iteration in ``[0, max_iter]``. ``0`` means ``z0`` already
        escapes; ``max_iter`` means the orbit stayed bounded.

    Raises:
        TypeError: if ``max_iter`` is not an ``int``.
        ValueError: if ``max_iter`` is not positive or any coordinate is not
            finite.

    References:
        Julia, G. (1918). Memoire sur l'iteration des fonctions rationnelles.
    """
    _validate_finite(z_real, "z_real")
    _validate_finite(z_imag, "z_imag")
    _validate_finite(c_real, "c_real")
    _validate_finite(c_imag, "c_imag")
    _validate_max_iter(max_iter)
    z_re = z_real
    z_im = z_imag
    for i in range(max_iter):
        if z_re * z_re + z_im * z_im > 4.0:
            return i
        z_re2 = z_re * z_re - z_im * z_im + c_real
        z_im2 = 2.0 * z_re * z_im + c_imag
        z_re, z_im = z_re2, z_im2
    return max_iter


def mandelbrot_image(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    width: int,
    height: int,
    max_iter: int = 256,
) -> list[list[int]]:
    """Dense Mandelbrot escape image over a rectangular region.

    Samples ``width * height`` points on a regular grid spanning
    ``[xmin, xmax] x [ymin, ymax]`` inclusive. Row ``0`` corresponds to
    ``ymin`` and row ``height-1`` to ``ymax`` (mathematical y increasing
    upward); columns run from ``xmin`` to ``xmax``. Each sample is evaluated
    with :func:`mandelbrot`.

    Args:
        xmin: left bound of the complex plane (must be finite).
        xmax: right bound (must be finite and greater than ``xmin``).
        ymin: bottom bound (must be finite).
        ymax: top bound (must be finite and greater than ``ymin``).
        width: number of columns, must be a positive ``int``.
        height: number of rows, must be a positive ``int``.
        max_iter: positive iteration limit forwarded to :func:`mandelbrot`.

    Returns:
        A ``height x width`` list of escape counts, ``image[row][col]``
        where ``row`` indexes ``y`` and ``col`` indexes ``x``. Values lie in
        ``[1, max_iter]`` (Mandelbrot never returns ``0``).

    Raises:
        TypeError: if ``width``, ``height`` or ``max_iter`` is not an ``int``.
        ValueError: if any bound is not finite, ``xmin >= xmax``,
            ``ymin >= ymax``, ``width <= 0``, ``height <= 0`` or
            ``max_iter <= 0``.

    References:
        Mandelbrot, B. B. (1980). Fractal aspects of the iteration.
    """
    _validate_finite(xmin, "xmin")
    _validate_finite(xmax, "xmax")
    _validate_finite(ymin, "ymin")
    _validate_finite(ymax, "ymax")
    if not isinstance(width, int) or isinstance(width, bool):
        raise TypeError("width must be an integer")
    if not isinstance(height, int) or isinstance(height, bool):
        raise TypeError("height must be an integer")
    if width <= 0:
        raise ValueError("width must be positive")
    if height <= 0:
        raise ValueError("height must be positive")
    if xmin >= xmax:
        raise ValueError("xmin must be less than xmax")
    if ymin >= ymax:
        raise ValueError("ymin must be less than ymax")
    _validate_max_iter(max_iter)
    image: list[list[int]] = []
    for iy in range(height):
        if height == 1:
            y = ymin
        else:
            y = ymin + (ymax - ymin) * iy / (height - 1)
        row: list[int] = []
        for ix in range(width):
            if width == 1:
                x = xmin
            else:
                x = xmin + (xmax - xmin) * ix / (width - 1)
            row.append(mandelbrot(x, y, max_iter=max_iter))
        image.append(row)
    return image


def barnsley_fern(n_points: int, seed: int | None = None) -> list[tuple[float, float]]:
    """Barnsley fern IFS attractor.

    Iterated function system with four affine maps (Barnsley 1988):

    * ``f1`` (p=0.01): ``(0, 0.16 y)``
    * ``f2`` (p=0.85): ``(0.85 x + 0.04 y, -0.04 x + 0.85 y + 1.6)``
    * ``f3`` (p=0.07): ``(0.2 x - 0.26 y, 0.23 x + 0.22 y + 1.6)``
    * ``f4`` (p=0.07): ``(-0.15 x + 0.28 y, 0.26 x + 0.24 y + 0.44)``

    Starts from ``(0, 0)`` and applies a random map each step. The random
    stream is a private :class:`random.Random` instance so the global RNG is
    untouched and results are reproducible for a given ``seed``.

    Args:
        n_points: number of points to generate, must be non-negative.
        seed: optional integer seed for determinism. ``None`` uses OS entropy.

    Returns:
        List of ``n_points`` ``(x, y)`` points. ``n_points == 0`` yields
        ``[]``. Each ``y`` lies in ``[0, 10)`` and ``x`` in ``[-3, 3)`` for
        typical runs.

    Raises:
        TypeError: if ``n_points`` or ``seed`` is not an ``int``.
        ValueError: if ``n_points`` is negative.

    References:
        Barnsley, M. F. (1988). Fractals Everywhere. Academic Press.
    """
    _validate_non_negative_int(n_points, "n_points")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise TypeError("seed must be an integer or None")
    rng = random.Random(seed)
    points: list[tuple[float, float]] = []
    x = 0.0
    y = 0.0
    for _ in range(n_points):
        r = rng.random()
        if r < 0.01:
            x_new = 0.0
            y_new = 0.16 * y
        elif r < 0.86:
            x_new = 0.85 * x + 0.04 * y
            y_new = -0.04 * x + 0.85 * y + 1.6
        elif r < 0.93:
            x_new = 0.2 * x - 0.26 * y
            y_new = 0.23 * x + 0.22 * y + 1.6
        else:
            x_new = -0.15 * x + 0.28 * y
            y_new = 0.26 * x + 0.24 * y + 0.44
        x, y = x_new, y_new
        points.append((x, y))
    return points


def sierpinski_triangle(n_points: int, seed: int | None = None) -> list[tuple[float, float]]:
    """Sierpinski triangle chaos game.

    Chaos-game IFS: start at ``(0, 0)`` and repeatedly move halfway toward a
    randomly chosen vertex of the equilateral triangle ``(0,0)``, ``(1,0)``,
    ``(0.5, sqrt(3)/2)`` (Sierpinski 1915). Uses a private
    :class:`random.Random` instance for reproducibility.

    Args:
        n_points: number of points to generate, must be non-negative.
        seed: optional integer seed for determinism. ``None`` uses OS entropy.

    Returns:
        List of ``n_points`` ``(x, y)`` points inside the triangle. ``[]`` if
        ``n_points == 0``.

    Raises:
        TypeError: if ``n_points`` or ``seed`` is not an ``int``.
        ValueError: if ``n_points`` is negative.

    References:
        Sierpinski, W. (1915). Sur une courbe dont tout point est un point de
        ramification.
    """
    _validate_non_negative_int(n_points, "n_points")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise TypeError("seed must be an integer or None")
    rng = random.Random(seed)
    vertices: list[tuple[float, float]] = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.5, math.sqrt(3.0) / 2.0),
    ]
    x = 0.0
    y = 0.0
    points: list[tuple[float, float]] = []
    for _ in range(n_points):
        r = rng.random()
        if r < 1.0 / 3.0:
            vx, vy = vertices[0]
        elif r < 2.0 / 3.0:
            vx, vy = vertices[1]
        else:
            vx, vy = vertices[2]
        x = (x + vx) * 0.5
        y = (y + vy) * 0.5
        points.append((x, y))
    return points
