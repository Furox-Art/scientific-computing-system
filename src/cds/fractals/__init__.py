"""Fractal geometry — escape-time and IFS fractals.

Re-exports the pure-Python fractal generators from :mod:`cds.fractals.sets`.

Available fractals:
    * Mandelbrot set (escape-time, :func:`mandelbrot` / :func:`mandelbrot_image`)
    * Julia set (escape-time, :func:`julia`)
    * Barnsley fern (IFS, :func:`barnsley_fern`)
    * Sierpinski triangle (chaos game, :func:`sierpinski_triangle`)

References:
    Mandelbrot, B. B. (1980). Fractal aspects of the iteration.
    Barnsley, M. F. (1988). Fractals Everywhere.
    Sierpinski, W. (1915). Sur une courbe dont tout point est un point de
        ramification.
"""

from __future__ import annotations

from cds.fractals.sets import (
    barnsley_fern,
    julia,
    mandelbrot,
    mandelbrot_image,
    sierpinski_triangle,
)

__all__ = [
    "barnsley_fern",
    "julia",
    "mandelbrot",
    "mandelbrot_image",
    "sierpinski_triangle",
]
