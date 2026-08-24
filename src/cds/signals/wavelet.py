"""Haar discrete wavelet transform (DWT) and its exact inverse.

Implements the single-level and multi-level Haar wavelet transform with the
orthonormal (``1 / sqrt(2)``) normalisation, so the transform preserves
signal energy exactly (Parseval identity holds without any scaling factor)
and :func:`idwt` inverts :func:`dwt` bit-for-bit up to floating-point
round-off.

For each adjacent pair ``(x[2k], x[2k + 1])`` the pair produces

* approximation ``a[k] = (x[2k] + x[2k + 1]) / sqrt(2)`` — a local average,
* detail      ``d[k] = (x[2k] - x[2k + 1]) / sqrt(2)`` — a local difference.

Multi-level analysis applies :func:`dwt` repeatedly to the approximation
band, splitting it into successively coarser averages plus one detail band
per level; synthesis reverses the chain level by level.

All routines are pure Python with no external dependencies.

References:
    - Haar, A. (1910). Zur Theorie der orthogonalen Funktionensysteme.
      Mathematische Annalen, 69, 331-371.
    - Mallat, S.G. (1989). A theory for multiresolution signal decomposition:
      the wavelet representation. IEEE Trans. PAMI, 11(7).
    - Daubechies, I. (1992). Ten Lectures on Wavelets, ch. 5 (the Haar
      multiresolution as the simplest orthonormal wavelet basis).
"""

from __future__ import annotations

import math

__all__ = ["dwt", "dwt_multi_level", "idwt"]

_INV_SQRT_2 = 1.0 / math.sqrt(2.0)


def _check_signal_length(n: int, name: str) -> None:
    """Validate that ``n`` is a power of two >= 2 for a wavelet input.

    Args:
        n: candidate length.
        name: parameter name used in error messages.

    Raises:
        ValueError: if ``n`` is not a power of two >= 2.
    """
    if n < 2 or (n & (n - 1)) != 0:
        raise ValueError(f"{name} length must be a power of two >= 2 (got {n})")


def dwt(signal: list[float]) -> tuple[list[float], list[float]]:
    """Single-level Haar discrete wavelet transform of a signal.

    Args:
        signal: input samples; length must be a power of two >= 2.

    Returns:
        Tuple ``(approx, detail)`` where both lists have length
        ``len(signal) // 2``: ``approx`` holds the coarse averages and
        ``detail`` the pairwise differences (each scaled by ``1/sqrt(2)``).

    Raises:
        ValueError: if the signal length is not a power of two >= 2.
    """
    _check_signal_length(len(signal), "signal")
    approx: list[float] = []
    detail: list[float] = []
    for k in range(len(signal) // 2):
        lo = float(signal[2 * k])
        hi = float(signal[2 * k + 1])
        approx.append((lo + hi) * _INV_SQRT_2)
        detail.append((lo - hi) * _INV_SQRT_2)
    return approx, detail


def idwt(approx: list[float], detail: list[float]) -> list[float]:
    """Single-level inverse Haar transform: exact inverse of :func:`dwt`.

    Args:
        approx: approximation coefficients from :func:`dwt`.
        detail: matching detail coefficients from :func:`dwt`.

    Returns:
        Reconstructed signal of length ``2 * len(approx)`` satisfying
        ``idwt(dwt(x)) == x`` up to floating-point round-off.

    Raises:
        ValueError: if ``approx`` and ``detail`` differ in length, are empty,
            or their common length is not a power of two.
    """
    n_coeff = len(approx)
    if n_coeff != len(detail):
        raise ValueError(f"approx and detail lengths differ ({n_coeff} != {len(detail)})")
    if n_coeff == 0:
        raise ValueError("approx must be non-empty")
    if (n_coeff & (n_coeff - 1)) != 0:
        raise ValueError(f"coefficient count must be a power of two (got {n_coeff})")

    reconstructed: list[float] = []
    for lo, hi in zip(approx, detail, strict=True):
        avg = float(lo)
        diff = float(hi)
        reconstructed.append((avg + diff) * _INV_SQRT_2)
        reconstructed.append((avg - diff) * _INV_SQRT_2)
    return reconstructed


def dwt_multi_level(
    signal: list[float],
    levels: int,
) -> list[tuple[list[float], list[float]]]:
    """Multi-level Haar decomposition by repeated approximation-band analysis.

    Level 1 transforms the full signal; every subsequent level transforms the
    previous level's approximation, yielding one ``(approx, detail)`` pair per
    level with geometrically shrinking bands.

    Args:
        signal: input samples; length must be a power of two >= 2.
        levels: number of decomposition levels, between 1 and ``log2(n)``
            inclusive where ``n == len(signal)``.

    Returns:
        One ``(approx, detail)`` tuple per level, coarsest last; the ``k``-th
        entry's bands have length ``n >> k``.

    Raises:
        ValueError: if the signal length is not a power of two >= 2, or
            ``levels`` is outside ``[1, log2(n)]``.
    """
    n = len(signal)
    _check_signal_length(n, "signal")
    max_levels = n.bit_length() - 1
    if levels < 1 or levels > max_levels:
        raise ValueError(
            f"levels must be between 1 and {max_levels} for length-{n} signal (got {levels})"
        )

    current = [float(sample) for sample in signal]
    coefficients: list[tuple[list[float], list[float]]] = []
    for _ in range(levels):
        approx, detail = dwt(current)
        coefficients.append((approx, detail))
        current = approx
    return coefficients
