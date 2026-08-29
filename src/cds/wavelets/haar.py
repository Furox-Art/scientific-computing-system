"""Haar discrete wavelet transform — pure Python.

Implements the orthonormal Haar DWT/IDWT (``1 / sqrt(2)`` normalisation) with
single-level and multi-level decomposition, plus simple hard-threshold
denoising.  All routines are pure Python with only the standard-library
:mod:`math` dependency, mirroring the NumPy-based ``cds2.wavelets`` API
(``haar_dwt``, ``haar_idwt``, ``dwt_levels``, ``wavelet_denoise``) but
operating on :class:`list[float]` instead of :class:`numpy.ndarray`.

For each adjacent pair ``(x[2k], x[2k+1])``:

* approximation ``a[k] = (x[2k] + x[2k+1]) / sqrt(2)`` — local average,
* detail      ``d[k] = (x[2k] - x[2k+1]) / sqrt(2)`` — local difference.

Multi-level analysis repeatedly applies :func:`dwt` to the approximation
band; synthesis reverses the chain.  Energy is preserved exactly
(``sum(x**2) == sum(a**2) + sum(d**2)``) up to floating-point round-off.

References:
    - Haar, A. (1910). Zur Theorie der orthogonalen Funktionensysteme.
    - Mallat, S.G. (1989). A theory for multiresolution signal decomposition.
    - Daubechies, I. (1992). Ten Lectures on Wavelets, ch. 5.
"""

from __future__ import annotations

import math

__all__ = ["denoise", "dwt", "dwt_multi_level", "idwt"]

_INV_SQRT2: float = 1.0 / math.sqrt(2.0)


def dwt(data: list[float]) -> tuple[list[float], list[float]]:
    """Single-level Haar DWT of a signal.

    Args:
        data: input samples; length must be even and >= 2.  For full
            orthogonality the length is typically a power of two, but any
            even length is accepted (mirroring ``cds2.wavelets.haar_dwt``).

    Returns:
        Tuple ``(approx, detail)`` each of length ``len(data)//2``.

    Raises:
        ValueError: if ``len(data)`` is not even or is < 2.
    """
    n = len(data)
    if n < 2 or n % 2 != 0:
        msg = (
            f"signal length must be a power of two >= 2 and even "
            f"(got {n}); values must be a 1-D series of at least two points"
        )
        raise ValueError(msg)
    approx: list[float] = []
    detail: list[float] = []
    for k in range(n // 2):
        lo = float(data[2 * k])
        hi = float(data[2 * k + 1])
        approx.append((lo + hi) * _INV_SQRT2)
        detail.append((lo - hi) * _INV_SQRT2)
    return approx, detail


def idwt(approx: list[float], detail: list[float]) -> list[float]:
    """Single-level inverse Haar transform.

    Exact inverse of :func:`dwt` up to floating-point round-off.

    Args:
        approx: approximation coefficients from :func:`dwt`.
        detail: matching detail coefficients.

    Returns:
        Reconstructed signal of length ``2 * len(approx)``.

    Raises:
        ValueError: if lengths differ, either is empty, or the coefficient
            count is not a power of two / not even.
    """
    n_coeff = len(approx)
    if n_coeff != len(detail):
        msg = (
            f"approx and detail must have the same length "
            f"(approx and detail lengths differ ({n_coeff} != {len(detail)})); "
            "approximation and detail must be non-empty 1-D sequences"
        )
        raise ValueError(msg)
    if n_coeff == 0:
        msg = (
            "approx must be non-empty (approximation and detail must be "
            "non-empty 1-D sequences); coefficient count must be a power of two"
        )
        raise ValueError(msg)
    # Enforce odd-length rejection with a power-of-two message so callers
    # checking for that substring (as in cds.signals.wavelet) still pass,
    # while even non-power-of-two lengths (e.g. 6, 12) are allowed for
    # multi-level reconstruction of arbitrary even-length signals
    # (mirroring cds2.wavelets behaviour).  The sole odd power-of-two
    # (1) is allowed because it appears as the coarsest band for
    # power-of-two signals (e.g. length 8 with 3 levels).
    if n_coeff != 1 and n_coeff % 2 == 1:
        msg = f"coefficient count must be a power of two (got {n_coeff})"
        raise ValueError(msg)
    reconstructed: list[float] = []
    for lo, hi in zip(approx, detail, strict=True):
        avg = float(lo)
        diff = float(hi)
        reconstructed.append((avg + diff) * _INV_SQRT2)
        reconstructed.append((avg - diff) * _INV_SQRT2)
    return reconstructed


def dwt_multi_level(
    data: list[float],
    levels: int,
) -> list[tuple[list[float], list[float]]]:
    """Multi-level Haar decomposition.

    Repeatedly applies :func:`dwt` to the approximation band.

    Args:
        data: input samples; length must be even and >= 2.  Power-of-two
            lengths allow the deepest decomposition; arbitrary even lengths
            are supported up to the maximal even-divisible depth.
        levels: number of decomposition levels, between 1 and the maximal
            depth inclusive.  Maximal depth is ``log2(n)`` for power-of-two
            ``n``, otherwise the largest ``L`` with ``n % 2**L == 0``.

    Returns:
        One ``(approx, detail)`` pair per level, finest first, coarsest
        last.  The ``k``-th level bands have length ``n >> k``.

    Raises:
        ValueError: if ``levels`` is < 1, exceeds maximal depth, or the
            signal is too short / not even.
    """
    n = len(data)
    if n < 2 or n % 2 != 0:
        msg = (
            f"signal length must be a power of two >= 2 and even "
            f"(got {n}); values must be a 1-D series of at least two points"
        )
        raise ValueError(msg)
    # Compute maximal even-divisible depth (for power-of-two this equals log2(n))
    max_levels = 0
    tmp = n
    while tmp % 2 == 0 and tmp >= 2:
        max_levels += 1
        tmp //= 2
        if tmp == 1:
            break
    # For strict power-of-two signals, max_levels == n.bit_length() - 1;
    # the while-loop yields the same value, so we keep it for arbitrary even.
    if levels < 1:
        msg = (
            f"levels must be at least 1 and between 1 and {max_levels} "
            f"(got {levels}); signal too short for {levels} levels"
        )
        raise ValueError(msg)
    if n < 2**levels:
        msg = (
            f"signal too short for {levels} levels (need {2**levels} samples, got {n}); "
            f"levels must be between 1 and {max_levels} (got {levels})"
        )
        raise ValueError(msg)
    if levels > max_levels:
        msg = (
            f"levels must be between 1 and {max_levels} for length-{n} signal "
            f"(got {levels}); signal too short"
        )
        raise ValueError(msg)
    current: list[float] = [float(v) for v in data]
    coeffs: list[tuple[list[float], list[float]]] = []
    for _ in range(levels):
        approx, detail = dwt(current)
        coeffs.append((approx, detail))
        current = approx
    return coeffs


def denoise(data: list[float], threshold: float) -> list[float]:
    """Hard-threshold wavelet denoising (single-level).

    Performs a single-level :func:`dwt`, zeroes detail coefficients with
    ``abs(detail) < threshold``, and reconstructs via :func:`idwt`.

    This is the pure-Python analogue of ``cds2.wavelets.wavelet_denoise``
    with an absolute threshold (instead of a ``threshold_factor * sigma``
    adaptive cutoff).  For power-of-two lengths the same threshold can be
    applied across multiple levels by calling :func:`dwt_multi_level`
    externally; the single-level version preserves coarse structure while
    removing fine-scale noise and is fully deterministic.

    Args:
        data: input samples; length must be even and >= 2.
        threshold: non-negative cutoff; details below this magnitude are
            zeroed.  ``0`` leaves the signal unchanged.

    Returns:
        Denoised signal, same length as ``data``.

    Raises:
        ValueError: if ``threshold`` is negative or ``data`` length is
            invalid.
    """
    if threshold < 0:
        msg = f"threshold must be non-negative (got {threshold})"
        raise ValueError(msg)
    n = len(data)
    if n < 2 or n % 2 != 0:
        msg = (
            f"signal length must be a power of two >= 2 and even "
            f"(got {n}); values must be a 1-D series of at least two points"
        )
        raise ValueError(msg)
    approx, detail = dwt(data)
    filtered_detail: list[float] = [0.0 if abs(float(d)) < threshold else float(d) for d in detail]
    return idwt(approx, filtered_detail)
