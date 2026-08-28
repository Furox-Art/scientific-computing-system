"""Short-time Fourier transform (STFT) with periodic Hann/Hamming windows.

Slices a real-valued sequence into ``n_fft``-long frames that advance by ``hop``
samples — so consecutive frames overlap by ``n_fft - hop`` — tapers each frame
with a periodic analysis window, and evaluates one radix-2 FFT per frame via
:func:`cds.signals.processing.fft_radix2` (reused verbatim — no FFT is
reimplemented here). The result is a time-frequency matrix restricted to the
non-redundant lower spectrum (DC through Nyquist, ``n_fft // 2 + 1`` bins), the
standard representation behind spectrograms of real signals.

Windows follow the periodic (DFT-even) convention ``w[k] = a - b *
cos(2 * pi * k / n)``, which tiles seamlessly across overlapping frames because
the wrap-around sample ``w[n]`` is excluded from the taper.

All routines are pure Python with no external dependencies.

References:
    - Allen, J.B. (1977). Short-time spectral analysis, synthesis, and
      modification by discrete Fourier transform. IEEE Trans. ASSP-25(3).
    - Harris, F.J. (1978). On the use of windows for harmonic analysis with
      the discrete Fourier transform. Proceedings of the IEEE, 66(1).
    - Oppenheim, A.V. & Schafer, R.W. Discrete-Time Signal Processing,
      ch. 10 (Fourier analysis of signals using the DFT).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from cds.signals.processing import fft_radix2

__all__ = ["STFTResult", "frame_signal", "stft", "window"]

# Symmetric-coefficient pairs ``(a, b)`` for ``w[k] = a - b * cos(2 pi k / n)``.
_WINDOW_COEFFICIENTS: dict[str, tuple[float, float]] = {
    "hann": (0.5, 0.5),
    "hamming": (0.54, 0.46),
}


@dataclass(frozen=True)
class STFTResult:
    """Time-frequency decomposition produced by :func:`stft`.

    Attributes:
        times: frame start indices in samples (one entry per frame).
        freqs: bin centre frequencies in cycles per sample (``k / n_fft``);
            length ``n_fft // 2 + 1``, covering DC through Nyquist.
        magnitude: ``|X[frame, bin]|`` matrix — rows are frames, columns are
            the first ``n_fft // 2 + 1`` FFT bins.
    """

    times: list[float]
    freqs: list[float]
    magnitude: list[list[float]]


def window(kind: str, n: int) -> list[float]:
    """Periodic Hann or Hamming taper of length ``n``.

    Args:
        kind: ``"hann"`` or ``"hamming"``.
        n: number of taps; must be >= 1.

    Returns:
        Samples of ``w[k] = a - b * cos(2 * pi * k / n)`` for ``k = 0 .. n-1``
        (periodic / DFT-even convention).

    Raises:
        ValueError: if ``kind`` is not a supported window name or ``n < 1``.
    """
    coefficients = _WINDOW_COEFFICIENTS.get(kind)
    if coefficients is None:
        raise ValueError(f"unknown window kind {kind!r}; expected 'hann' or 'hamming'")
    if n < 1:
        raise ValueError(f"window length must be >= 1 (got {n})")
    a, b = coefficients
    return [a - b * math.cos(2.0 * math.pi * k / n) for k in range(n)]


def frame_signal(signal: Sequence[float], n_fft: int, hop: int) -> list[list[complex]]:
    """Split a signal into overlapping ``n_fft``-long frames advancing by ``hop``.

    Each frame holds ``n_fft`` *consecutive* samples, and successive frames start
    ``hop`` samples apart, so they overlap by ``n_fft - hop``. Only the final
    frame is zero-padded, and only by however much the signal falls short.

    This previously took ``hop`` samples per frame and zero-padded the rest to
    ``n_fft``, which made frames non-overlapping and left the analysis window
    tapering zeros — at the default ``hop = n_fft // 4``, three quarters of the
    Hann window multiplied nothing. That is a block transform, not a short-time
    Fourier transform: the entire point of ``hop < n_fft`` is overlap, and the
    window's constant-overlap-add property is meaningless without it. The old
    behaviour was only correct in the single case ``hop == n_fft``, which is
    also the only case the spectral tests exercised.

    Args:
        signal: input samples.
        n_fft: frame length and FFT size, a power of two >= 2.
        hop: frame advance in samples, ``1 <= hop <= n_fft``.

    Returns:
        One ``n_fft``-long complex frame per window position.

    Raises:
        ValueError: if ``signal`` is empty, ``hop < 1``, ``n_fft`` is not a
            power of two >= 2, the signal is shorter than one hop, or ``hop``
            exceeds ``n_fft``.
    """
    if not signal:
        raise ValueError("signal must be non-empty")
    if hop < 1:
        raise ValueError(f"hop must be >= 1 (got {hop})")
    if n_fft < 2 or (n_fft & (n_fft - 1)) != 0:
        raise ValueError(f"n_fft must be a power of two >= 2 (got {n_fft})")
    if len(signal) < hop:
        raise ValueError(f"signal length {len(signal)} is shorter than one hop ({hop})")
    if hop > n_fft:
        raise ValueError(f"hop ({hop}) must not exceed n_fft ({n_fft})")

    total = len(signal)
    # Frame count in closed form: one frame at offset 0, plus however many hops
    # are needed for a frame to reach the end. Computing it rather than breaking
    # out of the loop keeps the loop free of an unreachable exit — because
    # ``hop <= n_fft``, the final start always satisfies ``start + n_fft >=
    # total``, so a ``break`` would make the for-else path dead code.
    span = total - n_fft
    n_frames = 1 + (max(0, -(-span // hop)) if span > 0 else 0)

    frames: list[list[complex]] = []
    for index in range(n_frames):
        start = index * hop
        frame: list[complex] = [complex(sample) for sample in signal[start : start + n_fft]]
        frame.extend([0j] * (n_fft - len(frame)))
        frames.append(frame)
    return frames


def stft(
    signal: Sequence[float],
    *,
    n_fft: int = 256,
    hop: int | None = None,
    window_kind: str = "hann",
) -> STFTResult:
    """Short-time Fourier transform: window each frame, then FFT it.

    Args:
        signal: input samples.
        n_fft: FFT size per frame, a power of two >= 2.
        hop: frame advance in samples; defaults to ``n_fft // 4`` when None.
        window_kind: ``"hann"`` or ``"hamming"`` analysis window.

    Returns:
        An :class:`STFTResult` whose ``magnitude`` has one row per frame and
        ``n_fft // 2 + 1`` columns (DC through Nyquist).

    Raises:
        ValueError: if arguments are invalid (see :func:`frame_signal` and
            :func:`window`).
    """
    step = n_fft // 4 if hop is None else hop
    frames = frame_signal(signal, n_fft, step)
    win = window(window_kind, n_fft)

    n_bins = n_fft // 2 + 1
    freqs = [bin_index / n_fft for bin_index in range(n_bins)]
    times: list[float] = []
    magnitude: list[list[float]] = []
    for frame_index, frame in enumerate(frames):
        windowed: list[float | complex] = [sample * weight for sample, weight in zip(frame, win)]
        spectrum = fft_radix2(windowed)
        magnitude.append([abs(value) for value in spectrum[:n_bins]])
        times.append(float(frame_index * step))
    return STFTResult(times=times, freqs=freqs, magnitude=magnitude)
