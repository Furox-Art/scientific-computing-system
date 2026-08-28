"""Digital IIR/FIR filters — Butterworth design and order-statistic denoising.

Extends the Fourier-domain tools in :mod:`cds.signals.processing` with classic
filter *design* and time-domain *application*. The Butterworth family is built
from scratch via the analog prototype + bilinear-transform recipe (the same
derivation scipy uses), so the coefficients are reproducible without any third-
party dependency. Band-pass and band-stop responses are realised as cascaded /
paralleled low- and high-pass sections, which keeps the maximally-flat -3 dB
edges exact while remaining numerically well-conditioned for hand-rolled
difference-equation evaluation.

All routines are pure Python with no external dependencies.

References:
    - Butterworth, S. (1930). On the theory of filter amplifiers.
    - Oppenheim, A.V. & Schafer, R.W. Discrete-Time Signal Processing,
      ch. 7 (IIR filter design via bilinear transformation).
    - Parks, T.W. & Burrus, C.S. (1987). Digital Filter Design.
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "FilterCoefficients",
    "butter_lowpass",
    "butter_highpass",
    "butter_bandpass",
    "butter_bandstop",
    "apply_filter",
    "moving_median",
]


# ---------------------------------------------------------------------------
# Coefficient container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FilterCoefficients:
    """A set of difference-equation coefficients ``b(z)/a(z)``.

    Represents the transfer function ::

        H(z) = (b0 + b1 z^-1 + ... + bM z^-M) / (a0 + a1 z^-1 + ... + aN z^-N)

    with ``a0`` normalised to 1. The companion :func:`apply_filter` evaluates
    the corresponding direct-form II difference equation in the time domain.

    Attributes:
        b: feed-forward (numerator) coefficients, ascending in z^-1.
        a: feed-back (denominator) coefficients, ascending in z^-1; ``a[0]``
            is always 1.0.
    """

    b: list[float]
    a: list[float]

    def __post_init__(self) -> None:
        if not self.b:
            raise ValueError("numerator coefficients 'b' must be non-empty")
        if not self.a:
            raise ValueError("denominator coefficients 'a' must be non-empty")
        if abs(self.a[0]) < 1e-15:
            raise ValueError("leading denominator coefficient a[0] must be non-zero")
        # Normalise so a[0] == 1.0; callers may pass un-normalised sections.
        if self.a[0] != 1.0:
            scale = self.a[0]
            object.__setattr__(self, "b", [x / scale for x in self.b])
            object.__setattr__(self, "a", [x / scale for x in self.a])

    @property
    def order(self) -> int:
        """Filter order (length of the longer of the two coefficient lists, minus one)."""
        return max(len(self.b), len(self.a)) - 1


# ---------------------------------------------------------------------------
# Polynomial helpers (operate directly on roots in the z^-1 representation)
# ---------------------------------------------------------------------------


def _poly_from_roots_zinv(roots: Sequence[complex]) -> list[complex]:
    """Expand ``prod(1 - r z^-1)`` over ``roots`` -> coeffs ascending in ``z^-1``.

    This is the z^-1 (causal) form: a constant term 1 followed by the
    coefficients of successive negative powers. Implemented as a running
    convolution so it stays pure-Python and dependency-free.
    """
    coeffs: list[complex] = [1.0]
    for r in roots:
        # Multiply the current polynomial by (1 - r z^-1).
        new_coeffs = [coeffs[0]]
        for i in range(1, len(coeffs)):
            new_coeffs.append(coeffs[i] - r * coeffs[i - 1])
        new_coeffs.append(-r * coeffs[-1])
        coeffs = new_coeffs
    return coeffs


def _bilinear_z(pole: complex) -> complex:
    """Map an analog s-plane pole/zero to the z-plane via the bilinear transform.

    Uses ``s = 2 (z - 1) / (z + 1)`` (i.e. ``T = 1``), so the forward map is
    ``z = (2 + s) / (2 - s)``.
    """
    return (2 + pole) / (2 - pole)


# ---------------------------------------------------------------------------
# Butterworth section design (low-pass / high-pass)
# ---------------------------------------------------------------------------


def _butter_section(order: int, cutoff: float, btype: str) -> FilterCoefficients:
    """Design a single low- or high-pass Butterworth section.

    Args:
        order: filter order ``N`` (>= 1).
        cutoff: normalised cutoff in ``(0, 1)``, where 1 is the Nyquist
            frequency. Interpreted as the -3 dB edge.
        btype: ``"low"`` or ``"high"``.

    Returns:
        Difference-equation coefficients for the section.
    """
    if order < 1:
        raise ValueError(f"filter order must be >= 1 (got {order})")
    if not 0.0 < cutoff < 1.0:
        raise ValueError(f"cutoff must be in the open interval (0, 1) (got {cutoff})")
    if btype not in ("low", "high"):
        raise ValueError(f"btype must be 'low' or 'high' (got {btype!r})")

    # Pre-warp the digital cutoff to the analog frequency (bilinear with T=1).
    omega_c = math.pi * cutoff
    analog_cutoff = 2.0 * math.tan(omega_c / 2.0)

    # Analog prototype poles on the unit circle (cutoff = 1 rad/s), LHP only,
    # for a maximally-flat (Butterworth) magnitude response.
    prototype = [cmath.exp(1j * math.pi * (2 * k + order + 1) / (2 * order)) for k in range(order)]

    if btype == "low":
        # Low-pass: scale prototype poles to the desired analog cutoff.
        analog_poles = [analog_cutoff * p for p in prototype]
        z_zeros = [-1.0] * order  # transmission zeros at Nyquist (z = -1)
    else:
        # High-pass: s -> analog_cutoff / s maps poles to analog_cutoff/p and
        # introduces order zeros at DC.
        analog_poles = [analog_cutoff / p for p in prototype]
        z_zeros = [1.0] * order  # transmission zeros at DC (z = 1)

    z_poles = [_bilinear_z(p) for p in analog_poles]

    a_coeffs = _poly_from_roots_zinv(z_poles)
    b_coeffs = _poly_from_roots_zinv(z_zeros)

    # Scale the numerator so the DC gain (low-pass) or Nyquist gain (high-pass)
    # is exactly unity. Evaluating b(z)/a(z) at z = +/-1 fixes the gain.
    z0 = 1.0 if btype == "low" else -1.0
    num_at = sum(c * z0 ** (-i) for i, c in enumerate(b_coeffs))
    den_at = sum(c * z0 ** (-i) for i, c in enumerate(a_coeffs))
    gain = den_at / num_at
    b_scaled = [gain * c for c in b_coeffs]

    # Imaginary parts are zero up to floating-point round-off: the conjugate-
    # symmetric pole/zero sets produce a purely real polynomial. Take the real
    # part so the coefficients carry an honest `float` type.
    return FilterCoefficients(
        b=[c.real for c in b_scaled],
        a=[c.real for c in a_coeffs],
    )


def butter_lowpass(order: int, cutoff: float) -> FilterCoefficients:
    """Design a Butterworth low-pass filter.

    Args:
        order: filter order ``N`` (>= 1). Higher orders give a steeper rolloff
            at the cost of longer transient ringing.
        cutoff: normalised cutoff frequency in ``(0, 1)``, where 1 is the
            Nyquist frequency (half the sample rate). This is the -3 dB edge.

    Returns:
        Difference-equation coefficients with unity DC gain.

    Example:
        >>> coef = butter_lowpass(order=4, cutoff=0.25)
        >>> smoothed = apply_filter(noisy_signal, coef)
    """
    return _butter_section(order, cutoff, "low")


def butter_highpass(order: int, cutoff: float) -> FilterCoefficients:
    """Design a Butterworth high-pass filter.

    Args:
        order: filter order ``N`` (>= 1).
        cutoff: normalised cutoff frequency in ``(0, 1)`` (-3 dB edge).

    Returns:
        Difference-equation coefficients with unity Nyquist gain.
    """
    return _butter_section(order, cutoff, "high")


# ---------------------------------------------------------------------------
# Band-pass / band-stop (cascade / parallel of low- and high-pass sections)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BandFilterCoefficients:
    """Coefficients for a band-pass or band-stop Butterworth filter.

    Band responses are realised as a *cascade* (band-pass) or *parallel sum*
    (band-stop) of two single-type sections, which preserves the exact -3 dB
    edges at ``low`` and ``high`` while staying well-conditioned for the
    direct-form difference equation in :func:`apply_filter`.

    Attributes:
        low: low-frequency -3 dB edge (normalised, in ``(0, 1)``).
        high: high-frequency -3 dB edge (normalised, in ``(0, 1)``).
        kind: ``"bandpass"`` or ``"bandstop"``.
        section1: first constituent :class:`FilterCoefficients`.
        section2: second constituent :class:`FilterCoefficients`.
    """

    low: float
    high: float
    kind: str
    section1: FilterCoefficients
    section2: FilterCoefficients

    def __post_init__(self) -> None:
        if self.kind not in ("bandpass", "bandstop"):
            raise ValueError(f"kind must be 'bandpass' or 'bandstop' (got {self.kind!r})")
        if not 0.0 < self.low < self.high < 1.0:
            raise ValueError(f"require 0 < low < high < 1 (got low={self.low}, high={self.high})")


def butter_bandpass(order: int, low: float, high: float) -> BandFilterCoefficients:
    """Design a Butterworth band-pass filter as a high-pass ∘ low-pass cascade.

    Passing ``order`` for each section yields an overall ``2*order`` response.
    The -3 dB edges fall exactly at ``low`` and ``high`` (normalised to Nyquist).

    Args:
        order: per-section order ``N`` (>= 1).
        low: low -3 dB edge, normalised in ``(0, 1)``.
        high: high -3 dB edge, normalised in ``(0, 1)``.

    Returns:
        A two-section cascade; apply with :func:`apply_filter`.

    Example:
        >>> coef = butter_bandpass(order=4, low=0.2, high=0.5)
        >>> isolated = apply_filter(signal, coef)
    """
    if not 0.0 < low < high < 1.0:
        raise ValueError(f"require 0 < low < high < 1 (got low={low}, high={high})")
    return BandFilterCoefficients(
        low=low,
        high=high,
        kind="bandpass",
        section1=butter_highpass(order, low),
        section2=butter_lowpass(order, high),
    )


def butter_bandstop(order: int, low: float, high: float) -> BandFilterCoefficients:
    """Band-stop response built as a parallel sum of a low-pass and a high-pass.

    Unity outside ``[low, high]``, attenuated inside it. Both branches are
    genuine Butterworth sections, but their **parallel sum is not** a
    Butterworth band-stop, and it is worth knowing what that costs before
    relying on the rejection figure.

    Measured for ``order=4, low=0.2, high=0.5`` against a true band-stop of the
    same order (``scipy.signal.butter(4, [0.2, 0.5], "bandstop")``):

    ======  ==========  =============
    freq    this        true 4th-order
    ======  ==========  =============
    0.05    1.000       1.000
    0.20    0.700       0.707
    0.35    0.061       0.00022
    0.50    0.700       0.707
    0.90    1.000       1.000
    ======  ==========  =============

    So the pass bands and the edges are right to within about 0.09 dB, but the
    stop band bottoms out near -24 dB where a true 4th-order band-stop reaches
    -73 dB. The reason is structural rather than a coding slip: away from its
    own cutoff each branch still leaks its skirt, and because the two cutoffs
    differ the branches are not power-complementary, so the leakages add instead
    of cancelling. No choice of coefficients fixes that; a real band-stop needs
    the analogue band-stop transformation ``s -> s*BW / (s^2 + w0^2)``, which
    turns an order-``N`` prototype into a single ``2N``-order IIR rather than
    two ``N``-order sections.

    Adequate for separating well-spaced components. Not adequate as a notch when
    you need the stated order's rejection.

    Args:
        order: per-section order ``N`` (>= 1).
        low: low -3 dB edge, normalised in ``(0, 1)``.
        high: high -3 dB edge, normalised in ``(0, 1)``.

    Returns:
        A two-section parallel sum; apply with :func:`apply_filter`.
    """
    if not 0.0 < low < high < 1.0:
        raise ValueError(f"require 0 < low < high < 1 (got low={low}, high={high})")
    return BandFilterCoefficients(
        low=low,
        high=high,
        kind="bandstop",
        section1=butter_lowpass(order, low),
        section2=butter_highpass(order, high),
    )


# ---------------------------------------------------------------------------
# Time-domain application (direct-form II transposed difference equation)
# ---------------------------------------------------------------------------


def _apply_iir(signal: list[float], coef: FilterCoefficients) -> list[float]:
    """Apply a single IIR section via the direct-form II transposed structure.

    This is the numerically preferable canonical form: it uses a single delay
    line of length ``max(len(b), len(a)) - 1`` and is stable for the same
    coefficient sets as direct-form I.
    """
    n_b = len(coef.b)
    n_a = len(coef.a)
    n_state = max(n_b, n_a) - 1
    states = [0.0] * n_state

    b = coef.b
    a = coef.a  # a[0] == 1.0 by construction

    out: list[float] = []
    for x in signal:
        # Direct-form II transposed: y[n] = b0*x[n] + s0
        y = b[0] * x + (states[0] if n_state > 0 else 0.0)
        for i in range(n_state - 1):
            states[i] = b[i + 1] * x - (a[i + 1] if (i + 1) < n_a else 0.0) * y + states[i + 1]
        if n_state > 0:
            states[n_state - 1] = (b[n_state] * x if n_state < n_b else 0.0) - (
                a[n_state] * y if n_state < n_a else 0.0
            )
        out.append(y)
    return out


def apply_filter(
    signal: list[float],
    coefficients: FilterCoefficients | BandFilterCoefficients,
) -> list[float]:
    """Apply a designed filter to a signal in the time domain.

    For a single :class:`FilterCoefficients` the direct-form II transposed
    difference equation is evaluated once. For a :class:`BandFilterCoefficients`
    a band-pass is applied as a cascade (high-pass then low-pass) and a
    band-stop as the parallel sum of the two section outputs.

    Args:
        signal: real-valued input samples.
        coefficients: a single section or a band section pair.

    Returns:
        The filtered samples, same length as ``signal``.
    """
    if not signal:
        return []

    if isinstance(coefficients, FilterCoefficients):
        return _apply_iir(signal, coefficients)

    # Band section pair.
    if coefficients.kind == "bandpass":
        # Cascade: high-pass first, then low-pass.
        stage1 = _apply_iir(signal, coefficients.section1)
        return _apply_iir(stage1, coefficients.section2)
    # Band-stop: parallel sum of the two sections.
    stage1 = _apply_iir(signal, coefficients.section1)
    stage2 = _apply_iir(signal, coefficients.section2)
    return [a + b for a, b in zip(stage1, stage2)]


# ---------------------------------------------------------------------------
# Nonlinear (order-statistic) denoising
# ---------------------------------------------------------------------------


def moving_median(signal: list[float], window: int) -> list[float]:
    """Smooth a signal with a centred running median filter.

    The median is a robust order statistic: unlike a moving average it rejects
    impulsive (salt-and-pepper) outliers without blurring genuine edges, which
    makes it the classical choice for spike-removal denoising.

    Boundary samples (where the window would extend past the ends of the
    signal) are produced from the truncated, available window — so the output
    always has the same length as the input.

    Args:
        signal: real-valued input samples.
        window: odd positive window length (e.g. 3, 5, 7). An even value is
            rejected because it has no single middle element.

    Returns:
        The median-smoothed samples, same length as ``signal``.

    Raises:
        ValueError: if ``window`` is not a positive odd integer, or if it
            exceeds ``2*len(signal) - 1`` (so at least one output is defined by
            a non-empty window).

    Example:
        >>> moving_median([1.0, 1.0, 9.0, 1.0, 1.0], window=3)
        [1.0, 1.0, 1.0, 1.0, 1.0]  # the spike at index 2 is removed
    """
    if window < 1:
        raise ValueError(f"window must be a positive integer (got {window})")
    if window % 2 == 0:
        raise ValueError(f"window must be odd (got {window})")
    n = len(signal)
    if n == 0:
        return []
    if window > 2 * n - 1:
        raise ValueError(
            f"window {window} too large for a signal of length {n} (must be <= {2 * n - 1})"
        )

    half = window // 2
    out: list[float] = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        block = sorted(signal[lo:hi])
        out.append(block[len(block) // 2])
    return out
