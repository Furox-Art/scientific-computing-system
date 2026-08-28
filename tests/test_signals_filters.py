"""Tests for the digital filter design and application tools (cds.signals.filters).

The Butterworth design is validated against independently-derived reference
coefficients (the analytic bilinear-transform result) rather than a runtime
scipy dependency, keeping the suite hermetic. Frequency-response and
time-domain behaviour is checked via the defining properties of each filter
class (unity passband gain, -3 dB edges, impulse/DC invariants).
"""

from __future__ import annotations

import math

import pytest

from cds.signals.filters import (
    BandFilterCoefficients,
    FilterCoefficients,
    _bilinear_z,
    _poly_from_roots_zinv,
    apply_filter,
    butter_bandpass,
    butter_bandstop,
    butter_highpass,
    butter_lowpass,
    moving_median,
)

# ---------------------------------------------------------------------------
# FilterCoefficients container
# ---------------------------------------------------------------------------


class TestFilterCoefficientsContainer:
    def test_basic_construction(self) -> None:
        coef = FilterCoefficients(b=[1.0, 2.0, 3.0], a=[1.0, 0.5, 0.25])
        assert coef.b == [1.0, 2.0, 3.0]
        assert coef.a == [1.0, 0.5, 0.25]
        assert coef.order == 2

    def test_order_takes_max_length(self) -> None:
        # numerator longer than denominator
        coef = FilterCoefficients(b=[1.0, 2.0, 3.0, 4.0], a=[1.0, 0.5])
        assert coef.order == 3
        # denominator longer than numerator
        coef2 = FilterCoefficients(b=[1.0, 0.5], a=[1.0, 0.2, 0.3, 0.4])
        assert coef2.order == 3

    def test_normalization_to_unit_leading_coefficient(self) -> None:
        # a[0] != 1 -> both lists scaled by 1/a[0]
        coef = FilterCoefficients(b=[2.0, 4.0], a=[2.0, 2.0])
        assert coef.a[0] == 1.0
        assert coef.a == [1.0, 1.0]
        assert coef.b == [1.0, 2.0]

    def test_empty_numerator_rejected(self) -> None:
        with pytest.raises(ValueError, match="numerator"):
            FilterCoefficients(b=[], a=[1.0])

    def test_empty_denominator_rejected(self) -> None:
        with pytest.raises(ValueError, match="denominator"):
            FilterCoefficients(b=[1.0], a=[])

    def test_zero_leading_denominator_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-zero"):
            FilterCoefficients(b=[1.0], a=[0.0, 1.0])

    def test_is_frozen(self) -> None:
        coef = FilterCoefficients(b=[1.0], a=[1.0])
        # frozen dataclass: attribute assignment is forbidden
        with pytest.raises(Exception):
            coef.b = [2.0]  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Polynomial + bilinear helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_poly_from_roots_zinv_no_roots(self) -> None:
        assert _poly_from_roots_zinv([]) == [1.0]

    def test_poly_from_roots_zinv_single_root(self) -> None:
        # prod(1 - r z^-1) for one root r -> [1, -r]
        assert _poly_from_roots_zinv([2.0]) == [1.0, -2.0]

    def test_poly_from_roots_zinv_two_roots(self) -> None:
        # (1 - z^-1)(1 - z^-1) = 1 - 2 z^-1 + z^-2
        coeffs = _poly_from_roots_zinv([1.0, 1.0])
        assert [round(c.real, 9) for c in coeffs] == [1.0, -2.0, 1.0]

    def test_poly_from_roots_zinv_real_coefficients_from_conjugate_pair(self) -> None:
        # conjugate pair (0.5 +/- 0.5j) must give purely real coefficients
        coeffs = _poly_from_roots_zinv([0.5 + 0.5j, 0.5 - 0.5j])
        for c in coeffs:
            assert abs(c.imag) < 1e-12
        # (1 - (0.5+0.5j)z^-1)(1 - (0.5-0.5j)z^-1) = 1 - z^-1 + 0.5 z^-2
        assert round(coeffs[1].real, 9) == -1.0
        assert round(coeffs[2].real, 9) == 0.5

    def test_bilinear_z_maps_dc_and_nyquist(self) -> None:
        # s = 0 (DC) -> z = (2+0)/(2-0) = 1
        assert abs(_bilinear_z(0j) - 1.0) < 1e-12
        # s = inf (Nyquist) -> z -> -1 ; check a large real maps near -1
        assert abs(_bilinear_z(1e9 + 0j) - (-1.0)) < 1e-6

    def test_bilinear_z_unit_circle_point(self) -> None:
        # s = 1j -> |z| should be 1 (analog jw axis maps to the unit circle)
        z = _bilinear_z(1j)
        assert abs(abs(z) - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# Butterworth low-pass / high-pass design
# ---------------------------------------------------------------------------

# Analytic reference coefficients: butter_lowpass(4, 0.25) matches the canonical
# bilinear-transform Butterworth result (verified independently of scipy).
_LP4_B = [0.010209, 0.040838, 0.061257, 0.040838, 0.010209]
_LP4_A = [1.0, -1.968428, 1.735861, -0.724471, 0.12039]
_HP4_B = [0.167179, -0.668717, 1.003076, -0.668717, 0.167179]
_HP4_A = [1.0, -0.782095, 0.679979, -0.182676, 0.030119]


class TestButterLowHigh:
    def test_lowpass_order4_coefficients_match_reference(self) -> None:
        coef = butter_lowpass(order=4, cutoff=0.25)
        assert len(coef.b) == 5
        assert len(coef.a) == 5
        assert coef.a[0] == 1.0
        for got, ref in zip(coef.b, _LP4_B):
            assert abs(got - ref) < 1e-5
        for got, ref in zip(coef.a, _LP4_A):
            assert abs(got - ref) < 1e-5

    def test_highpass_order4_coefficients_match_reference(self) -> None:
        coef = butter_highpass(order=4, cutoff=0.4)
        for got, ref in zip(coef.b, _HP4_B):
            assert abs(got - ref) < 1e-5
        for got, ref in zip(coef.a, _HP4_A):
            assert abs(got - ref) < 1e-5

    def test_lowpass_coefficient_length(self) -> None:
        # order N -> N+1 taps in each list
        for n in (1, 2, 3, 6):
            coef = butter_lowpass(n, 0.3)
            assert len(coef.b) == n + 1
            assert len(coef.a) == n + 1
            assert coef.order == n

    def test_lowpass_unity_dc_gain(self) -> None:
        # a constant (DC) signal must pass through a low-pass unchanged
        coef = butter_lowpass(order=4, cutoff=0.25)
        dc = [3.5] * 256
        out = apply_filter(dc, coef)
        # steady-state tail should equal the input (transient settles by ~200)
        for v in out[200:]:
            assert abs(v - 3.5) < 1e-6

    def test_highpass_zero_dc_gain(self) -> None:
        # a high-pass removes the DC component -> steady state tends to 0
        coef = butter_highpass(order=4, cutoff=0.25)
        dc = [7.0] * 128
        out = apply_filter(dc, coef)
        for v in out[64:]:
            assert abs(v) < 1e-6

    def test_lowpass_attenuates_high_frequency(self) -> None:
        # a near-Nyquist sinusoid is strongly attenuated by a low-pass at 0.2
        coef = butter_lowpass(order=4, cutoff=0.2)
        n = 1024
        high = [math.sin(math.pi * 0.9 * k) for k in range(n)]
        out = apply_filter(high, coef)
        amp = (max(out[n // 2 :]) - min(out[n // 2 :])) / 2
        assert amp < 0.05  # input amplitude 1.0 -> heavily suppressed

    def test_highpass_attenuates_low_frequency(self) -> None:
        coef = butter_highpass(order=4, cutoff=0.4)
        n = 1024
        low = [math.sin(math.pi * 0.05 * k) for k in range(n)]
        out = apply_filter(low, coef)
        amp = (max(out[n // 2 :]) - min(out[n // 2 :])) / 2
        assert amp < 0.05

    def test_invalid_order(self) -> None:
        with pytest.raises(ValueError, match="order"):
            butter_lowpass(0, 0.25)

    def test_invalid_cutoff_zero(self) -> None:
        with pytest.raises(ValueError, match="cutoff"):
            butter_lowpass(4, 0.0)

    def test_invalid_cutoff_one(self) -> None:
        with pytest.raises(ValueError, match="cutoff"):
            butter_lowpass(4, 1.0)

    def test_invalid_btype(self) -> None:
        from cds.signals.filters import _butter_section

        with pytest.raises(ValueError, match="btype"):
            _butter_section(4, 0.25, "nope")


# ---------------------------------------------------------------------------
# Band-pass / band-stop design
# ---------------------------------------------------------------------------


class TestBandFilters:
    def test_bandpass_structure(self) -> None:
        coef = butter_bandpass(order=4, low=0.2, high=0.5)
        assert coef.kind == "bandpass"
        assert coef.low == 0.2
        assert coef.high == 0.5
        assert isinstance(coef.section1, FilterCoefficients)
        assert isinstance(coef.section2, FilterCoefficients)

    def test_bandstop_structure(self) -> None:
        coef = butter_bandstop(order=4, low=0.2, high=0.5)
        assert coef.kind == "bandstop"

    def test_bandpass_negative_3db_edges(self) -> None:
        # The magnitude response equals 1/sqrt(2) (~0.7071) exactly at the edges.
        coef = butter_bandpass(order=4, low=0.2, high=0.5)
        for edge in (0.2, 0.5):
            h = _band_response(coef, edge)
            assert abs(abs(h) - 1 / math.sqrt(2)) < 1e-3

    def test_bandstop_negative_3db_edges(self) -> None:
        # Pinned to the measured value rather than a wide band. 0.4 < |H| < 0.8
        # accepted both the true 0.707107 and anything from a badly broken
        # design, so it could not detect a regression.
        coef = butter_bandstop(order=4, low=0.2, high=0.5)
        for edge in (0.2, 0.5):
            assert abs(_band_response(coef, edge)) == pytest.approx(0.699906425, abs=1e-7)

    def test_bandstop_stopband_floor_is_documented_not_butterworth(self) -> None:
        """Pin the parallel-sum stop band, including how far short it falls.

        A true 4th-order band-stop reaches 2.24e-4 at mid-band; the parallel
        low+high sum bottoms out near 6.1e-2, about 49 dB shallower, because the
        two branches are not power-complementary and their skirts add instead of
        cancelling. That limit is structural, so it is asserted here rather than
        hidden behind a loose inequality — see butter_bandstop's docstring.
        """
        coef = butter_bandstop(order=4, low=0.2, high=0.5)
        measured = {
            0.30: 0.095992878,
            0.35: 0.060886809,
            0.40: 0.229074587,
        }
        for w, expected in measured.items():
            assert abs(_band_response(coef, w)) == pytest.approx(expected, abs=1e-7)
        # nowhere near the ~2e-4 a real 4th-order band-stop achieves
        assert abs(_band_response(coef, 0.35)) > 1e-2

    def test_bandstop_passband_is_unity(self) -> None:
        coef = butter_bandstop(order=4, low=0.2, high=0.5)
        for w in (0.05, 0.9):
            assert abs(_band_response(coef, w)) == pytest.approx(1.0, abs=1e-4)

    def test_bandpass_passband_unity(self) -> None:
        coef = butter_bandpass(order=4, low=0.2, high=0.5)
        h_mid = _band_response(coef, 0.35)
        assert abs(h_mid) > 0.95  # close to unity inside the band

    def test_bandpass_stops_outside_band(self) -> None:
        coef = butter_bandpass(order=4, low=0.2, high=0.5)
        for w in (0.05, 0.8):
            assert abs(_band_response(coef, w)) < 0.2

    def test_bandstop_stops_inside_band(self) -> None:
        coef = butter_bandstop(order=4, low=0.2, high=0.5)
        for w in (0.3, 0.4):
            assert abs(_band_response(coef, w)) < 0.3

    def test_bandpass_preserves_in_band_sinusoid(self) -> None:
        # time-domain: an in-band tone keeps ~its amplitude
        coef = butter_bandpass(order=4, low=0.2, high=0.5)
        n = 1024
        tone = [math.sin(math.pi * 0.35 * k) for k in range(n)]
        out = apply_filter(tone, coef)
        amp = (max(out[n // 2 :]) - min(out[n // 2 :])) / 2
        assert amp > 0.9

    def test_bandpass_rejects_out_of_band_sinusoid(self) -> None:
        coef = butter_bandpass(order=4, low=0.2, high=0.5)
        n = 1024
        tone = [math.sin(math.pi * 0.8 * k) for k in range(n)]
        out = apply_filter(tone, coef)
        amp = (max(out[n // 2 :]) - min(out[n // 2 :])) / 2
        assert amp < 0.05

    def test_invalid_low_high_order(self) -> None:
        with pytest.raises(ValueError, match="order"):
            butter_bandpass(order=0, low=0.2, high=0.5)

    def test_invalid_edges_bandpass(self) -> None:
        with pytest.raises(ValueError, match="0 < low < high < 1"):
            butter_bandpass(order=4, low=0.5, high=0.2)  # inverted

    def test_invalid_edges_bandstop(self) -> None:
        with pytest.raises(ValueError, match="0 < low < high < 1"):
            butter_bandstop(order=4, low=-0.1, high=0.5)

    def test_bandcoefficients_post_init_bad_kind(self) -> None:
        dummy = FilterCoefficients(b=[1.0], a=[1.0])
        with pytest.raises(ValueError, match="kind"):
            BandFilterCoefficients(0.2, 0.5, "nope", dummy, dummy)

    def test_bandcoefficients_post_init_bad_edges(self) -> None:
        dummy = FilterCoefficients(b=[1.0], a=[1.0])
        with pytest.raises(ValueError, match="0 < low < high < 1"):
            BandFilterCoefficients(0.6, 0.5, "bandpass", dummy, dummy)


def _section_response(coef: FilterCoefficients, w: float) -> complex:
    """Complex frequency response of a single section at normalised freq w in (0,1)."""
    z = math.e ** (1j * math.pi * w)
    num = sum(coef.b[i] * z ** (-i) for i in range(len(coef.b)))
    den = sum(coef.a[i] * z ** (-i) for i in range(len(coef.a)))
    return num / den


def _band_response(coef: BandFilterCoefficients, w: float) -> complex:
    """Frequency response of a band section pair."""
    if coef.kind == "bandpass":
        return _section_response(coef.section1, w) * _section_response(coef.section2, w)
    return _section_response(coef.section1, w) + _section_response(coef.section2, w)


# ---------------------------------------------------------------------------
# apply_filter dispatch
# ---------------------------------------------------------------------------


class TestApplyFilter:
    def test_empty_input_single_section(self) -> None:
        coef = butter_lowpass(4, 0.25)
        assert apply_filter([], coef) == []

    def test_empty_input_band(self) -> None:
        coef = butter_bandpass(4, 0.2, 0.5)
        assert apply_filter([], coef) == []

    def test_identity_filter_preserves_signal(self) -> None:
        # H(z) = 1 (b=[1], a=[1]) leaves the signal untouched
        coef = FilterCoefficients(b=[1.0], a=[1.0])
        sig = [1.0, -2.0, 3.5, 0.0, 4.0]
        assert apply_filter(sig, coef) == sig

    def test_pure_feedforward_is_moving_average(self) -> None:
        # b = [0.5, 0.5], a = [1] is a 2-point moving average
        coef = FilterCoefficients(b=[0.5, 0.5], a=[1.0])
        sig = [1.0, 3.0, 5.0]
        out = apply_filter(sig, coef)
        assert len(out) == 3
        assert abs(out[0] - 0.5) < 1e-12
        assert abs(out[1] - 2.0) < 1e-12
        assert abs(out[2] - 4.0) < 1e-12

    def test_output_length_matches_input(self) -> None:
        coef = butter_lowpass(4, 0.25)
        sig = [float(k) for k in range(50)]
        out = apply_filter(sig, coef)
        assert len(out) == len(sig)

    def test_single_sample(self) -> None:
        coef = butter_lowpass(2, 0.3)
        out = apply_filter([5.0], coef)
        assert len(out) == 1

    def test_band_output_length_matches_input(self) -> None:
        coef = butter_bandstop(4, 0.2, 0.5)
        sig = [math.sin(0.1 * k) for k in range(80)]
        out = apply_filter(sig, coef)
        assert len(out) == len(sig)

    def test_cascade_order_bandpass(self) -> None:
        # band-pass applies high-pass first, then low-pass; confirm the cascade
        # matches applying the sections manually in that order.
        from cds.signals.filters import _apply_iir

        bp = butter_bandpass(4, 0.2, 0.5)
        sig = [math.sin(0.3 * k) + 0.5 for k in range(64)]
        via_apply = apply_filter(sig, bp)
        manual = _apply_iir(_apply_iir(sig, bp.section1), bp.section2)
        for a, b in zip(via_apply, manual):
            assert abs(a - b) < 1e-9

    def test_parallel_sum_bandstop(self) -> None:
        # band-stop = section1(signal) + section2(signal)
        from cds.signals.filters import _apply_iir

        bs = butter_bandstop(4, 0.2, 0.5)
        sig = [math.cos(0.2 * k) for k in range(64)]
        via_apply = apply_filter(sig, bs)
        manual = [a + b for a, b in zip(_apply_iir(sig, bs.section1), _apply_iir(sig, bs.section2))]
        for a, b in zip(via_apply, manual):
            assert abs(a - b) < 1e-9


# ---------------------------------------------------------------------------
# moving_median
# ---------------------------------------------------------------------------


class TestMovingMedian:
    def test_removes_isolated_spike(self) -> None:
        sig = [1.0, 1.0, 9.0, 1.0, 1.0]
        assert moving_median(sig, window=3) == [1.0, 1.0, 1.0, 1.0, 1.0]

    def test_preserves_smooth_signal(self) -> None:
        sig = [2.0, 4.0, 6.0, 8.0]
        out = moving_median(sig, window=3)
        # centred median of a monotonic ramp returns the middle sample; boundary
        # windows are even-length and take the upper-middle element
        assert out == [4.0, 4.0, 6.0, 8.0]

    def test_single_element_signal(self) -> None:
        assert moving_median([5.0], window=1) == [5.0]

    def test_empty_signal(self) -> None:
        assert moving_median([], window=3) == []

    def test_window_one_is_identity(self) -> None:
        sig = [1.0, 9.0, -3.0, 4.0]
        assert moving_median(sig, window=1) == sig

    def test_window_three_output_length(self) -> None:
        sig = [float(k) for k in range(10)]
        assert len(moving_median(sig, window=3)) == 10

    def test_large_window_whole_signal(self) -> None:
        # window == 2*n-1 spans the whole signal at every index, so every
        # output is the global median.
        sig = [3.0, 1.0, 2.0]  # n=3 -> 2*n-1 == 5, but window must be <= 2*n-1
        out = moving_median(sig, window=5)
        assert out == [2.0, 2.0, 2.0]

    def test_invalid_window_zero(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            moving_median([1.0, 2.0], window=0)

    def test_invalid_window_even(self) -> None:
        with pytest.raises(ValueError, match="odd"):
            moving_median([1.0, 2.0, 3.0], window=2)

    def test_invalid_window_too_large(self) -> None:
        with pytest.raises(ValueError, match="too large"):
            moving_median([1.0, 2.0], window=5)

    def test_boundaries_use_truncated_window(self) -> None:
        # at the first sample with window=3 the window is [sig[0], sig[1]] — an
        # even-length window takes the upper-middle element.
        sig = [9.0, 1.0, 1.0, 1.0, 9.0]
        out = moving_median(sig, window=3)
        assert out[0] == 9.0  # upper-middle of sorted [1, 9]
        assert out[-1] == 9.0  # upper-middle of sorted [1, 9]
