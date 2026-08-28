"""Tests for the short-time Fourier transform module."""

from __future__ import annotations

import math

import pytest

from cds.signals.processing import fft_radix2
from cds.signals.stft import STFTResult, frame_signal, stft, window

# --- window() ---


def test_window_hann_periodic_values() -> None:
    # periodic Hann of length 4: [0, 0.5, 1, 0.5] — w[0] == w[n] wrap value
    assert window("hann", 4) == pytest.approx([0.0, 0.5, 1.0, 0.5])


def test_window_formula_both_kinds() -> None:
    for kind, a, b in (("hann", 0.5, 0.5), ("hamming", 0.54, 0.46)):
        w = window(kind, 5)
        assert len(w) == 5
        for k, value in enumerate(w):
            expected = a - b * math.cos(2 * math.pi * k / 5)
            assert value == pytest.approx(expected)


def test_window_unknown_kind_raises() -> None:
    with pytest.raises(ValueError, match="unknown window kind"):
        window("blackman", 8)


def test_window_rejects_nonpositive_length() -> None:
    with pytest.raises(ValueError, match="window length must be >= 1"):
        window("hann", 0)


# --- frame_signal() ---


def test_frame_signal_exact_single_frame() -> None:
    frames = frame_signal([1.0, 2.0, 3.0, 4.0], 4, 4)
    assert len(frames) == 1
    assert frames[0] == [1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j]


def test_frame_signal_frames_overlap_by_n_fft_minus_hop() -> None:
    # Each frame carries n_fft consecutive samples and starts hop later, so
    # frame k and frame k+1 share their last/first n_fft - hop samples.
    frames = frame_signal([float(v) for v in range(1, 21)], 8, 4)
    assert len(frames) == 4
    assert frames[0] == [1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j, 5 + 0j, 6 + 0j, 7 + 0j, 8 + 0j]
    assert frames[1] == [5 + 0j, 6 + 0j, 7 + 0j, 8 + 0j, 9 + 0j, 10 + 0j, 11 + 0j, 12 + 0j]
    # the overlap is real: the tail of one frame is the head of the next
    for earlier, later in zip(frames, frames[1:]):
        assert earlier[4:] == later[:4]


def test_frame_signal_zero_pads_only_the_final_frame() -> None:
    signal = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    frames = frame_signal(signal, 8, 3)
    # 7 samples, 8-long frames: the very first frame already reaches the end,
    # so exactly one frame is emitted and only it is padded.
    assert len(frames) == 1
    assert frames[0] == [1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j, 5 + 0j, 6 + 0j, 7 + 0j, 0j]


def test_frame_signal_full_frames_are_not_padded() -> None:
    signal = [float(v) for v in range(16)]
    for frame in frame_signal(signal, 8, 8):
        assert all(value != 0j for value in frame[1:])


def test_frame_signal_rejects_empty_signal() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        frame_signal([], 8, 2)


def test_frame_signal_rejects_hop_below_one() -> None:
    with pytest.raises(ValueError, match="hop must be >= 1"):
        frame_signal([1.0, 2.0], 8, 0)


def test_frame_signal_rejects_n_fft_below_two() -> None:
    with pytest.raises(ValueError, match="power of two >= 2"):
        frame_signal([1.0, 2.0], 1, 1)


def test_frame_signal_rejects_non_power_of_two_n_fft() -> None:
    with pytest.raises(ValueError, match="power of two >= 2"):
        frame_signal([1.0, 2.0], 12, 2)


def test_frame_signal_rejects_signal_shorter_than_hop() -> None:
    with pytest.raises(ValueError, match="shorter than one hop"):
        frame_signal([1.0, 2.0], 8, 3)


def test_frame_signal_rejects_hop_above_n_fft() -> None:
    with pytest.raises(ValueError, match="must not exceed n_fft"):
        frame_signal([1.0] * 20, 8, 16)


# --- stft() ---


def test_stft_pure_cosine_lands_in_expected_bin() -> None:
    # full-window frames (hop == n_fft) see an integer number of cycles,
    # so a tone placed on the FFT grid peaks exactly in its own bin
    n_fft = 64
    tone_bin = 4
    signal = [math.sin(2 * math.pi * tone_bin * k / n_fft) for k in range(192)]
    result = stft(signal, n_fft=n_fft, hop=n_fft)
    assert len(result.magnitude) == 3
    for row in result.magnitude:
        assert len(row) == n_fft // 2 + 1
        peak = max(range(len(row)), key=row.__getitem__)
        assert peak == tone_bin


def test_stft_freqs_are_cycles_per_sample_and_times_are_frame_starts() -> None:
    n_fft = 32
    result = stft([math.sin(0.2 * k) for k in range(70)], n_fft=n_fft)
    assert result.freqs == pytest.approx([k / n_fft for k in range(n_fft // 2 + 1)])
    # hop defaults to n_fft // 4 = 8, frames are 32 long, and framing stops with
    # the frame that first reaches sample 70: starts 0, 8, 16, 24, 32, 40.
    assert result.times == [0.0, 8.0, 16.0, 24.0, 32.0, 40.0]


def test_stft_default_hop_matches_quarter_window() -> None:
    signal = [math.sin(0.2 * k) for k in range(70)]
    assert stft(signal, n_fft=32) == stft(signal, n_fft=32, hop=8)


def test_stft_constant_signal_dc_bin_dominates() -> None:
    signal = [2.5] * 96
    result = stft(signal, n_fft=32, hop=32, window_kind="hamming")
    for row in result.magnitude:
        assert row[0] == max(row)
        # Hamming's transform inherently places 0.46-scaled components in the
        # +/-1 neighbours (~18.4 here), so demand strict 2x dominance instead
        assert row[0] > 40.0
        assert all(row[0] > 2.0 * value for value in row[1:])


def test_stft_parseval_window_energy_identity() -> None:
    n_fft = 16
    signal = [
        math.sin(2 * math.pi * 3 * k / n_fft) + 0.25 * math.cos(2 * math.pi * k / n_fft)
        for k in range(n_fft)
    ]
    result = stft(signal, n_fft=n_fft, hop=n_fft)
    assert len(result.magnitude) == 1

    win = window("hann", n_fft)
    tapered: list[float] = [s * w for s, w in zip(signal, win)]
    windowed: list[float | complex] = [s * w for s, w in zip(signal, win)]
    spectrum = fft_radix2(windowed)

    # Parseval: sum |X[k]|^2 == n_fft * sum |x_w[n]|^2 (validates window scaling)
    time_energy = sum(value * value for value in tapered)
    spectrum_energy = sum(abs(bin_value) ** 2 for bin_value in spectrum)
    assert spectrum_energy == pytest.approx(n_fft * time_energy, rel=1e-9)

    # real input: mirror symmetry folds the full energy into the stored half
    stored = sum(m * m for m in result.magnitude[0])
    nyquist = abs(spectrum[n_fft // 2]) ** 2
    mirrored = 2 * stored - abs(spectrum[0]) ** 2 - nyquist
    assert mirrored == pytest.approx(spectrum_energy, rel=1e-9)


def test_stft_result_type_and_row_shape() -> None:
    result = stft(list(range(40)), n_fft=16, hop=8)
    assert isinstance(result, STFTResult)
    # 16-long frames stepping by 8 over 40 samples: starts 0, 8, 16, 24.
    assert len(result.times) == len(result.magnitude) == 4
    assert all(isinstance(t, float) for t in result.times)
    assert len(result.freqs) == 9


def test_stft_overlapping_frames_concentrate_a_tone_like_a_real_stft() -> None:
    """A windowed tone on the FFT grid puts 2/3 of its frame energy in one bin.

    That fraction is a property of the Hann window, not of this library: the
    taper splits a grid-aligned tone as 0.5 / 0.25 / 0.25 in amplitude across
    three bins, giving 0.5^2 / (0.5^2 + 0.25^2 + 0.25^2) = 2/3 in energy. When
    frames carried only ``hop`` real samples and padded the rest, the window was
    tapering zeros and this dropped to about 0.15.
    """
    n_fft, tone_bin = 256, 32
    signal = [math.sin(2 * math.pi * tone_bin * k / n_fft) for k in range(4096)]
    result = stft(signal, n_fft=n_fft, hop=64)

    row = result.magnitude[len(result.magnitude) // 2]
    assert max(range(len(row)), key=row.__getitem__) == tone_bin
    energy = sum(value * value for value in row)
    assert row[tone_bin] ** 2 / energy == pytest.approx(2.0 / 3.0, abs=0.02)


def test_stft_rejects_empty_signal() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        stft([], n_fft=16)


def test_stft_propagates_invalid_n_fft() -> None:
    with pytest.raises(ValueError, match="power of two >= 2"):
        stft([1.0] * 20, n_fft=18, hop=4)
