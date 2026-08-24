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


def test_frame_signal_zero_pads_partial_final_frame() -> None:
    signal = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    frames = frame_signal(signal, 8, 3)
    assert len(frames) == 3
    zeros = [0j] * 5
    assert frames[0] == [1 + 0j, 2 + 0j, 3 + 0j, *zeros]
    assert frames[1] == [4 + 0j, 5 + 0j, 6 + 0j, *zeros]
    assert frames[2] == [7 + 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j]


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
    # hop defaults to n_fft // 4 = 8; ceil(70 / 8) frames
    assert result.times == [float(start) for start in range(0, 70, 8)]


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
    assert len(result.times) == len(result.magnitude) == 5
    assert all(isinstance(t, float) for t in result.times)
    assert len(result.freqs) == 9


def test_stft_rejects_empty_signal() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        stft([], n_fft=16)


def test_stft_propagates_invalid_n_fft() -> None:
    with pytest.raises(ValueError, match="power of two >= 2"):
        stft([1.0] * 20, n_fft=18, hop=4)
