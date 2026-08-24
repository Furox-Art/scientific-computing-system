"""Tests for the Haar discrete wavelet transform module."""

from __future__ import annotations

import math
import random

import pytest

from cds.signals.wavelet import dwt, dwt_multi_level, idwt

_SQRT2 = math.sqrt(2.0)

# Deterministic pseudo-random signal (seeded LCG-style via random.Random).
_RNG = random.Random(20260824)
_RANDOM_16 = [_RNG.uniform(-5.0, 5.0) for _ in range(16)]


def _synthesise(
    coeffs: list[tuple[list[float], list[float]]],
) -> list[float]:
    """Invert a full multi-level decomposition back to the original signal."""
    current = coeffs[-1][0]
    for _approx, detail in reversed(coeffs):
        current = idwt(current, detail)
    return current


# --- dwt() ---


def test_dwt_known_values_length_four() -> None:
    # pairs (3, 1) and (0, 2): averages 4/sqrt2, 2/sqrt2; differences 2/sqrt2, -2/sqrt2
    approx, detail = dwt([3.0, 1.0, 0.0, 2.0])
    assert approx == pytest.approx([4.0 / _SQRT2, _SQRT2])
    assert detail == pytest.approx([_SQRT2, -_SQRT2])


def test_dwt_output_lengths_and_types() -> None:
    approx, detail = dwt(_RANDOM_16)
    assert isinstance(approx, list) and isinstance(detail, list)
    assert len(approx) == len(detail) == 8
    assert all(isinstance(value, float) for value in approx + detail)


def test_dwt_accepts_integer_input() -> None:
    approx, detail = dwt([3, 1])
    assert approx == pytest.approx([4.0 / _SQRT2])
    assert detail == pytest.approx([_SQRT2])


def test_dwt_constant_signal_has_zero_details() -> None:
    approx, detail = dwt([2.5] * 8)
    assert approx == pytest.approx([2.5 * _SQRT2] * 4)
    assert detail == [0.0] * 4


def test_dwt_rejects_empty_signal() -> None:
    with pytest.raises(ValueError, match="signal length must be a power of two >= 2"):
        dwt([])


def test_dwt_rejects_single_sample_signal() -> None:
    with pytest.raises(ValueError, match="signal length must be a power of two >= 2"):
        dwt([1.0])


def test_dwt_rejects_non_power_of_two_length() -> None:
    with pytest.raises(ValueError, match="signal length must be a power of two >= 2"):
        dwt([1.0, 2.0, 3.0])


# --- idwt() ---


def test_idwt_inverts_dwt_exactly_on_random_data() -> None:
    approx, detail = dwt(_RANDOM_16)
    reconstructed = idwt(approx, detail)
    assert reconstructed == pytest.approx(_RANDOM_16, abs=1e-12)


def test_idwt_known_values_length_two() -> None:
    assert idwt([4.0 / _SQRT2], [_SQRT2]) == pytest.approx([3.0, 1.0])


def test_idwt_round_trips_a_hand_computed_case() -> None:
    reconstructed = idwt([4.0 / _SQRT2, _SQRT2], [_SQRT2, -_SQRT2])
    assert reconstructed == pytest.approx([3.0, 1.0, 0.0, 2.0])


def test_idwt_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match=r"approx and detail lengths differ \(2 != 1\)"):
        idwt([1.0, 2.0], [1.0])


def test_idwt_rejects_empty_coefficients() -> None:
    with pytest.raises(ValueError, match="approx must be non-empty"):
        idwt([], [])


def test_idwt_rejects_non_power_of_two_coefficient_count() -> None:
    with pytest.raises(ValueError, match="coefficient count must be a power of two"):
        idwt([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])


# --- energy conservation (Parseval identity) ---


@pytest.mark.parametrize("n", [2, 4, 8, 16, 64])
def test_dwt_conserves_energy(n: int) -> None:
    rng = random.Random(1000 + n)
    signal = [rng.uniform(-10.0, 10.0) for _ in range(n)]
    approx, detail = dwt(signal)
    input_energy = sum(sample * sample for sample in signal)
    output_energy = sum(value * value for value in approx) + sum(value * value for value in detail)
    assert output_energy == pytest.approx(input_energy, rel=1e-9)


# --- dwt_multi_level() ---


def test_dwt_multi_level_band_shapes() -> None:
    coeffs = dwt_multi_level(_RANDOM_16, 3)
    assert len(coeffs) == 3
    assert [len(approx) for approx, _detail in coeffs] == [8, 4, 2]
    assert [len(detail) for _approx, detail in coeffs] == [8, 4, 2]


def test_dwt_multi_level_first_stage_matches_single_level() -> None:
    coeffs = dwt_multi_level(_RANDOM_16, 2)
    assert coeffs[0] == dwt(_RANDOM_16)


def test_dwt_multi_level_second_stage_transforms_first_approximation() -> None:
    coeffs = dwt_multi_level(_RANDOM_16, 2)
    first_approx = dwt(_RANDOM_16)[0]
    assert coeffs[1] == dwt(first_approx)


def test_dwt_multi_level_full_round_trip_on_random_data() -> None:
    coeffs = dwt_multi_level(_RANDOM_16, 4)
    assert _synthesise(coeffs) == pytest.approx(_RANDOM_16, abs=1e-12)


def test_dwt_multi_level_deepest_possible_decomposition() -> None:
    signal = [math.sin(0.7 * k) + 0.3 * math.cos(1.9 * k) for k in range(8)]
    coeffs = dwt_multi_level(signal, 3)
    assert len(coeffs[-1][0]) == 1
    assert _synthesise(coeffs) == pytest.approx(signal, abs=1e-12)


def test_dwt_multi_level_minimum_signal_single_level() -> None:
    approx, detail = dwt_multi_level([3.0, 1.0], 1)[0]
    assert approx == pytest.approx([4.0 / _SQRT2])
    assert detail == pytest.approx([_SQRT2])


def test_dwt_multi_level_rejects_zero_levels() -> None:
    with pytest.raises(ValueError, match="levels must be between"):
        dwt_multi_level([1.0, 2.0], 0)


def test_dwt_multi_level_rejects_levels_above_maximum() -> None:
    with pytest.raises(ValueError, match="levels must be between 1 and 2"):
        dwt_multi_level([1.0] * 4, 3)


def test_dwt_multi_level_rejects_invalid_signal() -> None:
    with pytest.raises(ValueError, match="signal length must be a power of two >= 2"):
        dwt_multi_level([1.0], 1)
