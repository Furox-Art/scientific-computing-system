"""Tests for the pure-Python Haar wavelet module (cds.wavelets)."""

from __future__ import annotations

import math
import random

import pytest

import cds.wavelets as wavelets_pkg
from cds.wavelets.haar import denoise, dwt, dwt_multi_level, idwt

_SQRT2 = math.sqrt(2.0)
_RNG = random.Random(20260824)
_RANDOM_16 = [_RNG.uniform(-5.0, 5.0) for _ in range(16)]


def test_wavelets_package_exports() -> None:
    assert hasattr(wavelets_pkg, "dwt")
    assert hasattr(wavelets_pkg, "idwt")
    assert hasattr(wavelets_pkg, "dwt_multi_level")
    assert hasattr(wavelets_pkg, "denoise")


def test_dwt_known_values_length_four() -> None:
    approx, detail = dwt([3.0, 1.0, 0.0, 2.0])
    assert approx == pytest.approx([4.0 / _SQRT2, _SQRT2])
    assert detail == pytest.approx([_SQRT2, -_SQRT2])


def test_dwt_constant_signal_has_zero_details() -> None:
    approx, detail = dwt([2.5] * 8)
    assert approx == pytest.approx([2.5 * _SQRT2] * 4)
    assert detail == [0.0] * 4


def test_dwt_rejects_odd_length() -> None:
    with pytest.raises(ValueError, match="power of two"):
        dwt([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="power of two"):
        dwt([])
    with pytest.raises(ValueError, match="power of two"):
        dwt([1.0])
    with pytest.raises(ValueError, match="power of two"):
        dwt([1.0, 2.0, 3.0, 4.0, 5.0])


def test_idwt_inverts_dwt_on_random_data() -> None:
    approx, detail = dwt(_RANDOM_16)
    reconstructed = idwt(approx, detail)
    assert reconstructed == pytest.approx(_RANDOM_16, abs=1e-12)


def test_idwt_known_values_length_two() -> None:
    assert idwt([4.0 / _SQRT2], [_SQRT2]) == pytest.approx([3.0, 1.0])


def test_idwt_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        idwt([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="non-empty"):
        idwt([], [])
    with pytest.raises(ValueError, match="power of two"):
        idwt([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    # deepest level with single coefficient must succeed (power-of-two 1 is allowed)
    assert idwt([2.0], [0.0]) == pytest.approx([2.0 / _SQRT2, 2.0 / _SQRT2])


def test_dwt_multi_level_band_shapes() -> None:
    coeffs = dwt_multi_level(_RANDOM_16, 3)
    assert len(coeffs) == 3
    assert [len(a) for a, _ in coeffs] == [8, 4, 2]
    assert [len(d) for _, d in coeffs] == [8, 4, 2]
    # non-power-of-two even length exercises the while-loop non-power-of-two path
    coeffs12 = dwt_multi_level([1.0] * 12, 2)
    assert len(coeffs12) == 2
    assert [len(a) for a, _ in coeffs12] == [6, 3]


def test_dwt_multi_level_round_trip() -> None:
    coeffs = dwt_multi_level(_RANDOM_16, 4)
    current = coeffs[-1][0]
    for _, detail in reversed(coeffs):
        current = idwt(current, detail)
    assert current == pytest.approx(_RANDOM_16, abs=1e-12)


def test_dwt_multi_level_rejects_bad_levels() -> None:
    with pytest.raises(ValueError, match="levels must be between"):
        dwt_multi_level([1.0, 2.0, 3.0, 4.0], 3)
    with pytest.raises(ValueError, match="levels must be at"):
        dwt_multi_level([1.0, 2.0], 0)
    with pytest.raises(ValueError, match="signal too short"):
        dwt_multi_level([1.0] * 12, 3)  # max is 2, triggers levels>max with n>=2**levels
    with pytest.raises(ValueError, match="power of two"):
        dwt_multi_level([1.0, 2.0, 3.0], 1)
    with pytest.raises(ValueError, match="power of two"):
        dwt_multi_level([], 1)


def test_denoise_preserves_constant_signal() -> None:
    data = [1.0] * 16
    result = denoise(data, threshold=0.5)
    assert result == pytest.approx(data, abs=1e-12)


def test_denoise_hard_threshold_known_values() -> None:
    # detail = (1-2)/sqrt2 = -0.707..., (3-4)/sqrt2 = -0.707...; threshold 0.8 zeros both
    data = [1.0, 2.0, 3.0, 4.0]
    result = denoise(data, threshold=0.8)
    assert result == pytest.approx([1.5, 1.5, 3.5, 3.5], abs=1e-12)
    # threshold 0 leaves unchanged
    assert denoise(data, threshold=0.0) == pytest.approx(data, abs=1e-12)
    with pytest.raises(ValueError, match="non-negative"):
        denoise(data, threshold=-0.1)
    with pytest.raises(ValueError, match="power of two"):
        denoise([1.0, 2.0, 3.0], threshold=0.5)
    with pytest.raises(ValueError, match="power of two"):
        denoise([], threshold=0.5)
