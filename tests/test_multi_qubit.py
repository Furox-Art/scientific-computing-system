"""Tests for multi-qubit quantum operations and entanglement."""

import math
from typing import cast

import pytest

from cds.quantum.multi_qubit import (
    QuantumRegister,
    bell_state,
    cnot,
    cz,
    ghz_state,
    h_gate,
    is_entangled,
    rz_gate,
    swap,
    toffoli,
    x_gate,
    y_gate,
    z_gate,
)

# --- QuantumRegister basics ---


def test_zeros_register() -> None:
    reg = QuantumRegister.zeros(2)
    assert reg.n_qubits == 2
    assert reg.size == 4
    assert abs(reg.amplitudes[0] - 1) < 1e-9
    assert all(abs(a) < 1e-9 for a in reg.amplitudes[1:])


def test_register_shape_and_value_validation() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        QuantumRegister.zeros(0)
    with pytest.raises(ValueError, match="positive integer"):
        QuantumRegister.zeros(True)
    with pytest.raises(ValueError, match="positive integer"):
        QuantumRegister.zeros(cast(int, 1.5))
    with pytest.raises(ValueError, match="exactly 2"):
        QuantumRegister(n_qubits=1, amplitudes=[1 + 0j])
    with pytest.raises(ValueError, match="amplitudes must be finite"):
        QuantumRegister(n_qubits=1, amplitudes=[complex(math.inf), 0j])


def test_from_bits() -> None:
    reg = QuantumRegister.from_bits(3, 5)
    assert abs(reg.amplitudes[5] - 1) < 1e-9
    assert sum(abs(a) ** 2 for a in reg.amplitudes) - 1.0 < 1e-9

    with pytest.raises(ValueError, match="value must be an integer"):
        QuantumRegister.from_bits(2, -1)
    with pytest.raises(ValueError, match="value must be an integer"):
        QuantumRegister.from_bits(2, 4)
    with pytest.raises(ValueError, match="value must be an integer"):
        QuantumRegister.from_bits(2, True)
    with pytest.raises(ValueError, match="value must be an integer"):
        QuantumRegister.from_bits(2, cast(int, 1.5))


def test_probabilities() -> None:
    reg = QuantumRegister.zeros(1)
    probs = reg.probabilities()
    assert abs(probs[0] - 1.0) < 1e-9
    assert abs(probs[1]) < 1e-9


def test_probabilities_normalize_equivalent_state_vectors() -> None:
    reg = QuantumRegister(n_qubits=1, amplitudes=[3 + 0j, 4 + 0j])
    assert reg.probabilities() == pytest.approx([9.0 / 25.0, 16.0 / 25.0])

    zero = QuantumRegister(n_qubits=1, amplitudes=[0j, 0j])
    with pytest.raises(ValueError, match="non-zero norm"):
        zero.probabilities()


def test_normalize() -> None:
    reg = QuantumRegister(n_qubits=1, amplitudes=[3 + 0j, 4 + 0j])
    reg.normalize()
    total = sum(abs(a) ** 2 for a in reg.amplitudes)
    assert abs(total - 1.0) < 1e-9


def test_normalize_zero_norm_is_noop() -> None:
    reg = QuantumRegister(n_qubits=1, amplitudes=[0 + 0j, 0 + 0j])
    reg.normalize()
    assert reg.amplitudes == [0, 0]


def test_normalize_rejects_overflowed_norm() -> None:
    reg = QuantumRegister(n_qubits=1, amplitudes=[complex(1e308), 0j])
    with pytest.raises(ValueError, match="norm must be finite"):
        reg.normalize()


def test_measure_shots_normalizes_equivalent_state_vectors() -> None:
    scaled = QuantumRegister(n_qubits=1, amplitudes=[1e-12 + 0j, 1e-12 + 0j])
    normalized = QuantumRegister(
        n_qubits=1,
        amplitudes=[complex(1 / math.sqrt(2)), complex(1 / math.sqrt(2))],
    )
    assert scaled.measure_shots(shots=200, seed=1) == normalized.measure_shots(shots=200, seed=1)


def test_measure_deterministic() -> None:
    reg = QuantumRegister.zeros(2)
    result = reg.measure(seed=42)
    assert result == 0


def test_measure_rejects_zero_norm_state() -> None:
    reg = QuantumRegister(n_qubits=1, amplitudes=[0j, 0j])
    with pytest.raises(ValueError, match="non-zero norm"):
        reg.measure(seed=42)


def test_measure_shots() -> None:
    reg = QuantumRegister.zeros(2)
    counts = reg.measure_shots(shots=100, seed=42)
    assert counts.get("00", 0) == 100


def test_measure_shots_rejects_invalid_shot_count() -> None:
    reg = QuantumRegister.zeros(1)
    with pytest.raises(ValueError, match="positive integer"):
        reg.measure_shots(shots=0)
    with pytest.raises(ValueError, match="positive integer"):
        reg.measure_shots(shots=-1)
    with pytest.raises(ValueError, match="positive integer"):
        reg.measure_shots(shots=True)
    with pytest.raises(ValueError, match="positive integer"):
        reg.measure_shots(shots=cast(int, 1.5))


def test_expectation_zero_state() -> None:
    reg = QuantumRegister.zeros(2)
    assert reg.expectation() == 0.0


def test_expectation_uses_normalized_probabilities() -> None:
    reg = QuantumRegister(n_qubits=1, amplitudes=[0j, 2 + 0j])
    assert reg.expectation() == 1.0


# --- Single-qubit gates on register ---


def test_x_gate_flips_qubit_0() -> None:
    reg = QuantumRegister.zeros(2)
    reg = x_gate(reg, 0)
    assert abs(reg.amplitudes[1] - 1) < 1e-9


def test_x_gate_flips_qubit_1() -> None:
    reg = QuantumRegister.zeros(2)
    reg = x_gate(reg, 1)
    assert abs(reg.amplitudes[2] - 1) < 1e-9


def test_single_qubit_gate_rejects_invalid_target() -> None:
    reg = QuantumRegister.zeros(2)
    with pytest.raises(ValueError, match="target index"):
        x_gate(reg, -1)
    with pytest.raises(ValueError, match="target index"):
        x_gate(reg, 2)
    with pytest.raises(ValueError, match="target index"):
        x_gate(reg, True)
    with pytest.raises(ValueError, match="target index"):
        x_gate(reg, cast(int, 0.5))


def test_h_gate_superposition() -> None:
    reg = QuantumRegister.zeros(1)
    reg = h_gate(reg, 0)
    probs = reg.probabilities()
    assert abs(probs[0] - 0.5) < 1e-9
    assert abs(probs[1] - 0.5) < 1e-9


def test_z_gate_on_zero() -> None:
    reg = QuantumRegister.zeros(1)
    reg = z_gate(reg, 0)
    assert abs(reg.amplitudes[0] - 1) < 1e-9


def test_z_gate_on_one() -> None:
    reg = QuantumRegister.from_bits(1, 1)
    reg = z_gate(reg, 0)
    assert abs(reg.amplitudes[1] - (-1)) < 1e-9


def test_y_gate() -> None:
    reg = QuantumRegister.zeros(1)
    reg = y_gate(reg, 0)
    assert abs(reg.amplitudes[0]) < 1e-9
    assert abs(reg.amplitudes[1] - 1j) < 1e-9


def test_rz_gate() -> None:
    reg = QuantumRegister.zeros(1)
    reg = h_gate(reg, 0)
    reg = rz_gate(reg, 0, math.pi)
    probs = reg.probabilities()
    assert abs(probs[0] - 0.5) < 1e-9
    assert abs(probs[1] - 0.5) < 1e-9


def test_rz_gate_rejects_non_finite_theta() -> None:
    with pytest.raises(ValueError, match="theta"):
        rz_gate(QuantumRegister.zeros(1), 0, math.nan)


def test_double_x_is_identity() -> None:
    reg = QuantumRegister.zeros(2)
    reg = x_gate(reg, 0)
    reg = x_gate(reg, 0)
    assert abs(reg.amplitudes[0] - 1) < 1e-9


def test_double_h_is_identity() -> None:
    reg = QuantumRegister.zeros(1)
    reg = h_gate(reg, 0)
    reg = h_gate(reg, 0)
    assert abs(reg.amplitudes[0] - 1) < 1e-9


# --- Controlled gates ---


def test_cnot_no_flip_when_control_zero() -> None:
    reg = QuantumRegister.zeros(2)
    reg = cnot(reg, 0, 1)
    assert abs(reg.amplitudes[0] - 1) < 1e-9


def test_cnot_flips_when_control_one() -> None:
    reg = QuantumRegister.zeros(2)
    reg = x_gate(reg, 0)
    reg = cnot(reg, 0, 1)
    assert abs(reg.amplitudes[3] - 1) < 1e-9


def test_controlled_gates_validate_indices_and_distinctness() -> None:
    reg = QuantumRegister.zeros(2)
    with pytest.raises(ValueError, match="control index"):
        cnot(reg, 2, 0)
    with pytest.raises(ValueError, match="target index"):
        cnot(reg, 0, 2)
    with pytest.raises(ValueError, match="distinct"):
        cnot(reg, 0, 0)
    with pytest.raises(ValueError, match="distinct"):
        cz(reg, 1, 1)


def test_cnot_creates_entanglement() -> None:
    reg = QuantumRegister.zeros(2)
    reg = h_gate(reg, 0)
    reg = cnot(reg, 0, 1)
    assert is_entangled(reg)


# --- CZ ---


def test_cz_on_11() -> None:
    reg = QuantumRegister.from_bits(2, 3)
    reg = cz(reg, 0, 1)
    assert abs(reg.amplitudes[3] - (-1)) < 1e-9


def test_cz_on_00() -> None:
    reg = QuantumRegister.zeros(2)
    reg = cz(reg, 0, 1)
    assert abs(reg.amplitudes[0] - 1) < 1e-9


# --- SWAP ---


def test_swap_01_to_10() -> None:
    reg = QuantumRegister.zeros(2)
    reg = x_gate(reg, 0)
    reg = swap(reg, 0, 1)
    assert abs(reg.amplitudes[2] - 1) < 1e-9


def test_swap_same_qubit_returns_equivalent_copy() -> None:
    reg = h_gate(QuantumRegister.zeros(2), 0)
    swapped = swap(reg, 0, 0)
    assert swapped is not reg
    assert swapped.amplitudes == reg.amplitudes


def test_swap_validates_indices() -> None:
    reg = QuantumRegister.zeros(2)
    with pytest.raises(ValueError, match="first index"):
        swap(reg, -1, 1)
    with pytest.raises(ValueError, match="second index"):
        swap(reg, 0, 2)


# --- Toffoli ---


def test_toffoli_flips_when_both_controls_set() -> None:
    reg = QuantumRegister.zeros(3)
    reg = x_gate(reg, 0)
    reg = x_gate(reg, 1)
    reg = toffoli(reg, 0, 1, 2)
    assert abs(reg.amplitudes[7] - 1) < 1e-9


def test_toffoli_no_flip_when_one_control() -> None:
    reg = QuantumRegister.zeros(3)
    reg = x_gate(reg, 0)
    reg = toffoli(reg, 0, 1, 2)
    assert abs(reg.amplitudes[1] - 1) < 1e-9


def test_toffoli_requires_three_distinct_valid_qubits() -> None:
    reg = QuantumRegister.zeros(3)
    with pytest.raises(ValueError, match="first control index"):
        toffoli(reg, 3, 1, 2)
    with pytest.raises(ValueError, match="second control index"):
        toffoli(reg, 0, 3, 2)
    with pytest.raises(ValueError, match="target index"):
        toffoli(reg, 0, 1, 3)
    with pytest.raises(ValueError, match="distinct"):
        toffoli(reg, 0, 0, 2)


# --- Bell states ---


def test_bell_phi_plus() -> None:
    reg = bell_state(0)
    s = 1 / math.sqrt(2)
    assert abs(reg.amplitudes[0] - s) < 1e-9
    assert abs(reg.amplitudes[3] - s) < 1e-9
    assert is_entangled(reg)


def test_bell_phi_minus() -> None:
    reg = bell_state(1)
    s = 1 / math.sqrt(2)
    assert abs(abs(reg.amplitudes[0]) - s) < 1e-9
    assert abs(abs(reg.amplitudes[3]) - s) < 1e-9
    assert is_entangled(reg)


def test_bell_psi_plus() -> None:
    reg = bell_state(2)
    s = 1 / math.sqrt(2)
    assert abs(abs(reg.amplitudes[1]) - s) < 1e-9
    assert abs(abs(reg.amplitudes[2]) - s) < 1e-9
    assert is_entangled(reg)


def test_bell_psi_minus() -> None:
    reg = bell_state(3)
    assert is_entangled(reg)


def test_bell_state_rejects_invalid_selector() -> None:
    with pytest.raises(ValueError, match="0 through 3"):
        bell_state(4)
    with pytest.raises(ValueError, match="0 through 3"):
        bell_state(True)
    with pytest.raises(ValueError, match="0 through 3"):
        bell_state(cast(int, 1.5))


def test_bell_measurements() -> None:
    reg = bell_state(0)
    counts = reg.measure_shots(shots=10000, seed=7)
    assert "00" in counts
    assert "11" in counts
    total = sum(counts.values())
    r00 = counts.get("00", 0) / total
    assert 0.45 < r00 < 0.55


# --- GHZ ---


def test_ghz_3_qubit() -> None:
    reg = ghz_state(3)
    probs = reg.probabilities()
    assert abs(probs[0] - 0.5) < 1e-9
    assert abs(probs[7] - 0.5) < 1e-9
    for i in range(1, 7):
        assert abs(probs[i]) < 1e-9


def test_ghz_4_qubit() -> None:
    reg = ghz_state(4)
    probs = reg.probabilities()
    assert abs(probs[0] - 0.5) < 1e-9
    assert abs(probs[15] - 0.5) < 1e-9


def test_ghz_requires_positive_qubit_count() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ghz_state(0)


# --- Entanglement checks ---


def test_separable_state() -> None:
    reg = QuantumRegister.zeros(2)
    assert not is_entangled(reg)


def test_product_superposition_not_entangled() -> None:
    reg = QuantumRegister.zeros(2)
    reg = h_gate(reg, 0)
    assert not is_entangled(reg)


def test_entangled_after_cnot() -> None:
    reg = QuantumRegister.zeros(2)
    reg = h_gate(reg, 0)
    reg = cnot(reg, 0, 1)
    assert is_entangled(reg)


def test_entanglement_is_scale_invariant_and_rejects_zero_norm() -> None:
    s = 3.0 / math.sqrt(2)
    scaled_bell = QuantumRegister(2, [complex(s), 0j, 0j, complex(s)])
    assert is_entangled(scaled_bell)

    zero = QuantumRegister(2, [0j, 0j, 0j, 0j])
    with pytest.raises(ValueError, match="non-zero norm"):
        is_entangled(zero)


def test_entanglement_requires_two_qubits() -> None:
    with pytest.raises(ValueError, match="2-qubit"):
        is_entangled(QuantumRegister.zeros(1))


# --- 3-qubit circuits ---


def test_3_qubit_circuit() -> None:
    reg = QuantumRegister.zeros(3)
    reg = h_gate(reg, 0)
    reg = h_gate(reg, 1)
    reg = h_gate(reg, 2)
    probs = reg.probabilities()
    for p in probs:
        assert abs(p - 0.125) < 1e-9


def test_quantum_teleportation_circuit() -> None:
    reg = QuantumRegister.zeros(3)
    reg = x_gate(reg, 0)
    reg = h_gate(reg, 1)
    reg = cnot(reg, 1, 2)
    reg = cnot(reg, 0, 1)
    reg = h_gate(reg, 0)
    total_prob = sum(abs(a) ** 2 for a in reg.amplitudes)
    assert abs(total_prob - 1.0) < 1e-9
