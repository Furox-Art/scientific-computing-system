"""Tests for quantum module."""

import math
from typing import cast

import pytest

from cds.quantum.circuit import (
    QuantumCircuit,
    QuantumGate,
    Qubit,
    hadamard,
    pauli_x,
    pauli_z,
    phase_gate,
)
from cds.quantum.simulator import measure, simulate


def test_qubit_default_is_zero_state() -> None:
    q = Qubit()
    p0, p1 = q.probabilities()
    assert p0 == 1.0
    assert p1 == 0.0


def test_qubit_probabilities_normalize_equivalent_state_vectors() -> None:
    q = Qubit(alpha=3 + 0j, beta=4 + 0j)
    p0, p1 = q.probabilities()
    assert p0 == pytest.approx(9.0 / 25.0)
    assert p1 == pytest.approx(16.0 / 25.0)
    assert p0 + p1 == pytest.approx(1.0)


def test_qubit_rejects_non_finite_and_zero_norm_measurement_state() -> None:
    with pytest.raises(ValueError, match="amplitudes must be finite"):
        Qubit(alpha=complex(math.inf), beta=0j)

    zero = Qubit(alpha=0j, beta=0j)
    with pytest.raises(ValueError, match="non-zero norm"):
        zero.probabilities()
    with pytest.raises(ValueError, match="non-zero norm"):
        measure(zero, seed=1)


def test_quantum_gate_validates_shape_finiteness_and_unitarity() -> None:
    identity = [1, 0, 0, 1]
    assert QuantumGate("I", identity).matrix == [1 + 0j, 0j, 0j, 1 + 0j]

    with pytest.raises(ValueError, match="name"):
        QuantumGate(" ", identity)
    with pytest.raises(ValueError, match="exactly four"):
        QuantumGate("bad", [1, 0, 0])
    with pytest.raises(ValueError, match="finite"):
        QuantumGate("bad", [1, 0, 0, complex(math.inf)])
    with pytest.raises(ValueError, match="unitary"):
        QuantumGate("bad", [2, 0, 0, 1])
    with pytest.raises(ValueError, match="unitary"):
        QuantumGate("bad", [1, 0, 0, 2])
    with pytest.raises(ValueError, match="unitary"):
        QuantumGate("bad", [1, 1, 0, 0])


def test_pauli_x_flips() -> None:
    q = pauli_x().apply(Qubit())
    p0, p1 = q.probabilities()
    assert abs(p0) < 1e-9
    assert abs(p1 - 1.0) < 1e-9


def test_hadamard_creates_superposition() -> None:
    q = hadamard().apply(Qubit())
    p0, p1 = q.probabilities()
    assert abs(p0 - 0.5) < 1e-9
    assert abs(p1 - 0.5) < 1e-9


def test_double_hadamard_is_identity() -> None:
    circuit = QuantumCircuit().add(hadamard()).add(hadamard())
    q = circuit.run()
    p0, p1 = q.probabilities()
    assert abs(p0 - 1.0) < 1e-9
    assert abs(p1) < 1e-9


def test_pauli_z_on_zero() -> None:
    q = pauli_z().apply(Qubit())
    assert abs(q.alpha - 1) < 1e-9


def test_phase_gate() -> None:
    g = phase_gate(math.pi)
    q = g.apply(Qubit(alpha=0, beta=1))
    assert abs(q.beta.real - (-1)) < 1e-6


def test_phase_gate_rejects_non_finite_theta() -> None:
    with pytest.raises(ValueError, match="theta"):
        phase_gate(math.inf)


def test_circuit_len() -> None:
    c = QuantumCircuit().add(hadamard()).add(pauli_x())
    assert len(c) == 2


def test_circuit_run_respects_explicit_initial_state() -> None:
    q = QuantumCircuit().run(Qubit(alpha=0, beta=1))
    assert q.probabilities() == (0.0, 1.0)


def test_simulate_pure_zero() -> None:
    c = QuantumCircuit()
    counts = simulate(c, shots=100, seed=42)
    assert counts.get(0, 0) == 100
    assert counts.get(1, 0) == 0


def test_simulate_hadamard_roughly_even() -> None:
    c = QuantumCircuit().add(hadamard())
    counts = simulate(c, shots=10000, seed=7)
    ratio = counts.get(0, 0) / 10000
    assert 0.45 < ratio < 0.55


def test_measure_seed_is_reproducible() -> None:
    q1 = hadamard().apply(Qubit())
    q2 = hadamard().apply(Qubit())
    assert measure(q1, seed=123) == measure(q2, seed=123)
    assert q1 == q2


def test_simulate_rejects_non_positive_or_non_integer_shots() -> None:
    circuit = QuantumCircuit()
    with pytest.raises(ValueError, match="positive integer"):
        simulate(circuit, shots=0)
    with pytest.raises(ValueError, match="positive integer"):
        simulate(circuit, shots=-1)
    with pytest.raises(ValueError, match="positive integer"):
        simulate(circuit, shots=True)
    with pytest.raises(ValueError, match="positive integer"):
        simulate(circuit, shots=cast(int, 1.5))


def test_qubit_normalize() -> None:
    q = Qubit(alpha=3 + 0j, beta=4 + 0j)
    q.normalize()
    p0, p1 = q.probabilities()
    assert abs(p0 + p1 - 1.0) < 1e-9


def test_qubit_normalize_zero_norm_is_noop() -> None:
    q = Qubit(alpha=0 + 0j, beta=0 + 0j)
    q.normalize()
    assert q.alpha == 0
    assert q.beta == 0


def test_double_pauli_x_is_identity() -> None:
    circuit = QuantumCircuit().add(pauli_x()).add(pauli_x())
    q = circuit.run()
    p0, p1 = q.probabilities()
    assert abs(p0 - 1.0) < 1e-9


def test_pauli_z_on_one() -> None:
    q = pauli_z().apply(pauli_x().apply(Qubit()))
    assert abs(abs(q.beta) - 1) < 1e-9


def test_circuit_empty() -> None:
    c = QuantumCircuit()
    q = c.run()
    p0, p1 = q.probabilities()
    assert abs(p0 - 1.0) < 1e-9


def test_simulate_pauli_x_always_one() -> None:
    c = QuantumCircuit().add(pauli_x())
    counts = simulate(c, shots=100, seed=42)
    assert counts.get(1, 0) == 100


def test_phase_gate_zero() -> None:
    g = phase_gate(0)
    q = g.apply(Qubit(alpha=0, beta=1))
    assert abs(q.beta - 1) < 1e-9


def test_circuit_three_gates() -> None:
    c = QuantumCircuit().add(hadamard()).add(pauli_x()).add(hadamard())
    assert len(c) == 3
    q = c.run()
    p0, p1 = q.probabilities()
    assert abs(p0 - 1.0) < 1e-6
