"""Tests for quantum module."""

import math
import random

import pytest

from cds.quantum.circuit import (
    QuantumCircuit,
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


def test_circuit_len() -> None:
    c = QuantumCircuit().add(hadamard()).add(pauli_x())
    assert len(c) == 2


def test_simulate_pure_zero() -> None:
    counts = simulate(QuantumCircuit(), shots=100, seed=42)
    assert counts.get(0, 0) == 100
    assert counts.get(1, 0) == 0


def test_simulate_hadamard_roughly_even() -> None:
    c = QuantumCircuit().add(hadamard())
    counts = simulate(c, shots=10000, seed=7)
    ratio = counts.get(0, 0) / 10000
    assert 0.45 < ratio < 0.55


def test_simulate_shared_rng_is_reproducible() -> None:
    c = QuantumCircuit().add(hadamard())
    assert simulate(c, shots=100, rng=random.Random(9)) == simulate(
        c, shots=100, rng=random.Random(9)
    )


def test_measure_seed_is_reproducible_and_collapses() -> None:
    first = hadamard().apply(Qubit())
    second = hadamard().apply(Qubit())
    outcome = measure(first, seed=123)
    assert outcome == measure(second, seed=123)
    assert first.probabilities() == ((1.0, 0.0) if outcome == 0 else (0.0, 1.0))


def test_measure_does_not_consume_global_rng() -> None:
    random.seed(55)
    expected = random.random()
    random.seed(55)
    measure(hadamard().apply(Qubit()), seed=4)
    assert random.random() == expected


def test_measure_and_simulate_reject_seed_plus_rng() -> None:
    with pytest.raises(ValueError, match="either seed or rng"):
        measure(Qubit(), seed=1, rng=random.Random(1))
    with pytest.raises(ValueError, match="either seed or rng"):
        simulate(QuantumCircuit(), seed=1, rng=random.Random(1))


def test_simulate_rejects_negative_shots_and_allows_zero() -> None:
    with pytest.raises(ValueError, match="shots"):
        simulate(QuantumCircuit(), shots=-1)
    assert simulate(QuantumCircuit(), shots=0, seed=1) == {}


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
    assert abs(p1) < 1e-9


def test_pauli_z_on_one() -> None:
    q = pauli_z().apply(pauli_x().apply(Qubit()))
    assert abs(abs(q.beta) - 1) < 1e-9


def test_circuit_empty() -> None:
    q = QuantumCircuit().run()
    p0, p1 = q.probabilities()
    assert abs(p0 - 1.0) < 1e-9
    assert abs(p1) < 1e-9


def test_simulate_pauli_x_always_one() -> None:
    c = QuantumCircuit().add(pauli_x())
    counts = simulate(c, shots=100, seed=42)
    assert counts.get(1, 0) == 100


def test_phase_gate_zero() -> None:
    q = phase_gate(0).apply(Qubit(alpha=0, beta=1))
    assert abs(q.beta - 1) < 1e-9


def test_circuit_three_gates() -> None:
    c = QuantumCircuit().add(hadamard()).add(pauli_x()).add(hadamard())
    assert len(c) == 3
    q = c.run()
    p0, p1 = q.probabilities()
    assert abs(p0 - 1.0) < 1e-6
