"""Basic quantum circuit representation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def _finite_complex(value: complex) -> bool:
    """Return whether both real and imaginary components are finite."""
    return math.isfinite(value.real) and math.isfinite(value.imag)


@dataclass
class Qubit:
    """Single qubit state as (alpha, beta) amplitudes."""

    alpha: complex = 1 + 0j
    beta: complex = 0 + 0j

    def __post_init__(self) -> None:
        if not _finite_complex(complex(self.alpha)) or not _finite_complex(complex(self.beta)):
            raise ValueError("qubit amplitudes must be finite")

    def probabilities(self) -> tuple[float, float]:
        """Return normalized ``(P(|0>), P(|1>))`` measurement probabilities."""
        p0 = abs(self.alpha) ** 2
        p1 = abs(self.beta) ** 2
        total = p0 + p1
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("qubit state must have a finite non-zero norm")
        return (p0 / total, p1 / total)

    def normalize(self) -> None:
        """Renormalize the state amplitudes in-place to unit length."""
        mag = (abs(self.alpha) ** 2) + (abs(self.beta) ** 2)
        norm = math.sqrt(mag)
        if norm > 0:
            self.alpha /= norm
            self.beta /= norm


@dataclass
class QuantumGate:
    """A validated 2x2 unitary gate stored as flat list [a, b, c, d]."""

    name: str
    matrix: list[complex]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("quantum gate name must not be empty")
        if len(self.matrix) != 4:
            raise ValueError("quantum gate matrix must contain exactly four entries")
        entries = tuple(complex(value) for value in self.matrix)
        if any(not _finite_complex(value) for value in entries):
            raise ValueError("quantum gate matrix entries must be finite")
        a, b, c, d = entries
        first_norm = abs(a) ** 2 + abs(c) ** 2
        second_norm = abs(b) ** 2 + abs(d) ** 2
        inner_product = a.conjugate() * b + c.conjugate() * d
        if (
            not math.isclose(first_norm, 1.0, rel_tol=1e-9, abs_tol=1e-12)
            or not math.isclose(second_norm, 1.0, rel_tol=1e-9, abs_tol=1e-12)
            or abs(inner_product) > 1e-9
        ):
            raise ValueError("quantum gate matrix must be unitary")
        self.matrix = list(entries)

    def apply(self, q: Qubit) -> Qubit:
        """Apply this gate to ``q`` and return a new Qubit without mutating it."""
        a, b, c, d = self.matrix
        new_alpha = a * q.alpha + b * q.beta
        new_beta = c * q.alpha + d * q.beta
        return Qubit(alpha=new_alpha, beta=new_beta)


# common gates
def hadamard() -> QuantumGate:
    """Hadamard gate H = (1/sqrt(2)) * [[1, 1], [1, -1]]."""
    s = 1 / math.sqrt(2)
    return QuantumGate("H", [s, s, s, -s])


def pauli_x() -> QuantumGate:
    """Pauli-X (NOT) gate X = [[0, 1], [1, 0]]."""
    return QuantumGate("X", [0, 1, 1, 0])


def pauli_z() -> QuantumGate:
    """Pauli-Z gate Z = [[1, 0], [0, -1]]."""
    return QuantumGate("Z", [1, 0, 0, -1])


def phase_gate(theta: float) -> QuantumGate:
    """Phase rotation gate P(theta) = diag(1, e^{i*theta})."""
    if not math.isfinite(theta):
        raise ValueError("theta must be finite")
    return QuantumGate(f"P({theta:.2f})", [1, 0, 0, complex(math.cos(theta), math.sin(theta))])


@dataclass
class QuantumCircuit:
    """Simple circuit that applies gates sequentially to a single qubit."""

    gates: list[QuantumGate] = field(default_factory=list)

    def add(self, gate: QuantumGate) -> QuantumCircuit:
        """Append a gate to the circuit (returns self for fluent chaining)."""
        self.gates.append(gate)
        return self

    def run(self, initial: Qubit | None = None) -> Qubit:
        """Apply all gates sequentially; starts from ``initial`` or |0>."""
        q = initial if initial is not None else Qubit()
        for gate in self.gates:
            q = gate.apply(q)
        return q

    def __len__(self) -> int:
        """Return the number of gates in the circuit."""
        return len(self.gates)
