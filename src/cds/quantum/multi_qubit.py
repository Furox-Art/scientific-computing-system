"""Multi-qubit quantum register with entanglement support."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from cds.core._numeric import CONCURRENCE_THRESHOLD


def _finite_complex(value: complex) -> bool:
    return math.isfinite(value.real) and math.isfinite(value.imag)


def _validate_qubit_count(n: int) -> None:
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("n_qubits must be a positive integer")


def _validate_qubit_index(reg: QuantumRegister, index: int, *, name: str = "qubit") -> None:
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < reg.n_qubits:
        raise ValueError(f"{name} index must be between 0 and {reg.n_qubits - 1}")


def _validate_distinct_indices(*indices: int) -> None:
    if len(set(indices)) != len(indices):
        raise ValueError("control and target qubit indices must be distinct")


@dataclass
class QuantumRegister:
    """N-qubit state vector with exactly ``2**n_qubits`` finite amplitudes."""

    n_qubits: int
    amplitudes: list[complex]

    def __post_init__(self) -> None:
        _validate_qubit_count(self.n_qubits)
        expected = 1 << self.n_qubits
        if len(self.amplitudes) != expected:
            raise ValueError(f"amplitudes must contain exactly {expected} entries")
        normalized_entries = [complex(amplitude) for amplitude in self.amplitudes]
        if any(not _finite_complex(amplitude) for amplitude in normalized_entries):
            raise ValueError("quantum register amplitudes must be finite")
        self.amplitudes = normalized_entries

    @classmethod
    def zeros(cls, n: int) -> QuantumRegister:
        """Return the all-zero computational basis state for ``n`` qubits."""
        _validate_qubit_count(n)
        amps: list[complex] = [0 + 0j] * (1 << n)
        amps[0] = 1 + 0j
        return cls(n_qubits=n, amplitudes=amps)

    @classmethod
    def from_bits(cls, n: int, value: int) -> QuantumRegister:
        """Return computational basis state ``|value>`` for ``n`` qubits."""
        _validate_qubit_count(n)
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < (1 << n):
            raise ValueError(f"value must be an integer between 0 and {(1 << n) - 1}")
        amps: list[complex] = [0 + 0j] * (1 << n)
        amps[value] = 1 + 0j
        return cls(n_qubits=n, amplitudes=amps)

    @property
    def size(self) -> int:
        """Number of amplitudes in the state vector (= 2**n_qubits)."""
        return len(self.amplitudes)

    def probabilities(self) -> list[float]:
        """Return normalized probabilities for all computational basis states."""
        raw = [abs(amplitude) ** 2 for amplitude in self.amplitudes]
        total = sum(raw)
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("quantum register state must have a finite non-zero norm")
        return [probability / total for probability in raw]

    def normalize(self) -> None:
        """Renormalize the state vector in-place to unit length."""
        squared_norm = sum(abs(amplitude) ** 2 for amplitude in self.amplitudes)
        if not math.isfinite(squared_norm):
            raise ValueError("quantum register norm must be finite")
        norm = math.sqrt(squared_norm)
        if norm > 0:
            self.amplitudes = [amplitude / norm for amplitude in self.amplitudes]

    def measure(self, seed: int | None = None) -> int:
        """Measure the register and collapse its state vector."""
        rng = random.Random(seed)
        probs = self.probabilities()
        r = rng.random()
        cumulative = 0.0
        for index, probability in enumerate(probs):
            cumulative += probability
            if r < cumulative:
                new_amps = [0.0 + 0j] * len(self.amplitudes)
                new_amps[index] = 1.0 + 0j
                self.amplitudes = new_amps
                return index

        # Normalized probabilities can miss 1.0 only by floating-point roundoff.
        final_idx = len(probs) - 1
        new_amps = [0.0 + 0j] * len(self.amplitudes)
        new_amps[final_idx] = 1.0 + 0j
        self.amplitudes = new_amps
        return final_idx

    def measure_shots(
        self,
        shots: int = 1000,
        seed: int | None = None,
    ) -> dict[str, int]:
        """Run multiple measurements and return counts as binary strings."""
        if isinstance(shots, bool) or not isinstance(shots, int) or shots <= 0:
            raise ValueError("shots must be a positive integer")
        rng = random.Random(seed)
        counts: dict[str, int] = {}
        probs = self.probabilities()
        for _ in range(shots):
            r = rng.random()
            cumulative = 0.0
            result = len(probs) - 1
            for index, probability in enumerate(probs):
                cumulative += probability
                if r < cumulative:
                    result = index
                    break
            label = format(result, f"0{self.n_qubits}b")
            counts[label] = counts.get(label, 0) + 1
        return counts

    def expectation(self) -> float:
        """Expected value treating the basis index as the eigenvalue."""
        return sum(index * probability for index, probability in enumerate(self.probabilities()))


def _gate_2x2(
    reg: QuantumRegister,
    target: int,
    matrix: list[complex],
) -> QuantumRegister:
    """Apply a 2x2 gate to a specific qubit in the register."""
    _validate_qubit_index(reg, target, name="target")
    n = reg.n_qubits
    new_amps = list(reg.amplitudes)
    step = 1 << target
    a, b, c, d = matrix

    for i in range(0, 1 << n, step << 1):
        for j in range(step):
            idx0 = i + j
            idx1 = idx0 + step
            v0 = reg.amplitudes[idx0]
            v1 = reg.amplitudes[idx1]
            new_amps[idx0] = a * v0 + b * v1
            new_amps[idx1] = c * v0 + d * v1

    return QuantumRegister(n_qubits=n, amplitudes=new_amps)


def h_gate(reg: QuantumRegister, target: int) -> QuantumRegister:
    """Hadamard on qubit ``target``."""
    s = 1 / math.sqrt(2)
    return _gate_2x2(reg, target, [s, s, s, -s])


def x_gate(reg: QuantumRegister, target: int) -> QuantumRegister:
    """Pauli-X (NOT) on qubit ``target``."""
    return _gate_2x2(reg, target, [0, 1, 1, 0])


def z_gate(reg: QuantumRegister, target: int) -> QuantumRegister:
    """Pauli-Z on qubit ``target``."""
    return _gate_2x2(reg, target, [1, 0, 0, -1])


def y_gate(reg: QuantumRegister, target: int) -> QuantumRegister:
    """Pauli-Y on qubit ``target``."""
    return _gate_2x2(reg, target, [0, -1j, 1j, 0])


def rz_gate(
    reg: QuantumRegister,
    target: int,
    theta: float,
) -> QuantumRegister:
    """Rotation around Z axis."""
    if not math.isfinite(theta):
        raise ValueError("theta must be finite")
    e_neg = complex(math.cos(theta / 2), -math.sin(theta / 2))
    e_pos = complex(math.cos(theta / 2), math.sin(theta / 2))
    return _gate_2x2(reg, target, [e_neg, 0, 0, e_pos])


def cnot(
    reg: QuantumRegister,
    control: int,
    target: int,
) -> QuantumRegister:
    """Controlled-NOT gate."""
    _validate_qubit_index(reg, control, name="control")
    _validate_qubit_index(reg, target, name="target")
    _validate_distinct_indices(control, target)
    n = reg.n_qubits
    new_amps = list(reg.amplitudes)
    for i in range(1 << n):
        if i & (1 << control):
            j = i ^ (1 << target)
            if j > i:
                new_amps[i], new_amps[j] = reg.amplitudes[j], reg.amplitudes[i]
    return QuantumRegister(n_qubits=n, amplitudes=new_amps)


def cz(
    reg: QuantumRegister,
    control: int,
    target: int,
) -> QuantumRegister:
    """Controlled-Z gate."""
    _validate_qubit_index(reg, control, name="control")
    _validate_qubit_index(reg, target, name="target")
    _validate_distinct_indices(control, target)
    n = reg.n_qubits
    new_amps = list(reg.amplitudes)
    for i in range(1 << n):
        if (i & (1 << control)) and (i & (1 << target)):
            new_amps[i] = -reg.amplitudes[i]
    return QuantumRegister(n_qubits=n, amplitudes=new_amps)


def swap(
    reg: QuantumRegister,
    q1: int,
    q2: int,
) -> QuantumRegister:
    """SWAP gate — exchange two qubits."""
    _validate_qubit_index(reg, q1, name="first")
    _validate_qubit_index(reg, q2, name="second")
    if q1 == q2:
        return QuantumRegister(reg.n_qubits, list(reg.amplitudes))
    reg = cnot(reg, q1, q2)
    reg = cnot(reg, q2, q1)
    reg = cnot(reg, q1, q2)
    return reg


def toffoli(
    reg: QuantumRegister,
    c1: int,
    c2: int,
    target: int,
) -> QuantumRegister:
    """Toffoli (CCNOT) gate — 3-qubit controlled-controlled-NOT."""
    _validate_qubit_index(reg, c1, name="first control")
    _validate_qubit_index(reg, c2, name="second control")
    _validate_qubit_index(reg, target, name="target")
    _validate_distinct_indices(c1, c2, target)
    n = reg.n_qubits
    new_amps = list(reg.amplitudes)
    for i in range(1 << n):
        if (i & (1 << c1)) and (i & (1 << c2)):
            j = i ^ (1 << target)
            if j > i:
                new_amps[i], new_amps[j] = (
                    reg.amplitudes[j],
                    reg.amplitudes[i],
                )
    return QuantumRegister(n_qubits=n, amplitudes=new_amps)


# ----- common state preparation -----


def bell_state(which: int = 0) -> QuantumRegister:
    """Create one of the four Bell states, indexed 0 through 3."""
    if isinstance(which, bool) or not isinstance(which, int) or which not in range(4):
        raise ValueError("which must be an integer from 0 through 3")
    reg = QuantumRegister.zeros(2)
    if which in (2, 3):
        reg = x_gate(reg, 1)
    reg = h_gate(reg, 0)
    reg = cnot(reg, 0, 1)
    if which in (1, 3):
        reg = z_gate(reg, 0)
    return reg


def ghz_state(n: int) -> QuantumRegister:
    """GHZ state: (|00...0> + |11...1>) / √2."""
    reg = QuantumRegister.zeros(n)
    reg = h_gate(reg, 0)
    for i in range(1, n):
        reg = cnot(reg, 0, i)
    return reg


def is_entangled(reg: QuantumRegister) -> bool:
    """Check whether a normalized-equivalent two-qubit pure state is entangled."""
    if reg.n_qubits != 2:
        raise ValueError("entanglement check only for 2-qubit states")
    norm_squared = sum(abs(amplitude) ** 2 for amplitude in reg.amplitudes)
    if not math.isfinite(norm_squared) or norm_squared <= 0.0:
        raise ValueError("quantum register state must have a finite non-zero norm")
    a, b, c, d = reg.amplitudes
    concurrence = 2 * abs(a * d - b * c) / norm_squared
    return concurrence > CONCURRENCE_THRESHOLD
