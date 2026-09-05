"""Run a circuit many times and collect measurement statistics."""

from __future__ import annotations

import random
from collections import Counter

from cds.quantum.circuit import QuantumCircuit, Qubit


def measure(
    q: Qubit,
    *,
    seed: int | None = None,
    rng: random.Random | None = None,
) -> int:
    """Measure a qubit and collapse its state vector.

    A caller may provide either a fixed ``seed`` for a one-off reproducible
    measurement or a private :class:`random.Random` instance for a reproducible
    stream of repeated measurements. The module-global RNG is never touched.
    """
    if seed is not None and rng is not None:
        raise ValueError("provide either seed or rng, not both")
    sampler = rng if rng is not None else random.Random(seed)
    p0, _ = q.probabilities()
    outcome = 0 if sampler.random() < p0 else 1

    # Quantum State Collapse
    if outcome == 0:
        q.alpha, q.beta = 1.0 + 0j, 0.0 + 0j
    else:
        q.alpha, q.beta = 0.0 + 0j, 1.0 + 0j

    return outcome


def simulate(circuit: QuantumCircuit, shots: int = 1000, seed: int | None = None) -> dict[int, int]:
    """Run a circuit many times and collect measurement statistics.

    The final state vector is computed once and then sampled with a private RNG.
    ``shots`` must be a positive integer; invalid shot counts fail explicitly
    rather than silently returning an empty distribution.
    """
    if isinstance(shots, bool) or not isinstance(shots, int) or shots <= 0:
        raise ValueError("shots must be a positive integer")
    rng = random.Random(seed)

    # Compute the final quantum state exactly once.
    q = circuit.run()
    p0, _ = q.probabilities()

    results = [0 if rng.random() < p0 else 1 for _ in range(shots)]
    return dict(Counter(results))
