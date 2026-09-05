"""Run quantum measurements with explicit, reproducible random streams."""

from __future__ import annotations

import random
from collections import Counter

from cds.quantum.circuit import QuantumCircuit, Qubit


def _rng(*, seed: int | None, rng: random.Random | None) -> random.Random:
    if seed is not None and rng is not None:
        raise ValueError("provide either seed or rng, not both")
    return rng if rng is not None else random.Random(seed)


def measure(
    q: Qubit,
    *,
    seed: int | None = None,
    rng: random.Random | None = None,
) -> int:
    """Measure and collapse a qubit using an isolated random generator.

    Supplying ``seed`` reproduces a single measurement.  Supplying ``rng`` lets
    a workflow share one run-level random stream across repeated measurements.
    With neither argument a private system-seeded ``random.Random`` instance is
    created; the process-global RNG is never read or mutated.
    """
    generator = _rng(seed=seed, rng=rng)
    p0, _ = q.probabilities()
    outcome = 0 if generator.random() < p0 else 1

    if outcome == 0:
        q.alpha, q.beta = 1.0 + 0j, 0.0 + 0j
    else:
        q.alpha, q.beta = 0.0 + 0j, 1.0 + 0j
    return outcome


def simulate(
    circuit: QuantumCircuit,
    shots: int = 1000,
    seed: int | None = None,
    *,
    rng: random.Random | None = None,
) -> dict[int, int]:
    """Run a circuit many times and collect reproducible measurement statistics."""
    if shots < 0:
        raise ValueError("shots must be non-negative")
    generator = _rng(seed=seed, rng=rng)
    q = circuit.run()
    p0, _ = q.probabilities()
    results = [0 if generator.random() < p0 else 1 for _ in range(shots)]
    return dict(Counter(results))
