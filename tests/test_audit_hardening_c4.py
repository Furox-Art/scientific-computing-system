"""C-4: ``estimate_pi`` default path must sample in this process.

The audited defect: with its default arguments ``estimate_pi`` fanned the
sample out to a :class:`ProcessPoolExecutor`, so every call paid for a fresh
interpreter (Windows ``spawn``), a module re-import and a pickle round-trip
before doing ≈14 ms of sampling. The wall-clock regression below is relative
to an equivalent serial loop in the same process, so it holds on any host.

The stronger, machine-independent guards are the *call-count* bound (the
sampler must draw exactly two random values per sample — a per-sample
dispatch schedule draws many more) and the *value* bound (the estimator's
hit count must equal a hand-rolled stream over the same seed).
"""

from __future__ import annotations

import math
import random
import time

import pytest

from cds.montecarlo import estimate_pi
from cds.montecarlo.methods import _count_hits, _pi_worker

_SERIAL_REFERENCE_SEED = 20_260_613


def _serial_hit_count(seed: int, samples: int) -> int:
    """Independent oracle: two draws per sample from one seeded stream."""
    rng = random.Random(seed)
    draw = rng.random
    inside = 0
    for _ in range(samples):
        x = draw()
        y = draw()
        if x * x + y * y <= 1.0:
            inside += 1
    return inside


class TestEstimatePiStreamContract:
    """The documented RNG stream is the contract, and it is now deterministic."""

    def test_hit_count_matches_a_serial_reference_stream(self) -> None:
        """The estimate must come from one in-process ``Random(seed)`` stream.

        Under the process pool the seeded result depended on the host's core
        count, because the chunk boundaries decided which draws belonged to
        which stream. The estimator now asserts what it actually computes.
        """
        samples = 4_000
        result = estimate_pi(n_samples=samples, seed=_SERIAL_REFERENCE_SEED)
        inside = _serial_hit_count(_SERIAL_REFERENCE_SEED, samples)
        assert result.samples == samples
        assert result.estimate == pytest.approx(4.0 * inside / samples)
        # Not just close — the same stream produces the same count exactly.
        assert result.estimate == pytest.approx(
            4.0 * _count_hits(random.Random(_SERIAL_REFERENCE_SEED), samples) / samples
        )

    def test_result_is_reproducible_across_calls(self) -> None:
        first = estimate_pi(n_samples=2_000, seed=1234)
        second = estimate_pi(n_samples=2_000, seed=1234)
        assert first == second
        assert first.std_error == second.std_error

    def test_estimate_is_a_usable_pi_approximation(self) -> None:
        """Value bounds: the 100k-sample estimate and its standard error.

        For a quarter-circle hit rate of ``p = π/4`` the binomial standard
        error is ``4·√(p(1-p)/n)``, which the estimator must reproduce
        independently of the realized hit count within sampling noise.
        """
        samples = 100_000
        result = estimate_pi(n_samples=samples, seed=_SERIAL_REFERENCE_SEED)
        assert math.isfinite(result.estimate)
        assert result.std_error > 0.0
        assert abs(result.estimate - math.pi) < 5.0 * result.std_error
        expected_se = 4.0 * math.sqrt((math.pi / 4.0) * (1.0 - math.pi / 4.0) / samples)
        assert result.std_error == pytest.approx(expected_se, rel=0.05)

    def test_pi_worker_still_counts_a_chunk(self) -> None:
        """The single-process worker keeps its ``(samples, seed)`` contract."""
        assert _pi_worker((1000, 42)) == _serial_hit_count(42, 1000)
        assert _pi_worker((500, 42)) == _serial_hit_count(42, 500)


class TestEstimatePiCallCountBound:
    """Exactly two RNG draws per sample — no per-sample dispatch schedule."""

    def test_default_path_draws_two_values_per_sample(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        samples = 500
        draws = 0
        original = random.Random.random

        def counting(self: random.Random) -> float:
            nonlocal draws
            draws += 1
            return original(self)

        monkeypatch.setattr(random.Random, "random", counting, raising=True)
        try:
            result = estimate_pi(n_samples=samples, seed=7)
        finally:
            monkeypatch.undo()

        assert result.samples == samples
        # A per-sample dispatch or a re-sharded stream would draw more.
        assert draws == 2 * samples

    def test_no_process_pool_is_created(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The dispatch machinery must not even be imported by the call path.

        If a future change reintroduces a pool, this fails before any timing
        assertion could flake.
        """
        import sys

        with pytest.MonkeyPatch.context() as patch:
            created: list[object] = []

            def boom(*args: object, **kwargs: object) -> object:
                created.append(args)
                raise AssertionError("estimate_pi must not create a process pool")

            for module_name in ("concurrent.futures.ProcessPoolExecutor",):
                if module_name in sys.modules:
                    patch.setattr(sys.modules[module_name], "ProcessPoolExecutor", boom)
            estimate_pi(n_samples=1_000, seed=3)
        assert created == []


class TestEstimatePiWallClock:
    """Relative wall-clock bound against an equivalent serial loop."""

    def test_default_path_is_not_slower_than_serial_sampling(self) -> None:
        """Relative bound: no more than 3× a plain Python loop plus 50 ms.

        The dispatch overhead this replaces was ≈1.4 s against ≈0.014 s of
        sampling — a ~100× multiplier. A 3× + 50 ms envelope keeps the test
        green on slow CI while still failing hard on any per-sample dispatch.
        """
        samples = 20_000

        reference_start = time.perf_counter()
        expected_inside = _serial_hit_count(_SERIAL_REFERENCE_SEED, samples)
        reference_seconds = time.perf_counter() - reference_start

        call_start = time.perf_counter()
        result = estimate_pi(n_samples=samples, seed=_SERIAL_REFERENCE_SEED)
        call_seconds = time.perf_counter() - call_start

        assert result.estimate == pytest.approx(4.0 * expected_inside / samples)
        assert call_seconds <= 3.0 * reference_seconds + 0.05, (
            f"estimate_pi({samples}) took {call_seconds:.3f}s against a "
            f"{reference_seconds:.3f}s serial reference "
            f"({call_seconds / max(reference_seconds, 1e-9):.1f}x)"
        )

    def test_validation_is_unchanged(self) -> None:
        with pytest.raises(ValueError, match="n_samples"):
            estimate_pi(n_samples=0)
        with pytest.raises(ValueError, match="n_samples"):
            estimate_pi(n_samples=True)
        assert estimate_pi(n_samples=1, seed=5).std_error == 0.0
