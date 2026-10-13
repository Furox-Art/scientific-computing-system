"""H-2: no two seeded tests may share an RNG seed, and seeded calls must
reproduce.

A shared seed is a masking mechanism. When two tests draw the same seed, a
defect in one is invisible in the other: whatever the sampler returned, both
see the *same* wrong value, so the pair either fails together (and looks like
one fault) or passes together (and hides a fault entirely). Before this guard,
``tests/test_montecarlo.py`` alone had twelve test functions drawing seed 42,
and ``tests/test_fractals.py`` had eight drawing seed 0.

Two things are asserted here, both mechanically:

1. **Disjointness** — within one file, no seed literal is passed to more than
   one test function. This is the property that makes failures independent of
   one another.
2. **Reproducibility** — a handful of representative seeded generators return
   bit-identical results for a repeated call under the same seed, so "the seed
   was changed" cannot quietly become "the generator stopped being
   deterministic".
"""

from __future__ import annotations

import ast
import math
from collections.abc import Callable
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# 1. Disjointness
# ---------------------------------------------------------------------------


def _seed_sites(path: Path) -> list[tuple[str, int, int]]:
    """``(function_name, seed_value, line)`` for every ``seed=<int>`` literal."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions: list[ast.FunctionDef] = [
        node for node in tree.body if isinstance(node, ast.FunctionDef)
    ]
    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
        functions.extend(f for f in cls.body if isinstance(f, ast.FunctionDef))

    sites: list[tuple[str, int, int]] = []
    for function in functions:
        for node in ast.walk(function):
            if (
                isinstance(node, ast.keyword)
                and node.arg == "seed"
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, int)
                and not isinstance(node.value.value, bool)
            ):
                sites.append((function.name, node.value.value, node.lineno))
    return sites


def test_no_seed_is_shared_between_two_tests() -> None:
    """Within a file, a seed literal may belong to exactly one test function.

    The audit's H-2: two tests sharing a seed mask each other's failures.
    """
    offenders: list[str] = []
    for path in sorted(TESTS_DIR.glob("*.py")):
        by_seed: dict[int, set[str]] = {}
        for function_name, seed, _line in _seed_sites(path):
            by_seed.setdefault(seed, set()).add(function_name)
        for seed, functions in sorted(by_seed.items()):
            if len(functions) > 1:
                offenders.append(f"{path.name}: seed {seed} -> {sorted(functions)}")

    assert not offenders, "tests share an RNG seed:\n" + "\n".join(offenders)


def test_seed_hygiene_scan_is_not_vacuous() -> None:
    """Guard the guard: the scan must actually find seeded tests.

    Otherwise a broken walker would pass by inspecting nothing.
    """
    scanned = 0
    total_sites = 0
    for path in sorted(TESTS_DIR.glob("*.py")):
        sites = _seed_sites(path)
        if sites:
            scanned += 1
            total_sites += len(sites)
    assert scanned >= 20, f"only {scanned} files carry seed literals; scan is broken"
    assert total_sites >= 100, f"only {total_sites} seed literals found; scan is broken"


def test_seed_values_are_plain_non_negative_integers() -> None:
    """A negative or boolean seed is never what a test means, and hides typos."""
    for path in sorted(TESTS_DIR.glob("*.py")):
        for function_name, seed, line in _seed_sites(path):
            assert seed >= 0, f"{path.name}:{line} uses negative seed {seed} in {function_name}"


# ---------------------------------------------------------------------------
# 2. Reproducibility of the seeded generators the tests depend on
# ---------------------------------------------------------------------------


def test_seeded_generators_reproduce_exactly() -> None:
    """Each seeded generator must return identical output for a repeated seed.

    If the de-duplication ever traded determinism for variety, this fails on a
    value, not on a theory about variety.
    """
    from cds.fractals import barnsley_fern, sierpinski_triangle
    from cds.montecarlo import estimate_pi, mc_integrate
    from cds.montecarlo.markov import metropolis_hastings
    from cds.probability import (
        exponential_sample,
        gaussian_sample,
        poisson_sample,
        uniform_sample,
    )

    def normal_log_pdf(x: float) -> float:
        """Unnormalised standard normal log-density."""
        return -0.5 * x * x

    # (name, callable) — each closes over its own dedicated seed.
    seeded_calls: list[tuple[str, Callable[[], object]]] = [
        ("barnsley_fern", lambda: barnsley_fern(50, seed=7919)),
        ("sierpinski_triangle", lambda: sierpinski_triangle(50, seed=7920)),
        ("estimate_pi", lambda: estimate_pi(n_samples=2_000, seed=7921)),
        ("mc_integrate", lambda: mc_integrate(lambda x: x, 0.0, 1.0, n_samples=500, seed=7922)),
        (
            "metropolis_hastings",
            lambda: metropolis_hastings(normal_log_pdf, 0.0, n_samples=50, seed=7923),
        ),
        ("uniform_sample", lambda: uniform_sample(0, 1, 20, seed=7924)),
        ("gaussian_sample", lambda: gaussian_sample(20, seed=7925)),
        ("exponential_sample", lambda: exponential_sample(20, lam=2.0, seed=7926)),
        ("poisson_sample", lambda: poisson_sample(20, lam=3.0, seed=7927)),
    ]

    for name, call in seeded_calls:
        assert call() == call(), f"{name} is not reproducible under a fixed seed"

    # A different seed must actually give a different stream, otherwise
    # "reproducible" would only mean "constant", which is also a defect.
    a = estimate_pi(n_samples=4_000, seed=1234)
    b = estimate_pi(n_samples=4_000, seed=5678)
    assert a.estimate != b.estimate
    assert math.isfinite(a.estimate) and math.isfinite(b.estimate)
