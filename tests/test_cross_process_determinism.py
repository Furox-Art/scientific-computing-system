"""Cross-process determinism: identical inputs must produce identical bytes.

The workload runs in two fresh interpreters whose only difference is
``PYTHONHASHSEED``. Any ordering that leaks from dict/set iteration into the
retrieval, search or seeded-sampling surface makes the two runs differ, so a
regression here is a hash-order dependence, not a flaky numerical result.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from functools import cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

CHILD = r'''
import json
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
sys.path.insert(0, str(repo_root / "src"))

from cds.fractals import barnsley_fern, sierpinski_triangle
from cds.knowledge import KnowledgeGraph, Notebook
from cds.knowledge.retrieval import rank_tfidf, search
from cds.montecarlo import estimate_pi

kg = KnowledgeGraph("det-graph")
for name, description in [
    ("Alpha", "quantum entanglement links particles"),
    ("Beta", "classical mechanics of particles"),
    ("Gamma", "telescope optics and spectroscopy"),
    ("Delta", "quantum tunneling in semiconductors"),
]:
    kg.add_concept(name, description=description, tags=["det", name.lower()])

notebook = Notebook("det-notes")
notebook.add_note("n1", "Lab log", "quantum tunneling observed today", tags=["spectroscopy"])
notebook.add_note("n2", "Optics lab", "telescope reading", tags=["optics"])

hits = [
    {
        "source": hit.source,
        "key": hit.key,
        "title": hit.title,
        "score": hit.score,
    }
    for hit in rank_tfidf(kg, notebook, "quantum particles telescope")
]
combined = [
    {
        "concept_name": hit.concept_name,
        "note_id": hit.note_id,
        "score": hit.score,
        "matched_on": hit.matched_on,
    }
    for hit in search(kg, notebook, "quantum")
]
payload = {
    "hits": hits,
    "search": combined,
    "pi": estimate_pi(n_samples=2_000, seed=20261024).estimate,
    "fern": barnsley_fern(40, seed=20261024),
    "triangle": sierpinski_triangle(40, seed=20261024),
}
print(json.dumps(payload, ensure_ascii=False))
'''


@cache
def _run_with_hash_seed(seed: str) -> str:
    env = {
        **os.environ,
        "PYTHONHASHSEED": seed,
        "PYTHONIOENCODING": "utf-8",
    }
    completed = subprocess.run(
        [sys.executable, "-c", CHILD, str(REPO_ROOT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def test_workload_is_byte_identical_across_hash_seeds() -> None:
    first = _run_with_hash_seed("0")
    second = _run_with_hash_seed("12345")
    assert first.strip()
    assert first.strip() == second.strip()


def test_workload_exercises_non_empty_results() -> None:
    payload = json.loads(_run_with_hash_seed("0"))
    assert payload["hits"], "retrieval produced no hits; the workload is vacuous"
    assert payload["search"], "combined search produced no hits"
    assert payload["pi"] > 0.0
    assert len(payload["fern"]) == 40
    assert len(payload["triangle"]) == 40
