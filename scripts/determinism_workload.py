"""Deterministic CDS workload sampled by the determinism matrix."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source_dir = repo_root / "src"
    if source_dir.is_dir():
        sys.path.insert(0, str(source_dir))

    from cds.fractals import barnsley_fern, sierpinski_triangle
    from cds.knowledge import KnowledgeGraph, Notebook
    from cds.knowledge.retrieval import rank_tfidf, search
    from cds.montecarlo import estimate_pi

    graph = KnowledgeGraph("det-graph")
    for name, description in [
        ("Alpha", "quantum entanglement links particles"),
        ("Beta", "classical mechanics of particles"),
        ("Gamma", "telescope optics and spectroscopy"),
        ("Delta", "quantum tunneling in semiconductors"),
    ]:
        graph.add_concept(name, description=description, tags=["det", name.lower()])
    notebook = Notebook("det-notes")
    notebook.add_note("n1", "Lab log", "quantum tunneling observed today", tags=["spectroscopy"])
    notebook.add_note("n2", "Optics lab", "telescope reading", tags=["optics"])
    payload = {
        "hits": [
            {"source": hit.source, "key": hit.key, "title": hit.title, "score": hit.score}
            for hit in rank_tfidf(graph, notebook, "quantum particles telescope")
        ],
        "search": [
            {
                "concept_name": hit.concept_name,
                "note_id": hit.note_id,
                "score": hit.score,
                "matched_on": hit.matched_on,
            }
            for hit in search(graph, notebook, "quantum")
        ],
        "pi": estimate_pi(n_samples=2_000, seed=20261024).estimate,
        "fern": barnsley_fern(40, seed=20261024),
        "triangle": sierpinski_triangle(40, seed=20261024),
    }
    if not payload["hits"] or not payload["search"]:
        raise SystemExit("workload produced empty retrieval results; it is vacuous")
    artifact_dir = os.environ.get("DETERMINISM_ARTIFACT_DIR")
    if artifact_dir:
        Path(artifact_dir, "payload.json").write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
