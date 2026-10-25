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

    from cds.core.models import Domain
    from cds.fractals import barnsley_fern, sierpinski_triangle
    from cds.hypothesis.generator import generate_hypotheses
    from cds.knowledge import KnowledgeGraph, Notebook
    from cds.knowledge.retrieval import rank_tfidf, search
    from cds.montecarlo import estimate_pi
    from cds.plot.charts import plot_series, save_figure

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

    def content(rows: list[dict]) -> list[dict]:
        # created_at records when the run happened, not what was generated.
        return [{key: value for key, value in row.items() if key != "created_at"} for row in rows]

    hypotheses = content(
        [h.to_dict() for h in generate_hypotheses("quantum decoherence", Domain.PHYSICS, n=3)]
    )
    repeated = content(
        [h.to_dict() for h in generate_hypotheses("quantum decoherence", Domain.PHYSICS, n=3)]
    )
    if hypotheses != repeated:
        raise SystemExit("hypothesis output differs between runs in one process")
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
        "hypotheses": hypotheses,
    }
    if not payload["hits"] or not payload["search"] or not payload["hypotheses"]:
        raise SystemExit("workload produced empty results; it is vacuous")
    artifact_dir = os.environ.get("DETERMINISM_ARTIFACT_DIR")
    if artifact_dir:
        artifacts = Path(artifact_dir)
        for suffix in (".pdf", ".svg"):
            save_figure(
                plot_series(
                    [3.0, 1.0, 2.0, 5.0, 4.0, 2.5],
                    title="Determinism",
                    xlabel="x",
                    ylabel="y",
                ),
                str(artifacts / f"series{suffix}"),
                close=True,
            )
        artifacts.joinpath("payload.json").write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
            newline="\n",
        )
        for name in ("payload.json", "series.pdf", "series.svg"):
            if artifacts.joinpath(name).stat().st_size == 0:
                raise SystemExit(f"artifact {name} is empty")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
