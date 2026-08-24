"""Knowledge organization — concept graphs, research notes, and structured retrieval.

A self-contained subsystem for organising research knowledge:

* :class:`KnowledgeGraph` of named :class:`Concept` nodes connected by typed,
  directed :class:`Relation` edges (``"is-a"``, ``"depends-on"``, …).
* :class:`Notebook` of :class:`Note` records linked to concept names.
* :func:`search` for structured retrieval across both, ranked by relevance.

All of it is pure Python (stdlib :mod:`json` for persistence) and decoupled
from :mod:`cds.graph`, whose dense integer-vertex, untyped-edge model is a
poor fit for named concepts with typed relationships.
"""

from cds.knowledge.graph import (
    Concept,
    KnowledgeGraph,
    Relation,
)
from cds.knowledge.notes import (
    Note,
    Notebook,
)
from cds.knowledge.retrieval import (
    RetrievalHit,
    SearchResult,
    rank_tfidf,
    search,
    search_concepts,
    search_notes,
)

__all__ = [
    "Concept",
    "Relation",
    "KnowledgeGraph",
    "Note",
    "Notebook",
    "SearchResult",
    "search",
    "search_concepts",
    "search_notes",
    "RetrievalHit",
    "rank_tfidf",
]
