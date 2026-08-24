"""TF-IDF ranked retrieval over a knowledge graph and its notes.

Given a free-text ``query``, :func:`rank_tfidf` scores every concept and
every note with a classic TF-IDF vector-space model and returns the hits
ranked by relevance:

* **Corpus** — one document per concept (its name plus its description
  text) and one per note (title + body + tags). Documents follow mapping
  insertion order: concepts first (in :attr:`KnowledgeGraph.concepts`
  order), then notes (in :attr:`Notebook.notes` order). Note tags are
  indexed; concept tags are not.
* **tf** — raw term count within the document.
* **idf** — ``ln(N / df)`` where ``N`` is the number of documents in the
  corpus and ``df`` the document frequency of the term across that same
  corpus. A term appearing in every document therefore has ``idf == 0``
  and can never promote a document on its own.
* **Scoring** — ``sum(tf(doc, t) * idf(t))`` over the query's token
  occurrences present in the document, **normalized by
  ``sqrt(number of query-token occurrences matched)``** so that long
  multi-term queries do not automatically dominate focused single-term
  ones. (Normalizing by document length instead would bias toward short
  documents; that alternative was considered and deliberately not chosen.)
* **Ordering** — descending score; ties keep corpus insertion order
  (Python's sort is stable). Only hits with ``score > 0`` are returned;
  a query that matches nothing yields an empty list.

Tokenization (:func:`_tokenize`) lowercases the text and splits it on
runs of non-alphanumeric characters.

The older substring-based structured search (:func:`search`,
:class:`SearchResult`, :func:`search_concepts`, :func:`search_notes`) is
kept unchanged further down for backwards compatibility; it is exercised
by ``tests/test_knowledge.py``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from cds.knowledge.graph import KnowledgeGraph
from cds.knowledge.notes import Notebook

_TOKEN_SEPARATOR = re.compile(r"[^a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase ``text`` and split it into alphanumeric tokens.

    Tokens are maximal runs of ASCII letters and digits; every other
    character (punctuation, whitespace, underscores, …) acts as a
    separator. Empty tokens produced by leading/trailing separators are
    dropped.

    Args:
        text: arbitrary input text.

    Returns:
        the lowercase token list (possibly empty).
    """
    return [token for token in _TOKEN_SEPARATOR.split(text.lower()) if token]


@dataclass(frozen=True)
class RetrievalHit:
    """A single ranked TF-IDF retrieval hit.

    Attributes:
        source: kind of the matched document — ``"concept"`` or ``"note"``.
        key: the concept name or note id identifying the hit inside its store.
        title: human-readable label (the concept name, or the note title).
        score: normalized TF-IDF relevance; higher is better, always positive.
    """

    source: str
    key: str
    title: str
    score: float


@dataclass(frozen=True)
class _Document:
    """Internal corpus entry: one scored unit with precomputed term counts.

    Attributes:
        source: ``"concept"`` or ``"note"``.
        key: concept name or note id.
        title: display title for the eventual hit.
        tf: raw term-frequency map for the document's indexed text.
    """

    source: str
    key: str
    title: str
    tf: dict[str, int]


def _build_corpus(kg: KnowledgeGraph, notebook: Notebook) -> list[_Document]:
    """Collect the TF-IDF corpus: one document per concept, then per note.

    Concepts contribute their name plus description text; notes contribute
    title, body, and tags. Insertion order of both mappings is preserved
    because it doubles as the tie-breaking order for ranking.
    """
    documents: list[_Document] = []
    for name, concept in kg.concepts.items():
        text = " ".join(filter(None, (name, concept.description)))
        documents.append(_Document(source="concept", key=name, title=name, tf=_count_terms(text)))
    for note_id, note in notebook.notes.items():
        text = " ".join([note.title, note.body, *note.tags])
        documents.append(
            _Document(source="note", key=note_id, title=note.title, tf=_count_terms(text))
        )
    return documents


def _count_terms(text: str) -> dict[str, int]:
    """Return raw term frequencies (``tf``) for the tokens of ``text``."""
    tf: dict[str, int] = {}
    for token in _tokenize(text):
        tf[token] = tf.get(token, 0) + 1
    return tf


def rank_tfidf(kg: KnowledgeGraph, notebook: Notebook, query: str) -> list[RetrievalHit]:
    """Rank every concept and note against ``query`` using TF-IDF.

    Each concept becomes one document (name + description) and each note
    another (title + body + tags). With ``N`` documents in the corpus and
    ``df(t)`` the number of documents containing term ``t``, a match on
    term ``t`` contributes ``tf(doc, t) * ln(N / df(t))``; the document's
    raw weight is the sum over matched query-token occurrences, divided by
    ``sqrt(number of query-token occurrences matched)`` (the documented
    normalization; see the module docstring). Terms occurring in every
    document have ``idf == 0`` and contribute nothing.

    Args:
        kg: the :class:`KnowledgeGraph` whose concepts form the concept half
            of the corpus.
        notebook: the :class:`Notebook` whose notes form the other half.
        query: free-text query; matched case-insensitively after
            tokenization.

    Returns:
        hits with ``score > 0``, sorted by descending score with ties
        broken by corpus insertion order (concepts first, then notes).
        Empty list when the corpus is empty or nothing matches.

    Raises:
        ValueError: if ``query`` contains no alphanumeric characters
            (i.e. it is empty after tokenization).
    """
    query_tokens = _tokenize(query)
    if not query_tokens:
        raise ValueError("empty query after tokenization")
    documents = _build_corpus(kg, notebook)
    total_docs = len(documents)
    if total_docs == 0:
        return []

    df: dict[str, int] = {}
    for document in documents:
        for term in document.tf:
            df[term] = df.get(term, 0) + 1

    hits: list[RetrievalHit] = []
    for document in documents:
        weight = 0.0
        matched = 0
        for token in query_tokens:
            term_tf = document.tf.get(token)
            if term_tf is None:
                continue
            matched += 1
            weight += term_tf * math.log(total_docs / df[token])
        if matched == 0:
            continue
        score = weight / math.sqrt(matched)
        if score > 0.0:
            hits.append(
                RetrievalHit(
                    source=document.source, key=document.key, title=document.title, score=score
                )
            )
    # Stable sort: equal scores keep corpus (insertion) order.
    hits.sort(key=lambda hit: -hit.score)
    return hits


# ---------------------------------------------------------------------- #
# Backwards-compatible structured search (substring-based)
# ---------------------------------------------------------------------- #
# Kept unchanged so `cds.knowledge.__init__` and existing callers keep
# working; the TF-IDF ranker above is the richer replacement.

# Score constants — exposed as module attributes for clarity in assertions.
NAME_TAG_SCORE = 1.0
SUBSTRING_SCORE = 0.5


@dataclass
class SearchResult:
    """A single ranked retrieval hit.

    Attributes:
        concept_name: the matched concept name, if the hit is a concept;
            ``None`` otherwise.
        note_id: the matched note id, if the hit is a note; ``None`` otherwise.
        score: relevance in ``[0, 1]`` — higher is better.
        matched_on: short label of the field that matched
            (e.g. ``"name"``, ``"description"``, ``"title"``).
    """

    concept_name: str | None
    note_id: str | None
    score: float
    matched_on: str


def search_concepts(
    graph: KnowledgeGraph,
    query: str,
    tag: str | None = None,
) -> list[SearchResult]:
    """Find concepts in ``graph`` matching ``query``.

    A concept matches if its name matches the query exactly (score 1.0) or
    its name or description contains the query as a substring (score 0.5).
    When ``tag`` is given, only concepts carrying that tag are considered.

    Args:
        graph: the :class:`KnowledgeGraph` to search.
        query: case-insensitive search text.
        tag: optional tag filter; ``None`` disables filtering.

    Returns:
        ranked :class:`SearchResult` list (best first, ties alphabetical).
    """
    needle = query.casefold()
    results: list[SearchResult] = []
    for name in sorted(graph.concepts):
        concept = graph.concepts[name]
        if tag is not None and tag not in concept.tags:
            continue
        name_folded = name.casefold()
        if name_folded == needle:
            results.append(
                SearchResult(
                    concept_name=name, note_id=None, score=NAME_TAG_SCORE, matched_on="name"
                )
            )
        elif needle in name_folded:
            results.append(
                SearchResult(
                    concept_name=name, note_id=None, score=SUBSTRING_SCORE, matched_on="name"
                )
            )
        elif concept.description is not None and needle in concept.description.casefold():
            results.append(
                SearchResult(
                    concept_name=name, note_id=None, score=SUBSTRING_SCORE, matched_on="description"
                )
            )
    results.sort(key=lambda r: (-r.score, r.concept_name or ""))
    return results


def search_notes(
    notebook: Notebook,
    query: str,
    tag: str | None = None,
) -> list[SearchResult]:
    """Find notes in ``notebook`` matching ``query``.

    A note matches if its title matches exactly (score 1.0) or its title or
    body contains the query as a substring (score 0.5). When ``tag`` is
    given, only notes carrying that tag are considered.

    Args:
        notebook: the :class:`Notebook` to search.
        query: case-insensitive search text.
        tag: optional tag filter; ``None`` disables filtering.

    Returns:
        ranked :class:`SearchResult` list (best first, ties alphabetical).
    """
    needle = query.casefold()
    results: list[SearchResult] = []
    for note_id in sorted(notebook.notes):
        note = notebook.notes[note_id]
        if tag is not None and tag not in note.tags:
            continue
        title_folded = note.title.casefold()
        if title_folded == needle:
            results.append(
                SearchResult(
                    concept_name=None, note_id=note_id, score=NAME_TAG_SCORE, matched_on="title"
                )
            )
        elif needle in title_folded:
            results.append(
                SearchResult(
                    concept_name=None, note_id=note_id, score=SUBSTRING_SCORE, matched_on="title"
                )
            )
        elif needle in note.body.casefold():
            results.append(
                SearchResult(
                    concept_name=None, note_id=note_id, score=SUBSTRING_SCORE, matched_on="body"
                )
            )
    results.sort(key=lambda r: (-r.score, r.note_id or ""))
    return results


def search(
    graph: KnowledgeGraph,
    notebook: Notebook,
    query: str,
    tag: str | None = None,
) -> list[SearchResult]:
    """Combined ranked search over both a graph's concepts and a notebook's notes.

    Results from :func:`search_concepts` and :func:`search_notes` are merged
    and re-ranked by score (desc) then by identifier (asc).

    Args:
        graph: the :class:`KnowledgeGraph` whose concepts to search.
        notebook: the :class:`Notebook` whose notes to search.
        query: case-insensitive search text.
        tag: optional tag filter applied to both concepts and notes.

    Returns:
        ranked :class:`SearchResult` list (best first, ties alphabetical).
    """
    combined = search_concepts(graph, query, tag) + search_notes(notebook, query, tag)
    combined.sort(key=lambda r: (-r.score, r.concept_name or r.note_id or ""))
    return combined
