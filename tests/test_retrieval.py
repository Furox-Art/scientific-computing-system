"""Tests for TF-IDF ranked retrieval in :mod:`cds.knowledge.retrieval`.

Covers the :func:`rank_tfidf` vector-space ranker (tokenization, idf
weighting, tag indexing, normalization, tie-breaking, determinism, and the
empty-input branches) plus smoke coverage of the legacy substring search
API kept for backwards compatibility.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from cds.knowledge import KnowledgeGraph, Notebook
from cds.knowledge.retrieval import (
    NAME_TAG_SCORE,
    SUBSTRING_SCORE,
    RetrievalHit,
    SearchResult,
    _tokenize,
    rank_tfidf,
    search,
    search_concepts,
    search_notes,
)


def _make_graph() -> KnowledgeGraph:
    """Small graph: two physics concepts plus one description-less concept."""
    kg = KnowledgeGraph("test-graph")
    kg.add_concept("Alpha", description="quantum entanglement links particles")
    kg.add_concept("Beta", description="classical mechanics of particles")
    kg.add_concept("Bare")
    return kg


def _make_notebook() -> Notebook:
    """Notebook with one quantum note and one optics note."""
    nb = Notebook("test-notes")
    nb.add_note("n1", "Lab log", "quantum tunneling observed today", tags=["spectroscopy"])
    nb.add_note("n2", "Optics lab", "telescope reading")
    return nb


# ---------------------------------------------------------------------- #
# _tokenize
# ---------------------------------------------------------------------- #


def test_tokenize_lowercases_and_splits_on_non_alphanumeric() -> None:
    assert _tokenize("Hello, World_42  FOo-BAR!") == ["hello", "world", "42", "foo", "bar"]


@pytest.mark.parametrize("text", ["", "!!!", "  ---  "])
def test_tokenize_returns_empty_for_separator_only_text(text: str) -> None:
    assert _tokenize(text) == []


# ---------------------------------------------------------------------- #
# rank_tfidf — ranking behavior
# ---------------------------------------------------------------------- #


def test_distinctive_term_surfaces_expected_document_top() -> None:
    hits = rank_tfidf(_make_graph(), _make_notebook(), "entanglement")
    assert [hit.key for hit in hits] == ["Alpha"]
    assert hits[0].source == "concept"
    assert hits[0].title == "Alpha"
    assert hits[0].score > 0


def test_query_matching_concept_and_note_orders_by_insertion_on_tie() -> None:
    hits = rank_tfidf(_make_graph(), _make_notebook(), "quantum")
    assert [(hit.source, hit.key) for hit in hits] == [("concept", "Alpha"), ("note", "n1")]


def test_rare_term_outranks_frequent_common_term() -> None:
    kg = KnowledgeGraph("idf-graph")
    kg.add_concept("Common", description="noise noise noise filler")
    kg.add_concept("Rare", description="signal")
    kg.add_concept("Extra", description="noise")
    nb = Notebook("idf-notes")
    nb.add_note("n1", "log", "noise")

    hits = rank_tfidf(kg, nb, "noise signal")

    assert [hit.key for hit in hits] == ["Rare", "Common", "Extra", "n1"]
    rare_score = math.log(4 / 1)
    common_score = 3 * math.log(4 / 3)
    assert hits[0].score == pytest.approx(rare_score)
    assert hits[1].score == pytest.approx(common_score)
    assert rare_score > common_score


def test_note_tags_are_indexed_but_concept_tags_are_not() -> None:
    nb = _make_notebook()
    by_tag = rank_tfidf(_make_graph(), nb, "spectroscopy")
    assert [(hit.source, hit.key) for hit in by_tag] == [("note", "n1")]

    kg = KnowledgeGraph("tagged-graph")
    kg.add_concept("Optics", description="light behaviour", tags=["photonics"])
    assert rank_tfidf(kg, nb, "photonics") == []


def test_insertion_order_breaks_score_ties() -> None:
    kg = KnowledgeGraph("tie-graph")
    kg.add_concept("Alpha", description="zebra plains")
    kg.add_concept("Beta", description="zebra plains")
    kg.add_concept("Gamma", description="nothing here")
    nb = Notebook("tie-notes")
    nb.add_note("n1", "Zebra watch", "giraffe")

    hits = rank_tfidf(kg, nb, "zebra")

    assert [(hit.source, hit.key) for hit in hits] == [
        ("concept", "Alpha"),
        ("concept", "Beta"),
        ("note", "n1"),
    ]
    assert len({hit.score for hit in hits}) == 1


def test_scores_match_closed_form_with_sqrt_matched_normalization() -> None:
    kg = KnowledgeGraph("closed-form")
    kg.add_concept("Pairs", description="alpha beta")
    kg.add_concept("Solo", description="alpha")

    hits = rank_tfidf(kg, Notebook("empty"), "alpha beta")

    # df(alpha) = 2 -> idf 0; df(beta) = 1 -> idf ln(2); Pairs matches both
    # query tokens, so its score is ln(2) / sqrt(2). Solo's only match has
    # idf 0, so it is filtered out by the score > 0 rule.
    assert [hit.key for hit in hits] == ["Pairs"]
    assert hits[0].score == pytest.approx(math.log(2) / math.sqrt(2))


def test_term_in_every_document_has_zero_idf_and_yields_nothing() -> None:
    kg = KnowledgeGraph("ubiquitous")
    kg.add_concept("One", description="gravity attracts mass")
    kg.add_concept("Two", description="gravity bends light")
    nb = Notebook("ubiquitous-notes")
    nb.add_note("n1", "Gravity diary", "gravity again")

    assert rank_tfidf(kg, nb, "gravity") == []


def test_repeated_calls_are_deterministic() -> None:
    kg, nb = _make_graph(), _make_notebook()
    first = rank_tfidf(kg, nb, "quantum particles telescope")
    assert first == rank_tfidf(kg, nb, "quantum particles telescope")
    assert first == rank_tfidf(kg, nb, "quantum particles telescope")


# ---------------------------------------------------------------------- #
# rank_tfidf — empty / invalid inputs
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize("query", ["", "   ", "!!! ---"])
def test_empty_query_after_tokenization_raises(query: str) -> None:
    with pytest.raises(ValueError, match="empty query after tokenization"):
        rank_tfidf(_make_graph(), _make_notebook(), query)


def test_empty_corpus_returns_no_hits() -> None:
    assert rank_tfidf(KnowledgeGraph("g"), Notebook("n"), "anything") == []


def test_query_without_matches_returns_empty_list() -> None:
    assert rank_tfidf(_make_graph(), _make_notebook(), "zzzqqq") == []


# ---------------------------------------------------------------------- #
# RetrievalHit dataclass
# ---------------------------------------------------------------------- #


def test_retrieval_hit_fields_equality_and_immutability() -> None:
    hit = RetrievalHit(source="note", key="n1", title="Lab log", score=0.5)
    assert hit == RetrievalHit(source="note", key="n1", title="Lab log", score=0.5)
    assert hit != RetrievalHit(source="note", key="n1", title="Lab log", score=0.6)
    with pytest.raises(dataclasses.FrozenInstanceError):
        hit.score = 1.0  # type: ignore[misc]


# ---------------------------------------------------------------------- #
# Legacy structured search (backwards compatibility)
# ---------------------------------------------------------------------- #


def _legacy_graph() -> KnowledgeGraph:
    kg = KnowledgeGraph("legacy")
    kg.add_concept("Gravity", description="attractive force between masses", tags=["physics"])
    kg.add_concept("Energy", description="capacity to do work", tags=["abstract"])
    kg.add_concept("Bare")
    return kg


def _legacy_notebook() -> Notebook:
    nb = Notebook("legacy-notes")
    nb.add_note("n1", "Gravity observations", "apple data", tags=["physics"])
    nb.add_note("n2", "Optics lab", "telescope reading")
    return nb


def test_search_concepts_exact_substring_description_and_filters() -> None:
    kg = _legacy_graph()
    exact = search_concepts(kg, "Gravity")
    assert (exact[0].concept_name, exact[0].score, exact[0].matched_on) == (
        "Gravity",
        NAME_TAG_SCORE,
        "name",
    )
    substring = search_concepts(kg, "Grav")
    assert (substring[0].concept_name, substring[0].score) == ("Gravity", SUBSTRING_SCORE)
    description = search_concepts(kg, "masses")
    assert (description[0].concept_name, description[0].matched_on) == (
        "Gravity",
        "description",
    )
    tagged_in = search_concepts(kg, "Energy", tag="abstract")
    assert [r.concept_name for r in tagged_in] == ["Energy"]
    assert search_concepts(kg, "Energy", tag="physics") == []
    assert search_concepts(kg, "nothing", tag=None) == []
    assert search_concepts(kg, "zzz") == []


def test_search_notes_title_body_and_tag_filtering() -> None:
    nb = _legacy_notebook()
    exact = search_notes(nb, "Gravity observations")
    assert (exact[0].note_id, exact[0].score, exact[0].matched_on) == (
        "n1",
        NAME_TAG_SCORE,
        "title",
    )
    substring = search_notes(nb, "Grav")
    assert (substring[0].note_id, substring[0].score) == ("n1", SUBSTRING_SCORE)
    body = search_notes(nb, "telescope")
    assert (body[0].note_id, body[0].matched_on) == ("n2", "body")
    tagged_in = search_notes(nb, "Gravity", tag="physics")
    assert [r.note_id for r in tagged_in] == ["n1"]
    assert search_notes(nb, "telescope", tag="physics") == []
    assert search_notes(nb, "Gravity", tag="no-such-tag") == []
    assert search_notes(nb, "zzz") == []


def test_search_merges_and_sorts_across_both_stores() -> None:
    results = search(_legacy_graph(), _legacy_notebook(), "grav")
    assert [(r.concept_name, r.note_id, r.score) for r in results] == [
        ("Gravity", None, SUBSTRING_SCORE),
        (None, "n1", SUBSTRING_SCORE),
    ]
    assert all(isinstance(r, SearchResult) for r in results)
