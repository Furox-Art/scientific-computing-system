"""Tests for cds.knowledge — concept graphs, notes, and structured retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from cds.knowledge import (
    Concept,
    KnowledgeGraph,
    Note,
    Notebook,
    Relation,
    SearchResult,
    search,
    search_concepts,
    search_notes,
)
from cds.knowledge.retrieval import NAME_TAG_SCORE, SUBSTRING_SCORE

# ---------------------------------------------------------------------- #
# Concept
# ---------------------------------------------------------------------- #


class TestConcept:
    def test_defaults_are_empty_not_none(self) -> None:
        c = Concept("x")
        assert c.name == "x"
        assert c.description is None
        assert c.tags == []
        assert c.metadata == {}

    def test_stores_fields(self) -> None:
        c = Concept("x", description="d", tags=["a", "b"], metadata={"k": "v"})
        assert c.description == "d"
        assert c.tags == ["a", "b"]
        assert c.metadata == {"k": "v"}

    def test_equality_by_name(self) -> None:
        assert Concept("x") == Concept("x", description="different")
        assert Concept("x") != Concept("y")

    def test_equality_with_non_concept(self) -> None:
        assert Concept("x") != "x"
        assert Concept("x") != 42
        assert Concept("x") != None  # noqa: E711

    def test_hash_is_stable_and_name_based(self) -> None:
        c1 = Concept("x", description="one")
        c2 = Concept("x", description="two")
        assert hash(c1) == hash(c2)
        # Usable as a set member / dict key.
        assert {c1, c2, Concept("y")} == {c1, Concept("y")}

    def test_to_dict_roundtrips_fields(self) -> None:
        c = Concept("x", description="d", tags=["a"], metadata={"k": "v"})
        d = c.to_dict()
        assert d == {"name": "x", "description": "d", "tags": ["a"], "metadata": {"k": "v"}}

    def test_to_dict_copies_mutable_fields(self) -> None:
        c = Concept("x", tags=["a"], metadata={"k": "v"})
        d = c.to_dict()
        cast(list[str], d["tags"]).append("b")
        cast(dict[str, str], d["metadata"])["z"] = "1"
        assert c.tags == ["a"]
        assert c.metadata == {"k": "v"}

    def test_from_dict_valid(self) -> None:
        c = Concept.from_dict(
            {"name": "x", "description": "d", "tags": ["a"], "metadata": {"k": "v"}}
        )
        assert c == Concept("x", description="d", tags=["a"], metadata={"k": "v"})

    def test_from_dict_null_description(self) -> None:
        c = Concept.from_dict({"name": "x", "description": None, "tags": [], "metadata": {}})
        assert c.description is None

    def test_from_dict_roundtrip(self) -> None:
        original = Concept("x", description="d", tags=["a"], metadata={"k": "v"})
        assert Concept.from_dict(original.to_dict()) == original

    def test_from_dict_invalid_name_type(self) -> None:
        with pytest.raises(ValueError, match="Invalid concept name"):
            Concept.from_dict({"name": 123, "description": None, "tags": [], "metadata": {}})

    def test_from_dict_invalid_description_type(self) -> None:
        with pytest.raises(ValueError, match="Invalid concept description"):
            Concept.from_dict({"name": "x", "description": 5, "tags": [], "metadata": {}})

    def test_from_dict_invalid_tags_not_list(self) -> None:
        with pytest.raises(ValueError, match="Invalid concept tags"):
            Concept.from_dict({"name": "x", "description": None, "tags": "a", "metadata": {}})

    def test_from_dict_invalid_tags_element(self) -> None:
        with pytest.raises(ValueError, match="Invalid concept tags"):
            Concept.from_dict({"name": "x", "description": None, "tags": [1], "metadata": {}})

    def test_from_dict_invalid_metadata_not_dict(self) -> None:
        with pytest.raises(ValueError, match="Invalid concept metadata"):
            Concept.from_dict({"name": "x", "description": None, "tags": [], "metadata": []})

    def test_from_dict_invalid_metadata_value_type(self) -> None:
        with pytest.raises(ValueError, match="Invalid concept metadata"):
            Concept.from_dict({"name": "x", "description": None, "tags": [], "metadata": {"k": 1}})

    def test_from_dict_missing_key_raises(self) -> None:
        with pytest.raises(KeyError):
            Concept.from_dict({"description": None, "tags": [], "metadata": {}})


# ---------------------------------------------------------------------- #
# Relation
# ---------------------------------------------------------------------- #


class TestRelation:
    def test_defaults_weight_to_one(self) -> None:
        r = Relation("a", "b", "is-a")
        assert r.weight == 1.0

    def test_stores_fields(self) -> None:
        r = Relation("a", "b", "is-a", weight=2.5)
        assert r.source == "a"
        assert r.target == "b"
        assert r.kind == "is-a"
        assert r.weight == 2.5

    def test_to_dict(self) -> None:
        r = Relation("a", "b", "is-a", weight=0.5)
        assert r.to_dict() == {"source": "a", "target": "b", "kind": "is-a", "weight": 0.5}

    def test_from_dict_valid(self) -> None:
        r = Relation.from_dict({"source": "a", "target": "b", "kind": "is-a", "weight": 1.0})
        assert r == Relation("a", "b", "is-a", weight=1.0)

    def test_from_dict_coerces_int_weight_to_float(self) -> None:
        r = Relation.from_dict({"source": "a", "target": "b", "kind": "is-a", "weight": 3})
        assert r.weight == 3.0
        assert isinstance(r.weight, float)

    def test_from_dict_roundtrip(self) -> None:
        original = Relation("a", "b", "is-a", weight=0.5)
        assert Relation.from_dict(original.to_dict()) == original

    def test_from_dict_invalid_source(self) -> None:
        with pytest.raises(ValueError, match="Invalid relation source"):
            Relation.from_dict({"source": 1, "target": "b", "kind": "is-a", "weight": 1.0})

    def test_from_dict_invalid_target(self) -> None:
        with pytest.raises(ValueError, match="Invalid relation target"):
            Relation.from_dict({"source": "a", "target": 1, "kind": "is-a", "weight": 1.0})

    def test_from_dict_invalid_kind(self) -> None:
        with pytest.raises(ValueError, match="Invalid relation kind"):
            Relation.from_dict({"source": "a", "target": "b", "kind": 1, "weight": 1.0})

    def test_from_dict_invalid_weight_type(self) -> None:
        with pytest.raises(ValueError, match="Invalid relation weight"):
            Relation.from_dict({"source": "a", "target": "b", "kind": "is-a", "weight": "x"})

    def test_from_dict_rejects_bool_weight(self) -> None:
        with pytest.raises(ValueError, match="Invalid relation weight"):
            Relation.from_dict({"source": "a", "target": "b", "kind": "is-a", "weight": True})

    def test_from_dict_missing_key_raises(self) -> None:
        with pytest.raises(KeyError):
            Relation.from_dict({"source": "a", "target": "b", "kind": "is-a"})


# ---------------------------------------------------------------------- #
# KnowledgeGraph — construction
# ---------------------------------------------------------------------- #


class TestGraphConstruction:
    def test_empty_graph(self) -> None:
        kg = KnowledgeGraph(name="empty")
        assert kg.concepts == {}
        assert kg.relations == []

    def test_add_concept_returns_stored_node(self) -> None:
        kg = KnowledgeGraph(name="g")
        c = kg.add_concept("x", description="d", tags=["t"], metadata={"k": "v"})
        assert c is kg.concepts["x"]
        assert c.name == "x"
        assert c.description == "d"

    def test_add_concept_is_idempotent(self) -> None:
        kg = KnowledgeGraph(name="g")
        first = kg.add_concept("x", description="first")
        second = kg.add_concept("x", description="second")
        assert first is second
        assert second.description == "first"  # not overwritten

    def test_add_concept_tags_default_empty_list(self) -> None:
        kg = KnowledgeGraph(name="g")
        c = kg.add_concept("x")
        assert c.tags == []
        assert c.metadata == {}

    def test_add_concept_copies_input_collections(self) -> None:
        kg = KnowledgeGraph(name="g")
        tags = ["a"]
        meta = {"k": "v"}
        c = kg.add_concept("x", tags=tags, metadata=meta)
        tags.append("b")
        meta["z"] = "1"
        assert c.tags == ["a"]
        assert c.metadata == {"k": "v"}

    def test_add_relation_requires_known_endpoints(self) -> None:
        kg = KnowledgeGraph(name="g")
        kg.add_concept("a")
        kg.add_concept("b")
        r = kg.add_relation("a", "b", "is-a", weight=2.0)
        assert r in kg.relations
        assert r.weight == 2.0

    def test_add_relation_unknown_source_raises(self) -> None:
        kg = KnowledgeGraph(name="g")
        kg.add_concept("b")
        with pytest.raises(KeyError, match="unknown source concept"):
            kg.add_relation("missing", "b", "is-a")

    def test_add_relation_unknown_target_raises(self) -> None:
        kg = KnowledgeGraph(name="g")
        kg.add_concept("a")
        with pytest.raises(KeyError, match="unknown target concept"):
            kg.add_relation("a", "missing", "is-a")

    def test_link_concepts_creates_missing(self) -> None:
        kg = KnowledgeGraph(name="g")
        kg.link_concepts("a", "b", "is-a")
        assert "a" in kg.concepts
        assert "b" in kg.concepts
        assert len(kg.relations) == 1

    def test_link_concepts_reuses_existing(self) -> None:
        kg = KnowledgeGraph(name="g")
        kg.add_concept("a", description="keep")
        kg.link_concepts("a", "b", "is-a")
        # Existing concept is preserved (idempotent add).
        assert kg.concepts["a"].description == "keep"


# ---------------------------------------------------------------------- #
# KnowledgeGraph — neighbor queries
# ---------------------------------------------------------------------- #


class TestGraphNeighbors:
    def _graph(self) -> KnowledgeGraph:
        # a --is-a--> b --is-a--> c
        # a --related-to--> c
        kg = KnowledgeGraph(name="g")
        kg.add_concept("a")
        kg.add_concept("b")
        kg.add_concept("c")
        kg.add_relation("a", "b", "is-a")
        kg.add_relation("b", "c", "is-a")
        kg.add_relation("a", "c", "related-to")
        return kg

    def test_neighbors_undirected_both_directions(self) -> None:
        kg = self._graph()
        assert set(kg.neighbors("a")) == {"b", "c"}
        assert set(kg.neighbors("b")) == {"a", "c"}
        assert set(kg.neighbors("c")) == {"a", "b"}

    def test_neighbors_filtered_by_kind(self) -> None:
        kg = self._graph()
        assert set(kg.neighbors("a", kind="is-a")) == {"b"}
        assert set(kg.neighbors("a", kind="related-to")) == {"c"}
        assert kg.neighbors("a", kind="nonexistent") == []

    def test_neighbors_dedupes(self) -> None:
        kg = KnowledgeGraph(name="g")
        kg.add_concept("a")
        kg.add_concept("b")
        # Two relations between the same pair: b should appear once.
        kg.add_relation("a", "b", "is-a")
        kg.add_relation("b", "a", "related-to")
        assert kg.neighbors("a") == ["b"]

    def test_neighbors_unknown_concept_raises(self) -> None:
        kg = self._graph()
        with pytest.raises(KeyError, match="unknown concept"):
            kg.neighbors("missing")

    def test_neighbors_out_only_outgoing(self) -> None:
        kg = self._graph()
        assert set(kg.neighbors_out("a")) == {"b", "c"}
        assert kg.neighbors_out("b") == ["c"]
        assert kg.neighbors_out("c") == []

    def test_neighbors_out_filtered_by_kind(self) -> None:
        kg = self._graph()
        assert kg.neighbors_out("a", kind="is-a") == ["b"]
        assert kg.neighbors_out("a", kind="related-to") == ["c"]

    def test_neighbors_out_unknown_raises(self) -> None:
        kg = self._graph()
        with pytest.raises(KeyError, match="unknown concept"):
            kg.neighbors_out("missing")

    def test_neighbors_in_only_incoming(self) -> None:
        kg = self._graph()
        assert kg.neighbors_in("a") == []
        assert set(kg.neighbors_in("b")) == {"a"}
        assert set(kg.neighbors_in("c")) == {"a", "b"}

    def test_neighbors_in_filtered_by_kind(self) -> None:
        kg = self._graph()
        assert kg.neighbors_in("c", kind="is-a") == ["b"]
        assert kg.neighbors_in("c", kind="related-to") == ["a"]

    def test_neighbors_in_unknown_raises(self) -> None:
        kg = self._graph()
        with pytest.raises(KeyError, match="unknown concept"):
            kg.neighbors_in("missing")


# ---------------------------------------------------------------------- #
# KnowledgeGraph — traversal (path, reachability, cycles)
# ---------------------------------------------------------------------- #


class TestGraphTraversal:
    def test_find_path_source_equals_target(self) -> None:
        kg = KnowledgeGraph(name="g")
        kg.add_concept("a")
        assert kg.find_path("a", "a") == ["a"]

    def test_find_path_direct_neighbor(self) -> None:
        kg = KnowledgeGraph(name="g")
        kg.link_concepts("a", "b", "is-a")
        assert kg.find_path("a", "b") == ["a", "b"]

    def test_find_path_multi_hop(self) -> None:
        kg = KnowledgeGraph(name="g")
        for pair in [("a", "b"), ("b", "c"), ("c", "d")]:
            kg.link_concepts(*pair, kind="is-a")
        assert kg.find_path("a", "d") == ["a", "b", "c", "d"]

    def test_find_path_traverses_against_direction(self) -> None:
        # Only edge is b -> a; path from a to b must go "backwards".
        kg = KnowledgeGraph(name="g")
        kg.add_concept("a")
        kg.add_concept("b")
        kg.add_relation("b", "a", "is-a")
        assert kg.find_path("a", "b") == ["a", "b"]

    def test_find_path_no_connection_returns_none(self) -> None:
        kg = KnowledgeGraph(name="g")
        kg.add_concept("a")
        kg.add_concept("b")
        assert kg.find_path("a", "b") is None

    def test_find_path_unknown_source_returns_none(self) -> None:
        kg = KnowledgeGraph(name="g")
        kg.add_concept("a")
        assert kg.find_path("missing", "a") is None

    def test_find_path_unknown_target_returns_none(self) -> None:
        kg = KnowledgeGraph(name="g")
        kg.add_concept("a")
        assert kg.find_path("a", "missing") is None

    def test_find_path_returns_shortest(self) -> None:
        # Triangle: a-b-c plus a direct shortcut a-c.
        kg = KnowledgeGraph(name="g")
        kg.link_concepts("a", "b", "is-a")
        kg.link_concepts("b", "c", "is-a")
        kg.link_concepts("a", "c", "is-a")
        # BFS finds the 2-hop [a, c] via the shortcut, not [a,b,c].
        assert len(kg.find_path("a", "c") or []) == 2  # noqa: PLR2004

    def test_reachable_includes_self(self) -> None:
        kg = KnowledgeGraph(name="g")
        kg.add_concept("a")
        assert kg.reachable("a") == {"a"}

    def test_reachable_traverses_graph(self) -> None:
        kg = KnowledgeGraph(name="g")
        kg.link_concepts("a", "b", "is-a")
        kg.link_concepts("b", "c", "is-a")
        assert kg.reachable("a") == {"a", "b", "c"}

    def test_reachable_unknown_concept_returns_empty(self) -> None:
        kg = KnowledgeGraph(name="g")
        assert kg.reachable("missing") == set()

    def test_find_cycles_empty(self) -> None:
        kg = KnowledgeGraph(name="g")
        kg.link_concepts("a", "b", "is-a")
        assert kg.find_cycles() == []

    def test_find_cycles_simple_triangle(self) -> None:
        kg = KnowledgeGraph(name="g")
        kg.link_concepts("a", "b", "is-a")
        kg.link_concepts("b", "c", "is-a")
        kg.link_concepts("c", "a", "is-a")
        cycles = kg.find_cycles()
        assert cycles == [["a", "b", "c"]]

    def test_find_cycles_normalized_to_smallest_member(self) -> None:
        # Same cycle entered from different nodes collapses to one entry.
        kg = KnowledgeGraph(name="g")
        kg.link_concepts("c", "a", "is-a")  # entered via c first
        kg.link_concepts("a", "b", "is-a")
        kg.link_concepts("b", "c", "is-a")
        cycles = kg.find_cycles()
        assert cycles == [["a", "b", "c"]]

    def test_find_cycles_self_loop(self) -> None:
        kg = KnowledgeGraph(name="g")
        kg.add_concept("a")
        kg.add_relation("a", "a", "is-a")
        assert kg.find_cycles() == [["a"]]

    def test_find_cycles_multiple_distinct(self) -> None:
        kg = KnowledgeGraph(name="g")
        for n in ["a", "b", "c", "d", "e", "f"]:
            kg.add_concept(n)
        # Cycle 1: a -> b -> c -> a
        kg.add_relation("a", "b", "is-a")
        kg.add_relation("b", "c", "is-a")
        kg.add_relation("c", "a", "is-a")
        # Cycle 2: d -> e -> f -> d
        kg.add_relation("d", "e", "is-a")
        kg.add_relation("e", "f", "is-a")
        kg.add_relation("f", "d", "is-a")
        cycles = kg.find_cycles()
        assert cycles == [["a", "b", "c"], ["d", "e", "f"]]

    def test_find_cycles_does_not_hit_recursion_limit(self) -> None:
        # A long chain with a closing edge back to the start.
        n = 5000
        kg = KnowledgeGraph(name="deep")
        for i in range(n):
            kg.add_concept(f"c{i}")
        for i in range(n - 1):
            kg.add_relation(f"c{i}", f"c{i + 1}", "is-a")
        kg.add_relation(f"c{n - 1}", "c0", "is-a")  # close the loop
        cycles = kg.find_cycles()
        assert len(cycles) == 1
        assert len(cycles[0]) == n
        assert cycles[0][0] == "c0"


# ---------------------------------------------------------------------- #
# KnowledgeGraph — rendering & serialization
# ---------------------------------------------------------------------- #


class TestGraphRendering:
    def test_to_markdown_full(self) -> None:
        kg = KnowledgeGraph(name="My Graph")
        kg.link_concepts("a", "b", "is-a")
        md = kg.to_markdown()
        assert "# Knowledge Graph: My Graph" in md
        assert "## Concepts" in md
        assert "## Relations" in md
        assert "**a**" in md
        assert "`a` --is-a--> `b`" in md

    def test_to_markdown_no_concepts(self) -> None:
        kg = KnowledgeGraph(name="empty")
        md = kg.to_markdown()
        assert "_No concepts._" in md
        assert "_No relations._" in md

    def test_to_markdown_concepts_sorted(self) -> None:
        kg = KnowledgeGraph(name="g")
        kg.add_concept("zeta")
        kg.add_concept("alpha")
        md = kg.to_markdown()
        assert md.index("**alpha**") < md.index("**zeta**")

    def test_to_dict_structure(self) -> None:
        kg = KnowledgeGraph(name="g")
        kg.link_concepts("a", "b", "is-a")
        d = kg.to_dict()
        assert d["name"] == "g"
        concepts = cast(list[dict[str, object]], d["concepts"])
        assert [c["name"] for c in concepts] == ["a", "b"]
        relations = cast(list[dict[str, object]], d["relations"])
        assert relations[0]["kind"] == "is-a"

    def test_from_dict_roundtrip(self) -> None:
        kg = KnowledgeGraph(name="g")
        kg.add_concept("a", description="d", tags=["t"])
        kg.link_concepts("a", "b", "is-a", weight=2.0)
        rebuilt = KnowledgeGraph.from_dict(kg.to_dict())
        assert rebuilt.name == "g"
        assert set(rebuilt.concepts) == {"a", "b"}
        assert rebuilt.concepts["a"].description == "d"
        assert len(rebuilt.relations) == 1
        assert rebuilt.relations[0].weight == 2.0

    def test_from_dict_preserves_concept_order(self) -> None:
        kg = KnowledgeGraph(name="g")
        kg.add_concept("z")
        kg.add_concept("a")
        rebuilt = KnowledgeGraph.from_dict(kg.to_dict())
        # concepts dict preserves insertion order from the serialized list.
        assert list(rebuilt.concepts.keys()) == ["z", "a"]

    def test_from_dict_invalid_name(self) -> None:
        with pytest.raises(ValueError, match="Invalid graph name"):
            KnowledgeGraph.from_dict({"name": 1, "concepts": [], "relations": []})

    def test_from_dict_invalid_concepts_list(self) -> None:
        with pytest.raises(ValueError, match="Invalid concepts list"):
            KnowledgeGraph.from_dict({"name": "g", "concepts": "x", "relations": []})

    def test_from_dict_invalid_relations_list(self) -> None:
        with pytest.raises(ValueError, match="Invalid relations list"):
            KnowledgeGraph.from_dict({"name": "g", "concepts": [], "relations": "x"})

    def test_from_dict_invalid_concept_entry(self) -> None:
        with pytest.raises(ValueError, match="Invalid concept entry"):
            KnowledgeGraph.from_dict({"name": "g", "concepts": ["not-a-dict"], "relations": []})

    def test_from_dict_invalid_relation_entry(self) -> None:
        with pytest.raises(ValueError, match="Invalid relation entry"):
            KnowledgeGraph.from_dict({"name": "g", "concepts": [], "relations": ["x"]})

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        kg = KnowledgeGraph(name="g")
        kg.link_concepts("a", "b", "is-a")
        path = tmp_path / "kg.json"
        kg.save(path)
        loaded = KnowledgeGraph.load(path)
        assert set(loaded.concepts) == {"a", "b"}
        assert len(loaded.relations) == 1

    def test_save_writes_indented_utf8_json(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        kg = KnowledgeGraph(name="g")
        kg.add_concept("λ")  # non-ASCII to verify ensure_ascii=False
        path = tmp_path / "kg.json"
        kg.save(path)
        text = path.read_text(encoding="utf-8")
        assert "λ" in text
        assert "\n" in text  # indented

    def test_load_non_dict_raises(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "bad.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid graph file"):
            KnowledgeGraph.load(path)

    def test_load_roundtrips_loaded_via_json_module(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        # Ensure the persisted form is plain JSON any reader can parse.
        kg = KnowledgeGraph(name="g")
        kg.link_concepts("a", "b", "is-a")
        path = tmp_path / "kg.json"
        kg.save(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw["name"] == "g"


# ---------------------------------------------------------------------- #
# Note
# ---------------------------------------------------------------------- #


class TestNote:
    def test_defaults(self) -> None:
        n = Note(id="n1", title="t", body="b")
        assert n.tags == []
        assert n.linked_concepts == []
        assert n.created is None

    def test_to_dict(self) -> None:
        n = Note(id="n1", title="t", body="b", tags=["x"], linked_concepts=["c"], created="2024")
        assert n.to_dict() == {
            "id": "n1",
            "title": "t",
            "body": "b",
            "tags": ["x"],
            "linked_concepts": ["c"],
            "created": "2024",
        }

    def test_to_dict_copies_mutable_fields(self) -> None:
        n = Note(id="n1", title="t", body="b", tags=["x"], linked_concepts=["c"])
        d = n.to_dict()
        cast(list[str], d["tags"]).append("y")
        cast(list[str], d["linked_concepts"]).append("z")
        assert n.tags == ["x"]
        assert n.linked_concepts == ["c"]

    def test_from_dict_valid(self) -> None:
        n = Note.from_dict(
            {
                "id": "n1",
                "title": "t",
                "body": "b",
                "tags": ["x"],
                "linked_concepts": ["c"],
                "created": "2024",
            }
        )
        assert n.id == "n1"
        assert n.tags == ["x"]
        assert n.created == "2024"

    def test_from_dict_null_created(self) -> None:
        n = Note.from_dict(
            {
                "id": "n1",
                "title": "t",
                "body": "b",
                "tags": [],
                "linked_concepts": [],
                "created": None,
            }
        )
        assert n.created is None

    def test_from_dict_roundtrip(self) -> None:
        original = Note(
            id="n1", title="t", body="b", tags=["x"], linked_concepts=["c"], created="2024"
        )
        rebuilt = Note.from_dict(original.to_dict())
        assert rebuilt.id == original.id
        assert rebuilt.tags == original.tags

    def test_from_dict_invalid_id(self) -> None:
        with pytest.raises(ValueError, match="Invalid note id"):
            Note.from_dict(
                {
                    "id": 1,
                    "title": "t",
                    "body": "b",
                    "tags": [],
                    "linked_concepts": [],
                    "created": None,
                }
            )

    def test_from_dict_invalid_title(self) -> None:
        with pytest.raises(ValueError, match="Invalid note title"):
            Note.from_dict(
                {
                    "id": "n1",
                    "title": 1,
                    "body": "b",
                    "tags": [],
                    "linked_concepts": [],
                    "created": None,
                }
            )

    def test_from_dict_invalid_body(self) -> None:
        with pytest.raises(ValueError, match="Invalid note body"):
            Note.from_dict(
                {
                    "id": "n1",
                    "title": "t",
                    "body": 1,
                    "tags": [],
                    "linked_concepts": [],
                    "created": None,
                }
            )

    def test_from_dict_invalid_tags(self) -> None:
        with pytest.raises(ValueError, match="Invalid note tags"):
            Note.from_dict(
                {
                    "id": "n1",
                    "title": "t",
                    "body": "b",
                    "tags": [1],
                    "linked_concepts": [],
                    "created": None,
                }
            )

    def test_from_dict_invalid_linked_concepts(self) -> None:
        with pytest.raises(ValueError, match="Invalid note linked_concepts"):
            Note.from_dict(
                {
                    "id": "n1",
                    "title": "t",
                    "body": "b",
                    "tags": [],
                    "linked_concepts": [1],
                    "created": None,
                }
            )

    def test_from_dict_invalid_created(self) -> None:
        with pytest.raises(ValueError, match="Invalid note created"):
            Note.from_dict(
                {
                    "id": "n1",
                    "title": "t",
                    "body": "b",
                    "tags": [],
                    "linked_concepts": [],
                    "created": 2024,
                }
            )

    def test_from_dict_missing_key_raises(self) -> None:
        with pytest.raises(KeyError):
            Note.from_dict(
                {"title": "t", "body": "b", "tags": [], "linked_concepts": [], "created": None}
            )

    def test_to_markdown_with_all_fields(self) -> None:
        n = Note(
            id="n1",
            title="t",
            body="b",
            tags=["x", "y"],
            linked_concepts=["c1", "c2"],
            created="2024",
        )
        md = n.to_markdown()
        assert md.startswith("# t")
        assert "_Created: 2024_" in md
        assert "`x`" in md and "`y`" in md
        assert "## Linked concepts" in md
        assert "`c1`" in md
        assert md.rstrip().endswith("b") is False or "b" in md  # body present

    def test_to_markdown_minimal(self) -> None:
        n = Note(id="n1", title="t", body="b")
        md = n.to_markdown()
        assert "_Created:" not in md
        assert "Tags:" not in md
        assert "## Linked concepts" not in md


# ---------------------------------------------------------------------- #
# Notebook
# ---------------------------------------------------------------------- #


class TestNotebook:
    def test_add_note_returns_stored(self) -> None:
        nb = Notebook(name="nb")
        n = nb.add_note("n1", "title", "body", tags=["t"], linked_concepts=["c"], created="2024")
        assert n is nb.notes["n1"]
        assert n.title == "title"

    def test_add_note_overwrites_existing(self) -> None:
        nb = Notebook(name="nb")
        nb.add_note("n1", "first", "b1")
        nb.add_note("n1", "second", "b2")
        assert len(nb.notes) == 1
        assert nb.notes["n1"].title == "second"

    def test_add_note_defaults_empty_collections(self) -> None:
        nb = Notebook(name="nb")
        n = nb.add_note("n1", "t", "b")
        assert n.tags == []
        assert n.linked_concepts == []
        assert n.created is None

    def test_add_note_copies_input_collections(self) -> None:
        nb = Notebook(name="nb")
        tags = ["a"]
        linked = ["c"]
        n = nb.add_note("n1", "t", "b", tags=tags, linked_concepts=linked)
        tags.append("b")
        linked.append("d")
        assert n.tags == ["a"]
        assert n.linked_concepts == ["c"]

    def test_get_note_found(self) -> None:
        nb = Notebook(name="nb")
        nb.add_note("n1", "t", "b")
        assert nb.get_note("n1").title == "t"

    def test_get_note_unknown_raises(self) -> None:
        nb = Notebook(name="nb")
        with pytest.raises(KeyError, match="unknown note"):
            nb.get_note("missing")

    def test_notes_for_concept(self) -> None:
        nb = Notebook(name="nb")
        nb.add_note("n1", "t1", "b", linked_concepts=["c1", "c2"])
        nb.add_note("n2", "t2", "b", linked_concepts=["c2"])
        nb.add_note("n3", "t3", "b", linked_concepts=[])
        assert [n.id for n in nb.notes_for_concept("c2")] == ["n1", "n2"]
        assert nb.notes_for_concept("c1") == [nb.notes["n1"]]
        assert nb.notes_for_concept("absent") == []

    def test_notes_by_tag(self) -> None:
        nb = Notebook(name="nb")
        nb.add_note("n1", "t1", "b", tags=["exp", "fail"])
        nb.add_note("n2", "t2", "b", tags=["exp"])
        assert [n.id for n in nb.notes_by_tag("exp")] == ["n1", "n2"]
        assert nb.notes_by_tag("fail") == [nb.notes["n1"]]
        assert nb.notes_by_tag("absent") == []

    def test_to_markdown_empty(self) -> None:
        nb = Notebook(name="empty")
        md = nb.to_markdown()
        assert "# Notebook: empty" in md
        assert "_No notes._" in md

    def test_to_markdown_with_notes_sorted_by_id(self) -> None:
        nb = Notebook(name="nb")
        nb.add_note("n2", "second", "b", tags=["t"])
        nb.add_note("n1", "first", "b")
        md = nb.to_markdown()
        assert md.index("**n1**") < md.index("**n2**")
        assert "(t)" in md  # tags rendered for n2

    def test_to_dict_structure(self) -> None:
        nb = Notebook(name="nb")
        nb.add_note("n1", "t", "b")
        d = nb.to_dict()
        assert d["name"] == "nb"
        assert cast(list[dict[str, object]], d["notes"])[0]["id"] == "n1"

    def test_from_dict_roundtrip(self) -> None:
        nb = Notebook(name="nb")
        nb.add_note("n1", "t", "b", tags=["x"], linked_concepts=["c"], created="2024")
        rebuilt = Notebook.from_dict(nb.to_dict())
        assert rebuilt.name == "nb"
        assert rebuilt.get_note("n1").title == "t"
        assert rebuilt.get_note("n1").tags == ["x"]

    def test_from_dict_invalid_name(self) -> None:
        with pytest.raises(ValueError, match="Invalid notebook name"):
            Notebook.from_dict({"name": 1, "notes": []})

    def test_from_dict_invalid_notes_list(self) -> None:
        with pytest.raises(ValueError, match="Invalid notes list"):
            Notebook.from_dict({"name": "nb", "notes": "x"})

    def test_from_dict_invalid_note_entry(self) -> None:
        with pytest.raises(ValueError, match="Invalid note entry"):
            Notebook.from_dict({"name": "nb", "notes": ["not-a-dict"]})

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        nb = Notebook(name="nb")
        nb.add_note("n1", "t", "b")
        path = tmp_path / "nb.json"
        nb.save(path)
        loaded = Notebook.load(path)
        assert loaded.get_note("n1").title == "t"

    def test_load_non_dict_raises(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "bad.json"
        path.write_text("[]", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid notebook file"):
            Notebook.load(path)

    def test_save_writes_utf8(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        nb = Notebook(name="café")
        nb.add_note("n1", "tître", "b")
        path = tmp_path / "nb.json"
        nb.save(path)
        text = path.read_text(encoding="utf-8")
        assert "café" in text
        assert "tître" in text


# ---------------------------------------------------------------------- #
# Retrieval — search_concepts
# ---------------------------------------------------------------------- #


class TestSearchConcepts:
    def _graph(self) -> KnowledgeGraph:
        kg = KnowledgeGraph(name="g")
        kg.add_concept("Gravity", description="attractive force between masses", tags=["physics"])
        kg.add_concept("Graviton", description="hypothetical quantum of gravity", tags=["physics"])
        kg.add_concept("Energy", description="capacity to do work", tags=["physics", "abstract"])
        kg.add_concept("Mass", tags=["physics"])
        return kg

    def test_exact_name_match_scores_highest(self) -> None:
        kg = self._graph()
        results = search_concepts(kg, "gravity")
        assert results[0].concept_name == "Gravity"
        assert results[0].score == NAME_TAG_SCORE
        assert results[0].matched_on == "name"

    def test_substring_name_match(self) -> None:
        kg = self._graph()
        results = search_concepts(kg, "grav")
        names = [r.concept_name for r in results]
        assert "Gravity" in names
        assert "Graviton" in names
        assert all(r.score == SUBSTRING_SCORE for r in results)

    def test_description_substring_match(self) -> None:
        kg = self._graph()
        results = search_concepts(kg, "masses")
        assert any(r.concept_name == "Gravity" and r.matched_on == "description" for r in results)

    def test_case_insensitive(self) -> None:
        kg = self._graph()
        assert search_concepts(kg, "GRAVITY")[0].concept_name == "Gravity"
        assert search_concepts(kg, "gravity")[0].concept_name == "Gravity"

    def test_no_match_returns_empty(self) -> None:
        kg = self._graph()
        assert search_concepts(kg, "nonexistent") == []

    def test_tag_filter_excludes_non_matching(self) -> None:
        kg = self._graph()
        results = search_concepts(kg, "energy", tag="abstract")
        assert [r.concept_name for r in results] == ["Energy"]
        # Without the abstract tag, Energy would still match.
        results2 = search_concepts(kg, "energy", tag="physics")
        assert "Energy" in [r.concept_name for r in results2]

    def test_tag_filter_no_matches_returns_empty(self) -> None:
        kg = self._graph()
        assert search_concepts(kg, "gravity", tag="absenttag") == []

    def test_ties_break_alphabetically(self) -> None:
        kg = self._graph()
        results = search_concepts(kg, "grav")
        # Gravity & Graviton both substring (0.5); ties break alphabetically:
        # "Graviton" < "Gravity" because 'i' < 'y'.
        assert [r.concept_name for r in results] == ["Graviton", "Gravity"]

    def test_results_sorted_by_score_then_name(self) -> None:
        kg = KnowledgeGraph(name="g")
        kg.add_concept("Zeta", description="contains needle")
        kg.add_concept("needle")  # exact match -> 1.0
        kg.add_concept("Alpha", description="contains needle")
        results = search_concepts(kg, "needle")
        # Exact (needle) first, then Alpha & Zeta alphabetical.
        assert [r.concept_name for r in results] == ["needle", "Alpha", "Zeta"]

    def test_description_none_does_not_match_substring(self) -> None:
        # 'Mass' has description=None; a query present only in it must not match.
        kg = self._graph()
        results = search_concepts(kg, "physics-label")
        assert not any(r.concept_name == "Mass" for r in results)


# ---------------------------------------------------------------------- #
# Retrieval — search_notes
# ---------------------------------------------------------------------- #


class TestSearchNotes:
    def _notebook(self) -> Notebook:
        nb = Notebook(name="nb")
        nb.add_note("n1", "Gravity", "discusses masses", tags=["physics"])
        nb.add_note("n2", "Misc", "gravity is mentioned in body", tags=["physics"])
        nb.add_note("n3", "Unrelated", "nothing here")
        return nb

    def test_exact_title_match_scores_highest(self) -> None:
        nb = self._notebook()
        results = search_notes(nb, "gravity")
        assert results[0].note_id == "n1"
        assert results[0].score == NAME_TAG_SCORE
        assert results[0].matched_on == "title"

    def test_substring_title_match(self) -> None:
        nb = Notebook(name="nb")
        nb.add_note("n1", "Gravitational waves", "b")
        results = search_notes(nb, "grav")
        assert results[0].note_id == "n1"
        assert results[0].matched_on == "title"
        assert results[0].score == SUBSTRING_SCORE

    def test_body_substring_match(self) -> None:
        nb = self._notebook()
        results = search_notes(nb, "gravity")
        ids = [r.note_id for r in results]
        assert "n2" in ids
        n2 = next(r for r in results if r.note_id == "n2")
        assert n2.matched_on == "body"
        assert n2.score == SUBSTRING_SCORE

    def test_title_match_preferred_over_body(self) -> None:
        # n1 matches title exactly (1.0), n2 matches body (0.5).
        nb = self._notebook()
        results = search_notes(nb, "gravity")
        assert results[0].note_id == "n1"

    def test_case_insensitive(self) -> None:
        nb = self._notebook()
        assert search_notes(nb, "GRAVITY")[0].note_id == "n1"

    def test_no_match_returns_empty(self) -> None:
        nb = self._notebook()
        assert search_notes(nb, "nonexistent") == []

    def test_tag_filter(self) -> None:
        nb = self._notebook()
        results = search_notes(nb, "gravity", tag="physics")
        assert {r.note_id for r in results} == {"n1", "n2"}
        # n3 has no 'physics' tag and wouldn't match anyway.

    def test_tag_filter_excludes(self) -> None:
        nb = Notebook(name="nb")
        nb.add_note("n1", "gravity", "b", tags=["a"])
        nb.add_note("n2", "gravity", "b", tags=["b"])
        results = search_notes(nb, "gravity", tag="a")
        assert [r.note_id for r in results] == ["n1"]

    def test_results_sorted_by_score_then_id(self) -> None:
        nb = Notebook(name="nb")
        nb.add_note("zzz", "needle", "b")  # exact 1.0
        nb.add_note("bbb", "x", "needle in body")  # substring 0.5
        nb.add_note("aaa", "x", "needle in body")  # substring 0.5
        results = search_notes(nb, "needle")
        assert [r.note_id for r in results] == ["zzz", "aaa", "bbb"]


# ---------------------------------------------------------------------- #
# Retrieval — combined search
# ---------------------------------------------------------------------- #


class TestSearchCombined:
    def test_merges_concepts_and_notes(self) -> None:
        kg = KnowledgeGraph(name="g")
        kg.add_concept("gravity")
        nb = Notebook(name="nb")
        nb.add_note("n1", "gravity note", "b")
        results = search(kg, nb, "gravity")
        assert {r.concept_name for r in results if r.concept_name} == {"gravity"}
        assert {r.note_id for r in results if r.note_id} == {"n1"}

    def test_combined_ranking(self) -> None:
        kg = KnowledgeGraph(name="g")
        kg.add_concept("needle", description="contains needle")  # exact 1.0
        nb = Notebook(name="nb")
        nb.add_note("n1", "needle", "b")  # exact 1.0
        nb.add_note("n2", "x", "needle body")  # substring 0.5
        results = search(kg, nb, "needle")
        # Two exact matches (1.0) first, ordered by identifier alphabetically,
        # then the 0.5 body match. Note id "n1" and concept name "needle" sort
        # before "n2"; "n1" < "needle" lexicographically.
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        assert results[-1].note_id == "n2"

    def test_combined_tag_filter_applied_to_both(self) -> None:
        kg = KnowledgeGraph(name="g")
        kg.add_concept("needle", tags=["keep"])
        kg.add_concept("needle2", tags=["drop"])
        nb = Notebook(name="nb")
        nb.add_note("n1", "needle", "b", tags=["keep"])
        nb.add_note("n2", "needle", "b", tags=["drop"])
        results = search(kg, nb, "needle", tag="keep")
        assert {r.concept_name for r in results if r.concept_name} == {"needle"}
        assert {r.note_id for r in results if r.note_id} == {"n1"}

    def test_empty_stores_return_empty(self) -> None:
        kg = KnowledgeGraph(name="g")
        nb = Notebook(name="nb")
        assert search(kg, nb, "anything") == []


# ---------------------------------------------------------------------- #
# Public API surface
# ---------------------------------------------------------------------- #


class TestPublicAPI:
    def test_search_result_dataclass_fields(self) -> None:
        r = SearchResult(concept_name="x", note_id=None, score=1.0, matched_on="name")
        assert r.concept_name == "x"
        assert r.note_id is None
        assert r.score == 1.0
        assert r.matched_on == "name"

    def test_search_result_note_variant(self) -> None:
        r = SearchResult(concept_name=None, note_id="n1", score=0.5, matched_on="body")
        assert r.concept_name is None
        assert r.note_id == "n1"

    def test_all_documented_names_exported(self) -> None:
        # cds.knowledge is already imported at module scope (above), so read
        # it back from sys.modules instead of re-``import``-ing, which would
        # mix ``import cds.knowledge`` with the module-scope ``from`` form.
        import sys

        mod = sys.modules["cds.knowledge"]

        expected = {
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
        }
        assert expected.issubset(set(dir(mod)))
        assert expected == set(mod.__all__)


# ---------------------------------------------------------------------- #
# Integration: the documented end-to-end workflow
# ---------------------------------------------------------------------- #


class TestIntegration:
    def test_full_workflow_roundtrip(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        kg = KnowledgeGraph(name="Cosmology")
        kg.add_concept("Dark Energy", description="drives expansion", tags=["cosmology"])
        kg.add_concept("Hubble Constant", tags=["cosmology"])
        kg.add_relation("Dark Energy", "Hubble Constant", kind="affects")

        nb = Notebook(name="Lab")
        nb.add_note(
            "n1",
            "Hubble Tension",
            "local vs cmb measurements",
            tags=["exp"],
            linked_concepts=["Hubble Constant"],
        )

        # Persistence round-trips for both.
        gpath = tmp_path / "g.json"
        npath = tmp_path / "n.json"
        kg.save(gpath)
        nb.save(npath)
        kg2 = KnowledgeGraph.load(gpath)
        nb2 = Notebook.load(npath)

        # Search works on reloaded data.
        results = search(kg2, nb2, "hubble")
        assert any(r.concept_name == "Hubble Constant" for r in results)
        assert any(r.note_id == "n1" for r in results)

        # Traversal survives the round-trip.
        assert kg2.find_path("Dark Energy", "Hubble Constant") == [
            "Dark Energy",
            "Hubble Constant",
        ]
