# Knowledge Organization Tutorial

`cds.knowledge` is a self-contained subsystem for organising research knowledge: a **knowledge graph** of named concepts connected by typed, directed relations, a **notebook** of free-form notes linked to concept names, and **ranked structured retrieval** across both. Everything is pure Python (stdlib `json` for persistence) and decoupled from `cds.graph`, whose dense integer-vertex model is a poor fit for named concepts with typed relationships.

## 1. Build a Concept Graph

Concepts are named nodes with optional description, tags, and metadata. Relations are typed, directed edges (`source --kind--> target`).

```python
from cds.knowledge import KnowledgeGraph

kg = KnowledgeGraph(name="Cosmology")
kg.add_concept(
    "Dark Energy",
    description="Energy driving the accelerated expansion of the universe.",
    tags=["cosmology", "energy"],
    metadata={"first_proposed": "1998"},
)
kg.add_concept("Hubble Constant", tags=["cosmology", "measurement"])
kg.add_concept("Cosmic Microwave Background", tags=["cosmology", "radiation"])

kg.add_relation("Dark Energy", "Hubble Constant", kind="affects")
kg.add_relation("Hubble Constant", "Cosmic Microwave Background", kind="constrains")
```

Relations are **directed**, `"A depends-on B"` is not symmetric, but the traversal helpers below follow edges in either direction when that's what you want.

## 2. Traverse the Graph

```python
print(kg.neighbors("Hubble Constant"))
# ['Dark Energy', 'Cosmic Microwave Background']  (undirected)

print(kg.neighbors_out("Dark Energy"))  # ['Hubble Constant']   (outgoing only)
print(kg.neighbors_in("Cosmic Microwave Background"))
# ['Hubble Constant']  (incoming only)

print(kg.find_path("Dark Energy", "Cosmic Microwave Background"))
# ['Dark Energy', 'Hubble Constant', 'Cosmic Microwave Background']

print(sorted(kg.reachable("Dark Energy")))  # transitive closure (undirected)
print(kg.find_cycles())  # directed cycles, normalized
```

`find_path` and `reachable` use BFS and traverse edges in either direction, so a path may run against a relation's direction. `find_cycles` works on the *directed* structure and reports each cycle once, normalized to start at its smallest member, deep graphs are safe because it uses an explicit stack rather than recursion.

You can filter neighbors by relation type:

```python
print(kg.neighbors("Hubble Constant", kind="constrains"))
# ['Cosmic Microwave Background']
```

## 3. Keep a Notebook of Notes

A `Note` is a free-form record (title + body + tags) that links to concept names by **string**, not by reference, so a note may mention a concept that isn't in any graph yet, exactly like a citation in a lab book.

```python
from cds.knowledge import Notebook

nb = Notebook(name="Cosmology Lab Book")
nb.add_note(
    note_id="n1",
    title="Hubble Tension",
    body="Local and CMB measurements of H0 disagree at ~5 sigma.",
    tags=["experiment", "tension"],
    linked_concepts=["Hubble Constant", "Cosmic Microwave Background"],
    created="2024-03-12",
)

print([n.id for n in nb.notes_for_concept("Hubble Constant")])  # ['n1']
print([n.id for n in nb.notes_by_tag("experiment")])  # ['n1']
```

## 4. Ranked Retrieval

`search` ranks matching concepts and notes by a simple, deterministic, explainable score: an exact name/title match is `1.0`, a substring match in the description/body/title is `0.5`, and ties break alphabetically.

```python
from cds.knowledge import search

for hit in search(kg, nb, query="hubble"):
    kind = "concept" if hit.concept_name else "note"
    ident = hit.concept_name or hit.note_id
    print(f"[{kind}] {ident}  score={hit.score}  matched_on={hit.matched_on}")
# [concept] Hubble Constant  score=0.5  matched_on=name
# [note] n1                 score=0.5  matched_on=title
```

Use `search_concepts(kg, query)` or `search_notes(nb, query)` to search one store at a time. Any of the three accepts an optional `tag=` filter.

## 5. Persist to JSON

Every entity serializes via `to_dict` / `from_dict`, and the owning graph and notebook read/write JSON via the stdlib.

```python
kg.save("cosmology.json")
reloaded = KnowledgeGraph.load("cosmology.json")
print(len(reloaded.concepts), len(reloaded.relations))  # 3 2
```

Both `KnowledgeGraph` and `Notebook` validate their input on load and raise `ValueError` on malformed data, so corrupted files fail loudly rather than silently.

## Rendering

For quick inspection or for pasting into a report, both collections render to Markdown:

```python
print(kg.to_markdown())  # structured list of concepts + relations
print(nb.to_markdown())  # compact index of notes
print(nb.get_note("n1").to_markdown())  # a single self-contained note
```

Run the full demo with `python examples/knowledge_demo.py`.
