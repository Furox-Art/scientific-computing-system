# Graph Algorithms Tutorial

`cds.graph` implements traversal, shortest paths, MST, topological sort, and
cycle detection on a small `Graph` class.

`Graph` is built over **integer vertex ids**, so every constructor call needs an
explicit `n_vertices`. Vertex labels are not supported — map your own strings to
integers at the call site if you need readable output.

## 1. Build a Graph

```python
from cds.graph import Graph

# Undirected weighted graph on vertices 0..5.
g = Graph(n_vertices=6, directed=False)
for u, v, w in [(0, 1, 4), (0, 2, 2), (1, 2, 1), (1, 3, 5), (2, 3, 8), (2, 4, 10), (3, 4, 2)]:
    g.add_edge(u, v, weight=w)
```

## 2. Traversal

```python
from cds.graph import bfs, dfs

print(bfs(g, 0))
print(dfs(g, 0))
```

## 3. Shortest Paths & MST

```python
from cds.graph import dijkstra, kruskal_mst

dists, prev = dijkstra(g, 0)  # (distances, predecessors)
print(dists)  # {vertex: distance}
edges, total = kruskal_mst(g)  # (selected edges, total weight)
print(edges)
print("total weight:", total)
```

## 4. Topological Sort & Cycles

Topological sort needs a directed acyclic graph, again with integer vertices.

```python
from cds.graph import topological_sort, has_cycle

dag = Graph(n_vertices=3, directed=True)
dag.add_edge(0, 1)
dag.add_edge(1, 2)

print(topological_sort(dag))  # a valid ordering
print(has_cycle(dag))  # False
```

Run the full demo, which prints every algorithm with worked output, with:

```bash
python examples/graph_demo.py
```
