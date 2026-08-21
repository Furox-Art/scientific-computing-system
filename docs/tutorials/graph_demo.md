# Graph Algorithms Tutorial

`cds.graph` implements traversal, shortest paths, MST, topo sort, and cycle detection on a small `Graph` class.

## 1. Build a Graph

```python
from cds.graph import Graph

g = Graph(directed=False)
for u, v, w in [("A", "B", 4), ("A", "C", 2), ("B", "C", 1), ("B", "D", 5), ("C", "D", 8)]:
    g.add_edge(u, v, weight=w)
```

## 2. Traversal

```python
from cds.graph import bfs, dfs

print(bfs(g, "A"))
print(dfs(g, "A"))
```

## 3. Shortest Paths & MST

```python
from cds.graph import dijkstra, kruskal_mst

print(dijkstra(g, "A"))  # {node: distance}
print(kruskal_mst(g))  # list of (u, v, weight)
```

## 4. Topological Sort & Cycles

```python
from cds.graph import Graph, topological_sort, has_cycle

dag = Graph(directed=True)
dag.add_edge("a", "b")
dag.add_edge("b", "c")
print(topological_sort(dag))  # valid ordering
print(has_cycle(dag))  # False
```

Run the full demo with `python examples/graph_demo.py`.
