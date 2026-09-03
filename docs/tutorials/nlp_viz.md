# Visualising NLP Internals

The `cds.nlp.viz` module renders the three things a learner most wants to
*see* when reading about transformers: attention, embeddings, and the
training-loss curve, as ASCII, so you need no plotting backend. Every
renderer returns a `str`, which means they compose under `print()`, log
cleanly, and are trivially testable.

## 1. The training-loss curve

```python
from cds.nlp import render_training_curve

losses = [3.5, 3.1, 2.8, 2.55, 2.3, 2.1, 1.95, 1.82, 1.7, 1.6]
print(render_training_curve(losses, width=50, height=10))
```

The curve is min-max normalised to the canvas. A single point or an
all-equal series is handled safely (no divide-by-zero); narrow widths
(`width < 10`) are also safe, the x-axis step label right-aligns rather
than raising a format error.

## 2. The attention heatmap

```python
from cds.nlp import render_attention_heatmap

attn = [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.05, 0.15, 0.8]]
tokens = ["the", "cat", "sat"]
print(render_attention_heatmap(attn, tokens, tokens))
```

Weights are normalised across the whole matrix per render, then mapped to
nine ASCII shades (`' '` → `'#'`). The diagonal of a causal or
identity-style attention pattern lights up as the darkest cells. Shape
mismatches between the matrix, the row tokens, and the column tokens each
raise a labelled `ValueError` so a wrong call is easy to spot.

## 3. The embedding projection

```python
from cds.nlp import render_embedding_projection

# Six 3-D embedding vectors; imagine six vocabulary entries.
emb = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1]]
labels = ["a", "b", "c", "ab", "bc", "ac"]
print(render_embedding_projection(emb, labels=labels, top_n=6))
```

The projection is **real PCA**, not a placeholder: the covariance matrix of
the embeddings is built in pure Python, then
[`cds.math_utils.linalg.power_iteration`](../api.md#cds.math_utils.power_iteration)
recovers the top-2 eigenvectors (the second via deflation). This is the
project's signature "slow but honest" trade-off, the math is exactly what
`sklearn.decomposition.PCA` does, just without the BLAS.

`top_n` keeps large vocabularies readable by plotting only the
highest-variance points along PC1 (pass `top_n=0` or a negative value to
render every point).

## Why ASCII?

Three reasons, in priority order:

1. **Zero-dependency.** No matplotlib import on the default path keeps
   `cds` installable with nothing but the standard library.
2. **Composability.** A `str` return value logs, prints, and diffs in a
   test exactly like any other value.
3. **Teaching clarity.** The renderer source is short enough to read in
   one sitting, which is the point of the whole `cds.nlp` track.

For publication-quality plots, export the matrices
(`attn_weights`, the `_pca_2d` result, `losses`) and plot them with your
own toolchain, the data path is the same. The runnable demo at
[`examples/nlp_viz_demo.py`](https://github.com/Furox-Art/scientific-computing-system/blob/main/examples/nlp_viz_demo.py)
wires all three renderers to a tiny live tokenizer + embedding pass.
