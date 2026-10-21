"""Scaled dot-product and multi-head attention in pure Python.

Implements the two attention building blocks from Vaswani et al. (2017)
"Attention Is All You Need" without depending on ``torch`` or any tensor
library:

* :func:`scaled_dot_product_attention` — the
  ``softmax(Q K^T / sqrt(d_k) + mask) V`` formula from §3.2.1.
* :func:`multi_head_attention` — the multi-head variant from §3.2.2
  that projects inputs to Q/K/V, splits the last dimension into
  ``n_heads`` slices, runs the per-head attention in parallel, then
  concatenates and re-projects with a learned output matrix.
* :func:`causal_mask` — an additive upper-triangular ``-inf`` mask that
  turns this block into a decoder self-attention layer (each position
  only attends to itself and earlier positions).

Performance note: ``matmul`` is O(m · p · n) on nested lists. This is
deliberately slow — it is the trade-off for the educational track. The
:class:`cds.nlp.embed.TokenEmbedding` weights are read-only at
inference time, so the attention block is still functional for
hand-rolled demos. There is no accelerated backend yet; this pure-Python
path is the supported implementation.

References:
    - Vaswani, A. et al. (2017). "Attention Is All You Need." NeurIPS.
      §3.2 (Scaled Dot-Product Attention) and §3.2.2 (Multi-Head).
    - Karpathy, A. (2023). "Let's build GPT: from scratch, in code,
      spelled out." (decoder-only attention pattern.)
"""

from __future__ import annotations

import math
from typing import Final

from cds.nlp.embed import _make_matrix  # reuse the matrix allocator

__all__ = [
    "softmax",
    "matmul",
    "transpose",
    "scaled_dot_product_attention",
    "split_heads",
    "merge_heads",
    "multi_head_attention",
    "causal_mask",
]


# ---------------------------------------------------------------------- #
# Building blocks
# ---------------------------------------------------------------------- #


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """Standard matrix multiply for nested-list matrices.

    Computes ``C = A @ B`` where ``A`` is ``(m, p)`` and ``B`` is
    ``(p, n)``. Returns an ``(m, n)`` matrix. Empty inputs return
    ``[]`` (no type check is more useful than letting the loops be
    a no-op). Raises :class:`ValueError` on shape mismatch.
    """
    if not a or not b:
        return []
    m = len(a)
    p = len(a[0])
    if p != len(b):
        raise ValueError(f"matmul shape mismatch: A is ({m}, {p}) but B has {len(b)} rows")
    n = len(b[0])
    # Allocate the result matrix once — this is the dominant memory
    # cost for the educational pipeline, so we don't double-allocate.
    result = _make_matrix(m, n)
    for i in range(m):
        row_a = a[i]
        row_result = result[i]
        for k in range(p):
            aik = row_a[k]
            row_b = b[k]
            for j in range(n):
                row_result[j] += aik * row_b[j]
    return result


def transpose(m: list[list[float]]) -> list[list[float]]:
    """Transpose a nested-list matrix."""
    if not m:
        return []
    rows = len(m)
    cols = len(m[0])
    return [[m[r][c] for r in range(rows)] for c in range(cols)]


def softmax(x: list[float]) -> list[float]:
    """Numerically stable softmax for a 1-D list.

    Subtracts the max before ``exp`` to avoid overflow on large
    inputs; the resulting distribution is invariant to the shift.
    Empty input returns ``[]``; the result always sums to 1.0
    (within float precision).
    """
    if not x:
        return []
    m = max(x)
    exps = [math.exp(xi - m) for xi in x]
    total = sum(exps)
    if total == 0.0 or math.isnan(total):
        # Degenerate: all inputs were -inf (``-inf - -inf`` is NaN, so
        # ``exp`` returns 0 / NaN and the sum is 0 or NaN). Return a
        # uniform distribution to keep downstream matmul finite.
        n = len(x)
        return [1.0 / n] * n
    return [e / total for e in exps]


# ---------------------------------------------------------------------- #
# Attention
# ---------------------------------------------------------------------- #


def scaled_dot_product_attention(
    q: list[list[float]],
    k: list[list[float]],
    v: list[list[float]],
    mask: list[list[float]] | None = None,
) -> list[list[float]]:
    """Compute ``softmax(Q K^T / sqrt(d_k) + mask) V``.

    Args:
        q: Query matrix of shape ``(n_q, d_k)``.
        k: Key matrix of shape ``(n_k, d_k)``.
        v: Value matrix of shape ``(n_k, d_v)``. ``n_k`` must equal
            ``n_q`` for self-attention; cross-attention uses a
            different ``n_k``.
        mask: Optional additive mask of shape ``(n_q, n_k)``. Use
            ``0.0`` to keep a position and ``-inf`` to suppress it.
            The mask is *added* to the scaled scores before softmax.

    Returns:
        A matrix of shape ``(n_q, d_v)``.
    """
    if not q or not k or not v:
        return []
    d_k = len(q[0])
    if d_k == 0:
        raise ValueError("q has zero width (d_k = 0)")
    n_q = len(q)
    n_k = len(k)
    if len(v) != n_k:
        raise ValueError(f"k has {n_k} rows but v has {len(v)}")
    if len(k[0]) != d_k:
        raise ValueError(f"q and k widths differ: {d_k} vs {len(k[0])}")
    if mask is not None and (len(mask) != n_q or len(mask[0]) != n_k):
        raise ValueError(
            f"mask shape {len(mask)}x{len(mask[0]) if mask else 0} "
            f"does not match attention shape {n_q}x{n_k}"
        )

    # scores = Q K^T / sqrt(d_k) — the scaling keeps the dot products
    # in a regime where softmax gradients are well-behaved.
    scale = 1.0 / math.sqrt(d_k)
    k_t = transpose(k)
    scores = matmul(q, k_t)
    for i in range(n_q):
        row = scores[i]
        for j in range(n_k):
            row[j] = row[j] * scale + (mask[i][j] if mask is not None else 0.0)

    # softmax row-wise, then multiply by V.
    attn_weights = [softmax(row) for row in scores]
    return matmul(attn_weights, v)


# ---------------------------------------------------------------------- #
# Multi-head
# ---------------------------------------------------------------------- #


def split_heads(
    x: list[list[float]],
    n_heads: int,
) -> list[list[list[float]]]:
    """Split the last dim of ``(n, d_model)`` into ``n_heads`` slices.

    Returns a list ``[n_heads][n][d_head]`` where ``d_head = d_model /
    n_heads``. Equivalent to ``x.view(n, n_heads, d_head).transpose(0, 1)``
    in PyTorch's convention.
    """
    if n_heads <= 0:
        raise ValueError(f"n_heads must be > 0, got {n_heads}")
    if not x:
        return [[] for _ in range(n_heads)]
    n = len(x)
    d_model = len(x[0])
    if d_model % n_heads != 0:
        raise ValueError(f"d_model {d_model} is not divisible by n_heads {n_heads}")
    d_head = d_model // n_heads
    return [
        [[x[i][h * d_head + j] for j in range(d_head)] for i in range(n)] for h in range(n_heads)
    ]


def merge_heads(
    heads: list[list[list[float]]],
) -> list[list[float]]:
    """Inverse of :func:`split_heads`: ``(n_heads, n, d_head) → (n, d_model)``."""
    if not heads:
        return []
    n_heads = len(heads)
    n = len(heads[0])
    if n == 0:
        return [[] for _ in range(n)]
    d_head = len(heads[0][0])
    d_model = n_heads * d_head
    out: list[list[float]] = _make_matrix(n, d_model)
    for h in range(n_heads):
        for i in range(n):
            for j in range(d_head):
                out[i][h * d_head + j] = heads[h][i][j]
    return out


def multi_head_attention(
    x: list[list[float]],
    w_q: list[list[float]],
    w_k: list[list[float]],
    w_v: list[list[float]],
    w_o: list[list[float]],
    n_heads: int,
    mask: list[list[float]] | None = None,
) -> list[list[float]]:
    """Multi-head self-attention (Vaswani 2017 §3.2.2).

    Args:
        x: Input sequence, shape ``(n, d_model)``.
        w_q: Projection matrix for queries, shape ``(d_model, d_model)``.
        w_k: Projection matrix for keys, shape ``(d_model, d_model)``.
        w_v: Projection matrix for values, shape ``(d_model, d_model)``.
        w_o: Output projection, shape ``(d_model, d_model)``.
        n_heads: Number of attention heads. Must divide ``d_model``.
        mask: Optional additive mask broadcast across all heads,
            shape ``(n, n)``.

    Returns:
        Output sequence, shape ``(n, d_model)``.
    """
    if not x:
        return []
    d_model = len(x[0])
    if d_model % n_heads != 0:
        raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")

    # 1. Project to Q, K, V.
    q = matmul(x, w_q)
    k = matmul(x, w_k)
    v = matmul(x, w_v)

    # 2. Split into heads, run per-head attention.
    qh = split_heads(q, n_heads)
    kh = split_heads(k, n_heads)
    vh = split_heads(v, n_heads)
    head_outputs = [scaled_dot_product_attention(qh[h], kh[h], vh[h], mask) for h in range(n_heads)]

    # 3. Concatenate heads, then project to the output space.
    merged = merge_heads(head_outputs)
    return matmul(merged, w_o)


# ---------------------------------------------------------------------- #
# Masks
# ---------------------------------------------------------------------- #


_NEG_INF: Final[float] = float("-inf")


def causal_mask(n: int) -> list[list[float]]:
    """Upper-triangular ``-inf`` mask for decoder self-attention.

    Position ``i`` may attend to positions ``0..=i`` and nothing else.
    The mask is added to the pre-softmax scores, so the ``-inf`` entries
    become zero probability after softmax.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    return [[0.0 if j <= i else _NEG_INF for j in range(n)] for i in range(n)]
