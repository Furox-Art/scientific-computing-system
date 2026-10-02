# Multi-Head Attention Tutorial

This demo extends the BPE + embeddings pipeline with a real **multi-head
self-attention block**, the core of the Transformer (Vaswani et al., 2017).
Everything runs in pure Python; the attention weights are dumped as an ASCII
heat map so the example is self-contained.

## 1. Tokenize and embed

Reuse the BPE tokenizer and positional embeddings from
[`nlp_bpe_demo.md`](nlp_bpe_demo.md):

```python
from cds.nlp import (
    PositionalEncoding,
    TokenEmbedding,
    add_positional,
    train_bpe,
)

corpus = (
    "the quick brown fox jumps over the lazy dog "
    "the quick brown fox jumps over the lazy dog "
    "she sells seashells by the seashore "
    "she sells seashells by the seashore "
)
tokenizer = train_bpe(corpus, vocab_size=80, min_frequency=2)

text = "the quick brown fox"
ids = tokenizer.encode(text)  # [33, 45, 50, 53]

d_model = 16
table = TokenEmbedding(vocab_size=tokenizer.vocab_size, d_model=d_model)
pe = PositionalEncoding(max_len=len(ids) + 4, d_model=d_model)
combined = add_positional(table.forward(ids), pe)
print(f"embedding shape: {len(combined)} x {len(combined[0])}")  # 4 x 16
```

## 2. Run multi-head attention with a causal mask

`multi_head_attention` takes linear projection weights (`w_q`, `w_k`, `w_v`,
`w_o`), each a `d_model × d_model` matrix, plus the head count. The weights are
not built by the library — you supply them — so this snippet initialises them
from a seeded RNG exactly as `examples/nlp_attention_demo.py` does.

The `mask` argument here is a **causal mask**: position `i` may only attend to
positions `<= i`, which is what makes this *decoder-style* attention (used in
GPT).

```python
import random

from cds.nlp import causal_mask, multi_head_attention

n_heads = 2

rng = random.Random(0xBEEF)
w_q, w_k, w_v, w_o = [
    [[rng.uniform(-0.3, 0.3) for _ in range(d_model)] for _ in range(d_model)] for _ in range(4)
]

mask = causal_mask(len(ids))
output = multi_head_attention(combined, w_q, w_k, w_v, w_o, n_heads, mask=mask)
print(f"attention output shape: {len(output)} x {len(output[0])}")  # 4 x 16
```

The scaled dot-product inside each head is
`softmax((Q Kᵀ) / √d_k) · V`, split across `n_heads` independent subspaces and
then re-projected through `w_o`.

## 3. Inspect the attention weights

The demo recomputes head-0's attention matrix and renders it as an ASCII heat
map (low → high: space `▁▂▃▄▅▆▇█`). Rows are query positions; bright cells are
high attention:

```text
attention weights, head 0 (rows = query position):
█
▄▃
▃▂▂
▂▂▁▁
```

The lower-triangular pattern is the causal mask made visible: each row has
non-zero weight only up to its own column.

## 4. Verify the causal property

A clean invariant check: position 0's output must depend **only** on `v[0]`, so
perturbing `v[1:]` must leave it unchanged. The demo asserts this with a
tolerance of `1e-9`:

Perturb the **input** at every position except 0, keep position 0 byte-identical,
and re-run. Because the causal mask limits position 0 to attend only to
position 0, its output cannot move; the later positions are free to change.

```python
rng2 = random.Random(0xCAFE)
perturbed = [list(row) for row in combined]
for i in range(1, len(perturbed)):
    perturbed[i] = [rng2.uniform(-1.0, 1.0) for _ in range(d_model)]

alt_output = multi_head_attention(perturbed, w_q, w_k, w_v, w_o, n_heads, mask=mask)

print(f"position 0 before: {output[0][0]:.9f}")
print(f"position 0 after:  {alt_output[0][0]:.9f}")
assert abs(output[0][0] - alt_output[0][0]) < 1e-9
print("OK — output at position 0 is invariant when the other positions change")
```

Real output:

```text
position 0 before: -0.053500394
position 0 after:  -0.053500394
OK — output at position 0 is invariant when the other positions change
```

`examples/nlp_attention_demo.py` performs the equivalent check internally by
perturbing head 1's value vectors and reports:

```text
verifying causal property at position 0…
  first output dim: -0.053500
  OK — output at position 0 is invariant under v[1:] changes
```

## Why it matters

This is the literal mechanism behind autoregressive language models. Because
the whole block is readable Python, you can step through the softmax, the
head-split, and the mask application to see exactly how information flows.
Continue to [`nlp_mini_gpt_demo.md`](nlp_mini_gpt_demo.md) to see attention
stacked into a full GPT and trained end-to-end.

Run the full demo:

```bash
python examples/nlp_attention_demo.py
```