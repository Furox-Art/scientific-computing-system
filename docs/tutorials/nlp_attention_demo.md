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
#   embedding shape: 4 x 16
```

## 2. Run multi-head attention with a causal mask

`multi_head_attention` takes linear projection weights (`w_q`, `w_k`, `w_v`,
`w_o`) and the number of heads. The `mask` argument here is a **causal mask**:
position `i` may only attend to positions `<= i`, which is what makes this a
*decoder-style* attention (used in GPT).

```python
from cds.nlp import causal_mask, multi_head_attention

n_heads = 2
# w_q, w_k, w_v, w_o are (d_model, d_model) weight matrices — see the
# example file for a small seeded initializer.
mask = causal_mask(len(ids))
output = multi_head_attention(combined, w_q, w_k, w_v, w_o, n_heads, mask=mask)
#   attention output shape: 4 x 16
```

The scaled dot-product inside each head is
`softmax((Q Kᵀ) / √d_k) · V`, split across `n_heads` independent subspaces and
then re-projected through `w_o`.

## 3. Inspect the attention weights

The demo recomputes head-0's attention matrix and renders it as an ASCII heat
map (low → high: space `▁▂▃▄▅▆▇█`). Rows are query positions; bright cells are
high attention:

```
attention weights, head 0 (rows = query position):
█
▄▃
▃▂▂
▂▂▁▁
```

The lower-triangular pattern is the causal mask made visible, each row only
has non-zero weight up to its own column.

## 4. Verify the causal property

A clean invariant check: position 0's output must depend **only** on `v[0]`.
If we perturb `v[1:]`, position 0's output should not change:

```python
# Swap v at positions > 0 with random values, re-run attention.
assert abs(output[0][0] - alt_output[0][0]) < 1e-9
print("OK — output at position 0 is invariant under v[1:] changes")
```

```
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
