# BPE Tokenizer & Embeddings Tutorial

`cds.nlp` is an educational NLP stack built from scratch: a Byte-Pair-Encoding
(BPE) tokenizer, sinusoidal positional encoding, and token embeddings, the
exact front-end a Transformer consumes. No HuggingFace, no NumPy.

## 1. Train a BPE tokenizer

`train_bpe` learns merge rules from a corpus by repeatedly merging the most
frequent adjacent pair, exactly as in the original Sennrich et al. (2016)
algorithm.

```python
from cds.nlp import train_bpe

corpus = (
    "the quick brown fox jumps over the lazy dog "
    "the quick brown fox jumps over the lazy dog "
    "she sells seashells by the seashore "
    "she sells seashells by the seashore "
)
tokenizer = train_bpe(corpus, vocab_size=80, min_frequency=2)
print(f"vocab size: {tokenizer.vocab_size}")  # 78
print(f"merges learned: {len(tokenizer.merges)}")
```

The first merges combine frequent character pairs:

```
first three merges:
  ('h', 'e') -> 'he'
  ('he', '</w>') -> 'he</w>'
  ('t', 'he</w>') -> 'the</w>'
```

(`</w>` is the end-of-word marker that lets BPE recover word boundaries.)

## 2. Encode and decode

```python
text = "the quick brown fox"
ids = tokenizer.encode(text)
print(f"ids:    {ids}")  # [33, 45, 50, 53]
print(f"tokens: {[tokenizer.id_to_token[i] for i in ids]}")
#   ['the</w>', 'quick</w>', 'brown</w>', 'fox</w>']
print(f"decoded: {tokenizer.decode(ids)!r}")  # 'the quick brown fox'
```

`decode` is a clean round-trip, verify with an assertion in your own code.

## 3. Token + positional embeddings

The embedding pipeline is two layers: a learned `TokenEmbedding` table plus
fixed sinusoidal `PositionalEncoding`, summed element-wise via
`add_positional`.

```python
from cds.nlp import (
    PositionalEncoding,
    TokenEmbedding,
    add_positional,
)

d_model = 16
table = TokenEmbedding(vocab_size=tokenizer.vocab_size, d_model=d_model)
pe = PositionalEncoding(max_len=len(ids) + 4, d_model=d_model)

tokens = table.forward(ids)  # (seq_len, d_model)
combined = add_positional(tokens, pe)  # token + position
print(f"embedding shape: {len(combined)} x {len(combined[0])}")  # 4 x 16
```

The first position's vector is the literal sum of its token and position rows:

```
first token position vector (first 4 dims):
  token:  [-0.1726, 0.1360, 0.1900, -0.0182]
  pos:    [0.0, 1.0, 0.0, 1.0]
  sum:    [-0.1726, 1.1360, 0.1900, 0.9818]
```

## 4. Save and reload

A trained tokenizer serializes to JSON and round-trips losslessly:

```python
tokenizer.save("examples/_demo_tokenizer.json")
loaded = type(tokenizer).load("examples/_demo_tokenizer.json")
assert loaded.encode(text) == ids
```

## Where this fits

The output of `add_positional` is exactly what `multi_head_attention` expects
(see [`nlp_attention_demo.md`](nlp_attention_demo.md)) and what `MiniGPT`
consumes (see [`nlp_mini_gpt_demo.md`](nlp_mini_gpt_demo.md)). The whole
front-to-back Transformer is readable pure Python.

Run the full demo:

```bash
python examples/nlp_bpe_demo.py
```
