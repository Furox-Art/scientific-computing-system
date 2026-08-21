# MiniGPT Training Tutorial

This is the capstone of the `cds.nlp` stack: a tiny **MiniGPT** (one causal
Transformer block, `d_model=32`) trained from scratch on a Shakespeare excerpt
using a scalar autograd engine and an Adam optimizer. Pure Python end to end.

## 1. The full pipeline

The demo runs four steps:

1. **Encode** the corpus to token ids via the bundled Shakespeare text.
2. **Build** the model — embedding + positional encoding + one causal
   multi-head-attention block + feed-forward + output head.
3. **Train** by sliding a window over the corpus, predicting the next
   character, and updating with Adam.
4. **Sample** text from the trained model.

```python
from cds.nlp.autograd.tensor import Tensor
from cds.nlp.data import TEXT, decode, encode, vocab_size
from cds.nlp.model import MiniGPT
from cds.nlp.optim import Adam
from cds.nlp.training import train_step

ids = encode(TEXT)
print(f"corpus: {len(ids)} chars, vocab_size={vocab_size}")

model = MiniGPT(
    vocab_size=vocab_size,
    d_model=32,
    n_heads=2,
    d_ff=64,
    max_len=32,
    seed=42,
)
```

## 2. The autograd engine

Unlike the float-matrix math in the other nlp demos, MiniGPT's parameters are
`Tensor` objects from `cds.nlp.autograd` — each op records its inputs on a tape
so `backward()` can compute gradients automatically. This is the same design
idea as micrograd / tinygrad, kept minimal so the source stays readable.

## 3. The training loop

The loop is explicit so you can see exactly what each step does:

```python
T = 24
params: list[Tensor] = model.parameters()  # type: ignore[assignment]
optimiser = Adam(params=params, lr=0.005)
n_steps = 200

import random

losses: list[float] = []
for step in range(n_steps):
    start = random.randint(0, max(0, len(ids) - T - 1))
    x = ids[start : start + T]
    y = ids[start + T]
    loss = train_step(model.forward, x, y, optimiser)
    losses.append(loss)
    if step % 20 == 0:
        avg = sum(losses[-20:]) / min(20, len(losses))
        print(f"  step {step:3d}: loss={avg:.4f}")
```

The loss is the cross-entropy of the next-character prediction. Over 200 steps
the moving-average loss falls steadily as the model learns character-level
statistics of the corpus.

> Note: `params: list[Tensor] = model.parameters()` is annotated with
> `# type: ignore[assignment]` because `parameters()` returns
> `list[Parameter]` (a `Tensor` subclass) and Python lists are invariant. The
> runtime assignment is sound; the cast only satisfies strict mypy.

## 4. Sampling

After training, seed the model with a prompt and let it generate one token at a
time, feeding each prediction back in (autoregressive decoding):

```python
prompt = encode("All the ")
out = model.sample(prompt, n_tokens=120)
print(decode(out))
```

The output won't be coherent Shakespeare — 200 steps on a tiny corpus is a
demonstration, not a real model — but it will reflect learned character
frequencies and common bigrams, which is the whole point: you can watch a
language model come into being from first principles.

## Why it matters

Most GPT tutorials hide the mechanics behind a framework. Here, the entire path
— BPE-free char encoding, positional encoding, causal multi-head attention,
feed-forward, layernorm, autograd, Adam, sampling — is open Python you can
read, modify, and instrument. Start from
[`nlp_bpe_demo.md`](nlp_bpe_demo.md) and [`nlp_attention_demo.md`](nlp_attention_demo.md)
for the building blocks.

> Note: the full demo trains for 200 steps and takes about a minute. Run it
> when you don't mind waiting.

Run the full demo:

```bash
python examples/nlp_mini_gpt_demo.py
```
