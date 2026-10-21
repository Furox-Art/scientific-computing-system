"""Natural language processing primitives in pure Python.

Educational, from-scratch implementations of the building blocks used in
modern language models — byte-pair encoding (BPE) tokenisation, the
sinusoidal token / positional embeddings from the original Transformer
paper, scaled dot-product and multi-head self-attention, the
Transformer encoder block (GeLU FFN, LayerNorm, residual), a
scalar-valued reverse-mode autograd engine with SGD/Adam optimisers,
and a high-level training helper.

Designed for teaching, prototyping, and small-model experiments, not
for production-scale training (no NumPy, no BLAS — performance is
deliberately the trade-off for full transparency). No accelerated backend
is bundled; the pure-Python path is the public surface.

Why this module exists inside CDS:
- Demonstrates that the *core ideas* of transformer-era NLP are short,
  readable pure-Python code.
- Provides a reproducible playground for tokenisation experiments
  without pulling in tokenizers / sentencepiece / torch.
- Closes the loop on the educational NLP track: BPE + embeddings +
  attention + autograd = a learner can now train a tiny GPT and
  *see* every gradient flowing back through the graph because the
  engine is ~250 lines, not 50,000.

Scope (v0.10.0b1):

* :func:`~cds.nlp.bpe.train_bpe` — train a BPE vocabulary from a corpus
* :class:`~cds.nlp.bpe.BPETokenizer` — encode / decode / save / load
* :class:`~cds.nlp.embed.TokenEmbedding` — token lookup table
* :class:`~cds.nlp.embed.PositionalEncoding` — sinusoidal positions
* :func:`~cds.nlp.attention.scaled_dot_product_attention`
* :func:`~cds.nlp.attention.multi_head_attention`
* :func:`~cds.nlp.attention.causal_mask` — decoder self-attention mask
* :func:`~cds.nlp.layers.gelu` / :func:`~cds.nlp.layers.layer_norm`
* :func:`~cds.nlp.layers.feed_forward`
* :func:`~cds.nlp.layers.transformer_block`
* :class:`~cds.nlp.autograd.Tensor` / :class:`Parameter` — scalar autograd
* :func:`~cds.nlp.autograd.matmul` — nested-Tensor matmul
* :class:`~cds.nlp.optim.SGD` / :class:`Adam` — optimisers
* :func:`~cds.nlp.training.cross_entropy` / :func:`train_step` — loss + loop
* :func:`~cds.nlp.viz.render_attention_heatmap`   — ASCII attention heatmap
* :func:`~cds.nlp.viz.render_embedding_projection` — ASCII 2-D PCA scatter
* :func:`~cds.nlp.viz.render_training_curve`      — ASCII loss curve

Out of scope for the educational track:
- Mixed precision (FP16 / bfloat16) — meaningful only with the
  Numba backend, deferred for now.
- Dropout, ALiBi, RoPE, GQA / MQA — modern attention refinements
  tracked for a later educational add-on.
- Subword sampling tricks (BPE-Dropout, Unigram LM).
- WordPiece / SentencePiece alternatives.

References:
    - Sennrich, R., Haddow, B., & Birch, A. (2016). "Neural Machine
      Translation of Rare Words with Subword Units." ACL.
    - Vaswani, A. et al. (2017). "Attention Is All You Need." NeurIPS.
    - Gage, P. (1994). "A New Algorithm for Data Compression." C
      Users Journal.
    - Kingma, D. P., & Ba, J. (2014). "Adam: A Method for Stochastic
      Optimization." arXiv:1412.6980.
    - Karpathy, A. (2020). micrograd — the scalar autograd engine
      this module imitates.
"""

from cds.nlp.attention import (
    causal_mask,
    merge_heads,
    multi_head_attention,
    scaled_dot_product_attention,
    softmax,  # scalar softmax (also re-exported from training)
    split_heads,
    transpose,
)
from cds.nlp.autograd import (
    Parameter,
    Tensor,
    add,
    div,
    exp,
    log,
    matmul,
    mul,
    neg,
    no_grad,
    relu,
    sub,
)
from cds.nlp.bpe import (
    BOS,
    EOS,
    PAD,
    SPECIAL_TOKENS,
    UNK,
    BPEMerge,
    BPETokenizer,
    train_bpe,
)
from cds.nlp.embed import (
    PositionalEncoding,
    TokenEmbedding,
    add_positional,
)
from cds.nlp.layers import (
    feed_forward,
    gelu,
    layer_norm,
    transformer_block,
)
from cds.nlp.optim import SGD, Adam, parameters
from cds.nlp.training import cross_entropy, train_step
from cds.nlp.viz import (
    render_attention_heatmap,
    render_embedding_projection,
    render_training_curve,
)

__all__ = [
    # bpe
    "UNK",
    "PAD",
    "BOS",
    "EOS",
    "SPECIAL_TOKENS",
    "BPEMerge",
    "BPETokenizer",
    "train_bpe",
    # embed
    "TokenEmbedding",
    "PositionalEncoding",
    "add_positional",
    # attention
    "softmax",
    "transpose",
    "scaled_dot_product_attention",
    "split_heads",
    "merge_heads",
    "multi_head_attention",
    "causal_mask",
    # layers
    "gelu",
    "layer_norm",
    "feed_forward",
    "transformer_block",
    # autograd
    "Tensor",
    "Parameter",
    "add",
    "sub",
    "mul",
    "div",
    "neg",
    "exp",
    "log",
    "relu",
    "no_grad",
    "matmul",
    # optim
    "SGD",
    "Adam",
    "parameters",
    # training
    "cross_entropy",
    "train_step",
    # viz
    "render_attention_heatmap",
    "render_embedding_projection",
    "render_training_curve",
]
