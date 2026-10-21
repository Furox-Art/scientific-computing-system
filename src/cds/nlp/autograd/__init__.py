"""Reverse-mode automatic differentiation in pure Python.

A from-scratch autograd engine inspired by Karpathy's micrograd,
organised as a small package:

* :mod:`cds.nlp.autograd.tensor` — the :class:`Tensor` graph node,
  :class:`Parameter`, and the elementwise arithmetic ops
  (:func:`add`, :func:`sub`, :func:`mul`, :func:`div`, :func:`neg`).
* :mod:`cds.nlp.autograd.ops` — transcendental unary ops
  (:func:`exp`, :func:`log`, :func:`relu`) and :func:`matmul` on
  nested :class:`Tensor` matrices.
* :mod:`cds.nlp.autograd._grad` — the grad-tracking context
  (:func:`no_grad`) and the shared :func:`_track` graph helper.

How it works:
- :class:`Tensor` wraps a Python ``float`` with a graph node
  (data, gradient, backward closure, child nodes).
- Ops (``+``, ``*``, ``@``, ``exp``, ``log``, ``relu``) record
  themselves on the graph via ``_backward`` closures during the
  forward pass.
- :meth:`Tensor.backward` runs a topological reverse traversal and
  fills ``.grad`` on every leaf.

Why this lives in CDS:
- Closes the loop on the educational NLP track: with BPE + embeddings
  + attention in hand, a learner can now watch a tiny GPT actually fit
  Shakespeare — and *see* every gradient flowing back through the graph
  because the engine is ~250 lines, not 50,000.
- The matmul hot-path is pure Python today: there is no Numba backend,
  so every gradient stays visible for the educational narrative.

References:
    - Karpathy, A. (2020). "micrograd — A tiny scalar-valued autograd
      engine and a neural net library on top of it."
      https://github.com/karpathy/micrograd
    - Karpathy, A. (2023). "Let's build GPT: from scratch, in code,
      spelled out." (the training pipeline this engine plugs into.)
    - PyTorch's autograd docs — for the ``requires_grad`` /
      ``backward()`` API conventions this module imitates.
"""

from __future__ import annotations

from cds.nlp.autograd._grad import no_grad
from cds.nlp.autograd.ops import exp, log, matmul, relu
from cds.nlp.autograd.tensor import (
    Parameter,
    Tensor,
    add,
    div,
    mul,
    neg,
    sub,
)

__all__ = [
    "Tensor",
    "Parameter",
    "no_grad",
    "add",
    "sub",
    "mul",
    "div",
    "neg",
    "exp",
    "log",
    "relu",
    "matmul",
]
