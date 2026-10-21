"""High-level training helpers for cds.nlp.

Combines the autograd engine (:mod:`cds.nlp.autograd`) and the
optimizers (:mod:`cds.nlp.optim`) into a small surface area for
the educational NLP track:

* :func:`cross_entropy` — softmax + negative log-likelihood, with
  built-in numerical stability (subtract the max logit before
  exponentiating).
* :func:`train_step` — one forward / backward / optimiser cycle for
  a model that exposes a ``forward(x) -> logits`` callable. Returns
  the loss scalar so callers can plot a learning curve.
* :func:`softmax` — vector softmax (Python list, NOT a :class:`Tensor`)
  used by the loss.

Scope:
- Mini-batch SGD on the educational track (batch size 4-32). For
  larger batches the user can call :func:`train_step` in a loop.
- No mixed precision, no gradient accumulation, no learning-rate
  scheduling — these are not implemented yet.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cds.nlp.autograd import Tensor
    from cds.nlp.optim import SGD, Adam


__all__ = [
    "cross_entropy",
    "softmax",
    "train_step",
]


# ---------------------------------------------------------------------- #
# Vector utilities (operate on plain Python lists, not Tensors)
# ---------------------------------------------------------------------- #


def softmax(logits: list[float]) -> list[float]:
    """Numerically stable softmax for a 1-D list of logits."""
    if not logits:
        return []
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


# ---------------------------------------------------------------------- #
# Loss
# ---------------------------------------------------------------------- #


def cross_entropy(
    logits: list[float] | list[Tensor],
    target: int,
) -> Tensor:
    """Softmax + negative log-likelihood for one example.

    Computes ``-log(softmax(logits)[target])`` in numerically stable
    form (subtract the max logit before exponentiating). The result
    is a :class:`cds.nlp.autograd.Tensor` so the optimiser can
    backpropagate through it.

    Args:
        logits: Output of the model's final linear layer (unnormalised
            log-probabilities), length ``V`` for vocab size ``V``. May
            be a list of Python floats (no autograd — useful for
            sanity checks) or :class:`cds.nlp.autograd.Tensor` values
            (loss is connected to the autograd graph).
        target: Index of the correct next token in ``[0, V)``.

    Returns:
        Scalar :class:`Tensor` — the cross-entropy loss for this
        example. ``backward()`` on it populates gradients on every
        model parameter that contributed (only if the logits were
        Tensors).
    """
    # Local import keeps the autograd module from being pulled into
    # the public ``cds.nlp`` namespace through this file.
    from cds.nlp.autograd import Tensor, exp, log

    if not logits:
        raise ValueError("cross_entropy: logits is empty")
    if not 0 <= target < len(logits):
        raise ValueError(f"cross_entropy: target {target} out of range [0, {len(logits)})")
    # Normalise to float values for the numerics — Tensor inputs
    # are unwrapped via ``.data``; float inputs pass through.
    raw = [li.data if isinstance(li, Tensor) else float(li) for li in logits]
    m = max(raw)
    m_const = Tensor(data=m, requires_grad=False)
    acc = Tensor(data=0.0, requires_grad=False)
    for li, v in zip(logits, raw):
        # Promote float logits to no-grad constants so the loss
        # itself isn't part of the graph when the user passes raw
        # floats (e.g. in tests). For Tensor logits, ``li - m_const``
        # uses the operator overload and keeps the graph.
        if isinstance(li, Tensor):
            shifted = li - m_const
        else:
            shifted = Tensor(data=v - m, requires_grad=False)
        acc = acc + exp(shifted)
    lse = m_const + log(acc)
    # Loss = LSE - logit[target] — keep the graph connection only
    # when the user passed Tensor logits.
    if isinstance(logits[target], Tensor):
        return lse - logits[target]
    return lse - Tensor(data=raw[target], requires_grad=False)


# ---------------------------------------------------------------------- #
# Training loop
# ---------------------------------------------------------------------- #


def train_step(
    model_fn: Callable[[list[int]], list[float] | list[Tensor]],
    x: list[int],
    y: int,
    optimiser: SGD | Adam,
) -> float:
    """Run one training step on a single example.

    Performs:
        1. ``logits = model_fn(x)`` (the user-supplied forward pass)
        2. ``loss = cross_entropy(logits, y)``
        3. ``optimiser.zero_grad(); loss.backward(); optimiser.step()``

    The model's parameters must be exposed somewhere the optimiser
    can see them — typically by collecting them into a list at
    construction time and passing that list to the optimiser.

    Args:
        model_fn: Pure function ``x -> logits`` for one example. The
            autograd graph is built inside this function (when the
            return type is ``list[Tensor]``); the function should
            return the model's pre-softmax output for the next-token
            prediction.
        x: Input token ids (length ``T`` for a ``T``-token context).
        y: Target next-token id.
        optimiser: :class:`cds.nlp.optim.SGD` or :class:`Adam` whose
            ``params`` list contains every :class:`Parameter` reachable
            from ``model_fn(x)``.

    Returns:
        The loss as a plain Python float (snapshot of ``loss.data``).
    """
    optimiser.zero_grad()
    logits = model_fn(x)
    loss = cross_entropy(logits, y)
    if not loss.requires_grad:
        raise RuntimeError(
            "train_step: model_fn returned plain floats — autograd "
            "needs the forward pass to produce Tensor logits so the "
            "loss can chain back to model parameters."
        )
    loss.backward()
    optimiser.step()
    return float(loss.data)
