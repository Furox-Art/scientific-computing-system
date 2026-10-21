"""Transcendental ops and matrix multiply for the autograd engine.

These ops depend only on :class:`~cds.nlp.autograd.tensor.Tensor` and
:func:`~cds.nlp.autograd._grad._track` (one-way edges), so they live
in their own module to keep the import graph acyclic.
"""

from __future__ import annotations

import math

from cds.nlp.autograd._grad import _track
from cds.nlp.autograd.tensor import Tensor

# ---------------------------------------------------------------------- #
# Transcendental / unary ops
# ---------------------------------------------------------------------- #


def exp(a: Tensor) -> Tensor:
    """``exp(a)`` with reverse-mode grad ``∂/∂a = exp(a) * out.grad``."""

    def _backward() -> None:
        a.grad += math.exp(a.data) * out.grad

    out = Tensor(data=math.exp(a.data))
    return _track(out, (a,), _backward)


def log(a: Tensor) -> Tensor:
    """Natural log. ``∂/∂a = out.grad / a.data``.

    Raises :class:`ValueError` for non-positive input — the gradient
    is undefined there.
    """

    def _backward() -> None:
        # Unreachable: the forward guard below rejects ``a.data <= 0`` before
        # any backward is queued, so ``a.data`` can never be 0 here. The check
        # is kept as a defensive assertion against future callers that bypass
        # the public ``log`` entrypoint.
        if a.data == 0.0:  # pragma: no cover
            raise ValueError("log(0) gradient is undefined")
        a.grad += out.grad / a.data

    if a.data <= 0:
        raise ValueError(f"log requires positive input, got {a.data}")
    out = Tensor(data=math.log(a.data))
    return _track(out, (a,), _backward)


def relu(a: Tensor) -> Tensor:
    """Rectified linear unit. ``∂/∂a = out.grad if a > 0 else 0``."""

    def _backward() -> None:
        if a.data > 0:
            a.grad += out.grad

    out = Tensor(data=max(0.0, a.data))
    return _track(out, (a,), _backward)


# ---------------------------------------------------------------------- #
# Vectorised op: matmul on nested Tensor lists
# ---------------------------------------------------------------------- #


def matmul(a: list[list[Tensor]], b: list[list[Tensor]]) -> list[list[Tensor]]:
    """Matrix multiply for nested :class:`Tensor` matrices.

    ``a`` has shape ``(m, p)``; ``b`` has shape ``(p, n)``. The result
    is an ``(m, n)`` matrix whose entries are :class:`Tensor` nodes
    connected to the inputs via the dep graph.

    Implementation is the textbook triple loop. Each inner product
    uses scalar autograd (one multiply + accumulate) so every entry
    in the result gets a backward fn that propagates to the
    contributing ``a`` and ``b`` entries.

    For a 50K-param model this is the hot path — the pure-Python
    implementation is the supported one; there is no Numba backend yet,
    and any future accelerator must preserve these autograd semantics.
    """
    if not a or not b or not a[0] or not b[0]:
        return []
    m = len(a)
    p = len(a[0])
    if len(b) != p:
        raise ValueError(f"matmul shape mismatch: a has {p} cols, b has {len(b)} rows")
    n = len(b[0])
    # Allocate result as a (m, n) matrix of zero-constant Tensors so
    # we can mutate them in place. The constant stays out of the
    # autograd graph because ``_track`` skips it (no grad children).
    zero = Tensor(data=0.0, requires_grad=False)
    out: list[list[Tensor]] = [[zero for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            acc = Tensor(data=0.0, requires_grad=False)
            for k in range(p):
                # acc += a[i][k] * b[k][j]  (scalar autograd chain)
                prod = _tracked_mul(a[i][k], b[k][j])
                acc = _tracked_add(acc, prod)
            out[i][j] = acc
    return out


def _tracked_add(a: Tensor, b: Tensor) -> Tensor:
    """Internal helper: ``a + b`` with autograd graph attached.

    Public :func:`~cds.nlp.autograd.tensor.add` works on Tensors but is
    re-exported with Python operator overloading; this function is the
    local version used by :func:`matmul`'s inner loop to avoid recursion
    through ``__add__``.
    """

    def _backward() -> None:
        a.grad += out.grad
        b.grad += out.grad

    out = Tensor(data=a.data + b.data)
    return _track(out, (a, b), _backward)


def _tracked_mul(a: Tensor, b: Tensor) -> Tensor:
    """Internal helper: ``a * b`` with autograd graph attached."""

    def _backward() -> None:
        a.grad += b.data * out.grad
        b.grad += a.data * out.grad

    out = Tensor(data=a.data * b.data)
    return _track(out, (a, b), _backward)
