"""Shared pytest fixtures for the CDS test suite.

These fixtures centralize object construction that was previously copy-pasted
across test modules (``test_hypothesis.py``, ``test_core.py``, and others), so
the tests can focus on behavior instead of boilerplate construction.

Scope rules
-----------
- ``function`` (default) is used for everything that could be mutated by a
  test, so each test gets a fresh instance and cannot leak state into the next.
- ``session`` would only be safe for truly immutable, expensive-to-build
  objects; nothing here qualifies, so every fixture stays ``function``-scoped.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator

import pytest

from cds.core.models import Domain, Hypothesis, HypothesisStatus


@pytest.fixture(autouse=True)
def _close_pyplot_figures() -> Iterator[None]:
    """Close every open pyplot figure after each test.

    ``matplotlib.pyplot`` keeps a module-level registry of open figures, so a
    suite that builds many charts (``test_plot_charts``, CLI ``--file`` tests)
    accumulates them until matplotlib emits its ``More than 20 figures``
    RuntimeWarning and memory grows. Closing everything per-test bounds memory
    and keeps the suite warning-clean.

    Guarded via ``sys.modules`` so environments without matplotlib (the lean
    ``[test]`` CI cells) never trigger the optional-import machinery — the
    fixture simply becomes a no-op there.
    """
    yield
    pyplot = sys.modules.get("matplotlib.pyplot")
    if pyplot is not None:
        pyplot.close("all")


@pytest.fixture
def make_hypothesis() -> Callable[..., Hypothesis]:
    """Factory fixture returning a configurable ``Hypothesis``.

    Call ``make_hypothesis(**overrides)`` to tweak any field; everything else
    defaults to a valid, fully-populated hypothesis. This keeps each test
    readable while letting callers vary exactly the field under test without
    re-stating the seven required arguments every time.

    Example
    -------
    >>> def test_x(make_hypothesis):
    ...     h = make_hypothesis(confidence=0.9)
    """

    def _build(**overrides: object) -> Hypothesis:
        defaults: dict[str, object] = {
            "id": "h-1",
            "statement": "Default hypothesis statement",
            "domain": Domain.PHYSICS,
            "research_question": "Default research question",
            "rationale": "Default rationale",
            "confidence": 0.5,
            "assumptions": ["assumption A"],
            "predictions": ["prediction A"],
            "tags": ["tag-A"],
            "sources": ["src-1"],
        }
        defaults.update(overrides)
        return Hypothesis(**defaults)  # type: ignore[arg-type]

    return _build


@pytest.fixture
def hypothesis(make_hypothesis: Callable[..., Hypothesis]) -> Hypothesis:
    """A single, fully-populated ``Hypothesis`` (all optional fields set).

    For tests that just need one canonical object. Use ``make_hypothesis``
    directly when you need to vary fields per test.
    """
    return make_hypothesis()


@pytest.fixture
def minimal_hypothesis(make_hypothesis: Callable[..., Hypothesis]) -> Hypothesis:
    """A ``Hypothesis`` carrying only its required fields.

    Exercises the empty-default paths (``rationale is None``, empty
    ``assumptions`` / ``predictions`` / ``tags``) without each test having to
    spell them out.
    """
    return make_hypothesis(
        rationale=None,
        assumptions=[],
        predictions=[],
        tags=[],
        sources=[],
    )


@pytest.fixture
def new_status() -> HypothesisStatus:
    """The default lifecycle status assigned to freshly-built hypotheses."""
    return HypothesisStatus.NEW
