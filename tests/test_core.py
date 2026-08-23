"""Focused regression tests for the ``cds.core`` package.

These tests cover the core data models (``Hypothesis``, ``Domain``,
``HypothesisStatus``) in isolation from the higher-level ``hypothesis``
generator/prompt-templating logic that ``test_hypothesis.py`` exercises. The
goal is a single place that pins down the *model* contract: validation
boundaries, enum round-tripping, default-factory isolation, and the full
``to_markdown`` branch matrix.

Shared instances come from ``tests/conftest.py`` (``make_hypothesis``,
``hypothesis``, ``minimal_hypothesis``, ``new_status``).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import timezone

import pytest

from cds.core import Domain, Hypothesis, HypothesisStatus


# --------------------------------------------------------------------------- #
# Domain / HypothesisStatus enums
# --------------------------------------------------------------------------- #
class TestDomainEnum:
    def test_all_domains_distinct_values(self) -> None:
        """No two domains may share a value (used as stable string keys)."""
        values = [d.value for d in Domain]
        assert len(values) == len(set(values))

    def test_domain_round_trip(self) -> None:
        """``Domain(value)`` must recover the member from its string value."""
        for d in list(Domain):
            assert Domain(d.value) is d

    def test_domain_is_str_subclass(self) -> None:
        """Domains are ``str, Enum``: their value is a plain str."""
        assert Domain.PHYSICS.value == "physics"
        assert isinstance(Domain.COSMOLOGY, str)


class TestHypothesisStatusEnum:
    def test_all_statuses_distinct_values(self) -> None:
        values = [s.value for s in HypothesisStatus]
        assert len(values) == len(set(values))

    def test_status_round_trip(self) -> None:
        for s in list(HypothesisStatus):
            assert HypothesisStatus(s.value) is s

    def test_status_is_str_subclass(self) -> None:
        assert HypothesisStatus.NEW.value == "new"
        assert isinstance(HypothesisStatus.REJECTED, str)


# --------------------------------------------------------------------------- #
# Hypothesis validation
# --------------------------------------------------------------------------- #
class TestHypothesisValidation:
    def test_required_fields_enforced(self) -> None:
        """Missing a required field must raise a TypeError (dataclass contract)."""
        # Mypy can't see that this is intentionally malformed; the whole
        # point is that the dataclass rejects the missing required fields at
        # runtime. ``call-arg`` is the expected, signal-bearing error here.
        with pytest.raises(TypeError):
            Hypothesis(  # no statement / domain / research_question
                id="h-1",
            )  # type: ignore[call-arg]

    def test_confidence_below_zero_rejected(self) -> None:
        with pytest.raises(ValueError):
            Hypothesis(
                id="h-1",
                statement="s",
                domain=Domain.PHYSICS,
                research_question="q",
                confidence=-0.01,
            )

    def test_confidence_above_one_rejected(self) -> None:
        with pytest.raises(ValueError):
            Hypothesis(
                id="h-1",
                statement="s",
                domain=Domain.PHYSICS,
                research_question="q",
                confidence=1.01,
            )

    @pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
    def test_confidence_boundary_values_accepted(self, confidence: float) -> None:
        """The closed interval [0.0, 1.0] is valid at both endpoints."""
        h = Hypothesis(
            id="h-1",
            statement="s",
            domain=Domain.PHYSICS,
            research_question="q",
            confidence=confidence,
        )
        assert h.confidence == confidence

    def test_invalid_domain_rejected(self) -> None:
        with pytest.raises(ValueError):
            Hypothesis(
                id="h-1",
                statement="s",
                domain="not-a-domain",  # type: ignore[arg-type]
                research_question="q",
                confidence=0.5,
            )


# --------------------------------------------------------------------------- #
# Hypothesis defaults & isolation
# --------------------------------------------------------------------------- #
class TestHypothesisDefaults:
    def test_defaults_applied(self, minimal_hypothesis: Hypothesis) -> None:
        """Required-only construction fills in documented defaults."""
        h = minimal_hypothesis
        assert h.status == HypothesisStatus.NEW
        assert h.confidence == 0.5
        assert h.assumptions == []
        assert h.predictions == []
        assert h.tags == []
        assert h.sources == []
        assert h.rationale is None
        assert h.metadata == {}

    def test_created_at_is_utc_aware(self, minimal_hypothesis: Hypothesis) -> None:
        """``created_at`` defaults to a timezone-aware UTC ``datetime``."""
        ts = minimal_hypothesis.created_at
        assert ts.tzinfo is not None
        assert ts.utcoffset() == timezone.utc.utcoffset(ts)

    def test_list_defaults_are_not_shared(self, make_hypothesis: Callable[..., Hypothesis]) -> None:
        """``default_factory`` must give each instance its own list.

        A common footgun is ``default=[]`` (shared mutable default);
        ``field(default_factory=list)`` avoids it. This test guards against a
        regression that would silently cross-contaminate instances.
        """
        a = make_hypothesis(id="a", assumptions=[])
        b = make_hypothesis(id="b", assumptions=[])
        a.assumptions.append("only-a")
        assert b.assumptions == []
        assert a.assumptions == ["only-a"]

    def test_dict_default_is_not_shared(self, make_hypothesis: Callable[..., Hypothesis]) -> None:
        """Same isolation guarantee for the ``metadata`` dict default."""
        a = make_hypothesis(id="a")
        b = make_hypothesis(id="b")
        a.metadata["k"] = "v"
        assert b.metadata == {}


# --------------------------------------------------------------------------- #
# to_markdown rendering — full branch matrix
# --------------------------------------------------------------------------- #
class TestToMarkdown:
    def test_full_render_includes_all_sections(self, hypothesis: Hypothesis) -> None:
        """A fully-populated hypothesis renders every optional section."""
        md = hypothesis.to_markdown()
        assert md.startswith("# Hypothesis: h-1")
        assert "**Statement**: Default hypothesis statement" in md
        assert "**Domain**: physics" in md
        assert "**Status**: new" in md
        assert "## Rationale" in md
        assert "Default rationale" in md
        assert "## Assumptions" in md
        assert "- assumption A" in md
        assert "## Predictions / Testable Consequences" in md
        assert "- prediction A" in md
        assert "**Tags**: tag-A" in md

    def test_minimal_render_omits_optional_sections(self, minimal_hypothesis: Hypothesis) -> None:
        """Required-only hypothesis renders header + nothing else optional.

        Exercises every ``if <field>:`` False-branch in the renderer.
        """
        md = minimal_hypothesis.to_markdown()
        assert "# Hypothesis: h-1" in md
        assert "## Rationale" not in md
        assert "## Assumptions" not in md
        assert "## Predictions" not in md
        assert "**Tags**" not in md

    def test_confidence_formatted_two_decimals(
        self, make_hypothesis: Callable[..., Hypothesis]
    ) -> None:
        """Confidence renders with exactly two decimal places."""
        h = make_hypothesis(confidence=0.875)
        md = h.to_markdown()
        assert "**Confidence**: 0.88" in md

    def test_domain_uses_value_not_name(self, make_hypothesis: Callable[..., Hypothesis]) -> None:
        """Markdown shows ``domain.value`` (e.g. ``cosmology``), not the enum name."""
        h = make_hypothesis(domain=Domain.COSMOLOGY)
        assert "**Domain**: cosmology" in h.to_markdown()
        assert "COSMOLOGY" not in h.to_markdown()

    def test_multiple_assumptions_all_listed(
        self, make_hypothesis: Callable[..., Hypothesis]
    ) -> None:
        h = make_hypothesis(assumptions=["a1", "a2", "a3"])
        md = h.to_markdown()
        assert "- a1" in md
        assert "- a2" in md
        assert "- a3" in md

    def test_multiple_tags_comma_separated(
        self, make_hypothesis: Callable[..., Hypothesis]
    ) -> None:
        h = make_hypothesis(tags=["x", "y", "z"])
        assert "**Tags**: x, y, z" in h.to_markdown()

    def test_empty_tags_omitted(self, make_hypothesis: Callable[..., Hypothesis]) -> None:
        """An empty tags list must not render the Tags line at all."""
        h = make_hypothesis(tags=[])
        assert "**Tags**" not in h.to_markdown()

    def test_empty_assumptions_omitted(self, make_hypothesis: Callable[..., Hypothesis]) -> None:
        h = make_hypothesis(assumptions=[])
        assert "## Assumptions" not in h.to_markdown()

    def test_empty_predictions_omitted(self, make_hypothesis: Callable[..., Hypothesis]) -> None:
        h = make_hypothesis(predictions=[])
        assert "## Predictions" not in h.to_markdown()

    def test_none_rationale_omitted(self, make_hypothesis: Callable[..., Hypothesis]) -> None:
        h = make_hypothesis(rationale=None)
        assert "## Rationale" not in h.to_markdown()


class TestFromDictRoundTrip:
    """``from_dict`` completes the JSON round trip started by ``to_dict``."""

    def test_full_round_trip_preserves_every_field(
        self, make_hypothesis: Callable[..., Hypothesis]
    ) -> None:
        h = make_hypothesis(
            status=HypothesisStatus.TESTABLE,
            confidence=0.73,
        )
        assert Hypothesis.from_dict(h.to_dict()) == h

    def test_missing_optional_keys_fall_back_to_defaults(self) -> None:
        partial = Hypothesis.from_dict(
            {"id": "x", "statement": "y", "domain": "biology", "research_question": "q"}
        )
        assert partial.status == HypothesisStatus.NEW
        assert partial.confidence == 0.5
        assert partial.assumptions == []
        assert partial.rationale is None

    def test_invalid_status_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="olmayan"):
            Hypothesis.from_dict(
                {
                    "id": "x",
                    "statement": "y",
                    "domain": "physics",
                    "research_question": "q",
                    "status": "olmayan",
                }
            )

    def test_json_dumps_output_is_loadable(self, hypothesis: Hypothesis) -> None:
        import json

        loaded = Hypothesis.from_dict(json.loads(json.dumps(hypothesis.to_dict())))
        assert loaded == hypothesis
