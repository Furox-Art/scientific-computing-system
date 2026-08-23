"""Core data models for CDS.

The data models here are plain :mod:`dataclasses` — no third-party validation
library. Field constraints that previously relied on ``pydantic`` (the
``[0.0, 1.0]`` confidence range, enum coercion of ``domain``) are enforced in
``Hypothesis.__post_init__``, which raises :class:`ValueError` on violations.
This keeps the whole ``cds`` package zero-dependency at runtime while
preserving the construction contract the test suite pins down.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class Domain(str, Enum):
    """Broad scientific domains supported by CDS."""

    PHYSICS = "physics"
    COSMOLOGY = "cosmology"
    MATHEMATICS = "mathematics"
    BIOLOGY = "biology"
    CHEMISTRY = "chemistry"
    GENERAL_SCIENCE = "general_science"


class HypothesisStatus(str, Enum):
    """Lifecycle states for a Hypothesis."""

    NEW = "new"
    REFINED = "refined"
    CRITIQUED = "critiqued"
    TESTABLE = "testable"
    VALIDATED = "validated"
    REJECTED = "rejected"
    ARCHIVED = "archived"


@dataclass
class Hypothesis:
    """A scientific hypothesis with metadata and traceability.

    Attributes:
        id: Unique identifier.
        statement: The core hypothesis statement.
        domain: Scientific domain; accepts a :class:`Domain` or its string value.
        research_question: The research question this hypothesis addresses.
        rationale: Optional reasoning behind the hypothesis.
        assumptions: Preconditions assumed to hold.
        predictions: Testable consequences of the hypothesis.
        status: Current point in the hypothesis lifecycle.
        confidence: Subjective belief in the hypothesis, in the closed
            interval ``[0.0, 1.0]``.
        created_at: Timezone-aware UTC timestamp of creation.
        tags: Free-form categorical tags.
        sources: References or retrieval sources backing the hypothesis.
        metadata: Arbitrary string-keyed/string-valued extra information.
    """

    id: str
    statement: str
    domain: Domain
    research_question: str
    rationale: str | None = None
    assumptions: list[str] = field(default_factory=list)
    predictions: list[str] = field(default_factory=list)
    status: HypothesisStatus = HypothesisStatus.NEW
    confidence: float = 0.5
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Coerce ``domain`` to its enum and validate ``confidence`` range.

        Mirrors the coercion/validation pydantic performed previously: a
        bare string like ``"physics"`` is accepted and converted to
        :class:`Domain.PHYSICS`; ``confidence`` must lie in ``[0.0, 1.0]``.
        Unknown values raise ``ValueError`` (matching the old
        ``ValidationError`` behaviour).
        """
        # Coerce string values into the enum members so callers can pass either
        # ``Domain.PHYSICS`` or the bare string ``"physics"``. This mirrors the
        # coercion pydantic performed previously; an unknown value raises
        # ValueError, matching the old ValidationError behaviour.
        if not isinstance(self.domain, Domain):
            try:
                self.domain = Domain(self.domain)
            except ValueError as exc:
                raise ValueError(f"{self.domain!r} is not a valid Domain") from exc

        # The closed interval [0.0, 1.0] is the only constraint on confidence.
        # Out-of-range values raise ValueError, matching the former pydantic
        # ``ge=0.0, le=1.0`` contract that the test suite pins down.
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize this hypothesis to a JSON-friendly dict.

        Mirrors the previous ``pydantic`` ``model_dump(mode="json")`` output:
        enums become their string ``value`` and ``created_at`` is rendered as
        an ISO-8601 string, so the result is directly :func:`json.dumps`-able.
        """
        return {
            "id": self.id,
            "statement": self.statement,
            "domain": self.domain.value,
            "research_question": self.research_question,
            "rationale": self.rationale,
            "assumptions": list(self.assumptions),
            "predictions": list(self.predictions),
            "status": self.status.value,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "tags": list(self.tags),
            "sources": list(self.sources),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Hypothesis:
        """Rebuild a :class:`Hypothesis` from :meth:`to_dict` output.

        Completes the JSON round trip: enum strings are coerced back into
        their :class:`Domain` / :class:`HypothesisStatus` members and
        ``created_at`` is parsed from its ISO-8601 rendering. Missing
        optional keys fall back to the same defaults as the constructor, so
        partially-specified records load cleanly. Unknown ``status`` values
        raise ``ValueError``.
        """
        payload = dict(data)  # copy so we never mutate the caller's dict
        created_at = payload.get("created_at")
        if isinstance(created_at, str):
            payload["created_at"] = datetime.fromisoformat(created_at)
        status = payload.get("status")
        if status is not None and not isinstance(status, HypothesisStatus):
            payload["status"] = HypothesisStatus(status)
        return cls(**payload)

    def to_markdown(self) -> str:
        """Render this hypothesis as a structured Markdown document."""
        lines = [
            f"# Hypothesis: {self.id}",
            "",
            f"**Statement**: {self.statement}",
            "",
            f"**Domain**: {self.domain.value}",
            f"**Research Question**: {self.research_question}",
            f"**Status**: {self.status.value} | **Confidence**: {self.confidence:.2f}",
            "",
        ]
        if self.rationale:
            lines += ["## Rationale", self.rationale, ""]
        if self.assumptions:
            lines += ["## Assumptions"] + [f"- {a}" for a in self.assumptions] + [""]
        if self.predictions:
            preds = [f"- {p}" for p in self.predictions]
            lines += ["## Predictions / Testable Consequences"] + preds + [""]
        if self.tags:
            lines += [f"**Tags**: {', '.join(self.tags)}"]
        return "\n".join(lines)
