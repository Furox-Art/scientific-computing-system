"""
Scientific hypothesis generator for the Scientific Computing System.

This module turns research questions into structured, falsifiable
hypotheses with clear assumptions and predictions.

It provides ready-to-use prompt templates and a simple offline
generator for demos and testing. The HypothesisGenerator Protocol
makes it straightforward for researchers to supply their own
generator implementations for specialized workflows.

Key goals:
- High-quality, reusable prompt templates for scientific reasoning
- Clean separation between templates and generation logic
- Simple Protocol for custom generator implementations
- Usable offline as a starting point

See the HypothesisGenerator protocol and examples for how to
provide a custom generator.
"""

from __future__ import annotations

import uuid
from typing import Protocol

from cds.core.models import Domain, Hypothesis, HypothesisStatus


class HypothesisGenerator(Protocol):
    """Interface for hypothesis generators."""

    def generate(
        self,
        research_question: str,
        domain: Domain | str = Domain.GENERAL_SCIENCE,
        n: int = 3,
        **kwargs: object,
    ) -> list[Hypothesis]:
        """Generate `n` hypotheses for the given research question."""
        # Protocol methods carry no body — the docstring alone satisfies the
        # required suite (PEP 544). Previously a bare ``...`` here triggered
        # CodeQL's ineffectual-statement query; the docstring is both clearer
        # and the idiomatic body for an abstract/protocol method.


class PromptTemplate:
    """Prompt templates for different providers / use cases."""

    SYSTEM = (
        "You are an expert research scientist and rigorous thinker. "
        "Your goal is to propose high-quality, falsifiable, "
        "novel-yet-grounded scientific hypotheses. "
        "Always make assumptions explicit. Prioritize testability and clarity. "
        "Respond ONLY in the requested structured format."
    )

    USER_BASE = """Research Question: {research_question}

Domain focus: {domain}

Generate {n} distinct hypotheses.

For each hypothesis provide:
- Clear one-sentence statement
- Short rationale (2-4 sentences) connecting to known science
- Key assumptions (bullet list)
- Specific, measurable predictions or consequences (bullet list)
- Estimated confidence (0-1) with brief justification

Format each as:
ID: H-<number>
Statement: ...
Rationale: ...
Assumptions:
- ...
Predictions:
- ...
Confidence: 0.xx
"""

    @classmethod
    def render(cls, research_question: str, domain: Domain, n: int = 3) -> str:
        """Format the user-side prompt for a hypothesis generation request."""
        return cls.USER_BASE.format(
            research_question=research_question,
            domain=domain.value,
            n=n,
        )


class SimpleOfflineGenerator:
    """
    A deterministic offline generator for demos and early development.

    It creates plausible but generic hypotheses. Researchers can replace
    or wrap it with a custom implementation of HypothesisGenerator
    tailored to their domain or data sources.
    """

    def __init__(self) -> None:
        """Load the built-in hypothesis templates keyed by :class:`Domain`."""
        self.templates = {
            Domain.COSMOLOGY: [
                (
                    "Late-time modifications to gravity can mimic "
                    "dark energy while altering structure growth."
                ),
                (
                    "A time-varying dark energy equation of state "
                    "w(a) with a sharp transition at z~0.5 "
                    "explains current tensions."
                ),
                (
                    "Primordial non-Gaussianity of local type at "
                    "f_NL ~ 5-10 is detectable with next-gen "
                    "surveys and resolves sigma8 tension."
                ),
            ],
            Domain.PHYSICS: [
                (
                    "A hidden sector with light mediators can "
                    "resolve the muon g-2 anomaly without "
                    "conflicting with collider bounds."
                ),
                (
                    "Modified dispersion relations at Planck scale "
                    "suppress high-energy cosmic rays in a "
                    "characteristic energy-dependent way."
                ),
            ],
            Domain.MATHEMATICS: [
                (
                    "A new family of special functions between "
                    "hypergeometric and q-hypergeometric satisfies "
                    "a novel functional equation."
                ),
            ],
        }
        # Reusable assumption/prediction pools. Pooling (rather than a single
        # hard-coded pair repeated across every hypothesis) ensures generated
        # output varies and remains a useful starting point. ``_cycled`` draws
        # from these pools so duplicates only recur once the pool is exhausted.
        self._assumption_pool = [
            "Background model is approximately correct at low energies.",
            "New physics at observable scales doesn't violate existing constraints.",
            "Systematic uncertainties in the measurement are subdominant to statistics.",
            "The effect is not an artifact of selection bias or confounders.",
            "Approximations made in the theoretical framework hold in the relevant regime.",
            "The relevant degrees of freedom are captured by the chosen variables.",
        ]
        self._prediction_pool = [
            "A measurable deviation in observable O at scale S with amplitude A.",
            "Correlation between two previously uncorrelated datasets D1 and D2.",
            "A characteristic scaling exponent distinct from the null hypothesis.",
            "Anomalous behavior concentrated in a specific sub-population or energy bin.",
            "A detectable phase shift or timing offset relative to the reference model.",
            "Convergence of independent estimators toward a common anomalous value.",
        ]

    @staticmethod
    def _cycled(pool: list[str], start: int, count: int) -> list[str]:
        """Return ``count`` items from ``pool`` starting at ``start`` (mod).

        Cycling the pool (rather than repeating the same fixed pair on every
        hypothesis) is what keeps the generated assumptions/predictions
        varied once we exceed the pool length.

        Caller invariant: ``pool`` is non-empty (the generator only ever
        feeds the fixed ``_assumption_pool`` / ``_prediction_pool``). A
        zero-length pool would raise ``ZeroDivisionError`` on the modulo —
        that's the intended failure mode for a programmer error, not a
        silent empty-list return.
        """
        m = len(pool)
        return [pool[(start + k) % m] for k in range(count)]

    @staticmethod
    def _generic_statement(research_question: str, variant: int) -> str:
        """A parameterized generic hypothesis, distinct per ``variant``.

        The previous fallback multiplied a single sentence, producing
        identical statements when ``n`` exceeded the template count.
        Varying the phrasing keeps fallback hypotheses distinct so callers
        don't receive obvious duplicates.
        """
        leads = [
            "A yet-untested factor influencing",
            "An interaction between latent variables governing",
            "A previously overlooked boundary condition affecting",
            "A nonlinear coupling hidden in",
            "A second-order correction to the standard model of",
            "A structural regularity underlying",
            "An emergent constraint shaping",
            "A time-dependent modulation superimposed on",
        ]
        suffixes = [
            "produces a measurable, reproducible effect.",
            "yields a deviation distinguishable from the null hypothesis.",
            "manifests as a detectable signature in available data.",
            "predicts a correlation absent from current baselines.",
            "implies a scaling law testable under controlled conditions.",
            "introduces an asymmetry observable across measurement regimes.",
            "narrows the allowed range of a key free parameter.",
            "is consistent with prior upper limits but not with zero signal.",
        ]
        m = len(leads)
        return f"{leads[variant % m]} {research_question} {suffixes[(variant // m) % m]}"

    def generate(
        self,
        research_question: str,
        domain: Domain | str = Domain.GENERAL_SCIENCE,
        n: int = 3,
        **kwargs: object,
    ) -> list[Hypothesis]:
        """Generate `n` hypotheses from the built-in domain templates."""
        # Ensure domain is a Domain enum instance. ``Domain`` subclasses
        # ``str``, so the isinstance guard is True for both plain strings and
        # enum members; the False branch (skip mapping) is therefore
        # unreachable from typed callers — it remains as a defensive seam for
        # hypothetical non-str subclasses and is excluded from coverage.
        if isinstance(domain, str):  # pragma: no branch
            try:
                # Case-insensitive mapping for better UX
                domain = Domain(domain.lower())
            except ValueError:
                domain = Domain.GENERAL_SCIENCE

        ideas = self.templates.get(domain, self.templates[Domain.PHYSICS])[:n]
        if len(ideas) < n:
            # The built-in templates only cover a few domains. For any other
            # domain (or when more hypotheses are requested than templates
            # exist), fall back to parameterized generic statements derived
            # from the research question. Each fallback variant is distinct
            # (see ``_generic_statement``) so the output stays a usable
            # starting point without obvious duplication.
            deficit = n - len(ideas)
            ideas += [self._generic_statement(research_question, v) for v in range(deficit)]

        hypos: list[Hypothesis] = []
        for i, idea in enumerate(ideas[:n]):
            # Confidence increases with rank but is clamped to [0.4, 0.9].
            # The previous formula 0.45 + i*0.05 overflowed Hypothesis's
            # le=1.0 constraint for n >= 12 (raising ValidationError); the
            # clamp keeps confidence in-range for any n.
            confidence = min(0.9, 0.4 + i * 0.05)
            h = Hypothesis(
                id=f"H-{uuid.uuid4().hex[:8]}",
                statement=idea,
                domain=domain,
                research_question=research_question,
                rationale=(
                    "Builds on known tensions in the literature "
                    "and proposes a falsifiable deviation."
                ),
                assumptions=self._cycled(self._assumption_pool, start=i, count=2),
                predictions=self._cycled(self._prediction_pool, start=i, count=2),
                status=HypothesisStatus.NEW,
                confidence=confidence,
                tags=[domain.value, "early-draft"],
            )
            hypos.append(h)
        return hypos


def generate_hypotheses(
    research_question: str,
    domain: Domain = Domain.GENERAL_SCIENCE,
    n: int = 3,
    generator: HypothesisGenerator | None = None,
) -> list[Hypothesis]:
    """Convenience entrypoint."""
    gen = generator or SimpleOfflineGenerator()
    return gen.generate(research_question=research_question, domain=domain, n=n)
