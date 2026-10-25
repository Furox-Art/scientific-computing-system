"""Tests for hypothesis generation module."""

from cds.core.models import Domain, Hypothesis, HypothesisStatus
from cds.hypothesis.generator import (
    PromptTemplate,
    SimpleOfflineGenerator,
    generate_hypotheses,
)


def test_generate_hypotheses_returns_list() -> None:
    hypos = generate_hypotheses(
        "Hubble tension resolution mechanisms",
        domain=Domain.COSMOLOGY,
        n=2,
    )
    assert len(hypos) == 2
    for h in hypos:
        assert isinstance(h, Hypothesis)
        assert h.research_question == "Hubble tension resolution mechanisms"
        assert h.domain == Domain.COSMOLOGY
        assert h.status == HypothesisStatus.NEW
        assert 0.0 <= h.confidence <= 1.0


def test_prompt_template_renders() -> None:
    prompt = PromptTemplate.render("Test question about dark energy", Domain.COSMOLOGY, n=3)
    assert "Research Question: Test question about dark energy" in prompt
    assert "Domain focus: cosmology" in prompt
    assert "Generate 3 distinct hypotheses" in prompt


def test_offline_generator_is_deterministic() -> None:
    gen = SimpleOfflineGenerator()
    h1 = gen.generate("foo", domain=Domain.COSMOLOGY, n=1)
    h2 = gen.generate("foo", domain=Domain.COSMOLOGY, n=1)
    assert len(h1) == 1 and len(h2) == 1
    assert h1[0].domain == Domain.COSMOLOGY
    # created_at records when the hypothesis was made, not what it says.
    first = {k: v for k, v in h1[0].to_dict().items() if k != "created_at"}
    second = {k: v for k, v in h2[0].to_dict().items() if k != "created_at"}
    assert first == second
    assert h1[0].id.startswith("H-")
    other = gen.generate("bar", domain=Domain.COSMOLOGY, n=1)
    assert other[0].id != h1[0].id


def test_generate_single_hypothesis() -> None:
    hypos = generate_hypotheses("dark matter", domain=Domain.PHYSICS, n=1)
    assert len(hypos) == 1
    assert hypos[0].domain == Domain.PHYSICS


def test_generate_multiple_hypotheses() -> None:
    hypos = generate_hypotheses("protein folding", domain=Domain.BIOLOGY, n=5)
    assert len(hypos) == 5
    for h in hypos:
        assert h.domain == Domain.BIOLOGY


def test_hypothesis_to_markdown() -> None:
    h = Hypothesis(
        id="test-1",
        statement="Test statement",
        domain=Domain.PHYSICS,
        research_question="test question",
        rationale="test rationale",
        confidence=0.8,
        assumptions=["a1", "a2"],
        predictions=["p1"],
    )
    md = h.to_markdown()
    assert "test-1" in md
    assert "Test statement" in md
    assert "test rationale" in md
    assert "a1" in md
    assert "p1" in md


def test_hypothesis_to_markdown_minimal() -> None:
    """to_markdown() with no rationale / assumptions / predictions / tags.

    Exercises every ``if <field>:`` False-branch in the renderer so the
    minimal hypothesis (only the required fields) renders without the
    optional sections.
    """
    h = Hypothesis(
        id="minimal-1",
        statement="Bare statement",
        domain=Domain.MATHEMATICS,
        research_question="bare question",
        confidence=0.5,
    )
    md = h.to_markdown()
    assert "Bare statement" in md
    assert "mathematics" in md
    # None of the optional sections should appear
    assert "## Rationale" not in md
    assert "## Assumptions" not in md
    assert "## Predictions" not in md
    assert "**Tags**" not in md


def test_hypothesis_model_fields() -> None:
    h = Hypothesis(
        id="h-1",
        statement="stmt",
        domain=Domain.CHEMISTRY,
        research_question="q",
        confidence=0.5,
    )
    assert h.confidence == 0.5
    assert h.status == HypothesisStatus.NEW
    assert h.tags == []
    assert h.sources == []


def test_prompt_template_different_domains() -> None:
    # __members__.values() is the Enum API's explicit member-collection view.
    # Same members/order as iterating the EnumType, but a mapping view whose
    # type CodeQL's non-iterable query can resolve (the EnumType iterator
    # could not).
    for domain in Domain.__members__.values():
        prompt = PromptTemplate.render("test", domain, n=1)
        assert domain.value in prompt


def test_generate_general_science() -> None:
    hypos = generate_hypotheses("climate", domain=Domain.GENERAL_SCIENCE, n=2)
    assert len(hypos) == 2
    assert hypos[0].domain == Domain.GENERAL_SCIENCE


def test_generate_with_domain_enum_instance_skips_str_mapping() -> None:
    # Passing a Domain enum instance (not a str) leaves the `isinstance(domain, str)`
    # guard False, so the try/except str→Domain mapping is skipped entirely
    # (generator.py 149 -> 156 edge). The returned hypotheses carry that domain.
    gen = SimpleOfflineGenerator()
    hypos = gen.generate("neutrino mass ordering", domain=Domain.PHYSICS, n=2)
    assert len(hypos) == 2
    assert all(h.domain == Domain.PHYSICS for h in hypos)


def test_evaluator_goodness_of_fit_with_explicit_expected() -> None:
    # Supplying a non-None `expected` list skips the uniform-distribution
    # fallback (evaluator.py 164 -> 168 edge) and feeds the caller's counts
    # straight into chi_square_gof.
    from cds.hypothesis.evaluator import HypothesisEvaluator

    h = Hypothesis(
        id="gof-1",
        statement="Loaded die is fair",
        domain=Domain.MATHEMATICS,
        research_question="die fairness",
        confidence=0.5,
    )
    evaluator = HypothesisEvaluator()
    result = evaluator.goodness_of_fit(h, observed=[10, 20, 30], expected=[20, 20, 20])
    assert result.hypothesis_id == "gof-1"
    assert "Chi-square goodness-of-fit" in result.test_name


def test_generate_large_n_does_not_overflow_confidence() -> None:
    # Regression: confidence was computed as 0.45 + i*0.05, which exceeded
    # Hypothesis's le=1.0 constraint once i >= 12 (n >= 12), raising a
    # Pydantic ValidationError. Confidence must stay in [0, 1] for any n.
    hypos = generate_hypotheses("broad question", domain=Domain.GENERAL_SCIENCE, n=20)
    assert len(hypos) == 20
    for h in hypos:
        assert 0.0 <= h.confidence <= 1.0


def test_generate_does_not_duplicate_statements() -> None:
    # When more hypotheses are requested than built-in templates exist, the
    # generator previously repeated the SAME generic statement via list
    # multiplication. Returned statements must all be distinct so the output
    # is usable as a starting point rather than obvious duplication.
    hypos = generate_hypotheses("quantum gravity", domain=Domain.PHYSICS, n=8)
    assert len(hypos) == 8
    statements = [h.statement for h in hypos]
    assert len(set(statements)) == len(statements), "Duplicate statements found"


def test_generate_n_one_returns_single_hypothesis() -> None:
    # Edge: n=1 must return exactly one hypothesis without error (previously
    # worked, but now guarded against confidence floor bugs).
    hypos = generate_hypotheses("lone question", domain=Domain.COSMOLOGY, n=1)
    assert len(hypos) == 1
    assert hypos[0].confidence >= 0.0


def test_offline_generator_string_domain_case_insensitive() -> None:
    # Passing a string domain (not a Domain enum) exercises the
    # case-insensitive str -> Domain mapping in the generator.
    gen = SimpleOfflineGenerator()
    hypos = gen.generate("q", domain="Cosmology", n=2)
    assert len(hypos) == 2
    assert all(h.domain == Domain.COSMOLOGY for h in hypos)


def test_offline_generator_invalid_string_falls_back_to_general() -> None:
    # An unrecognized domain string must fall back to GENERAL_SCIENCE, not
    # raise, so callers cannot accidentally crash the pipeline.
    gen = SimpleOfflineGenerator()
    hypos = gen.generate("q", domain="not_a_real_domain_xyz", n=2)
    assert len(hypos) == 2
    assert all(h.domain == Domain.GENERAL_SCIENCE for h in hypos)


def test_generated_hypotheses_carry_distinct_ids() -> None:
    # Each hypothesis should have a unique id, even when many are requested.
    hypos = generate_hypotheses("ids must be unique", domain=Domain.BIOLOGY, n=6)
    ids = [h.id for h in hypos]
    assert len(set(ids)) == len(ids)


def test_generated_hypotheses_assumptions_nonempty() -> None:
    # Quality guard: every generated hypothesis must carry at least one
    # assumption and one prediction (a falsifiable hypothesis needs them).
    hypos = generate_hypotheses("rich question", domain=Domain.CHEMISTRY, n=4)
    for h in hypos:
        assert len(h.assumptions) >= 1, f"No assumptions on {h.id}"
        assert len(h.predictions) >= 1, f"No predictions on {h.id}"
