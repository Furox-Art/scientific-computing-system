"""
Hypothesis generation and evaluation module for Cognitive Discovery.

Provides tools to generate structured scientific hypotheses from
research questions. Includes prompt templates and an offline
generator for immediate use, plus a clear Protocol for supplying
custom generator implementations for specialized research needs.

The focus is on making hypotheses falsifiable, with explicit
assumptions, predictions, and confidence estimates.

Example:
    from cds.hypothesis import generate_hypotheses

    hypos = generate_hypotheses(
        "Why do we observe the Hubble tension?",
        domain="cosmology",
        n=3
    )
"""

from cds.core.models import Domain, Hypothesis, HypothesisStatus
from cds.hypothesis.evaluator import (
    ChiSquareGofPayload,
    EvaluationData,
    EvaluationResult,
    HypothesisEvaluator,
)
from cds.hypothesis.generator import (
    HypothesisGenerator,
    PromptTemplate,
    SimpleOfflineGenerator,
    generate_hypotheses,
)
from cds.hypothesis.mining import MinedHypothesis, mine_correlations

__all__ = [
    "Domain",
    "Hypothesis",
    "HypothesisStatus",
    "HypothesisGenerator",
    "PromptTemplate",
    "SimpleOfflineGenerator",
    "generate_hypotheses",
    "HypothesisEvaluator",
    "EvaluationData",
    "ChiSquareGofPayload",
    "EvaluationResult",
    "MinedHypothesis",
    "mine_correlations",
]
