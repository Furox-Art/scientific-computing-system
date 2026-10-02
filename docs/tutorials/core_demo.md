# Core Data Models Tutorial

`cds.core` defines the `Domain`, `Hypothesis`, and `HypothesisStatus` types used across CDS, particularly by the hypothesis engine.

## 1. Domains

```python
from cds.core import Domain

for d in Domain:
    print(d.name, d.value)
```

## 2. Building a Hypothesis

The status enum has no `PROPOSED` member — the lifecycle values are `NEW`,
`TESTABLE`, `VALIDATED`, `REFINED`, `REJECTED`, `CRITIQUED`, and `ARCHIVED`. A
newly constructed `Hypothesis` defaults to `NEW`.

```python
from cds.core import Domain, Hypothesis, HypothesisStatus

h = Hypothesis(
    id="H-001",
    statement="A cosmological constant (Λ) drives late-time acceleration.",
    domain=Domain.COSMOLOGY,
    research_question="Why is the Hubble expansion accelerating?",
    status=HypothesisStatus.TESTABLE,
    confidence=0.62,
    assumptions=["GR holds on cosmological scales"],
    predictions=["(m-M) vs z bends downward"],
    tags=["dark-energy"],
)
print(h.to_markdown())
```

The same record type is what `cds.hypothesis.generate_hypotheses` returns, so you can inspect, persist, or feed it to the evaluator.

Run the full demo with `python examples/core_demo.py`.
