# Hypothesis Generation

This tutorial demonstrates the unique `cds.hypothesis` module, which automates the scientific reasoning process by generating structured hypotheses.

## 1. Generating Hypotheses

You can generate falsifiable hypotheses from a simple research question:

```python
from cds.hypothesis import generate_hypotheses, Domain

question = "What are the potential causes of the Hubble Tension?"
hypos = generate_hypotheses(question, domain=Domain.COSMOLOGY, n=2)

for h in hypos:
    print(f"--- Hypothesis ID: {h.id} ---")
    print(h.statement)
    print("\nAssumptions:")
    for a in h.assumptions:
        print(f"  - {a}")
    print("\n")
```

## 2. Autonomous Evaluation

We can also use the `HypothesisEvaluator` to test these ideas against empirical data using statistical tests:

```python
from cds.hypothesis import HypothesisEvaluator

evaluator = HypothesisEvaluator(alpha=0.05)

# Mock data comparison (Late Universe vs Early Universe H0)
data = {
    "groups": [
        [70.1, 71.2, 69.5, 70.8],  # Group A
        [67.4, 68.2, 67.8, 67.1],  # Group B
    ]
}

result = evaluator.evaluate(hypos[0], data)
print(f"Result: {result.conclusion}")
print(f"p-value: {result.p_value:.6f}")
```
