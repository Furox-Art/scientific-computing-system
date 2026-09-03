# Hypothesis Generation + Stats Tutorial

CDS's distinctive feature is pairing **structured hypothesis generation**
(`cds.hypothesis`) with **classical statistics** (`cds.stats`) so a research
question can move from idea to testable experiment in a few lines.

## 1. Generate structured hypotheses

`generate_hypotheses` turns a research question into falsifiable hypotheses,
each with explicit predictions and a confidence score.

```python
from cds.core.models import Domain
from cds.hypothesis import generate_hypotheses

question = "Does a new catalyst improve reaction yield?"
hypos = generate_hypotheses(question, domain=Domain.CHEMISTRY, n=2)

for h in hypos:
    print(f"Hypothesis: {h.statement}")
    print(f"Predictions: {h.predictions}\n")
```

```
Hypothesis: A hidden sector with light mediators can resolve the muon g-2 anomaly ...
Predictions: ['A measurable deviation in observable O at scale S with amplitude A.', ...]
```

> The generator is domain-aware but uses template-driven statements, so the
> text reads generically. The value is the **structure**: statement,
> predictions, assumptions, confidence, ready to attach data to.

## 2. Validate against experimental data

Once you have predictions, simulate or collect data and test them. Here a
two-group yield comparison becomes a t-test:

```python
from cds.stats import mean, one_sample_ttest, stdev

control_yields = [72.1, 71.8, 73.2, 70.9, 72.5]
catalyst_yields = [78.4, 79.1, 77.8, 80.2, 78.9]

print("Control group mean yield:", mean(control_yields))
print("Catalyst group mean yield:", mean(catalyst_yields))
print("Std dev (catalyst):", stdev(catalyst_yields))

result = one_sample_ttest(catalyst_yields, popmean=mean(control_yields))
print(f"One-sample t-test vs control mean: t={result.statistic:.2f}, p={result.p_value:.4f}")
```

```
Control group mean yield: 72.1
Catalyst group mean yield: 78.88
Std dev (catalyst): 0.8927...
One-sample t-test vs control mean: t=16.98, p=0.0001
```

The tiny p-value strongly supports the catalyst improving yield.

## 3. Why this workflow matters

The pairing turns a generated hypothesis into a **testable experiment sketch**
in one script:

1. **Generate** candidate explanations with explicit predictions.
2. **Collect** data (simulated here, real in practice).
3. **Test** with the appropriate statistical procedure.
4. **Decide**: keep or reject the hypothesis.

This is the "cognitive discovery" loop CDS is built around: structure the
thinking, then close it with numbers. See
[`hypothesis_demo.md`](hypothesis_demo.md) for the generator alone and
[`hypothesis_tests_demo.md`](hypothesis_tests_demo.md) for the stats in depth.

Run the full demo:

```bash
python examples/hypothesis_with_stats_demo.py
```
