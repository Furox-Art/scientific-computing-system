# Using CDS in Research Workflows

CDS is designed to be easy to embed in larger research scripts and analysis pipelines. All modules are pure Python with no heavy external dependencies, so they are straightforward to read, audit, and adapt.

## Structured Hypothesis Generation

The `cds.hypothesis` module helps turn a research question into a set of falsifiable hypotheses with explicit assumptions, predictions, and confidence estimates.

- Use the built-in offline generator for quick exploration and demos.
- For real work, implement the `HypothesisGenerator` Protocol (see `examples/hypothesis_custom_generator.py` for a minimal example). This lets you inject domain knowledge, load templates from files, apply constraints from your data, or call any backend you control.

Example flow:
1. Generate candidate hypotheses for a question.
2. Use `cds.stats` or `cds.montecarlo` to perform quick checks or simulations against the predictions.
3. Use `cds.optimization` or `cds.diffeq` to explore parameter regimes suggested by the hypotheses.

See:
- `examples/hypothesis_demo.py`
- `examples/hypothesis_with_stats_demo.py`
- `examples/hypothesis_custom_generator.py`
- `cds hypothesis` CLI command

## Combining Modules

Because every module exposes simple functions and classes, they compose naturally:

```python
from cds.hypothesis import generate_hypotheses
from cds.stats import one_sample_ttest
from cds.montecarlo import estimate_pi  # or your own simulation

hypos = generate_hypotheses("What causes the observed tension?", domain="cosmology", n=3)

# Example: treat a numeric prediction from a hypothesis as a test value
# and run a quick statistical check with real or simulated data
...
```

The same pattern works for optimization (tune a model suggested by a hypothesis), differential equations (explore dynamics), graph algorithms, etc.

## Custom Generators and Extension Points

The `HypothesisGenerator` Protocol is the main extension point for the hypothesis part:

```python
from cds.hypothesis import HypothesisGenerator, Domain, Hypothesis, generate_hypotheses


class MyDomainGenerator:
    def generate(
        self, research_question: str, domain: Domain = Domain.GENERAL_SCIENCE, n: int = 3, **kwargs
    ) -> list[Hypothesis]:
        ...
        return hypos


custom_hypos = generate_hypotheses(question, generator=MyDomainGenerator(), n=5)
```

You can wrap any source of ideas (literature database, your own heuristics, external service, etc.) as long as it returns `Hypothesis` objects with the expected fields.

Other modules are similarly simple to subclass or wrap when you need domain-specific behavior.

## Pure Python Advantages for Research

- Easy to inspect the exact implementation of any calculation.
- No hidden C extensions or large dependency trees that complicate reproducibility or auditing.
- Straightforward to modify a single function for a one-off research need and still pass the existing test suite.
- Low barrier to embedding inside bigger simulation, data-analysis, or experimental-design scripts.

## Citing

If you use CDS in a paper, thesis, or research pipeline, please cite it using the metadata in `CITATION.cff` (or the preferred-citation section inside it). This gives proper credit and helps others discover the platform.

## Next Steps

- Browse the full example set in `examples/`.
- Read the module reference in `docs/api.md`.
- See current performance numbers in `docs/benchmarks.md`.
- Check `ROADMAP.md` for planned directions.

Contributions that improve a module's usability for a specific research domain, add a well-tested example workflow, or expand the test coverage are always welcome.
