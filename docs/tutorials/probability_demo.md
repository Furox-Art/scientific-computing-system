# Probability Distributions Tutorial

`cds.probability` covers common continuous PDFs and discrete PMFs, plus reproducible sampling.

## 1. Continuous PDFs

```python
from cds.probability import gaussian_pdf, uniform_pdf, exponential_pdf

print(gaussian_pdf(0.0, mu=0.0, sigma=1.0))  # peak ≈ 0.399
print(uniform_pdf(0.5, a=0.0, b=1.0))  # 1.0 on support
print(exponential_pdf(1.0, lambda_=2.0))
```

## 2. Discrete PMFs

```python
from cds.probability import binomial_pmf, poisson_pmf

for k in range(11):
    print(k, binomial_pmf(k, n=10, p=0.5))  # symmetric around 5

for k in range(6):
    print(k, poisson_pmf(k, lambda_=3.0))
```

## 3. Reproducible Sampling

```python
from cds.probability import uniform_sample

print(uniform_sample(low=0.0, high=1.0, n=5, seed=42))  # deterministic
```

Run the full demo with `python examples/probability_demo.py`.
