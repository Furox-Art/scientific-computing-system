# Monte Carlo Methods Tutorial

`cds.montecarlo` estimates integrals, π, and random walks by sampling.

Every function takes `n_samples` / `n_throws` (not `samples` / `drops`) and an
optional `seed`. Passing a `seed` makes the run reproducible; the estimators
return an `MCResult` dataclass carrying `estimate` plus diagnostics, and the
random walks return plain lists of positions.

## 1. Estimating π

```python
import math
from cds.montecarlo import estimate_pi

r = estimate_pi(n_samples=100_000, seed=42)
print(f"pi ~ {r.estimate:.6f}  (error {abs(r.estimate - math.pi):.6f})")
```

## 2. Integrating by Sampling

```python
from cds.montecarlo import mc_integrate

# ∫_0^1 x^2 dx = 1/3
r = mc_integrate(lambda x: x**2, a=0.0, b=1.0, n_samples=50_000, seed=7)
print(f"estimate {r.estimate:.6f}  exact 0.333333")
```

A Monte-Carlo integral converges slowly. 50,000 samples land within ~0.001 of
the exact value, and that error shrinks only as `1/sqrt(n)` — for smooth
integrands prefer `cds.numerical_integration`, which is deterministic and much
more accurate for the same cost.

## 3. Random Walks

The walkers return the **full position history**, so the result is a list you
index — there is no `.position`, `.x`, or `.y` attribute.

```python
from cds.montecarlo import random_walk_1d, random_walk_2d

path1 = random_walk_1d(steps=100, step_size=1.0, seed=1)
print(f"{len(path1)} positions, final x = {path1[-1]}")

path2 = random_walk_2d(steps=100, step_size=1.0, seed=2)
x, y = path2[-1]
print(f"final position = ({x:.4f}, {y:.4f})")
```

## 4. Buffon's Needle

```python
from cds.montecarlo import buffon_needle

r = buffon_needle(needle_length=0.5, line_spacing=1.0, n_throws=10_000, seed=3)
print(f"pi ~ {r.estimate:.4f}")
```

Run the full demo with:

```bash
python examples/montecarlo_demo.py
```
