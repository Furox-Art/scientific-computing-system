# Differential Equations Tutorial

`cds.diffeq` solves initial-value problems with classical schemes plus an adaptive RK45 integrator.

## 1. Single ODE

Solve `dy/dt = -y`, `y(0) = 1` whose exact value at `t=1` is `e^-1 ≈ 0.3679`:

```python
from cds.diffeq import euler_method, midpoint_method, rk4, rk45

f = lambda t, y: -y
print(euler_method(f, t0=0.0, y0=1.0, t_end=1.0, n=100).value)
print(midpoint_method(f, t0=0.0, y0=1.0, t_end=1.0, n=100).value)
print(rk4(f, t0=0.0, y0=1.0, t_end=1.0, n=10).value)  # very accurate
print(rk45(f, t0=0.0, y0=1.0, t_end=1.0, rtol=1e-6).value)  # adaptive
```

## 2. System of ODEs

```python
from cds.diffeq import solve_system


def lotka(t, state):
    x, y = state
    return [1.1 * x - 0.4 * x * y, 0.1 * x * y - 0.4 * y]  # Lotka-Volterra


sol = solve_system(lotka, t0=0.0, y0=[10.0, 5.0], t_end=15.0, n=300)
print(sol.value)  # final [prey, predator]
print(len(sol.ts))  # trajectory points
```

Run the full demo with `python examples/diffeq_demo.py`.
