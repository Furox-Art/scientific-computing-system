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


ts, ys = solve_system(lotka, t0=0.0, y0=[10.0, 5.0], t_end=15.0, dt=0.05)
print(ys[-1])       # final [prey, predator]
print(len(ts))      # trajectory points (301 with this step)
```

## 3. Stiff problems: implicit solvers

When explicit methods need absurdly small steps, the implicit solvers stay
stable. See the [Stiff ODE Solvers](stiff_ode_demo.md) tutorial for the full
walkthrough.

```python
from cds.diffeq import backward_euler

sol = backward_euler(lambda t, y: -1000.0 * (y - 1.0),
                     t0=0.0, y0=0.0, t_end=0.05, dt=0.001)
print(f"{sol.y[-1]:.6f}")   # ≈ 1.000000 — explicit Euler diverges here
```

Run the full demo with `python examples/diffeq_demo.py`.
