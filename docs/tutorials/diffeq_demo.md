# Differential Equations Tutorial

`cds.diffeq` solves initial-value problems with classical schemes plus an
adaptive RK45 integrator.

The fixed-step solvers take the step size as **`dt`**, not `n`, and return an
`ODESolution` dataclass whose solution values are read from `.y` (a list, index
`-1` for the final value) and whose grid is `.t`. `rk45` additionally accepts
`rtol` / `atol` and chooses its own steps.

## 1. Single ODE

Solve `dy/dt = -y`, `y(0) = 1`, whose exact value at `t=1` is
`e^-1 ≈ 0.36787944`:

```python
from cds.diffeq import euler_method, midpoint_method, rk4, rk45

f = lambda t, y: -y
print(f"euler      {euler_method(f, 0.0, 1.0, 1.0, dt=0.01).y[-1]:.8f}")
print(f"midpoint   {midpoint_method(f, 0.0, 1.0, 1.0, dt=0.01).y[-1]:.8f}")
print(f"rk4        {rk4(f, 0.0, 1.0, 1.0, dt=0.01).y[-1]:.8f}")
print(f"rk45       {rk45(f, 0.0, 1.0, 1.0, rtol=1e-9).y[-1]:.8f}")
print(f"exact      {2.718281828459045**-1:.8f}")
```

The accuracy ladder is visible in the output: Euler is off in the fourth
decimal, midpoint in the sixth, and both RK4 and adaptive RK45 match the exact
value to the precision printed.

## 2. System of ODEs

`solve_system` returns `(t_values, y_values)` rather than a result object.

```python
from cds.diffeq import solve_system


def lotka(t, state):
    x, y = state
    return [1.1 * x - 0.4 * x * y, 0.1 * x * y - 0.4 * y]  # Lotka-Volterra


ts, ys = solve_system(lotka, t0=0.0, y0=[10.0, 5.0], t_end=15.0, dt=0.05)
print(f"{len(ts)} points")  # 301 with this step
print(f"final [prey, predator] = {[round(v, 4) for v in ys[-1]]}")
```

## 3. Stiff problems: implicit solvers

When explicit methods need absurdly small steps, the implicit solvers stay
stable. See the [Stiff ODE Solvers](stiff_ode_demo.md) tutorial for the full
walkthrough.

```python
from cds.diffeq import backward_euler

sol = backward_euler(lambda t, y: -1000.0 * (y - 1.0), 0.0, 0.0, 0.05, dt=0.001)
print(f"{sol.y[-1]:.6f}")  # 1.000000 — explicit Euler diverges here
```

Run the full demo with:

```bash
python examples/diffeq_demo.py
```
