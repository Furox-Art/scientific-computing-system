# Stiff ODE Solvers Tutorial

`cds.diffeq._implicit` adds two A-stable implicit methods to the explicit
solvers: **backward Euler** (θ = 1) and the **trapezoidal / Crank–Nicolson
method** (θ = ½). Both solve their implicit stage equation with Newton
iteration, using an analytic Jacobian when you supply one and central
differences otherwise.

## 1. Why implicit? The stiff decay problem

For `dy/dt = -1000·(y − 1)` any explicit method diverges unless
`dt < 2/1000`. Backward Euler stays stable at the same step:

```python
from cds.diffeq import backward_euler

k = 1000.0
sol = backward_euler(lambda t, y: -k * (y - 1.0), t0=0.0, y0=0.0,
                     t_end=0.05, dt=0.001)
print(f"{sol.y[-1]:.6f}")   # ≈ 1.000000 — exact is 1 − e⁻⁵⁰ ≈ 1.0
```

## 2. Trapezoid (Crank–Nicolson): stable AND second-order

Backward Euler buys stability with only first-order accuracy. The trapezoidal
rule averages the explicit and implicit slopes for O(dt²) error while staying
A-stable — on smooth problems its error shrinks ~4× when you halve `dt`.

```python
import math
from cds.diffeq import backward_euler, trapezoid_method

f = lambda t, y: -y
be = backward_euler(f, 0.0, 1.0, 2.0, dt=0.05).y[-1]
cn = trapezoid_method(f, 0.0, 1.0, 2.0, dt=0.05).y[-1]
exact = math.exp(-2.0)
print(abs(cn - exact) < abs(be - exact))   # True
```

## 3. Systems: the harmonic oscillator

The `_system` variants take vector states and return `(t_values, y_values)`
just like :func:`cds.diffeq.solve_system`.

```python
import math
from cds.diffeq import trapezoid_method_system

def harmonic(t, y):
    return [y[1], -y[0]]          # x'' = -x as a system

ts, ys = trapezoid_method_system(harmonic, 0.0, [1.0, 0.0],
                                 t_end=4 * math.pi, dt=0.01)
amplitude = math.hypot(ys[-1][0], ys[-1][1])
print(abs(amplitude - 1.0) < 0.01)   # True — energy nearly conserved
```

## 4. Supplying an analytic Jacobian

Newton converges in one step on linear problems either way, but a Jacobian
removes the finite-difference evaluations:

```python
from cds.diffeq import backward_euler_system

jac = lambda t, y: [[-1000.0]]           # df/dy
ts, ys = backward_euler_system(
    lambda t, y: [-1000.0 * (y[0] - 1.0)],
    t0=0.0, y0=[0.0], t_end=0.05, dt=0.001, jac=jac,
)
```

## Notes

- Direction follows `sign(t_end - t0)` exactly like the explicit solvers;
  `dt` is always a magnitude.
- Newton failures raise `ValueError` with a clear message ("singular
  Jacobian" / "did not converge within max_iter") instead of returning
  garbage.
- Rule of thumb: use RK45 until stiffness hurts, then reach for these.
