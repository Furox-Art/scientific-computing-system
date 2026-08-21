# Numerical Integration Tutorial

`cds.numerical_integration` provides deterministic quadrature rules. A canonical test: `∫_0^1 e^x dx = e - 1`.

## 1. Newton–Cotes Rules

```python
import math
from cds.numerical_integration import trapezoid, simpson, simpson_38

f = math.exp
print(trapezoid(f, 0, 1, 1000))  # O(h^2)
print(simpson(f, 0, 1, 100))  # O(h^4)
print(simpson_38(f, 0, 1, 99))  # 3/8 variant
```

## 2. Higher-Order & Adaptive

```python
from cds.numerical_integration import gaussian_quadrature, romberg, adaptive_simpson

print(gaussian_quadrature(f, 0, 1, 8).value)  # ~1e-16 error
print(romberg(f, 0, 1).value)  # Richardson extrapolation
print(adaptive_simpson(f, 0, 1).value)  # tolerance-driven bisection
```

**Why it matters:** you can watch error drop from `1e-7` (trapezoid) to `1e-16` (Gauss-Legendre) on the same problem — a clear lesson in quadrature convergence.

## 3. 2-D Tensor-Product Quadrature

Both `simpson_2d` and `gaussian_quadrature_2d` integrate a bivariate
function over a rectangle `[ax, bx] × [ay, by]` by taking the tensor product
of a 1-D rule in each axis.

```python
import math
from cds.numerical_integration import simpson_2d, gaussian_quadrature_2d

# ∬_{[0,1]^2} e^{x+y} dx dy = (e-1)^2 ≈ 2.9525
f = lambda x, y: math.exp(x + y)
exact = (math.e - 1) ** 2

# Composite Simpson 1/3 in each axis — O(h_x^4 + h_y^4).
print(simpson_2d(f, 0, 1, 0, 1, 50, 50))  # ~2.9525

# Tensor-product Gauss-Legendre: n nodes per axis integrate polynomials
# up to degree 2n-1 in *each* variable exactly. 5 nodes is far past exact
# for the smooth exponential, so it lands essentially on the closed form.
print(gaussian_quadrature_2d(f, 0, 1, 0, 1, 5))  # ~2.9525
```

**Exactness check.** `gaussian_quadrature_2d` with `n=3` integrates
`x^5 · y^5` over `[0,1]^2` to machine precision, since each axis is exact to
degree `2n-1 = 5`:

```python
# Analytic value: (1/6)·(1/6) = 1/36 ≈ 0.027778
val = gaussian_quadrature_2d(lambda x, y: x**5 * y**5, 0, 1, 0, 1, 3)
assert abs(val - 1 / 36) < 1e-12
```

**Reversed limits** are honoured consistently: flipping one axis flips the
sign once, flipping both flips it twice (net unchanged), exactly as in 1-D.

Run the full demo with `python examples/numerical_integration_demo.py`.
