# Math Utilities (Calculus & Linear Algebra) Tutorial

`cds.math_utils` provides pure-Python calculus primitives and a compact linear-algebra toolkit.

## 1. Calculus

```python
import math
from cds.math_utils import derivative, integral, gradient

print(derivative(lambda x: x**2, x=3.0))  # ≈ 6
print(integral(lambda x: math.sin(x), a=0.0, b=math.pi))  # ≈ 2
print(gradient(lambda v: v[0] ** 2 + v[1] ** 2, point=[1.0, 2.0]))  # [2, 4]
```

## 2. Linear Algebra Basics

```python
from cds.math_utils import mat_mul, transpose, determinant, solve_linear

A = [[2.0, 1.0], [1.0, 3.0]]
print(mat_mul(A, [[1, 0], [0, 1]]))  # back to A
print(transpose(A))  # [[2,1],[1,3]]
print(determinant(A))  # 5.0
print(solve_linear(A, [3.0, 4.0]))  # solution of A·x = b
```

## 3. Decompositions

```python
from cds.math_utils import lu_decomposition, qr_decomposition, power_iteration

L, U, P = lu_decomposition(A)  # PLU factorisation
Q, R = qr_decomposition(A)  # QR factorisation
eigval, eigvec = power_iteration(A)  # dominant eigenpair via power iteration
```

## 4. Vector ops

```python
from cds.math_utils import dot

print(dot([1.0, 2.0], [3.0, 4.0]))  # 11.0
```

Run the full demo with `python examples/math_utils_demo.py`.
