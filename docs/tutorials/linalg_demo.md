# Linear Algebra Tutorial

`cds.math_utils` provides from-scratch numerical linear algebra, QR and
Cholesky decompositions, eigenvalues via power iteration, linear-system
solving, and matrix inverse. All O(N³) implementations in readable pure Python.

## 1. QR decomposition (Householder)

Factor `A = QR` into an orthogonal matrix `Q` and upper-triangular `R`.

```python
from cds.math_utils import qr_decomposition

A = [[12.0, -51.0, 4.0], [6.0, 167.0, -68.0], [-4.0, 24.0, -41.0]]
Q, R = qr_decomposition(A)
```

```
Q (orthogonal):
  -0.8571   0.3943   0.3314
  -0.4286  -0.9029  -0.0343
   0.2857  -0.1714   0.9429
R (upper triangular):
   -14.0000   -21.0000    14.0000
     0.0000  -175.0000    70.0000
     0.0000    -0.0000   -35.0000
```

QR is the backbone of stable least-squares and eigenvalue algorithms.

## 2. Cholesky decomposition

For a symmetric positive-definite matrix, `A = L Lᵀ` with `L` lower-triangular.
Cheaper and more numerically stable than LU.

```python
from cds.math_utils import cholesky

M = [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]]
L = cholesky(M)
```

```
L (lower triangular), A = L Lᵀ:
    2.00    0.00    0.00
    6.00    1.00    0.00
   -8.00    5.00    3.00
```

## 3. Dominant eigenvalue (power iteration)

Iterative method that converges to the largest-magnitude eigenvalue and its
eigenvector.

```python
from cds.math_utils import power_iteration

S = [[2.0, 1.0], [1.0, 3.0]]
eigval, eigvec = power_iteration(S)
print(f"Dominant eigenvalue: {eigval:.6f}")  # 3.618034 (the golden ratio + 1)
print(f"Eigenvector:         [{eigvec[0]:.4f}, {eigvec[1]:.4f}]")
```

## 4. Solving a linear system

`solve_linear` uses LU decomposition with partial pivoting (O(N³)), far
better than naive determinant expansion (O(N!)).

```python
from cds.math_utils import solve_linear

A2 = [[3.0, 2.0], [1.0, 2.0]]
b = [12.0, 8.0]
x = solve_linear(A2, b)
print(f"Solution x = [{x[0]:.4f}, {x[1]:.4f}]")  # [2.0000, 3.0000]
```

## 5. Matrix inverse

`A · A⁻¹` should give the identity.

```python
from cds.math_utils import mat_mul, matrix_inverse

inv = matrix_inverse(A2)
check = mat_mul(A2, inv)
```

```
A · A⁻¹ =
   1.0000   0.0000
   0.0000   1.0000
```

Composing these primitives is straightforward, `transpose`, `mat_mul`, and
the solvers interoperate on plain `list[list[float]]` inputs. See
[`math_utils_demo.md`](math_utils_demo.md) for the calculus side of the module.

Run the full demo:

```bash
python examples/linalg_demo.py
```
