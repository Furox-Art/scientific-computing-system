# Mathematical Modeling Tutorial

`cds.modeling` provides a small symbolic algebra: build expressions, differentiate them symbolically, simplify, export to LaTeX, group them into a `MathModel`, and solve numerically.

## 1. Build & Evaluate an Expression

```python
from cds.modeling import Variable, Constant

x = Variable("x")
expr = x**2 + 3 * x  # x^2 + 3x
print(expr.evaluate({"x": 2}))  # 10.0
print(expr.to_str())  # ((x ** 2) + (3 * x))
```

## 2. Symbolic Differentiation

```python
from cds.modeling import Sin, Log, Exp

x = Variable("x")
f = Sin(x) * Exp(x)  # sin(x) * e^x
df = f.diff("x")  # symbolic derivative
print(df.to_str())
print(df.evaluate({"x": 0.0}))  # 1.0  (analytically e^0*(sin0+cos0))
```

The chain, product, and quotient rules are all built in — for `Sin`, `Cos`, `Exp`, `Log`, `Sqrt`, `Pow`, and the arithmetic operators.

## 3. Simplify & Export to LaTeX

```python
from cds.modeling import Variable

x = Variable("x")
print((x + 0).simplify().to_str())  # x
print((x * 1).simplify().to_str())  # x
print((x**0).simplify().to_str())  # 1

expr = (Variable("x") ** Constant(2.0)) / Variable("y")
print(expr.to_latex())  # \frac{x^{2}}{y}
```

## 4. Group Equations into a MathModel

```python
from cds.modeling import MathModel, Variable

t, a = Variable("t"), Variable("a")
v0 = Variable("v0")
model = MathModel(
    name="Kinematics",
    description="Constant-acceleration motion",
    parameters={"a": 9.81},
    variables=["t", "v0", "x0"],
)
model.add_equation("velocity", v0 + a * t)
model.add_equation("position", Variable("x0") + v0 * t + 0.5 * a * t**2)

print(model.evaluate({"t": 2.0, "v0": 0.0, "x0": 0.0}))
# {'velocity': 19.62, 'position': 19.62}

# Jacobian column: how every equation depends on t
for label, d in model.jacobian("t").items():
    print(label, d.to_str())
```

## 5. Solve & Fit

```python
from cds.modeling import Variable, solve_equation

# Solve x^2 - 2 = 0  =>  x = sqrt(2)
x = Variable("x")
result = solve_equation(x**2 - 2, variable="x", x0=1.0)
print(result.x)  # ~1.4142
print(result.converged)  # True
```

For fitting a model's parameters to data, see `cds.modeling.fit_parameters` — it minimizes the residual sum of squares via `cds.optimization.gradient_descent`.

Run the full demo with `python examples/modeling_demo.py`.
