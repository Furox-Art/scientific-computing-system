"""Normalized adapters for optional SciPy, SymPy, and Z3 backends."""

from __future__ import annotations

import ast
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from types import ModuleType
from typing import Protocol, SupportsFloat, cast

from cds.tools.registry import ToolRegistry, default_registry


@dataclass(frozen=True)
class OptimizationResult:
    """Backend-neutral result returned by :func:`scipy_minimize`."""

    parameters: tuple[float, ...]
    objective: float
    success: bool
    message: str
    iterations: int | None


class Satisfiability(str, Enum):
    """Normalized SMT solver outcome."""

    SAT = "sat"
    UNSAT = "unsat"
    UNKNOWN = "unknown"


class _OptimizeModule(Protocol):
    def minimize(
        self,
        function: Callable[[object], float],
        x0: Sequence[float],
        *,
        method: str | None = None,
        options: dict[str, object] | None = None,
    ) -> object: ...


class _SympyModule(Protocol):
    def sympify(self, expression: str) -> object: ...

    def simplify(self, expression: object) -> object: ...


class _Subtractable(Protocol):
    def __sub__(self, other: object) -> object: ...


class _Solver(Protocol):
    def add(self, *constraints: object) -> object: ...

    def check(self) -> object: ...


_SAFE_SYMPY_FUNCTIONS = frozenset(
    {
        "Abs",
        "acos",
        "asin",
        "atan",
        "cos",
        "cosh",
        "exp",
        "log",
        "sin",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
    }
)
_SAFE_BINARY_OPERATORS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
_SAFE_UNARY_OPERATORS = (ast.UAdd, ast.USub)


def _validate_sympy_expression(expression: str) -> None:
    """Reject Python-evaluation features before handing text to SymPy."""
    if len(expression) > 4096:
        raise ValueError("identity expression is too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("identity expression is not valid arithmetic syntax") from exc

    nodes = list(ast.walk(tree))
    if len(nodes) > 256:
        raise ValueError("identity expression is too complex")

    for node in nodes:
        if isinstance(node, (ast.Expression, ast.Load)):
            continue
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float, complex)):
                raise ValueError("identity expressions may contain only numeric constants")
            continue
        if isinstance(node, ast.Name):
            if node.id.startswith("_"):
                raise ValueError("private names are not allowed in identity expressions")
            continue
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, _SAFE_BINARY_OPERATORS):
                raise ValueError("unsupported operator in identity expression")
            continue
        if isinstance(node, _SAFE_BINARY_OPERATORS):
            continue
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, _SAFE_UNARY_OPERATORS):
                raise ValueError("unsupported unary operator in identity expression")
            continue
        if isinstance(node, _SAFE_UNARY_OPERATORS):
            continue
        if isinstance(node, ast.Call):
            if (
                not isinstance(node.func, ast.Name)
                or node.func.id not in _SAFE_SYMPY_FUNCTIONS
                or node.keywords
            ):
                raise ValueError("only approved mathematical function calls are allowed")
            continue
        raise ValueError(f"unsupported syntax in identity expression: {type(node).__name__}")


def _registry(registry: ToolRegistry | None) -> ToolRegistry:
    return registry if registry is not None else default_registry()


def scipy_minimize(
    objective: Callable[[Sequence[float]], float],
    initial: Sequence[float],
    *,
    method: str | None = None,
    options: dict[str, object] | None = None,
    registry: ToolRegistry | None = None,
) -> OptimizationResult:
    """Minimize a scalar objective through optional SciPy and normalize the result."""
    x0 = tuple(float(value) for value in initial)
    if not x0:
        raise ValueError("initial parameters must not be empty")
    if any(not math.isfinite(value) for value in x0):
        raise ValueError("initial parameters must be finite")
    if method is not None and not method.strip():
        raise ValueError("method must not be empty when provided")

    scipy = _registry(registry).load("scipy")
    optimize = cast(_OptimizeModule, getattr(scipy, "optimize"))

    def wrapped(values: object) -> float:
        return float(objective(cast(Sequence[float], values)))

    raw = optimize.minimize(wrapped, x0, method=method, options=options)
    raw_parameters: object = getattr(raw, "x")
    parameters = tuple(float(value) for value in cast(Iterable[SupportsFloat], raw_parameters))
    objective_value = float(getattr(raw, "fun"))
    if any(not math.isfinite(value) for value in parameters) or not math.isfinite(objective_value):
        raise ValueError("SciPy returned non-finite optimization output")

    iterations_raw: object = getattr(raw, "nit", None)
    iterations = (
        iterations_raw
        if isinstance(iterations_raw, int) and not isinstance(iterations_raw, bool)
        else None
    )
    return OptimizationResult(
        parameters=parameters,
        objective=objective_value,
        success=bool(getattr(raw, "success")),
        message=str(getattr(raw, "message")),
        iterations=iterations,
    )


def sympy_verify_identity(
    left: str,
    right: str,
    *,
    registry: ToolRegistry | None = None,
) -> bool:
    """Return whether SymPy can simplify ``left - right`` exactly to zero.

    String expressions are restricted to arithmetic syntax and a small
    allowlist of mathematical functions before they reach SymPy. This keeps
    agent/user-provided expressions from exposing Python evaluation features.
    """
    if not left.strip() or not right.strip():
        raise ValueError("identity expressions must not be empty")
    _validate_sympy_expression(left)
    _validate_sympy_expression(right)
    sympy = cast(_SympyModule, _registry(registry).load("sympy"))
    left_expr = sympy.sympify(left)
    right_expr = sympy.sympify(right)
    difference = cast(_Subtractable, left_expr) - right_expr
    simplified = sympy.simplify(difference)
    return bool(simplified == 0)


def z3_satisfiability(
    build_constraints: Callable[[ModuleType], Sequence[object]],
    *,
    registry: ToolRegistry | None = None,
) -> Satisfiability:
    """Build constraints with optional Z3 and return a backend-neutral solver state."""
    z3 = _registry(registry).load("z3")
    solver_factory = cast(Callable[[], _Solver], getattr(z3, "Solver"))
    solver = solver_factory()
    solver.add(*tuple(build_constraints(z3)))
    state = str(solver.check()).lower()
    if state == "sat":
        return Satisfiability.SAT
    if state == "unsat":
        return Satisfiability.UNSAT
    return Satisfiability.UNKNOWN
