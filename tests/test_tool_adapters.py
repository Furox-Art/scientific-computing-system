"""Tests for normalized optional scientific-tool adapters."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from types import ModuleType

import pytest

from cds.tools import ToolRegistry
from cds.tools.adapters import (
    OptimizationResult,
    Satisfiability,
    scipy_minimize,
    sympy_verify_identity,
    z3_satisfiability,
)


def _registry_with_module(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
) -> ToolRegistry:
    registry = ToolRegistry()

    def fake_load(_name: str) -> ModuleType:
        return module

    monkeypatch.setattr(registry, "load", fake_load)
    return registry


class _OptimizeResult:
    def __init__(
        self,
        parameters: Sequence[float],
        objective: float,
        *,
        success: bool = True,
        message: str = "ok",
        iterations: object = 4,
    ) -> None:
        self.x = list(parameters)
        self.fun = objective
        self.success = success
        self.message = message
        self.nit = iterations


class _Optimize:
    def minimize(
        self,
        function: Callable[[object], float],
        x0: Sequence[float],
        *,
        method: str | None = None,
        options: dict[str, object] | None = None,
    ) -> object:
        objective = function(list(x0))
        if method == "nonfinite":
            return _OptimizeResult([math.inf], objective)
        iterations: object = True if method == "boolnit" else 4
        message = str(options or {})
        return _OptimizeResult([value / 2.0 for value in x0], objective / 2.0, message=message, iterations=iterations)


def test_scipy_minimize_normalizes_backend_result(monkeypatch: pytest.MonkeyPatch) -> None:
    scipy = ModuleType("scipy")
    setattr(scipy, "optimize", _Optimize())
    registry = _registry_with_module(monkeypatch, scipy)

    result = scipy_minimize(
        lambda values: sum(value * value for value in values),
        [2.0, 4.0],
        options={"maxiter": 10},
        registry=registry,
    )
    assert result == OptimizationResult((1.0, 2.0), 10.0, True, "{'maxiter': 10}", 4)

    bool_nit = scipy_minimize(lambda values: sum(values), [2.0], method="boolnit", registry=registry)
    assert bool_nit.iterations is None


def test_scipy_minimize_validation_and_nonfinite_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    scipy = ModuleType("scipy")
    setattr(scipy, "optimize", _Optimize())
    registry = _registry_with_module(monkeypatch, scipy)

    with pytest.raises(ValueError, match="must not be empty"):
        scipy_minimize(lambda _values: 0.0, [], registry=registry)
    with pytest.raises(ValueError, match="must be finite"):
        scipy_minimize(lambda _values: 0.0, [math.nan], registry=registry)
    with pytest.raises(ValueError, match="method must not be empty"):
        scipy_minimize(lambda _values: 0.0, [1.0], method=" ", registry=registry)
    with pytest.raises(ValueError, match="non-finite"):
        scipy_minimize(lambda values: sum(values), [1.0], method="nonfinite", registry=registry)


class _Expr:
    def __init__(self, value: int) -> None:
        self.value = value

    def __sub__(self, other: object) -> object:
        if not isinstance(other, _Expr):
            return NotImplemented
        return _Expr(self.value - other.value)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, int):
            return self.value == other
        if isinstance(other, _Expr):
            return self.value == other.value
        return False


def test_sympy_verify_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    sympy = ModuleType("sympy")

    def sympify(expression: str) -> object:
        return _Expr(int(expression))

    def simplify(expression: object) -> object:
        return expression

    setattr(sympy, "sympify", sympify)
    setattr(sympy, "simplify", simplify)
    registry = _registry_with_module(monkeypatch, sympy)

    assert sympy_verify_identity("2", "2", registry=registry)
    assert not sympy_verify_identity("2", "3", registry=registry)
    with pytest.raises(ValueError, match="must not be empty"):
        sympy_verify_identity("", "2", registry=registry)
    with pytest.raises(ValueError, match="must not be empty"):
        sympy_verify_identity("2", " ", registry=registry)


class _FakeSolver:
    def __init__(self, state: str) -> None:
        self.state = state
        self.constraints: tuple[object, ...] = ()

    def add(self, *constraints: object) -> object:
        self.constraints = constraints
        return None

    def check(self) -> object:
        return self.state


def _z3_module(state: str) -> ModuleType:
    module = ModuleType("z3")

    def solver_factory() -> _FakeSolver:
        return _FakeSolver(state)

    setattr(module, "Solver", solver_factory)
    return module


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("sat", Satisfiability.SAT),
        ("unsat", Satisfiability.UNSAT),
        ("unknown", Satisfiability.UNKNOWN),
        ("unexpected", Satisfiability.UNKNOWN),
    ],
)
def test_z3_satisfiability_states(
    monkeypatch: pytest.MonkeyPatch,
    state: str,
    expected: Satisfiability,
) -> None:
    registry = _registry_with_module(monkeypatch, _z3_module(state))
    seen: list[ModuleType] = []

    def build(module: ModuleType) -> Sequence[object]:
        seen.append(module)
        return (object(), object())

    assert z3_satisfiability(build, registry=registry) is expected
    assert seen and seen[0].__name__ == "z3"
