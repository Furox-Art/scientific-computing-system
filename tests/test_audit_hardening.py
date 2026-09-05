"""Regression tests for repository-wide audit hardening."""

from __future__ import annotations

import math
import random
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

from cds.data_io import iter_csv_batches, profile_file
from cds.quantum.circuit import Qubit
from cds.quantum.simulator import measure, simulate
from cds.quantum import QuantumCircuit, hadamard
from cds.tools import ToolRegistry, ToolSpec
from cds.tools.adapters import _validate_sympy_expression, sympy_verify_identity


def _plus_qubit() -> Qubit:
    amplitude = 1.0 / math.sqrt(2.0)
    return Qubit(alpha=complex(amplitude), beta=complex(amplitude))


def test_measure_uses_private_reproducible_randomness() -> None:
    assert measure(_plus_qubit(), seed=19) == measure(_plus_qubit(), seed=19)

    rng_a = random.Random(7)
    rng_b = random.Random(7)
    outcomes_a = [measure(_plus_qubit(), rng=rng_a) for _ in range(8)]
    outcomes_b = [measure(_plus_qubit(), rng=rng_b) for _ in range(8)]
    assert outcomes_a == outcomes_b

    with pytest.raises(ValueError, match="either seed or rng"):
        measure(_plus_qubit(), seed=1, rng=random.Random(1))


@pytest.mark.parametrize("shots", [True, 0, -1, cast(int, 1.5)])
def test_simulate_rejects_invalid_shot_counts(shots: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        simulate(QuantumCircuit().add(hadamard()), shots=shots, seed=1)


def test_profile_file_respects_hard_memory_budget(tmp_path: Path) -> None:
    path = tmp_path / "sample.bin"
    path.write_bytes(b"x" * 4096)

    with pytest.raises(ValueError, match="at least min_block_size"):
        profile_file(path, memory_budget_bytes=1024, min_block_size=2048)

    profile = profile_file(
        path,
        memory_budget_bytes=4096,
        min_block_size=1024,
        max_block_size=8192,
    )
    assert profile.recommended_block_size <= 4096


def test_csv_delimiter_contract_is_explicit(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")

    for delimiter in ("", "::"):
        with pytest.raises(ValueError, match="exactly one character"):
            list(iter_csv_batches(path, delimiter=delimiter))


def test_tool_spec_rejects_ambiguous_metadata() -> None:
    with pytest.raises(ValueError, match="purpose"):
        ToolSpec("tool", "module", "dist", ("cap",), " ")
    with pytest.raises(ValueError, match="empty values"):
        ToolSpec("tool", "module", "dist", ("cap", " "), "purpose")
    with pytest.raises(ValueError, match="unique"):
        ToolSpec("tool", "module", "dist", ("cap", "cap"), "purpose")


def test_sympy_expression_guard_accepts_arithmetic_surface() -> None:
    for expression in (
        "x + 2*y - 3",
        "-x / 2",
        "+x",
        "x**2",
        "sin(x) + sqrt(4) + Abs(-2)",
        "1j + E + pi",
    ):
        _validate_sympy_expression(expression)


def test_sympy_expression_guard_rejects_python_evaluation_features() -> None:
    invalid = (
        "True",
        "'text'",
        "_private + 1",
        "x % 2",
        "not x",
        "unknown(x)",
        "sin(x, evaluate=False)",
        "x.real",
        "x[0]",
        "x < 1",
        "lambda x: x",
    )
    for expression in invalid:
        with pytest.raises(ValueError):
            _validate_sympy_expression(expression)

    with pytest.raises(ValueError, match="valid arithmetic syntax"):
        _validate_sympy_expression("x +")
    with pytest.raises(ValueError, match="too long"):
        _validate_sympy_expression("x" * 4097)
    with pytest.raises(ValueError, match="too complex"):
        _validate_sympy_expression(" + ".join(["x"] * 130))


def test_sympy_verify_identity_guards_before_backend_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    sympy = ModuleType("sympy")

    class Expr:
        def __init__(self, value: int) -> None:
            self.value = value

        def __sub__(self, other: object) -> object:
            assert isinstance(other, Expr)
            return Expr(self.value - other.value)

        def __eq__(self, other: object) -> bool:
            return isinstance(other, int) and self.value == other

    setattr(sympy, "sympify", lambda text: Expr(int(text)))
    setattr(sympy, "simplify", lambda expression: expression)
    registry = ToolRegistry()
    monkeypatch.setattr(registry, "load", lambda _name: sympy)

    assert sympy_verify_identity("2", "2", registry=registry)
    with pytest.raises(ValueError, match="approved mathematical function"):
        sympy_verify_identity("__import__('os')", "0", registry=registry)
