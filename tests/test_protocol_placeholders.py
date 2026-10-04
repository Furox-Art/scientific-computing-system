"""The ``typing.Protocol`` bodies in ``src/cds`` are placeholders, not code.

Nine ``py/ineffectual-statement`` alerts across ``src/cds/ml/model_selection.py``,
``src/cds/ml/voting.py`` and ``src/cds/tools/adapters.py`` came from the bare
``...`` that conventionally terminates a protocol method. The finding was a false
positive in the strict sense -- nothing was discarded and no ``return`` was
missing -- but the placeholder still had to change, because a statement that
static analysis can see and that never runs is worth removing.

Each body is now a docstring. That is behaviourally identical to ``...``, and
these tests pin the properties that make it safe, so a future edit that puts real
code into one of those bodies fails here instead of passing unnoticed.

The shape check is done on the AST rather than on bytecode deliberately: a
docstring-only body and a docstring followed by ``pass`` compile to almost the
same instructions, so only the syntax tree distinguishes "placeholder" from
"placeholder plus a statement".
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Sequence
from pathlib import Path
from typing import Any, get_type_hints

from cds.ml.metrics import Label
from cds.ml.model_selection import SupervisedModel
from cds.ml.voting import SoftVotingModel
from cds.tools.adapters import _OptimizeModule, _Solver, _Subtractable, _SympyModule

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SOURCES = (
    ROOT / "src" / "cds" / "ml" / "model_selection.py",
    ROOT / "src" / "cds" / "ml" / "voting.py",
    ROOT / "src" / "cds" / "tools" / "adapters.py",
)

PROTOCOLS = (_OptimizeModule, _SympyModule, _Subtractable, _Solver, SupervisedModel)

#: Only these classes are structural contracts; the same files also contain
#: ordinary classes with ordinary method bodies, which must not be constrained.
PROTOCOL_CLASS_NAMES = frozenset({protocol.__name__ for protocol in (*PROTOCOLS, SoftVotingModel)})


def _is_docstring_only(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True when the body is exactly one string-expression and nothing else."""
    body = node.body
    return (
        len(body) == 1
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    )


def test_no_protocol_method_body_carries_a_statement() -> None:
    """Every method of these protocols must remain a pure placeholder.

    A bare ``...`` is an expression statement with no effect -- exactly what the
    alerts flagged. ``pass`` or ``raise`` would be equally executable and equally
    wrong here, so the assertion is on the shape rather than on the old spelling.
    """
    offenders: list[str] = []
    for path in PROTOCOL_SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in PROTOCOL_CLASS_NAMES:
                for member in node.body:
                    if isinstance(member, ast.FunctionDef | ast.AsyncFunctionDef):
                        label = f"{path.name}:{member.lineno} {node.name}.{member.name}"
                        if not _is_docstring_only(member):
                            offenders.append(label)
                        elif not ast.get_docstring(member):
                            offenders.append(f"{label} (no docstring)")
    assert not offenders, f"protocol bodies must stay docstring-only: {offenders}"


def test_no_bare_ellipsis_statements_remain_in_the_protocol_sources() -> None:
    """Direct regression for the reported finding."""
    found: list[str] = []
    for path in PROTOCOL_SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and node.value.value is Ellipsis
            ):
                found.append(f"{path.name}:{node.lineno}")
    assert not found, f"bare `...` expressions are the reported defect: {found}"


class _ConformsToEverything:
    """A single object that structurally satisfies every protocol in play.

    Declaring the annotated locals below (``model: SupervisedModel`` and so on)
    makes mypy verify structural conformance at the assignment, so this class is
    type-checked as a real implementation rather than waved through with a cast.
    """

    def fit(self, X: list[list[float]], y: Sequence[Label]) -> object:
        return self

    def predict(self, x: list[float]) -> Label:
        return "a"

    def predict_proba(self, x: list[float]) -> dict[Label, float]:
        return {"a": 1.0}

    def add(self, *constraints: object) -> object:
        return self

    def check(self) -> object:
        return "sat"

    def __sub__(self, other: object) -> object:
        return self

    def sympify(self, expression: str) -> object:
        return expression

    def simplify(self, expression: object) -> object:
        return expression


def test_protocol_bodies_still_return_none_when_called_directly() -> None:
    """Pins the placeholder semantics that ``...`` provided.

    The calls go through the unbound protocol functions rather than an instance,
    because an instance would dispatch to the conformer's implementation and
    never touch the placeholder body this test is about.
    """
    model: SupervisedModel = _ConformsToEverything()
    voting: SoftVotingModel = _ConformsToEverything()
    solver: _Solver = _ConformsToEverything()

    assert SupervisedModel.fit(model, [], []) is None
    assert SupervisedModel.predict(model, []) is None
    assert SoftVotingModel.predict_proba(voting, []) is None
    assert _Solver.add(solver, "c") is None
    assert _Solver.check(solver) is None


def test_runtime_checkable_protocol_is_still_a_structural_contract() -> None:
    """The library relies on ``isinstance`` against ``SoftVotingModel``.

    ``runtime_checkable`` ``isinstance`` inspects attribute *presence* and never
    invokes a body, which is why replacing ``...`` with a docstring cannot affect
    it. Only this protocol is runtime-checkable, so it is the only one where
    ``isinstance`` is meaningful.
    """

    class _Conforms:
        def fit(self, X: list[list[float]], y: list[Label]) -> object:
            return self

        def predict(self, x: list[float]) -> Label:
            return "a"

        def predict_proba(self, x: list[float]) -> dict[Label, float]:
            return {"a": 1.0}

    assert getattr(SoftVotingModel, "_is_runtime_protocol", False) is True
    assert isinstance(_Conforms(), SoftVotingModel)
    assert not isinstance(object(), SoftVotingModel)

    class _MissingProba:
        def fit(self, X: list[list[float]], y: list[Label]) -> object:
            return self

        def predict(self, x: list[float]) -> Label:
            return "a"

    assert not isinstance(_MissingProba(), SoftVotingModel)


def test_the_other_protocols_are_type_only_and_still_usable_by_cast() -> None:
    """The non-runtime protocols are consumed via ``cast``, which needs no body."""
    from typing import cast

    for protocol in (_OptimizeModule, _SympyModule, _Subtractable, _Solver):
        assert getattr(protocol, "_is_protocol", False) is True
        assert getattr(protocol, "_is_runtime_protocol", False) is False
        assert cast(Any, protocol) is protocol


def test_protocol_signatures_and_annotations_are_intact() -> None:
    """A docstring body must not have disturbed the declared signatures."""
    assert list(inspect.signature(SupervisedModel.fit).parameters) == ["self", "X", "y"]
    assert list(inspect.signature(SupervisedModel.predict).parameters) == ["self", "x"]
    assert list(inspect.signature(SoftVotingModel.predict_proba).parameters) == ["self", "x"]
    assert list(inspect.signature(_Solver.add).parameters) == ["self", "constraints"]
    assert list(inspect.signature(_Subtractable.__sub__).parameters) == ["self", "other"]

    minimize = inspect.signature(_OptimizeModule.minimize)
    assert list(minimize.parameters) == ["self", "function", "x0", "method", "options"]
    assert minimize.parameters["method"].kind is inspect.Parameter.KEYWORD_ONLY

    # `Label` is the `str | int` alias, so resolving hints must still work.
    hints: dict[str, Any] = get_type_hints(SupervisedModel.predict)
    assert hints == {"x": list[float], "return": Label}
    assert get_type_hints(SoftVotingModel.predict_proba)["return"] == dict[Label, float]
