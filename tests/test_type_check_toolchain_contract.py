"""Keep the Python support floor and the type-checking toolchain consistent.

Three settings describe the same decision and drift apart easily:

* ``[project] requires-python`` -- the interpreters users may run CDS on;
* ``[tool.mypy] python_version`` -- the syntax and stdlib mypy may assume;
* ``[tool.ruff] target-version`` -- the syntax ruff may rewrite toward.

Issue #62 capped numpy below 2.5 in the ``[dev]`` extra because the numpy 2.5
stub (``numpy/__init__.pyi``) uses PEP 695 ``type`` aliases, and mypy releases
before 2.4.1 reject those aliases whenever ``python_version`` is below 3.12.
The cap is gone now that the support floor is 3.11 and the mypy floor is
2.4.1, which parses the numpy 2.5 stubs at a 3.11 target. Without a test, the
next floor change could silently reintroduce the cap -- or drop the mypy
version that makes the uncapped stub parseable -- and only fail in CI's
``mypy src/`` step with a ``[syntax]`` error that names numpy, not this
contract.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any, cast

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"

# numpy 2.5 introduced PEP 695 `type` aliases in numpy/__init__.pyi.
STUB_TYPE_ALIAS_RELEASE = (2, 5)
# First mypy release verified to parse those aliases at a 3.11/3.12 target.
MYPY_FLOOR = (2, 4, 1)

_NAME = re.compile(r"^([A-Za-z0-9][\w.-]*)")


def _pyproject() -> dict[str, Any]:
    payload: dict[str, Any] = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return payload


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", text))


def _extras() -> dict[str, list[str]]:
    project = _pyproject()["project"]
    return cast(dict[str, list[str]], project["optional-dependencies"])


def _requirement_name(requirement: str) -> str:
    match = _NAME.match(requirement.strip())
    assert match is not None, requirement
    return match.group(1).lower()


def _requires_python() -> tuple[int, ...]:
    return _version_tuple(cast(str, _pyproject()["project"]["requires-python"]))


def _mypy_target() -> tuple[int, ...]:
    return _version_tuple(cast(str, _pyproject()["tool"]["mypy"]["python_version"]))


def _ruff_target() -> tuple[int, ...]:
    target = cast(str, _pyproject()["tool"]["ruff"]["target-version"])
    assert target.startswith("py"), target
    digits = target.removeprefix("py")
    return (int(digits[0]), int(digits[1:]))


def _mypy_floors() -> list[tuple[int, ...]]:
    """Every ``mypy>=X`` floor declared across the extras groups."""
    floors: list[tuple[int, ...]] = []
    for requirements in _extras().values():
        for requirement in requirements:
            if _requirement_name(requirement) != "mypy":
                continue
            match = re.search(r">=\s*([\d.]+)", requirement)
            assert match is not None, requirement
            floors.append(_version_tuple(match.group(1)))
    return floors


def _upper_bound(requirement: str) -> tuple[int, ...] | None:
    """The parsed ``<`` upper bound of a requirement, or None when unbounded."""
    match = re.search(r"<\s*([\d.]+)", requirement)
    return _version_tuple(match.group(1)) if match else None


def _numpy_requirements() -> list[str]:
    return [
        requirement
        for requirements in _extras().values()
        for requirement in requirements
        if _requirement_name(requirement) == "numpy"
    ]


def test_support_floor_drops_python_310() -> None:
    floor = _requires_python()
    assert floor[:2] >= (3, 11), f"requires-python is {floor}"


def test_classifiers_match_the_support_floor() -> None:
    classifiers = cast(list[str], _pyproject()["project"]["classifiers"])
    python_classifiers = [
        classifier
        for classifier in classifiers
        if classifier.startswith("Programming Language :: Python :: 3.")
    ]
    assert "Programming Language :: Python :: 3.11" in python_classifiers
    assert not [c for c in python_classifiers if c.endswith("3.10")], python_classifiers


def test_mypy_target_matches_the_support_floor() -> None:
    assert _mypy_target()[:2] == _requires_python()[:2], (
        f"mypy python_version is {_mypy_target()} but requires-python is {_requires_python()}; "
        "mypy would type-check against a Python the package no longer supports"
    )


def test_ruff_target_matches_the_support_floor() -> None:
    assert _ruff_target() == _requires_python()[:2], (
        f"ruff target-version is {_ruff_target()} but requires-python is {_requires_python()}"
    )


def test_dev_extra_does_not_pin_numpy() -> None:
    """numpy reaches the dev environment only through pandas-stubs."""
    dev = _extras()["dev"]
    pinned = [requirement for requirement in dev if _requirement_name(requirement) == "numpy"]
    assert pinned == [], f"[dev] must not pin numpy directly, found {pinned}"


def test_no_numpy_requirement_carries_an_upper_bound() -> None:
    capped = [
        requirement
        for requirement in _numpy_requirements()
        if _upper_bound(requirement) is not None
    ]
    assert capped == [], (
        f"numpy requirements carry an upper bound again: {capped}. The <{STUB_TYPE_ALIAS_RELEASE[0]}."
        f"{STUB_TYPE_ALIAS_RELEASE[1]} cap existed only for a mypy that could not parse the stub; "
        "see issue #62."
    )


def test_mypy_floor_parses_the_numpy_25_stub() -> None:
    floors = _mypy_floors()
    assert floors, "mypy must be declared in the dev extras"
    assert min(floors) >= MYPY_FLOOR, (
        f"the lowest declared mypy floor is {min(floors)}, but numpy {STUB_TYPE_ALIAS_RELEASE[0]}."
        f"{STUB_TYPE_ALIAS_RELEASE[1]}+ stubs need >= {MYPY_FLOOR} to parse their PEP 695 "
        "type aliases at a 3.11 target"
    )


@pytest.mark.parametrize(
    ("requirement", "expected"),
    [
        ("numpy>=2.0,<2.5", (2, 5)),
        ("numpy>=2.0,<2.4", (2, 4)),
        ("numpy>=1.26", None),
        ("numpy", None),
        ("numpy>=2.5", None),
    ],
)
def test_upper_bound_parsing_handles_real_specifier_spellings(
    requirement: str, expected: tuple[int, ...] | None
) -> None:
    assert _upper_bound(requirement) == expected
