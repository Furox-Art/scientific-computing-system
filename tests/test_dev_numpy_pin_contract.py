"""Enforce the dev-only numpy cap described in issue #62.

`[tool.mypy] python_version` is deliberately ``"3.10"`` because CDS supports
Python 3.10+. The numpy 2.5 stub (``numpy/__init__.pyi``) uses the ``type``
statement, which only parses under a 3.13+ mypy target, so ``[dev]`` caps numpy
at ``<2.5``. Without the cap, ``mypy --strict src/`` aborts with a ``[syntax]``
error before any real type checking happens.

The cap is dev-only: CDS never imports numpy at runtime, so lifting it changes
nothing for end users. Issue #62 says to drop the cap "once mypy's default
python_version advances to 3.13+" -- and that coupling was previously only
documented in a comment, which is exactly how tech debt rots silently. This test
makes the condition executable in both directions.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import tomllib

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"

# mypy's numpy stub needs a 3.13+ target before the `type` statement parses.
STUB_REQUIRES_PYTHON = (3, 13)
# The numpy release whose stub introduced the `type` statement.
CAP_BELOW = (2, 5)


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", text))


def _pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _mypy_python_version() -> tuple[int, ...]:
    config = _pyproject().get("tool", {}).get("mypy", {})
    assert "python_version" in config, "pyproject.toml must pin [tool.mypy] python_version"
    return _version_tuple(str(config["python_version"]))


def _dev_numpy_requirement() -> str:
    dev = _pyproject()["project"]["optional-dependencies"]["dev"]
    matches = [req for req in dev if re.match(r"\s*numpy\b", req)]
    assert len(matches) == 1, f"expected exactly one numpy requirement in [dev], found {matches}"
    return matches[0]


def _upper_cap(requirement: str) -> tuple[int, ...] | None:
    """Return the parsed ``<`` upper bound of a requirement, or None when unbounded."""
    match = re.search(r"<\s*([\d.]+)", requirement)
    return _version_tuple(match.group(1)) if match else None


def _has_upper_cap(requirement: str) -> bool:
    return _upper_cap(requirement) is not None


def test_dev_numpy_requirement_is_parseable():
    assert _dev_numpy_requirement().strip()


def test_mypy_target_stays_below_stub_requirement():
    """CDS supports 3.10+, so the mypy target must not silently outrun the stubs."""
    target = _mypy_python_version()
    requires_python = _pyproject()["project"]["requires-python"]
    supported = _version_tuple(requires_python)
    assert min(supported) <= 3.10, f"requires-python drifted to {requires_python}"
    assert target < STUB_REQUIRES_PYTHON, (
        f"mypy python_version is {target}, which is >= {STUB_REQUIRES_PYTHON}. The numpy 2.5 stub now "
        "parses natively, so issue #62 says to drop the <2.5 cap from [dev] and update this test."
    )


def test_numpy_cap_matches_mypy_target():
    """The cap is required below 3.13 and forbidden at or above it."""
    requirement = _dev_numpy_requirement()
    capped = _has_upper_cap(requirement)
    needs_cap = _mypy_python_version() < STUB_REQUIRES_PYTHON
    if needs_cap:
        assert capped, (
            f"[dev] numpy requirement {requirement!r} lost its <{CAP_BELOW[0]}.{CAP_BELOW[1]} cap. "
            f"mypy targets {_mypy_python_version()}, so `mypy --strict src/` will now fail with a "
            "[syntax] error from the numpy stub. See issue #62."
        )
    else:
        assert not capped, (
            f"[dev] numpy requirement {requirement!r} still caps numpy below {CAP_BELOW[0]}.{CAP_BELOW[1]} "
            "but mypy now targets 3.13+. Issue #62 says to remove it."
        )


@pytest.mark.parametrize(
    "requirement,expected_cap",
    [
        ("numpy>=2.0,<2.5", (2, 5)),
        ("numpy>=2.0,<2.4", (2, 4)),
        ("numpy>=2.0", None),
        ("numpy", None),
        ("numpy>=2.0", None),
    ],
)
def test_upper_cap_parsing_handles_real_specifier_spellings(requirement, expected_cap):
    assert _upper_cap(requirement) == expected_cap


def test_cap_detection_agrees_with_parsed_cap():
    assert _has_upper_cap("numpy>=2.0,<2.5") is True
    assert _has_upper_cap("numpy>=2.0") is False


def test_lower_bounds_are_not_mistaken_for_caps():
    """``>=`` must never read as an upper cap, or the guard would pass vacuously."""
    assert _upper_cap("numpy>=2.5") is None
    assert _has_upper_cap("numpy>=2.5") is False
