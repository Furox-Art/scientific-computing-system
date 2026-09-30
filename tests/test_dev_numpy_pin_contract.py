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

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"

# mypy's numpy stub needs a 3.13+ target before the `type` statement parses.
STUB_REQUIRES_PYTHON = (3, 13)
# The numpy release whose stub introduced the `type` statement.
CAP_BELOW = (2, 5)

_REQUIREMENT = re.compile(r"""["']([^"']+)["']""")


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", text))


def _table_body(text: str, header: str) -> list[str]:
    """Return the lines of a single TOML table, stopping at the next table header.

    ``tomllib`` is deliberately not used: it only exists on Python 3.11+, and CI
    runs this test file on 3.10 too (the repo hit that exact wall once already,
    in the commit 'fix(ci): drop tomllib from 3.10 tests'). The repo also takes
    no test dependency on a backport, so the extraction stays textual.
    """
    lines = text.splitlines()
    body: list[str] = []
    inside = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("["):
            if inside:
                break
            inside = stripped == header
            continue
        if inside:
            body.append(line)
    if not inside:
        raise AssertionError(f"pyproject.toml has no {header} table")
    return body


def _extras() -> list[str]:
    """Return every requirement string under [project.optional-dependencies].

    The extras arrays span many lines, so scan the whole table body instead of
    only the line that carries the ``=``.
    """
    body = "\n".join(
        _table_body(PYPROJECT.read_text(encoding="utf-8"), "[project.optional-dependencies]")
    )
    return _REQUIREMENT.findall(body)


def _dev_numpy_requirement() -> str:
    """Return the single numpy requirement from the [dev] extra's own array."""
    lines = _table_body(PYPROJECT.read_text(encoding="utf-8"), "[project.optional-dependencies]")
    collecting = False
    found: list[str] = []
    for line in lines:
        if re.match(r"^dev\s*=", line):
            collecting = True
            tail = line.partition("=")[2]
            found.extend(_REQUIREMENT.findall(tail))
            continue
        if collecting:
            if line.strip().startswith("]"):
                break
            found.extend(_REQUIREMENT.findall(line))
    matches = [req for req in found if re.match(r"\s*numpy\b", req)]
    assert len(matches) == 1, f"expected exactly one numpy requirement in [dev], found {matches}"
    return matches[0]


def _mypy_python_version() -> tuple[int, ...]:
    for line in _table_body(PYPROJECT.read_text(encoding="utf-8"), "[tool.mypy]"):
        match = re.match(r"""^python_version\s*=\s*["']([^"']+)["']""", line)
        if match:
            return _version_tuple(match.group(1))
    raise AssertionError("pyproject.toml must pin [tool.mypy] python_version")


def _requires_python() -> tuple[int, ...]:
    for line in _table_body(PYPROJECT.read_text(encoding="utf-8"), "[project]"):
        match = re.match(r"""^requires-python\s*=\s*["']([^"']+)["']""", line)
        if match:
            return _version_tuple(match.group(1))
    raise AssertionError("pyproject.toml must declare requires-python")


def _upper_cap(requirement: str) -> tuple[int, ...] | None:
    """Return the parsed ``<`` upper bound of a requirement, or None when unbounded."""
    match = re.search(r"<\s*([\d.]+)", requirement)
    return _version_tuple(match.group(1)) if match else None


def _has_upper_cap(requirement: str) -> bool:
    return _upper_cap(requirement) is not None


def test_dev_numpy_requirement_is_parseable() -> None:
    assert _dev_numpy_requirement().strip()


def test_table_extraction_reads_real_pyproject() -> None:
    """The textual extractors must see the real file, not silently return nothing."""
    assert "numpy>=2.0,<2.5" in _dev_numpy_requirement()
    assert _mypy_python_version() < STUB_REQUIRES_PYTHON
    assert _requires_python()[0] >= 3


def test_extras_capture_every_extras_group() -> None:
    """A regression guard on the extractor itself: a dropped table must not pass quietly."""
    every = _extras()
    assert len(every) > 40, f"expected the full extras list, extracted only {len(every)}"
    assert any(req.startswith("pandas-stubs") for req in every)
    assert any(req.startswith("mypy") for req in every)


def test_mypy_target_stays_below_stub_requirement() -> None:
    """CDS supports 3.10+, so the mypy target must not silently outrun the stubs."""
    target = _mypy_python_version()
    supported = _requires_python()
    assert min(supported) <= 3.10, f"requires-python drifted to {supported}"
    assert target < STUB_REQUIRES_PYTHON, (
        f"mypy python_version is {target}, which is >= {STUB_REQUIRES_PYTHON}. The numpy 2.5 stub now "
        "parses natively, so issue #62 says to drop the <2.5 cap from [dev] and update this test."
    )


def test_numpy_cap_matches_mypy_target() -> None:
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
    ("requirement", "expected_cap"),
    [
        ("numpy>=2.0,<2.5", (2, 5)),
        ("numpy>=2.0,<2.4", (2, 4)),
        ("numpy>=2.0", None),
        ("numpy", None),
        ("numpy>=2.0", None),
    ],
)
def test_upper_cap_parsing_handles_real_specifier_spellings(
    requirement: str, expected_cap: tuple[int, ...] | None
) -> None:
    assert _upper_cap(requirement) == expected_cap


def test_cap_detection_agrees_with_parsed_cap() -> None:
    assert _has_upper_cap("numpy>=2.0,<2.5") is True
    assert _has_upper_cap("numpy>=2.0") is False


def test_lower_bounds_are_not_mistaken_for_caps() -> None:
    """``>=`` must never read as an upper cap, or the guard would pass vacuously."""
    assert _upper_cap("numpy>=2.5") is None
    assert _has_upper_cap("numpy>=2.5") is False
