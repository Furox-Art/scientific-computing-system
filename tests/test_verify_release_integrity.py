"""The GitHub API host check must be an exact hostname comparison.

CodeQL flagged ``scripts/verify_release_integrity.py`` with
``py/incomplete-url-substring-sanitization`` (high): the code decided whether a
URL was the GitHub API by testing whether the host text occurred *anywhere* in
the URL. A containment test on a URL is the wrong comparison, and this file pins
the corrected behaviour in both directions -- the hosts it used to wave through,
and the legitimate hosts it used to reject.

This file contains no containment test of its own. An earlier version asserted
the old behaviour by performing the old containment check in the test body, which
made the test that defends the fix an instance of the defect it defends against
(alert #97, same rule, on this file). The reasoning is recorded as prose and the
verdicts are asserted against the shipped function; see
``DIVERGENT_CASES`` below.

The alert on ``scripts/verify_release_integrity.py`` is recorded against
``main``, so it clears only once that fix reached the default branch.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_release_integrity.py"
_MODULE_NAME = "_verify_release_integrity"

# `scripts/verify_release_integrity.py` imports `tomllib`, which is stdlib from
# Python 3.11. The script is a release gate that only ever runs on the release
# interpreter, so the suite skips it below that rather than the script growing a
# 3.10 compatibility shim. Same guard, and same reason, as
# `tests/test_codemeta_version_lockstep.py`.
SCRIPT_MIN_PYTHON = (3, 11)

needs_python_311 = pytest.mark.skipif(
    sys.version_info < SCRIPT_MIN_PYTHON,
    reason=(
        "scripts/verify_release_integrity.py imports tomllib, added in Python 3.11; "
        "the release-integrity job runs on a newer interpreter"
    ),
)


def _load() -> ModuleType:
    """Import the script by path.

    ``scripts/`` is not a package and is absent from the mypy search path, so a
    plain ``import verify_release_integrity`` is an ``import-not-found`` under
    ``mypy tests/``. The module is therefore loaded explicitly and typed as
    ``ModuleType``.

    It is registered in ``sys.modules`` before execution because the script uses
    ``@dataclass``, and ``dataclasses`` resolves ``cls.__module__`` through
    ``sys.modules`` while the class body runs -- without this, every dataclass in
    the script raises ``AttributeError: 'NoneType' object has no attribute
    '__dict__'``.
    """
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, SCRIPT)
    assert spec is not None, f"cannot build an import spec for {SCRIPT}"
    assert spec.loader is not None, f"no loader for {SCRIPT}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _is_github_api() -> Callable[[str], bool]:
    """Return the script's host predicate, typed for use in assertions."""
    predicate: Callable[[str], bool] = getattr(_load(), "_is_github_api")
    return predicate


# --- the exact host, and equivalent spellings of it ------------------------

EXACT_HOST = (
    "https://api.github.com/repos/Furox-Art/scientific-computing-system",
    "https://api.github.com/repos/Furox-Art/scientific-computing-system/releases/tags/v2.2.1",
    "http://api.github.com/repos/x/y",  # scheme is not part of the host test
    "https://API.GITHUB.COM/repos/x/y",  # hostnames are case-insensitive
    "https://user:token@api.github.com/repos/x/y",  # userinfo is not the host
    "https://api.github.com:443/repos/x/y",  # explicit default port
)

# --- everything the substring form got wrong, or anything else -------------

REJECTED = (
    # The two that made the substring test unsafe: the string appears, the host
    # does not.
    "https://evil.example/?api.github.com",
    "https://evil.example/api.github.com",
    "https://api.github.com.evil.example/x",
    # Suffix and prefix lookalikes.
    "https://notapi.github.com/repos/x/y",
    "https://api.github.com.co/repos/x/y",
    "https://myapi.github.com.attacker.test/x",
    "https://github.com/repos/x/y",
    "https://raw.githubusercontent.com/x/y",
    "https://pypi.org/pypi/scientific-computing-system/json",
    # Credentials in the authority section: the real host is the part after `@`.
    "https://api.github.com@evil.example/x",
    # A trailing dot resolves to the same host, but nothing here builds one, so
    # the comparison stays exact and fails closed.
    "https://api.github.com./repos/x/y",
    # Not absolute URLs at all.
    "not-a-url",
    "/repos/x/y",
    "",
)


# --- the inputs where a containment check and a hostname check disagree ------
#
# These are the only two inputs in the tables above where the two approaches
# reach opposite conclusions, and they are what a future reader needs in order to
# understand why the comparison is a hostname comparison. The third element is
# prose, used only to explain a failure.
#
# The expected verdicts are asserted against the real predicate. This file
# deliberately contains **no** containment test over a URL: the previous version
# asserted the old behaviour by writing the old expression, which is the very
# pattern CodeQL's py/incomplete-url-substring-sanitization flags -- so the test
# defending the fix became an instance of the defect. The reasoning belongs in
# prose; the verdicts belong in assertions on the shipped function.

DIVERGENT_CASES = (
    (
        "https://evil.example/?api.github.com",
        False,
        "a containment check would match the host text in the query string and "
        "wrongly treat this as the GitHub API -- the too-permissive direction",
    ),
    (
        "https://API.GITHUB.COM/repos/x/y",
        True,
        "hostnames are case-insensitive, so this is the GitHub API; a containment "
        "check would miss it and wrongly treat it as some other host -- the "
        "too-strict direction",
    ),
)


@needs_python_311
def test_accepts_only_the_exact_github_api_host() -> None:
    check = _is_github_api()
    for url in EXACT_HOST:
        assert check(url) is True, f"should be recognised as the GitHub API: {url}"


@needs_python_311
def test_rejects_lookalike_hosts_and_non_absolute_urls() -> None:
    check = _is_github_api()
    for url in REJECTED:
        assert check(url) is False, f"must not be treated as the GitHub API: {url}"


@needs_python_311
def test_the_two_directions_a_containment_check_gets_wrong() -> None:
    """Pin *why* this exists, by asserting the shipped function on both.

    Each case below is one where checking whether the host text merely appears
    somewhere in the URL disagrees with checking the URL's hostname. Asserting the
    correct verdict on the real predicate catches a regression in either
    direction: an edit back to a containment test flips the first case to True and
    the second to False, and this test fails on both.

    The previous version of this test asserted the *premise* -- that a
    containment check would have decided differently -- by performing that
    containment check in the test body. That is the pattern this whole change
    exists to remove, so asserting it here reintroduced the defect inside the file
    meant to be free of it. The premise is now recorded in the prose above and in
    each case's explanation, which a reader can check by inspection.
    """
    check = _is_github_api()
    for url, expected, why in DIVERGENT_CASES:
        assert check(url) is expected, f"{url}: {why}"


@needs_python_311
def test_read_json_selects_the_accept_header_by_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The decision the check exists for must actually flow from it.

    Behavioural rather than a source grep: an earlier version of this test
    asserted on the file's text and passed only by accident, because the
    explanation lives in a docstring that quotes the old expression.
    """
    module = _load()
    captured: list[str] = []

    class _Response:
        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def read(self) -> bytes:
            return b"{}"

    def fake_urlopen(request: object, timeout: int = 0) -> _Response:
        captured.append(request.headers.get("Accept", ""))  # type: ignore[attr-defined]
        return _Response()

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)

    module._read_json("https://api.github.com/repos/x/y")
    module._read_json("https://evil.example/?api.github.com")
    module._read_json("https://api.github.com.evil.example/x")

    assert captured == [
        "application/vnd.github+json",
        "application/json",
        "application/json",
    ], captured
