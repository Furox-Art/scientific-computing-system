"""The GitHub API host check must be an exact hostname comparison.

CodeQL flagged ``scripts/verify_release_integrity.py`` with
``py/incomplete-url-substring-sanitization`` (high): the code decided whether a
URL was the GitHub API with ``"api.github.com" in url``. A substring test on a URL
is the wrong comparison, and this file pins the corrected behaviour in both
directions -- the hosts it used to wave through, and the legitimate hosts it used
to reject.

The alert is recorded against ``main``, so it will only clear once this fix
reaches the default branch.
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


def test_accepts_only_the_exact_github_api_host() -> None:
    check = _is_github_api()
    for url in EXACT_HOST:
        assert check(url) is True, f"should be recognised as the GitHub API: {url}"


def test_rejects_lookalike_hosts_and_non_absolute_urls() -> None:
    check = _is_github_api()
    for url in REJECTED:
        assert check(url) is False, f"must not be treated as the GitHub API: {url}"


def test_the_substring_form_would_have_been_wrong() -> None:
    """Pin *why* this exists: show the old comparison failing on both sides.

    Without this, the exactness of the hostname comparison looks like pedantry
    and a future edit back to `"api.github.com" in url` would pass review.
    """
    check = _is_github_api()

    # Too permissive: the string is present, the host is not GitHub.
    lookalike = "https://evil.example/?api.github.com"
    assert "api.github.com" in lookalike, "premise: the substring test matches"
    assert check(lookalike) is False, "the hostname test must not"

    # Too strict: a legitimate GitHub URL the substring test rejected.
    uppercase = "https://API.GITHUB.COM/repos/x/y"
    assert "api.github.com" not in uppercase, "premise: the substring test misses"
    assert check(uppercase) is True, "the hostname test must not"


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
