"""Retry behaviour of ``scripts/verify_release_integrity.py``.

Release run 40231188282 (v2.2.20) was a false red in this gate: the PyPI upload
had succeeded, but the JSON API still answered 404 when the twelve-attempt,
five-second budget ran out; a rerun minutes later verified the same release.
The tests below pin a budget wide enough for that propagation lag, and the
retry semantics around it -- a missing release is retried with a pause between
attempts, the final failed attempt does not sleep, and a policy violation is
not retried because waiting cannot change it.
"""

from __future__ import annotations

import importlib.util
import sys
import time
import urllib.error
from collections.abc import Callable
from email.message import Message
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_release_integrity.py"
_MODULE_NAME = "_verify_release_integrity_retry"

MINIMUM_BUDGET_SECONDS = 300.0
REPOSITORY = "Furox-Art/scientific-computing-system"
VERSION = "2.2.20"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, SCRIPT)
    assert spec is not None and spec.loader is not None, f"cannot import {SCRIPT}"
    module = importlib.util.module_from_spec(spec)
    # Registered before execution because the script uses @dataclass, which
    # resolves ``cls.__module__`` through sys.modules while the class body runs.
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _verifier(module: ModuleType) -> Callable[..., None]:
    verify: Callable[..., None] = getattr(module, "verify_release_integrity")
    return verify


def _not_found() -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        f"https://pypi.org/pypi/scientific-computing-system/{VERSION}/json",
        404,
        "Not Found",
        Message(),
        None,
    )


def _recording_sleep(delays: list[float]) -> Callable[[float], None]:
    def sleep(seconds: float) -> None:
        delays.append(seconds)

    return sleep


def _no_forbidden_assets(repository: str, tag: str, *, token: str | None = None) -> tuple[str, ...]:
    return ()


def test_the_default_budget_survives_a_five_minute_pypi_lag() -> None:
    module = _load()
    attempts = int(getattr(module, "DEFAULT_ATTEMPTS"))
    delay = float(getattr(module, "DEFAULT_DELAY_SECONDS"))
    assert attempts * delay >= MINIMUM_BUDGET_SECONDS, (attempts, delay)


def test_a_missing_pypi_release_is_retried_until_it_appears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load()
    seen: list[str] = []
    delays: list[float] = []

    def appears_on_third_attempt(version: str) -> dict[str, object]:
        seen.append(version)
        if len(seen) < 3:
            raise _not_found()
        return {}

    monkeypatch.setattr(module, "pypi_manifest", appears_on_third_attempt)
    monkeypatch.setattr(module, "github_distribution_assets", _no_forbidden_assets)
    monkeypatch.setattr(module.time, "sleep", _recording_sleep(delays))

    _verifier(module)(VERSION, REPOSITORY, attempts=4, delay_seconds=1.0)

    assert seen == [VERSION] * 3
    assert delays == [1.0, 1.0]


def test_the_budget_exhausts_without_sleeping_after_the_last_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load()
    seen: list[str] = []
    delays: list[float] = []

    def always_missing(version: str) -> dict[str, object]:
        seen.append(version)
        raise _not_found()

    monkeypatch.setattr(module, "pypi_manifest", always_missing)
    monkeypatch.setattr(module.time, "sleep", _recording_sleep(delays))

    with pytest.raises(RuntimeError, match="release integrity could not be verified"):
        _verifier(module)(VERSION, REPOSITORY, attempts=3, delay_seconds=1.0)

    assert seen == [VERSION] * 3
    assert delays == [1.0, 1.0]


def test_a_distribution_asset_fails_immediately_without_retrying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load()
    delays: list[float] = []

    def forbidden(repository: str, tag: str, *, token: str | None = None) -> tuple[str, ...]:
        return (f"scientific_computing_system-{VERSION}.tar.gz",)

    monkeypatch.setattr(module, "pypi_manifest", lambda version: {})
    monkeypatch.setattr(module, "github_distribution_assets", forbidden)
    monkeypatch.setattr(module.time, "sleep", _recording_sleep(delays))

    with pytest.raises(ValueError, match="must not contain wheel/sdist"):
        _verifier(module)(VERSION, REPOSITORY, attempts=3, delay_seconds=0.1)

    assert delays == []


def test_the_pause_never_runs_past_the_wall_clock_budget() -> None:
    """Requests can each take 30 s, so the wait must be capped by a deadline."""
    module = _load()
    pause: Callable[[float, float], float | None] = getattr(module, "_retry_pause")
    now = time.monotonic()

    assert pause(now + 30.0, 5.0) == 5.0
    partial = pause(now + 1.0, 5.0)
    assert partial is not None and 0 < partial <= 1.0
    assert pause(now - 1.0, 5.0) is None
