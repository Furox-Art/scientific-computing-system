#!/usr/bin/env python3
"""Confirm a just-published npm version has become visible on the registry.

Why this exists
---------------

``npm publish`` returns as soon as the registry accepts the upload. The version
is then propagated to the read path asynchronously, and for roughly a minute to a
few minutes afterwards a read of the *same* version still answers 404. This
repository's post-publish check did a single immediate read and turned that
window into a red job:

    npm error code E404
    npm error 404 No match found for version 2.2.0

...about ninety seconds before the registry recorded the upload. The publish had
succeeded; the check reported it as a failure. A maintainer reading that run has
every reason to believe the release broke. Three sibling repositories hit the
same false failure.

So this polls with backoff, succeeds on the first match, and still fails closed
if the version never appears.

Why not ``npm view``
-------------------

``npm view`` was the original implementation and it is a poor fit for a
post-publish probe:

* **It conflates "absent" with "denied".** A missing version, a missing package
  and an authorisation problem all surface as E404, so the check cannot tell
  propagation lag from a credential problem.
* **It drags credentials into the probe.** The failing run's log shows
  ``npm view`` resolving and emitting ``npm warn Unknown user config
  "always-auth"``, i.e. it read the userconfig this workflow had just created
  for publishing. A verification read has no need for write credentials.
* **Its failure text is unhelpful for triage.** The E404 block is a stack of
  install suggestions, none of which apply to "it was just published".

The registry's own JSON API answers the question unambiguously and with a
payload that can be asserted on: ``GET /{package}`` returns every published
version. A version absent from ``versions`` means not-yet-visible, full stop.

Exit codes
----------

0  the version is present in the registry (first match, immediately)
1  the deadline passed without the version appearing; a ``::warning::``
   annotation is emitted first so a slow-but-successful publish is never
   silently swallowed
2  the invocation itself was wrong (bad arguments, unparseable registry payload)

Usage
-----

    python scripts/verify_npm_publication.py --version 2.2.0

The attempt count and delays default to a budget of several minutes, chosen to
cover npm's documented CDN propagation rather than to make a test fast. Tests
override them with small values and point ``--registry`` at a local stub.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

# Defaults are deliberately generous. npm publishes to a write path and serves
# reads from a CDN; the gap is usually seconds but is not bounded by npm, and
# under load has been observed in the minutes. Anything shorter converts a
# successful publish into a false failure, which is the failure mode this script
# exists to remove.
DEFAULT_ATTEMPTS = 12
DEFAULT_INITIAL_DELAY = 5.0
DEFAULT_GROWTH = 1.5
DEFAULT_MAX_DELAY = 30.0
DEFAULT_TIMEOUT = 30.0
DEFAULT_REGISTRY = "https://registry.npmjs.org"


class VerificationError(Exception):
    """The registry answered, but not with something we can judge."""


def package_url(registry: str, package: str) -> str:
    """Registry read URL for a package, with scoped names encoded correctly."""
    encoded = urllib.parse.quote(package, safe="")
    return f"{registry.rstrip('/')}/{encoded}"


def fetch_versions(url: str, timeout: float) -> dict[str, object] | None:
    """Return the registry's version map, or None if the package is not visible yet.

    None means "ask again later": a 404 (package or version not yet served) and a
    transport-level error are both transient from this script's point of view.
    A 200 that is not the expected JSON shape raises instead, because that is a
    real problem worth failing on rather than retrying.
    """
    request = urllib.request.Request(
        url,
        headers={"accept": "application/json", "user-agent": "cds-npm-publish-verify"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                return None
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        if error.code in (403, 404, 408, 429, 500, 502, 503, 504):
            return None
        raise VerificationError(f"unexpected HTTP {error.code} from {url}") from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        # Transport hiccup: indistinguishable from "not visible yet" for our
        # purposes, and retrying is the right response.
        print(f"    registry unreachable ({error}); will retry", flush=True)
        return None
    except json.JSONDecodeError as error:
        raise VerificationError(f"registry returned non-JSON for {url}: {error}") from error

    versions = payload.get("versions") if isinstance(payload, dict) else None
    if not isinstance(versions, dict):
        raise VerificationError(f"registry payload for {url} has no `versions` object")
    return versions


def sleep_for(attempt: int, initial: float, growth: float, maximum: float) -> float:
    """Backoff before ``attempt`` (1-based), capped at ``maximum``."""
    if attempt <= 1:
        return 0.0
    delay = initial * (growth ** (attempt - 2))
    return min(delay, maximum)


def verify(
    version: str,
    *,
    package: str,
    registry: str,
    attempts: int,
    initial_delay: float,
    growth: float,
    max_delay: float,
    timeout: float,
) -> int:
    """Poll until ``version`` is visible. Returns a process exit code."""
    url = package_url(registry, package)
    print(f"Verifying {package}@{version} is visible on {registry}")

    for attempt in range(1, attempts + 1):
        delay = sleep_for(attempt, initial_delay, growth, max_delay)
        if delay:
            print(f"    attempt {attempt}/{attempts}: sleeping {delay:.0f}s", flush=True)
            time.sleep(delay)

        versions = fetch_versions(url, timeout)
        if versions is None:
            print(
                f"    attempt {attempt}/{attempts}: {package} not served by the registry yet",
                flush=True,
            )
            continue
        if version in versions:
            print(f"    attempt {attempt}/{attempts}: {package}@{version} is published")
            print(f"PASS: the registry serves {package}@{version}")
            return 0
        served = sorted(versions)
        print(
            f"    attempt {attempt}/{attempts}: {package}@{version} not visible; "
            f"registry serves {served or 'no versions'}",
            flush=True,
        )

    # The upload succeeded -- this step only runs after `npm publish` returned 0 --
    # so a deadline miss is almost certainly propagation lag rather than a failed
    # release. Say so as a warning, then still fail: a version that truly never
    # appeared must not be reported as success.
    print(
        "::warning::The npm upload succeeded but "
        f"{package}@{version} was still not visible after {attempts} attempts "
        f"(~{budget_seconds(attempts, initial_delay, growth, max_delay):.0f}s). "
        "This is normally registry propagation delay, not a failed publish. "
        "Check https://www.npmjs.com/package/"
        f"{package}?activeTab=versions before retrying."
    )
    print(f"FAIL: {package}@{version} never appeared on {registry}", file=sys.stderr)
    return 1


def budget_seconds(attempts: int, initial: float, growth: float, maximum: float) -> float:
    """Total planned sleep for ``attempts`` attempts (excluding request time)."""
    return sum(sleep_for(a, initial, growth, maximum) for a in range(1, attempts + 1))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="exact version to confirm")
    parser.add_argument("--package", default="scientific-computing-system")
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS)
    parser.add_argument("--initial-delay", type=float, default=DEFAULT_INITIAL_DELAY)
    parser.add_argument("--growth", type=float, default=DEFAULT_GROWTH)
    parser.add_argument("--max-delay", type=float, default=DEFAULT_MAX_DELAY)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    args = parser.parse_args(argv)

    if args.attempts < 1:
        print("ERROR: --attempts must be at least 1", file=sys.stderr)
        return 2
    if args.initial_delay < 0 or args.max_delay < 0 or args.growth < 1:
        print("ERROR: delays must be non-negative and growth at least 1", file=sys.stderr)
        return 2

    budget = budget_seconds(args.attempts, args.initial_delay, args.growth, args.max_delay)
    print(
        f"Budget: {args.attempts} attempts, ~{budget:.0f}s of planned backoff "
        f"(plus up to {args.timeout:.0f}s per request)"
    )

    try:
        return verify(
            args.version,
            package=args.package,
            registry=args.registry,
            attempts=args.attempts,
            initial_delay=args.initial_delay,
            growth=args.growth,
            max_delay=args.max_delay,
            timeout=args.timeout,
        )
    except VerificationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
