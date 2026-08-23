#!/usr/bin/env python3
"""Clean up stale ``pypi`` deployment records on GitHub.

Usage:
    python scripts/cleanup_deployments.py                  # dry-run (list only)
    python scripts/cleanup_deployments.py --apply           # actually delete
    python scripts/cleanup_deployments.py --apply --env pypi

Why this exists: every ``v*`` tag push used to make GitHub auto-create a
``pypi`` deployment record (via ``environment: pypi`` in release.yml), but
nothing ever closed it with a ``success``/``failure`` status. Those orphaned
records pile up on the repo's Deployments page and show as "failed" once
GitHub times them out. release.yml now creates + statuses deployments
explicitly, so new releases produce exactly one green record — but the 38
historical ``pypi`` entries need to be cleared once.

This script lists (dry-run) or deletes those records. It touches ONLY the
``--env`` environment (default ``pypi``); ``github-pages`` and any other
environment are left alone. Deletion requires marking the deployment
``inactive`` first (GitHub rejects DELETE on a deployment whose latest
status is still ``in_progress``/``queued`` for protected environments).

Requires the ``gh`` CLI, authenticated with repo access:
    gh auth status
    gh auth login   # if not logged in
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any

DEFAULT_REPO = "Furox-Art/scientific-computing-system"
PYPI_URL = "https://pypi.org/project/scientific-computing-system/"

# Resolved at runtime from --repo; defaults to DEFAULT_REPO.
REPO = DEFAULT_REPO


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a subprocess, return completed process. Raise on non-zero if check."""
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def gh(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a ``gh`` CLI command, returning the completed process."""
    return run(["gh", *args], check=check)


def gh_api(
    method: str, endpoint: str, *, fields: list[str] | None = None, check: bool = True
) -> Any:
    """Call ``gh api`` with the given HTTP method and endpoint."""
    cmd = ["gh", "api", "--method", method, endpoint]
    if fields:
        for f in fields:
            cmd += ["-f", f]
    cp = run(cmd, check=check)
    if not cp.stdout.strip():
        return None
    try:
        return json.loads(cp.stdout)
    except json.JSONDecodeError:
        return cp.stdout


def list_deployments(env: str) -> list[dict[str, Any]]:
    """Return all deployments for ``env`` (paginated)."""
    out = gh(
        "api",
        f"repos/{REPO}/deployments",
        "--paginate",
        "-q",
        f'[.[] | select(.environment == "{env}")]',
    )
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError as exc:
        sys.exit(f"Failed to parse deployments list: {exc}\n{out.stdout[:500]}")


def latest_status(deployment_id: int) -> str:
    """Return the latest status state for a deployment, or 'none'."""
    cp = gh(
        "api", f"repos/{REPO}/deployments/{deployment_id}/statuses", "-q", ".[0].state", check=False
    )
    state = cp.stdout.strip()
    return state if state else "none"


def mark_inactive(deployment_id: int) -> bool:
    """POST an ``inactive`` status so the deployment can be deleted."""
    res = gh_api(
        "POST",
        f"repos/{REPO}/deployments/{deployment_id}/statuses",
        fields=["state=inactive"],
        check=False,
    )
    return isinstance(res, dict)


def delete_deployment(deployment_id: int) -> bool:
    """DELETE a deployment by id. Returns True on success."""
    cp = gh("api", "--method", "DELETE", f"repos/{REPO}/deployments/{deployment_id}", check=False)
    return cp.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Delete stale pypi deployment records from GitHub."
    )
    parser.add_argument("--env", default="pypi", help="environment to clean (default: pypi)")
    parser.add_argument(
        "--apply", action="store_true", help="actually delete (default: dry-run, list only)"
    )
    parser.add_argument(
        "--repo", default=DEFAULT_REPO, help=f"owner/repo (default: {DEFAULT_REPO})"
    )
    args = parser.parse_args()

    global REPO
    REPO = args.repo

    # Sanity: gh must be authenticated.
    auth = gh("auth", "status", check=False)
    if auth.returncode != 0:
        sys.exit("gh CLI is not authenticated. Run `gh auth login` first.")

    print(f"Listing deployments for environment '{args.env}' in {REPO} ...")
    deployments = list_deployments(args.env)
    if not deployments:
        print(f"No '{args.env}' deployments found. Nothing to do.")
        return 0

    print(f"Found {len(deployments)} '{args.env}' deployment(s):\n")
    print(f"{'ID':<14}{'REF':<14}{'CREATED':<22}{'LAST STATUS':<14}")
    print("-" * 64)
    for d in deployments:
        state = latest_status(d["id"])
        created = d.get("created_at", "?")[:19]
        ref = d.get("ref", "?")
        print(f"{d['id']:<14}{ref:<14}{created:<22}{state:<14}")

    if not args.apply:
        print(f"\nDRY RUN: {len(deployments)} deployment(s) would be deleted.")
        print("Re-run with --apply to actually delete them.")
        return 0

    print(f"\nApplying: deleting {len(deployments)} deployment(s) ...")
    deleted = 0
    for d in deployments:
        dep_id = d["id"]
        ref = d.get("ref", "?")
        # Mark inactive first (idempotent — ignore failure, attempt delete anyway).
        mark_inactive(dep_id)
        if delete_deployment(dep_id):
            print(f"  [deleted] #{dep_id} ({ref})")
            deleted += 1
        else:
            print(f"  [FAILED]  #{dep_id} ({ref})")

    print(f"\nDone: {deleted}/{len(deployments)} deleted.")
    # Verify.
    remaining = list_deployments(args.env)
    print(f"Remaining '{args.env}' deployments: {len(remaining)}")
    return 0 if deleted == len(deployments) else 1


if __name__ == "__main__":
    raise SystemExit(main())
