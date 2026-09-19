#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import time
import traceback
from pathlib import Path

import requests

REMOTE = "https://github.com/OpenNeuroDatasets/ds002278.git"
COMMIT = "ed03c47c368000e511401021c1d13c3fb916b470"
BASE = "https://s3.amazonaws.com/openneuro.org/ds002278/"
MANIFEST = Path(".public-runner/p08/raw_preprocessing_input_manifest_2026-09-18.csv")
ROOT = Path(".public-runner/p08/sensitivity")
CHUNK = 16 * 1024 * 1024
PARTICIPANTS = {"sub-Bubbles", "sub-Buttercup", "sub-PILOT02"}


def digest_file(path: Path, algo: str) -> str:
    h = hashlib.md5() if algo == "md5" else hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()


def download_exact(session: requests.Session, url: str, dest: Path, expected_bytes: int, algo: str, expected_hash: str) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    h = hashlib.md5() if algo == "md5" else hashlib.sha256()
    start = 0
    requests_n = 0
    hosts = set()
    with tmp.open("wb") as out:
        while start < expected_bytes:
            end = min(start + CHUNK - 1, expected_bytes - 1)
            response = None
            for attempt in range(1, 5):
                try:
                    response = session.get(
                        url,
                        headers={
                            "Range": f"bytes={start}-{end}",
                            "Accept-Encoding": "identity",
                            "User-Agent": "github-actions-p08-sensitivity/1.0",
                        },
                        timeout=(30, 240),
                        allow_redirects=True,
                    )
                    if response.status_code not in (200, 206):
                        raise RuntimeError(f"HTTP {response.status_code}; final={response.url}")
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(min(2 ** (attempt - 1), 8))
            assert response is not None
            requests_n += 1
            if response.url:
                hosts.add(requests.utils.urlparse(response.url).hostname)
            body = response.content
            if response.status_code == 206:
                need = end - start + 1
                if len(body) != need:
                    raise RuntimeError(f"short range for {url}: {len(body)} != {need}")
                cr = response.headers.get("Content-Range", "")
                if not cr.lower().startswith(f"bytes {start}-{end}/".lower()):
                    raise RuntimeError(f"bad Content-Range for {url}: {cr!r}")
                out.write(body)
                h.update(body)
                start = end + 1
            elif response.status_code == 200 and start == 0 and len(body) == expected_bytes:
                out.write(body)
                h.update(body)
                start = expected_bytes
            else:
                raise RuntimeError(
                    f"range ignored unexpectedly for {url}: status={response.status_code}, bytes={len(body)}"
                )
    got = h.hexdigest()
    if start != expected_bytes:
        raise RuntimeError(f"size mismatch for {dest}: {start} != {expected_bytes}")
    if got.lower() != expected_hash.lower():
        raise RuntimeError(f"hash mismatch for {dest}: {got} != {expected_hash}")
    tmp.replace(dest)
    return {
        "bytes": start,
        "hash_algorithm": algo,
        "hash": got,
        "requests": requests_n,
        "hosts": sorted(x for x in hosts if x),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", required=True, choices=sorted(PARTICIPANTS))
    ap.add_argument("--root", default=str(ROOT), help="Output root; defaults to the original sensitivity location.")
    ap.add_argument("--session", default=None, help="Optional BIDS session (e.g. ses-01) for outcome-blind recovery sharding.")
    ap.add_argument("--anatomy-only", action="store_true", help="Reconstruct only root metadata plus the participant frozen T1w pair.")
    args = ap.parse_args()

    participant = args.participant
    session_filter = args.session
    if session_filter is not None and not session_filter.startswith("ses-"):
        raise SystemExit("--session must be a BIDS session label such as ses-01")
    workroot = Path(args.root) / participant
    bids = workroot / "bids"
    source = workroot / "source_repo"
    result_path = workroot / "input_reconstruction.json"
    if workroot.exists():
        shutil.rmtree(workroot)
    bids.mkdir(parents=True)

    result = {
        "status": "STARTED",
        "participant": participant,
        "frozen_dataset": "OpenNeuro ds002278 v2.0.0",
        "frozen_commit": COMMIT,
        "manifest": str(MANIFEST),
        "files": [],
        "all_exact_verified": False,
        "neural_effect_computed": False,
    }

    try:
        with MANIFEST.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if args.anatomy_only:
            selected = []
            for r in rows:
                p = r["path"]
                if not p.startswith("sub-"):
                    selected.append(r)
                    continue
                if p.startswith(participant + "/") and ("_T1w.nii.gz" in p or "_T1w.json" in p):
                    selected.append(r)
            result["anatomy_only"] = True
        elif session_filter is None:
            selected = [r for r in rows if not r["path"].startswith("sub-") or r["path"].startswith(participant + "/")]
        else:
            session_prefix = f"{participant}/{session_filter}/"
            selected = []
            for r in rows:
                p = r["path"]
                if not p.startswith("sub-"):
                    selected.append(r)
                    continue
                if p.startswith(session_prefix):
                    selected.append(r)
                    continue
                # Keep the participant's single frozen T1w even when it lives in
                # another session; the session shard still requires the same
                # anatomical normalization as the full participant pipeline.
                if p.startswith(participant + "/") and ("_T1w.nii.gz" in p or "_T1w.json" in p):
                    selected.append(r)
            result["session_filter"] = session_filter
        if not selected:
            raise RuntimeError("no selected manifest rows")

        source.mkdir(parents=True)
        run("git", "init", cwd=source)
        run("git", "remote", "add", "origin", REMOTE, cwd=source)
        run("git", "fetch", "--depth", "1", "origin", COMMIT, cwd=source)
        run("git", "checkout", "--detach", "FETCH_HEAD", cwd=source)
        got_commit = run("git", "rev-parse", "HEAD", cwd=source)
        if got_commit != COMMIT:
            raise RuntimeError(f"source commit mismatch: {got_commit}")

        session = requests.Session()
        total_bytes = 0
        for i, row in enumerate(selected, 1):
            rel = Path(row["path"])
            dest = bids / rel
            expected_bytes = int(row["expected_bytes"])
            algo = row["hash_algorithm"]
            want = row["expected_hash"]
            t0 = time.time()
            if row["storage"] == "git_blob":
                src = source / rel
                data = src.read_bytes()
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                got = digest_file(dest, algo)
                if len(data) != expected_bytes or got.lower() != want.lower():
                    raise RuntimeError(f"git blob verification failed: {rel}")
                rec = {"bytes": len(data), "hash_algorithm": algo, "hash": got, "requests": 0, "hosts": []}
            elif row["storage"] == "git_annex":
                rec = download_exact(session, BASE + rel.as_posix(), dest, expected_bytes, algo, want)
            else:
                raise RuntimeError(f"unknown storage type {row['storage']!r}")

            total_bytes += rec["bytes"]
            rec.update({
                "path": rel.as_posix(),
                "role": row["role"],
                "storage": row["storage"],
                "elapsed_seconds": time.time() - t0,
            })
            result["files"].append(rec)
            print(f"[{i}/{len(selected)}] exact {rel} {rec['bytes']} B", flush=True)

        shutil.rmtree(source)
        result.update({
            "status": "P08_SENSITIVITY_BIDS_SUBSET_EXACT_RECONSTRUCTED",
            "selected_files": len(selected),
            "selected_bytes": total_bytes,
            "all_exact_verified": True,
            "bids_dir": str(bids),
        })
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: result[k] for k in ["status", "participant", "selected_files", "selected_bytes", "all_exact_verified"]}, indent=2))
        return 0
    except Exception as e:
        result.update({
            "status": "P08_SENSITIVITY_BIDS_RECONSTRUCTION_FAILED",
            "error": {"type": type(e).__name__, "message": str(e), "traceback": traceback.format_exc()},
        })
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
