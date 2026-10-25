"""Run a workload under hash-seed/thread/locale/timezone variants and compare output digests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

TOOL_VERSION = "0.3.0"
SCHEMA_VERSION = 1
EXIT_OK = 0
EXIT_USAGE = 2
EXIT_GATE_FAILED = 3
EXIT_WORKLOAD_ERROR = 4
TIMEOUT_EXIT_CODE = 124
INHERIT = "inherit"
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
LOCALE_VARIABLES = ("LC_ALL", "LANG", "LANGUAGE")
THREAD_PROBE = r"""
import importlib.util
import json
import os

variables = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
report = {
    "environment": {name: os.environ.get(name) for name in variables},
    "threadpoolctl_available": importlib.util.find_spec("threadpoolctl") is not None,
}
if report["threadpoolctl_available"]:
    import threadpoolctl

    report["threadpools"] = [
        {
            "library": info.get("user_api"),
            "num_threads": info.get("num_threads"),
            "internal_name": info.get("internal_name"),
        }
        for info in threadpoolctl.threadpool_info()
    ]
print(json.dumps(report, sort_keys=True))
"""
ENVIRONMENT_PROBE = r"""
import json
import locale
import os
import time

report = {
    "environment": {
        name: os.environ.get(name)
        for name in ("LC_ALL", "LANG", "LANGUAGE", "TZ")
    },
    "locale": locale.setlocale(locale.LC_ALL, None),
    "tzname": list(time.tzname),
    "timezone": time.timezone,
}
print(json.dumps(report, sort_keys=True))
"""


def host_metadata() -> dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "cpu_count": os.cpu_count(),
    }


def variant_key(run: dict[str, Any]) -> str:
    return (
        f"seed={run['hash_seed']}|threads={run['threads']}"
        f"|locale={run.get('locale', INHERIT)}"
        f"|timezone={run.get('timezone', INHERIT)}"
        f"|repeat={run['repeat']}"
    )


def comparison_digest(run: dict[str, Any]) -> str:
    material = f"{run['stdout_sha256']}\n{run.get('artifacts_sha256', '')}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def normalized_bytes_digest(payload: bytes) -> str:
    normalized = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def hash_artifacts(root: Path) -> list[dict[str, Any]]:
    records = []
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    for path in files:
        relative = path.relative_to(root).as_posix()
        payload = path.read_bytes()
        normalized = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        records.append(
            {
                "path": relative,
                "bytes": len(normalized),
                "sha256": hashlib.sha256(normalized).hexdigest(),
                "raw_bytes": len(payload),
                "raw_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return records


def artifacts_digest(artifacts: list[dict[str, Any]]) -> str:
    canonical = json.dumps(
        [
            {"path": artifact["path"], "bytes": artifact["bytes"], "sha256": artifact["sha256"]}
            for artifact in artifacts
        ],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def variant_environment(
    seed: str,
    threads: int,
    locale: str = INHERIT,
    timezone: str = INHERIT,
) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = seed
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for variable in THREAD_VARIABLES:
        env[variable] = str(threads)
    for variable in LOCALE_VARIABLES:
        env.pop(variable, None)
    if locale != INHERIT:
        for variable in LOCALE_VARIABLES:
            env[variable] = locale
    env.pop("TZ", None)
    if timezone != INHERIT:
        env["TZ"] = timezone
    return env


def probe_threads(threads: int, timeout: float) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [sys.executable, "-c", THREAD_PROBE],
            env=variant_environment("0", threads),
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"requested_threads": threads, "error": "thread probe timed out"}
    if completed.returncode != 0:
        return {
            "requested_threads": threads,
            "error": completed.stderr.decode("utf-8", "replace")[-500:],
        }
    report: dict[str, Any] = json.loads(completed.stdout.decode("utf-8"))
    report["requested_threads"] = threads
    report["environment_matches_request"] = all(
        report["environment"][name] == str(threads) for name in THREAD_VARIABLES
    )
    return report


def probe_environment(locale: str, timezone: str, timeout: float) -> dict[str, Any]:
    requested = {"requested_locale": locale, "requested_timezone": timezone}
    try:
        completed = subprocess.run(
            [sys.executable, "-c", ENVIRONMENT_PROBE],
            env=variant_environment("0", 1, locale, timezone),
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {**requested, "error": "environment probe timed out"}
    if completed.returncode != 0:
        return {
            **requested,
            "error": completed.stderr.decode("utf-8", "replace")[-500:],
        }
    report: dict[str, Any] = json.loads(completed.stdout.decode("utf-8"))
    report.update(requested)
    report["environment_matches_request"] = all(
        report["environment"][name] == (None if locale == INHERIT else locale)
        for name in LOCALE_VARIABLES
    ) and report["environment"]["TZ"] == (None if timezone == INHERIT else timezone)
    return report


def run_variant(
    workload: Path,
    workload_args: list[str],
    seed: str,
    threads: int,
    repeat: int,
    timeout: float,
    cwd: Path | None,
    locale: str,
    timezone: str,
) -> dict[str, Any]:
    command = [sys.executable, str(workload), *workload_args]
    with tempfile.TemporaryDirectory(prefix="determinism-artifacts-") as artifact_dir:
        env = variant_environment(seed, threads, locale, timezone)
        env["DETERMINISM_ARTIFACT_DIR"] = artifact_dir
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                cwd=str(cwd) if cwd else None,
                env=env,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
            stdout, stderr = completed.stdout, completed.stderr
            exit_code, timed_out = completed.returncode, False
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or b""
            stderr = exc.stderr or b""
            exit_code, timed_out = TIMEOUT_EXIT_CODE, True
        artifacts = hash_artifacts(Path(artifact_dir))
    record: dict[str, Any] = {
        "hash_seed": seed,
        "threads": threads,
        "locale": locale,
        "timezone": timezone,
        "repeat": repeat,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "duration_seconds": round(time.perf_counter() - started, 3),
        "stdout_bytes": len(stdout),
        "stdout_sha256": normalized_bytes_digest(stdout),
        "stdout_raw_sha256": hashlib.sha256(stdout).hexdigest(),
        "artifacts": artifacts,
        "artifacts_sha256": artifacts_digest(artifacts),
    }
    if exit_code != 0:
        record["stderr_tail"] = stderr.decode("utf-8", "replace")[-2000:]
    return record


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    digests: dict[str, list[str]] = {}
    for run in runs:
        digests.setdefault(comparison_digest(run), []).append(variant_key(run))
    return {
        "run_count": len(runs),
        "distinct_digests": len(digests),
        "identical": len(digests) == 1,
        "digest_groups": [
            {"comparison_sha256": digest, "variants": sorted(variants)}
            for digest, variants in sorted(digests.items())
        ],
    }


def command_run(args: argparse.Namespace) -> int:
    workload = Path(args.workload).resolve()
    if not workload.is_file():
        print(f"workload not found: {workload}", file=sys.stderr)
        return EXIT_USAGE
    cwd = Path(args.cwd).resolve() if args.cwd else None
    runs = []
    for seed in args.seeds:
        for threads in args.threads:
            for locale in args.locales:
                for timezone in args.timezones:
                    for repeat in range(args.repeats):
                        record = run_variant(
                            workload,
                            args.workload_args,
                            seed,
                            threads,
                            repeat,
                            args.timeout,
                            cwd,
                            locale,
                            timezone,
                        )
                        runs.append(record)
                        print(
                            f"seed={seed:<6} threads={threads} locale={locale:<8} "
                            f"tz={timezone:<16} repeat={repeat} "
                            f"exit={record['exit_code']} "
                            f"sha256={record['stdout_sha256'][:16]}",
                            flush=True,
                        )
    failures = [run for run in runs if run["exit_code"] != 0]
    summary = summarize(runs)
    thread_probes = {
        str(threads): probe_threads(threads, args.timeout) for threads in sorted(set(args.threads))
    }
    environment_probes = {
        f"locale={locale}|tz={timezone}": probe_environment(locale, timezone, args.timeout)
        for locale in args.locales
        for timezone in args.timezones
    }
    probe_failures = [
        f"threads={threads}"
        for threads, probe in thread_probes.items()
        if probe.get("error") or not probe.get("environment_matches_request")
    ] + [key for key, probe in environment_probes.items() if probe.get("error")]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "audit": "determinism-matrix",
        "tool_version": TOOL_VERSION,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "digest_normalization": (
            "stdout, workload source, and artifact bytes newlines normalized to LF; "
            "artifact paths sorted by POSIX relative path"
        ),
        "host": host_metadata(),
        "workload": {
            "path": str(workload),
            "args": args.workload_args,
            "cwd": str(cwd) if cwd else None,
            # Checkouts on Windows may use CRLF; normalize so identical
            # sources compare equal across hosts.
            "sha256": normalized_bytes_digest(workload.read_bytes()),
        },
        "variants": {
            "hash_seeds": args.seeds,
            "thread_counts": args.threads,
            "locales": args.locales,
            "timezones": args.timezones,
            "repeats": args.repeats,
            "thread_variables": list(THREAD_VARIABLES),
            "locale_variables": list(LOCALE_VARIABLES),
        },
        "thread_probe": thread_probes,
        "environment_probe": environment_probes,
        "runs": runs,
        "summary": summary,
        "gate": {
            "mode": args.gate,
            "passed": summary["identical"] and not failures and not probe_failures,
            "probe_failures": probe_failures,
        },
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"runs={summary['run_count']} distinct_digests={summary['distinct_digests']} "
        f"manifest={output}"
    )
    if failures:
        print(f"{len(failures)} workload run(s) exited non-zero", file=sys.stderr)
        return EXIT_WORKLOAD_ERROR
    if probe_failures:
        print(f"environment probe failed: {', '.join(probe_failures)}", file=sys.stderr)
        return EXIT_WORKLOAD_ERROR
    if args.gate == "within-host" and not summary["identical"]:
        print("gate failed: output digests differ across variants", file=sys.stderr)
        return EXIT_GATE_FAILED
    return EXIT_OK


def workload_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    workload = manifest.get("workload", {})
    return {
        "sha256": workload.get("sha256"),
        "args": workload.get("args"),
        "cwd": workload.get("cwd"),
    }


def manifest_errors(manifest: dict[str, Any], label: str) -> list[str]:
    errors = []
    if manifest.get("audit") != "determinism-matrix":
        errors.append(f"{label}: unsupported audit {manifest.get('audit')!r}")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"{label}: schema_version={manifest.get('schema_version')!r}, expected {SCHEMA_VERSION}"
        )
    runs = manifest.get("runs")
    if not isinstance(runs, list) or not runs:
        errors.append(f"{label}: no runs recorded")
        return errors
    for run in runs:
        if run.get("exit_code") != 0 or run.get("timed_out"):
            errors.append(f"{label}: variant {variant_key(run)} exited {run.get('exit_code')}")
    return errors


def artifact_digests(run: dict[str, Any]) -> dict[str, str]:
    return {
        artifact["path"]: artifact["sha256"]
        for artifact in run.get("artifacts", [])
        if "path" in artifact and "sha256" in artifact
    }


def mismatch_details(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    left = artifact_digests(baseline)
    right = artifact_digests(candidate)
    paths = sorted(set(left) | set(right))
    changed = [path for path in paths if left.get(path) != right.get(path)]
    return {
        "stdout_mismatch": baseline.get("stdout_sha256") != candidate.get("stdout_sha256"),
        "artifacts_mismatch": bool(changed),
        "artifact_paths": changed,
    }


def render_report(report: dict[str, Any], out: str | None) -> None:
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if out:
        Path(out).write_text(rendered, encoding="utf-8")
    print(rendered, end="")


def command_compare(args: argparse.Namespace) -> int:
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    errors = manifest_errors(baseline, "baseline") + manifest_errors(candidate, "candidate")
    if not errors and workload_identity(baseline) != workload_identity(candidate):
        errors.append("manifests describe different workloads or working directories")
    if errors:
        report = {
            "audit": "determinism-matrix-compare",
            "schema_version": SCHEMA_VERSION,
            "comparable": False,
            "errors": errors,
            "baseline_host": baseline.get("host"),
            "candidate_host": candidate.get("host"),
            "baseline_tool_version": baseline.get("tool_version"),
            "candidate_tool_version": candidate.get("tool_version"),
            "shared_variants": 0,
            "baseline_only_variants": [],
            "candidate_only_variants": [],
            "mismatches": [],
            "identical": False,
        }
        render_report(report, args.out)
        for error in errors:
            print(error, file=sys.stderr)
        return EXIT_USAGE
    baseline_runs = {variant_key(run): run for run in baseline["runs"]}
    candidate_runs = {variant_key(run): run for run in candidate["runs"]}
    baseline_digests = {key: comparison_digest(run) for key, run in baseline_runs.items()}
    candidate_digests = {key: comparison_digest(run) for key, run in candidate_runs.items()}
    shared = sorted(set(baseline_digests) & set(candidate_digests))
    mismatches = [
        {
            "variant": key,
            "baseline": baseline_digests[key],
            "candidate": candidate_digests[key],
            **mismatch_details(baseline_runs[key], candidate_runs[key]),
        }
        for key in shared
        if baseline_digests[key] != candidate_digests[key]
    ]
    baseline_only = sorted(set(baseline_digests) - set(candidate_digests))
    candidate_only = sorted(set(candidate_digests) - set(baseline_digests))
    identical = bool(shared) and not mismatches and not baseline_only and not candidate_only
    report = {
        "audit": "determinism-matrix-compare",
        "schema_version": SCHEMA_VERSION,
        "comparable": True,
        "errors": [],
        "baseline_host": baseline.get("host"),
        "candidate_host": candidate.get("host"),
        "baseline_tool_version": baseline.get("tool_version"),
        "candidate_tool_version": candidate.get("tool_version"),
        "shared_variants": len(shared),
        "baseline_only_variants": baseline_only,
        "candidate_only_variants": candidate_only,
        "mismatches": mismatches,
        "identical": identical,
    }
    render_report(report, args.out)
    if not shared:
        print("no shared variants to compare", file=sys.stderr)
        return EXIT_USAGE
    if args.gate and not identical:
        return EXIT_GATE_FAILED
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="execute the variant matrix")
    run_parser.add_argument("--workload", required=True)
    run_parser.add_argument("--out", required=True)
    run_parser.add_argument("--seeds", nargs="+", default=["0", "12345", "random"])
    run_parser.add_argument("--threads", nargs="+", type=int, default=[1, 4])
    run_parser.add_argument("--locales", nargs="+", default=[INHERIT])
    run_parser.add_argument("--timezones", nargs="+", default=[INHERIT])
    run_parser.add_argument("--repeats", type=int, default=1)
    run_parser.add_argument("--timeout", type=float, default=300.0)
    run_parser.add_argument("--cwd")
    run_parser.add_argument("--gate", choices=["none", "within-host"], default="none")
    run_parser.add_argument("--workload-args", nargs="*", default=[])
    run_parser.set_defaults(handler=command_run)
    compare_parser = subparsers.add_parser("compare", help="compare two manifests")
    compare_parser.add_argument("baseline")
    compare_parser.add_argument("candidate")
    compare_parser.add_argument("--out")
    compare_parser.add_argument("--gate", action="store_true")
    compare_parser.set_defaults(handler=command_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = args.handler
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
