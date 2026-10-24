"""End-to-end checks for the determinism matrix engine and its CDS workload."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE = REPO_ROOT / "scripts" / "determinism_matrix.py"
WORKLOAD = REPO_ROOT / "scripts" / "determinism_workload.py"


def _run_engine(*args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ENGINE), *(str(arg) for arg in args)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
        check=False,
    )


def _manifest(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def test_matrix_output_is_identical_across_seeds_and_threads(tmp_path: Path) -> None:
    out = tmp_path / "manifest.json"
    completed = _run_engine(
        "run",
        "--workload",
        WORKLOAD,
        "--out",
        out,
        "--seeds",
        "0",
        "12345",
        "--threads",
        1,
        2,
        "--gate",
        "within-host",
    )
    assert completed.returncode == 0, completed.stderr
    data = _manifest(out)
    assert data["summary"]["identical"] is True
    assert data["summary"]["run_count"] == 4
    assert data["gate"] == {"mode": "within-host", "passed": True}
    assert all(run["exit_code"] == 0 for run in data["runs"])
    assert [item["path"] for item in data["runs"][0]["artifacts"]] == ["payload.json"]
    assert data["runs"][0]["artifacts"][0]["bytes"] > 0


def test_stdout_newlines_are_normalized_before_hashing(tmp_path: Path) -> None:
    crlf = tmp_path / "crlf.py"
    crlf.write_text("import sys\nsys.stdout.buffer.write(b'a\\r\\nb\\n')\n", encoding="utf-8")
    lf = tmp_path / "lf.py"
    lf.write_text("import sys\nsys.stdout.buffer.write(b'a\\nb\\n')\n", encoding="utf-8")
    crlf_out = tmp_path / "crlf.json"
    lf_out = tmp_path / "lf.json"
    for workload, out in ((crlf, crlf_out), (lf, lf_out)):
        completed = _run_engine(
            "run", "--workload", workload, "--out", out, "--seeds", "0", "--threads", 1
        )
        assert completed.returncode == 0, completed.stderr
    crlf_run = _manifest(crlf_out)["runs"][0]
    lf_run = _manifest(lf_out)["runs"][0]
    assert crlf_run["stdout_sha256"] == lf_run["stdout_sha256"]
    assert crlf_run["stdout_raw_sha256"] != lf_run["stdout_raw_sha256"]


def test_matrix_records_the_requested_thread_environment(tmp_path: Path) -> None:
    out = tmp_path / "probe.json"
    completed = _run_engine(
        "run",
        "--workload",
        WORKLOAD,
        "--out",
        out,
        "--seeds",
        "0",
        "--threads",
        1,
        "--gate",
        "within-host",
    )
    assert completed.returncode == 0, completed.stderr
    probe = _manifest(out)["thread_probe"]["1"]
    assert probe["requested_threads"] == 1
    assert probe["environment_matches_request"] is True
    assert probe["environment"]["OMP_NUM_THREADS"] == "1"


def test_matrix_gate_detects_a_hash_order_leak(tmp_path: Path) -> None:
    leaky = tmp_path / "leaky.py"
    leaky.write_text(
        "import json\nprint(json.dumps(list({'alpha', 'beta', 'gamma', 'delta', 'epsilon'})))\n",
        encoding="utf-8",
    )
    out = tmp_path / "leaky.json"
    completed = _run_engine(
        "run",
        "--workload",
        leaky,
        "--out",
        out,
        "--seeds",
        "0",
        "12345",
        "--threads",
        1,
        "--gate",
        "within-host",
    )
    assert completed.returncode == 3
    assert _manifest(out)["summary"]["identical"] is False


def test_matrix_reports_a_workload_error_separately(tmp_path: Path) -> None:
    broken = tmp_path / "broken.py"
    broken.write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    out = tmp_path / "broken.json"
    completed = _run_engine(
        "run", "--workload", broken, "--out", out, "--seeds", "0", "--threads", 1
    )
    assert completed.returncode == 4
    assert _manifest(out)["runs"][0]["exit_code"] == 1
