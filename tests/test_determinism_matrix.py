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


def _run_with_marker(tmp_path: Path, marker: str, name: str, *extra: object) -> tuple[Path, Path]:
    workload = tmp_path / f"{name}_workload.py"
    workload.write_text(
        "from pathlib import Path\n"
        "marker = Path(__file__).with_name('marker.txt')\n"
        "print(marker.read_text(encoding='utf-8'))\n",
        encoding="utf-8",
    )
    (tmp_path / "marker.txt").write_text(marker, encoding="utf-8")
    out = tmp_path / f"{name}.json"
    completed = _run_engine(
        "run",
        "--workload",
        workload,
        "--out",
        out,
        "--seeds",
        "0",
        "--threads",
        1,
        *extra,
    )
    assert completed.returncode == 0, completed.stderr
    return workload, out


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
    assert data["schema_version"] == 1
    assert data["summary"]["identical"] is True
    assert data["summary"]["run_count"] == 4
    assert data["gate"] == {"mode": "within-host", "passed": True, "probe_failures": []}
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


def test_matrix_records_locale_and_timezone_variants(tmp_path: Path) -> None:
    probe = tmp_path / "environment_report.py"
    probe.write_text(
        "import json\nimport os\n"
        "keys = ('LC_ALL', 'LANG', 'LANGUAGE', 'TZ')\n"
        "print(json.dumps({key: os.environ.get(key) for key in keys}))\n",
        encoding="utf-8",
    )
    out = tmp_path / "environment.json"
    completed = _run_engine(
        "run",
        "--workload",
        probe,
        "--out",
        out,
        "--seeds",
        "0",
        "--threads",
        1,
        "--locales",
        "inherit",
        "C",
        "--timezones",
        "UTC",
        "Europe/Istanbul",
    )
    assert completed.returncode == 0, completed.stderr
    data = _manifest(out)
    assert data["variants"]["locales"] == ["inherit", "C"]
    assert data["variants"]["timezones"] == ["UTC", "Europe/Istanbul"]
    assert data["summary"]["run_count"] == 4
    assert data["summary"]["identical"] is False
    runs = {(run["locale"], run["timezone"]): run for run in data["runs"]}
    assert runs[("C", "UTC")]["stdout_sha256"] != runs[("inherit", "UTC")]["stdout_sha256"]
    report = data["environment_probe"]["locale=C|tz=UTC"]
    assert report["requested_locale"] == "C"
    assert report["requested_timezone"] == "UTC"
    assert report["environment"]["LC_ALL"] == "C"
    assert report["environment"]["TZ"] == "UTC"


def test_compare_gate_detects_stdout_mismatch(tmp_path: Path) -> None:
    _, baseline = _run_with_marker(tmp_path, "one\n", "baseline")
    (tmp_path / "marker.txt").write_text("two\n", encoding="utf-8")
    out = tmp_path / "candidate.json"
    completed = _run_engine(
        "run",
        "--workload",
        tmp_path / "baseline_workload.py",
        "--out",
        out,
        "--seeds",
        "0",
        "--threads",
        1,
    )
    assert completed.returncode == 0, completed.stderr
    compared = _run_engine("compare", baseline, out, "--gate")
    assert compared.returncode == 3
    report = json.loads(compared.stdout)
    assert report["comparable"] is True
    assert report["identical"] is False
    assert report["mismatches"][0]["stdout_mismatch"] is True
    assert report["mismatches"][0]["artifacts_mismatch"] is False


def test_compare_gate_detects_artifact_mismatch(tmp_path: Path) -> None:
    workload = tmp_path / "artifact_workload.py"
    workload.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "marker = Path(__file__).with_name('marker.txt').read_text(encoding='utf-8')\n"
        "out = Path(os.environ['DETERMINISM_ARTIFACT_DIR'], 'payload.txt')\n"
        "out.write_text(marker, encoding='utf-8', newline='\\n')\n"
        "print('constant')\n",
        encoding="utf-8",
    )
    (tmp_path / "marker.txt").write_text("one\n", encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    first = _run_engine(
        "run", "--workload", workload, "--out", baseline, "--seeds", "0", "--threads", 1
    )
    assert first.returncode == 0, first.stderr
    (tmp_path / "marker.txt").write_text("two\n", encoding="utf-8")
    second = _run_engine(
        "run", "--workload", workload, "--out", candidate, "--seeds", "0", "--threads", 1
    )
    assert second.returncode == 0, second.stderr
    compared = _run_engine("compare", baseline, candidate, "--gate")
    assert compared.returncode == 3
    report = json.loads(compared.stdout)
    assert report["identical"] is False
    assert report["mismatches"][0]["stdout_mismatch"] is False
    assert report["mismatches"][0]["artifacts_mismatch"] is True
    assert report["mismatches"][0]["artifact_paths"] == ["payload.txt"]


def test_compare_normalizes_crlf_artifacts(tmp_path: Path) -> None:
    workload = tmp_path / "newline_workload.py"
    workload.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "mode = Path(__file__).with_name('mode.txt').read_text(encoding='utf-8').strip()\n"
        "newline = b'\\r\\n' if mode == 'crlf' else b'\\n'\n"
        "out = Path(os.environ['DETERMINISM_ARTIFACT_DIR'], 'payload.txt')\n"
        "out.write_bytes(b'a' + newline + b'b' + newline)\n"
        "print('constant')\n",
        encoding="utf-8",
    )
    (tmp_path / "mode.txt").write_text("crlf\n", encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    first = _run_engine(
        "run", "--workload", workload, "--out", baseline, "--seeds", "0", "--threads", 1
    )
    assert first.returncode == 0, first.stderr
    (tmp_path / "mode.txt").write_text("lf\n", encoding="utf-8")
    second = _run_engine(
        "run", "--workload", workload, "--out", candidate, "--seeds", "0", "--threads", 1
    )
    assert second.returncode == 0, second.stderr
    baseline_artifact = _manifest(baseline)["runs"][0]["artifacts"][0]
    candidate_artifact = _manifest(candidate)["runs"][0]["artifacts"][0]
    assert baseline_artifact["raw_sha256"] != candidate_artifact["raw_sha256"]
    assert baseline_artifact["sha256"] == candidate_artifact["sha256"]
    compared = _run_engine("compare", baseline, candidate, "--gate")
    assert compared.returncode == 0, compared.stderr
    assert json.loads(compared.stdout)["identical"] is True


def test_compare_gate_detects_missing_variants(tmp_path: Path) -> None:
    workload = tmp_path / "constant.py"
    workload.write_text("print('constant')\n", encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    first = _run_engine(
        "run",
        "--workload",
        workload,
        "--out",
        baseline,
        "--seeds",
        "0",
        "--threads",
        1,
        "--locales",
        "inherit",
        "C",
    )
    assert first.returncode == 0, first.stderr
    second = _run_engine(
        "run",
        "--workload",
        workload,
        "--out",
        candidate,
        "--seeds",
        "0",
        "--threads",
        1,
        "--locales",
        "inherit",
    )
    assert second.returncode == 0, second.stderr
    compared = _run_engine("compare", baseline, candidate, "--gate")
    assert compared.returncode == 3
    report = json.loads(compared.stdout)
    assert report["identical"] is False
    assert report["mismatches"] == []
    assert report["baseline_only_variants"]


def test_compare_rejects_schema_mismatch(tmp_path: Path) -> None:
    workload = tmp_path / "constant.py"
    workload.write_text("print('constant')\n", encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    for out in (baseline, candidate):
        completed = _run_engine(
            "run", "--workload", workload, "--out", out, "--seeds", "0", "--threads", 1
        )
        assert completed.returncode == 0, completed.stderr
    data = _manifest(candidate)
    data["schema_version"] = 999
    candidate.write_text(json.dumps(data), encoding="utf-8")
    compared = _run_engine("compare", baseline, candidate, "--gate")
    assert compared.returncode == 2
    report = json.loads(compared.stdout)
    assert report["comparable"] is False
    assert any("schema_version" in error for error in report["errors"])


def test_compare_rejects_failed_runs(tmp_path: Path) -> None:
    broken = tmp_path / "broken.py"
    broken.write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    for out in (baseline, candidate):
        completed = _run_engine(
            "run", "--workload", broken, "--out", out, "--seeds", "0", "--threads", 1
        )
        assert completed.returncode == 4
    compared = _run_engine("compare", baseline, candidate, "--gate")
    assert compared.returncode == 2
    report = json.loads(compared.stdout)
    assert report["comparable"] is False
    assert any("exited 1" in error for error in report["errors"])
