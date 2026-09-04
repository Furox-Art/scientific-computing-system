"""Tests for reproducibility manifests and local checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cds.provenance import (
    RunManifest,
    load_checkpoint,
    save_checkpoint,
    sha256_bytes,
    sha256_file,
    sha256_text,
)


def test_manifest_creation_recording_and_roundtrip() -> None:
    manifest = RunManifest.create(
        "Fit a physical model",
        seed=7,
        run_id="run-1",
        created_utc="2026-09-05T00:00:00+00:00",
    )
    assert manifest.question == "Fit a physical model"
    assert manifest.run_id == "run-1"
    assert manifest.seed == 7
    assert "python" in manifest.environment

    digest = sha256_text("dataset")
    manifest.record_data_hash("raw-data", digest.upper())
    manifest.record_tool("scipy", "1.14.0")
    manifest.record_decision(
        "use robust loss", "outliers affect least squares", approved_by_user=True
    )
    manifest.metadata["language"] = "both"

    assert manifest.data_hashes["raw-data"] == digest
    assert manifest.tool_versions == {"scipy": "1.14.0"}
    assert manifest.decisions[0].approved_by_user

    restored = RunManifest.from_json(manifest.to_json(indent=None))
    assert restored == manifest
    assert isinstance(manifest.to_dict(), dict)


def test_manifest_creation_defaults_and_validation() -> None:
    generated = RunManifest.create("Question")
    assert generated.run_id
    assert generated.created_utc
    assert generated.seed is None

    with pytest.raises(ValueError, match="question"):
        RunManifest.create("   ")

    manifest = RunManifest.create("Q")
    with pytest.raises(ValueError, match="hash name"):
        manifest.record_data_hash(" ", "0" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        manifest.record_data_hash("data", "abc")
    with pytest.raises(ValueError, match="SHA-256"):
        manifest.record_data_hash("data", "g" * 64)
    with pytest.raises(ValueError, match="tool name"):
        manifest.record_tool("", "1")
    with pytest.raises(ValueError, match="tool name"):
        manifest.record_tool("tool", " ")
    with pytest.raises(ValueError, match="decision"):
        manifest.record_decision("", "why", approved_by_user=False)
    with pytest.raises(ValueError, match="decision"):
        manifest.record_decision("action", " ", approved_by_user=False)


def test_manifest_from_json_rejects_malformed_payloads() -> None:
    with pytest.raises(ValueError, match="JSON must be an object"):
        RunManifest.from_json("[]")

    base: dict[str, object] = {
        "question": "Q",
        "run_id": "r",
        "created_utc": "t",
        "seed": None,
        "data_hashes": {},
        "tool_versions": {},
        "environment": {},
        "decisions": [],
        "metadata": {},
    }

    invalid_decisions = dict(base)
    invalid_decisions["decisions"] = {}
    with pytest.raises(ValueError, match="decisions must be a list"):
        RunManifest.from_json(json.dumps(invalid_decisions))

    non_object_decision = dict(base)
    non_object_decision["decisions"] = [1]
    with pytest.raises(ValueError, match="each decision"):
        RunManifest.from_json(json.dumps(non_object_decision))

    bad_decision = dict(base)
    bad_decision["decisions"] = [{"action": 1, "rationale": "why", "approved_by_user": True}]
    with pytest.raises(ValueError, match="invalid decision"):
        RunManifest.from_json(json.dumps(bad_decision))

    bad_map = dict(base)
    bad_map["data_hashes"] = ["not", "a", "map"]
    with pytest.raises(ValueError, match="data_hashes"):
        RunManifest.from_json(json.dumps(bad_map))

    bad_map_value = dict(base)
    bad_map_value["tool_versions"] = {"tool": 1}
    with pytest.raises(ValueError, match="tool_versions"):
        RunManifest.from_json(json.dumps(bad_map_value))

    bad_identity = dict(base)
    bad_identity["question"] = 1
    with pytest.raises(ValueError, match="identity"):
        RunManifest.from_json(json.dumps(bad_identity))

    bad_seed = dict(base)
    bad_seed["seed"] = "7"
    with pytest.raises(ValueError, match="seed"):
        RunManifest.from_json(json.dumps(bad_seed))

    boolean_seed = dict(base)
    boolean_seed["seed"] = True
    with pytest.raises(ValueError, match="seed"):
        RunManifest.from_json(json.dumps(boolean_seed))


def test_sha256_helpers_and_streaming_file_hash(tmp_path: Path) -> None:
    data = b"abcdef"
    path = tmp_path / "data.bin"
    path.write_bytes(data)

    expected = sha256_bytes(data)
    assert sha256_text("abcdef") == expected
    assert sha256_file(path, chunk_size=2) == expected

    with pytest.raises(ValueError, match="chunk_size"):
        sha256_file(path, chunk_size=0)


def test_checkpoint_roundtrip_and_validation(tmp_path: Path) -> None:
    manifest = RunManifest.create(
        "Q",
        run_id="run",
        created_utc="2026-09-05T00:00:00+00:00",
    )
    destination = tmp_path / "nested" / "run.json"
    save_checkpoint(destination, manifest, {"step": 3, "complete": False})

    restored, state = load_checkpoint(destination)
    assert restored == manifest
    assert state == {"step": 3, "complete": False}
    assert destination.read_text(encoding="utf-8").endswith("\n")

    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_checkpoint(scalar)

    missing = tmp_path / "missing.json"
    missing.write_text('{"manifest": {}, "state": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="manifest and state"):
        load_checkpoint(missing)


def test_checkpoint_removes_temporary_file_if_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = RunManifest.create("Q")
    destination = tmp_path / "run.json"

    def fail_replace(self: Path, target: Path) -> Path:
        assert target == destination
        raise OSError("replace failed")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        save_checkpoint(destination, manifest, {"x": 1})

    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []
