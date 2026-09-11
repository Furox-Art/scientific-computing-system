"""Regression tests for provenance integrity and checkpoint hygiene."""

import json
from pathlib import Path

import pytest

from cds.provenance import RunManifest, save_checkpoint


def test_manifest_rejects_nonfinite_identity_metadata_shapes() -> None:
    with pytest.raises(ValueError, match="question"):
        RunManifest.create("   ")
    with pytest.raises(ValueError, match="seed"):
        RunManifest.create("question", seed=True)
    with pytest.raises(ValueError, match="timezone-aware"):
        RunManifest.create("question", created_utc="2026-09-05T12:00:00")


def test_from_json_revalidates_hashes_and_nonempty_metadata() -> None:
    manifest = RunManifest.create(
        "question",
        run_id="audit-run",
        created_utc="2026-09-05T12:00:00+00:00",
    )
    raw = manifest.to_dict()
    raw["data_hashes"] = {"dataset": "not-a-sha256"}
    with pytest.raises(ValueError, match="SHA-256"):
        RunManifest.from_json(json.dumps(raw))

    raw = manifest.to_dict()
    raw["tool_versions"] = {"sympy": ""}
    with pytest.raises(ValueError, match="non-empty strings"):
        RunManifest.from_json(json.dumps(raw))


def test_failed_checkpoint_serialization_leaves_no_partial_file(tmp_path: Path) -> None:
    destination = tmp_path / "checkpoint.json"
    manifest = RunManifest.create("question")

    with pytest.raises(TypeError):
        save_checkpoint(destination, manifest, {"not_json": object()})

    assert not destination.exists()
    assert list(tmp_path.glob(".checkpoint.json.*.tmp")) == []
