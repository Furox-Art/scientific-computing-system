"""Local-first provenance records for reproducible scientific runs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import uuid4


@dataclass(frozen=True)
class DecisionRecord:
    """One scientifically meaningful method/data decision."""

    action: str
    rationale: str
    approved_by_user: bool


@dataclass
class RunManifest:
    """Portable record of the inputs, environment, tools, and decisions of a run."""

    question: str
    run_id: str
    created_utc: str
    seed: int | None = None
    data_hashes: dict[str, str] = field(default_factory=dict)
    tool_versions: dict[str, str] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)
    decisions: list[DecisionRecord] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        question: str,
        *,
        seed: int | None = None,
        run_id: str | None = None,
        created_utc: str | None = None,
    ) -> RunManifest:
        """Create a manifest with a snapshot of the local Python environment."""
        if not question.strip():
            raise ValueError("question must not be empty")
        return cls(
            question=question,
            run_id=run_id or str(uuid4()),
            created_utc=created_utc or datetime.now(UTC).isoformat(),
            seed=seed,
            environment={
                "python": platform.python_version(),
                "implementation": platform.python_implementation(),
                "platform": platform.platform(),
                "executable": sys.executable,
            },
        )

    def record_data_hash(self, name: str, digest: str) -> None:
        """Associate a logical dataset/input name with a SHA-256 digest."""
        if not name.strip():
            raise ValueError("data hash name must not be empty")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest.lower()):
            raise ValueError("digest must be a SHA-256 hex digest")
        self.data_hashes[name] = digest.lower()

    def record_tool(self, name: str, version: str) -> None:
        """Record the exact version of a scientific backend used in the run."""
        if not name.strip() or not version.strip():
            raise ValueError("tool name and version must not be empty")
        self.tool_versions[name] = version

    def record_decision(self, action: str, rationale: str, *, approved_by_user: bool) -> None:
        """Append an auditable scientific decision."""
        if not action.strip() or not rationale.strip():
            raise ValueError("decision action and rationale must not be empty")
        self.decisions.append(
            DecisionRecord(
                action=action,
                rationale=rationale,
                approved_by_user=approved_by_user,
            )
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return cast(dict[str, object], asdict(self))

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the manifest deterministically for storage or comparison."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_json(cls, payload: str) -> RunManifest:
        """Restore a manifest from ``to_json`` output."""
        raw = json.loads(payload)
        if not isinstance(raw, dict):
            raise ValueError("manifest JSON must be an object")
        obj = cast(dict[str, object], raw)

        decisions_raw = obj.get("decisions", [])
        if not isinstance(decisions_raw, list):
            raise ValueError("decisions must be a list")
        decisions: list[DecisionRecord] = []
        for item in decisions_raw:
            if not isinstance(item, dict):
                raise ValueError("each decision must be an object")
            decision = cast(dict[str, object], item)
            action = decision.get("action")
            rationale = decision.get("rationale")
            approved = decision.get("approved_by_user")
            if not isinstance(action, str) or not isinstance(rationale, str) or not isinstance(approved, bool):
                raise ValueError("invalid decision record")
            decisions.append(DecisionRecord(action, rationale, approved))

        def string_map(name: str) -> dict[str, str]:
            value = obj.get(name, {})
            if not isinstance(value, dict) or any(
                not isinstance(key, str) or not isinstance(item, str) for key, item in value.items()
            ):
                raise ValueError(f"{name} must map strings to strings")
            return cast(dict[str, str], value)

        question = obj.get("question")
        run_id = obj.get("run_id")
        created_utc = obj.get("created_utc")
        seed = obj.get("seed")
        if not isinstance(question, str) or not isinstance(run_id, str) or not isinstance(created_utc, str):
            raise ValueError("manifest identity fields must be strings")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise ValueError("seed must be an integer or null")

        return cls(
            question=question,
            run_id=run_id,
            created_utc=created_utc,
            seed=cast(int | None, seed),
            data_hashes=string_map("data_hashes"),
            tool_versions=string_map("tool_versions"),
            environment=string_map("environment"),
            decisions=decisions,
            metadata=string_map("metadata"),
        )


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 digest for in-memory bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str, *, encoding: str = "utf-8") -> str:
    """Return the SHA-256 digest for text with explicit encoding."""
    return sha256_bytes(text.encode(encoding))


def sha256_file(path: str | os.PathLike[str], *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file incrementally so large research datasets need not fit in memory."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(
    path: str | os.PathLike[str],
    manifest: RunManifest,
    state: dict[str, object],
) -> None:
    """Atomically save a JSON checkpoint without silently overwriting mid-write."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"manifest": manifest.to_dict(), "state": state}
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    try:
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def load_checkpoint(path: str | os.PathLike[str]) -> tuple[RunManifest, dict[str, object]]:
    """Load a checkpoint created by ``save_checkpoint``."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("checkpoint must contain a JSON object")
    obj = cast(dict[str, object], raw)
    manifest_raw = obj.get("manifest")
    state_raw = obj.get("state")
    if not isinstance(manifest_raw, dict) or not isinstance(state_raw, dict):
        raise ValueError("checkpoint must contain manifest and state objects")
    manifest = RunManifest.from_json(json.dumps(manifest_raw))
    return manifest, cast(dict[str, object], state_raw)
