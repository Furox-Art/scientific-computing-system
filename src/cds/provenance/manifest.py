"""Local-first provenance records for reproducible scientific runs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import cast
from uuid import uuid4


@dataclass(frozen=True)
class DecisionRecord:
    """One scientifically meaningful method/data decision."""

    action: str
    rationale: str
    approved_by_user: bool

    def __post_init__(self) -> None:
        if not self.action.strip() or not self.rationale.strip():
            raise ValueError("decision action and rationale must not be empty")
        if not isinstance(self.approved_by_user, bool):
            raise ValueError("approved_by_user must be a boolean")


def _is_hex_sha(value: str) -> bool:
    return 7 <= len(value) <= 64 and all(char in "0123456789abcdef" for char in value.lower())


def detect_git_sha() -> str | None:
    """Resolve the current source revision from CI metadata or the local checkout."""
    for key in ("GITHUB_SHA", "CI_COMMIT_SHA", "GIT_COMMIT"):
        value = os.environ.get(key, "").strip()
        if value and _is_hex_sha(value):
            return value.lower()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value.lower() if result.returncode == 0 and _is_hex_sha(value) else None


def _canonicalize(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _canonicalize(asdict(value))
    if isinstance(value, dict):
        return {str(key): _canonicalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported canonical provenance value: {type(value).__name__}")


def canonical_sha256(value: object) -> str:
    """Hash a deterministic JSON representation of supported scientific metadata."""
    payload = json.dumps(
        _canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return sha256_text(payload)


def _capture_lock_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in ("requirements.lock", "requirements-dev.lock"):
        path = root / name
        if path.is_file():
            hashes[f"lock.{name}.sha256"] = sha256_file(path)
    return hashes


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

    def __post_init__(self) -> None:
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("question must not be empty")
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("run_id must not be empty")
        if not isinstance(self.created_utc, str) or not self.created_utc.strip():
            raise ValueError("created_utc must not be empty")
        try:
            created = datetime.fromisoformat(self.created_utc)
        except ValueError as exc:
            raise ValueError("created_utc must be a valid ISO-8601 timestamp") from exc
        if created.tzinfo is None or created.utcoffset() is None:
            raise ValueError("created_utc must be timezone-aware")
        if self.seed is not None and (not isinstance(self.seed, int) or isinstance(self.seed, bool)):
            raise ValueError("seed must be an integer or null")

        for name, digest in self.data_hashes.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("data hash names must not be empty")
            if not isinstance(digest, str) or len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest.lower()
            ):
                raise ValueError("data hashes must be SHA-256 hex digests")

        for mapping_name, mapping in (
            ("tool_versions", self.tool_versions),
            ("environment", self.environment),
            ("metadata", self.metadata),
        ):
            if any(
                not isinstance(key, str)
                or not key.strip()
                or not isinstance(value, str)
                or not value.strip()
                for key, value in mapping.items()
            ):
                raise ValueError(f"{mapping_name} must map non-empty strings to non-empty strings")

        if any(not isinstance(decision, DecisionRecord) for decision in self.decisions):
            raise ValueError("decisions must contain DecisionRecord values")

    @classmethod
    def create(
        cls,
        question: str,
        *,
        seed: int | None = None,
        run_id: str | None = None,
        created_utc: str | None = None,
        project_root: str | os.PathLike[str] | None = None,
    ) -> RunManifest:
        """Create a manifest with an automatic source/environment snapshot."""
        root = Path(project_root) if project_root is not None else Path.cwd()
        metadata_values = _capture_lock_hashes(root)
        git_sha = detect_git_sha()
        if git_sha is not None:
            metadata_values["source.git_sha"] = git_sha
        return cls(
            question=question,
            run_id=run_id or str(uuid4()),
            created_utc=created_utc or datetime.now(timezone.utc).isoformat(),
            seed=seed,
            environment={
                "python": platform.python_version(),
                "implementation": platform.python_implementation(),
                "platform": platform.platform(),
                "executable": sys.executable,
            },
            metadata=metadata_values,
        )

    def record_data_hash(self, name: str, digest: str) -> None:
        """Associate a logical dataset/input name with a SHA-256 digest."""
        if not name.strip():
            raise ValueError("data hash name must not be empty")
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest.lower()
        ):
            raise ValueError("digest must be a SHA-256 hex digest")
        self.data_hashes[name] = digest.lower()

    def record_tool(self, name: str, version: str) -> None:
        """Record the exact version of a scientific backend used in the run."""
        if not name.strip() or not version.strip():
            raise ValueError("tool name and version must not be empty")
        self.tool_versions[name] = version

    def record_decision(self, action: str, rationale: str, *, approved_by_user: bool) -> None:
        """Append an auditable scientific decision."""
        self.decisions.append(
            DecisionRecord(
                action=action,
                rationale=rationale,
                approved_by_user=approved_by_user,
            )
        )

    def record_plan_hash(self, plan: object) -> str:
        """Bind this run to the canonical analysis plan used for execution."""
        digest = canonical_sha256(plan)
        self.metadata["plan.sha256"] = digest
        return digest

    def record_environment_lock(self, name: str, path: str | os.PathLike[str]) -> str:
        """Bind an additional environment/lock file to the manifest."""
        if not name.strip():
            raise ValueError("environment lock name must not be empty")
        digest = sha256_file(path)
        self.metadata[f"lock.{name}.sha256"] = digest
        return digest

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return cast(dict[str, object], asdict(self))

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the manifest deterministically for storage or comparison."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_json(cls, payload: str) -> RunManifest:
        """Restore and revalidate a manifest from ``to_json`` output."""
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
            if (
                not isinstance(action, str)
                or not isinstance(rationale, str)
                or not isinstance(approved, bool)
            ):
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
        if (
            not isinstance(question, str)
            or not isinstance(run_id, str)
            or not isinstance(created_utc, str)
        ):
            raise ValueError("manifest identity fields must be strings")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise ValueError("seed must be an integer or null")

        return cls(
            question=question,
            run_id=run_id,
            created_utc=created_utc,
            seed=seed,
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
    """Atomically save a JSON checkpoint and remove partial files on any failure."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"manifest": manifest.to_dict(), "state": state}
    temporary: Path | None = None
    try:
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
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    except Exception:
        if temporary is not None:
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
