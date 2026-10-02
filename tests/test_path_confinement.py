"""Path-traversal confinement tests for caller-supplied provenance paths.

The threat: a manifest field or checkpoint name that the caller did not author
(``../../../../etc/shadow``, a sibling directory sharing a textual prefix, a
symlink pointing outward) is passed to a provenance API that hashes or writes
it. Without confinement the library happily reads or writes outside the
directory the caller intended to bound.

These tests pin three properties:

* confinement is **opt-in** -- omitting ``root`` preserves historical behaviour,
  so no existing caller breaks;
* confinement is **sound** -- traversal, shared-prefix, absolute, and symlink
  escapes are all rejected;
* rejection happens **before** any filesystem side effect, so a rejected
  ``save_checkpoint`` leaves no directory and no partial file behind.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cds._paths import resolve_target, resolve_within_root
from cds.provenance import RunManifest, load_checkpoint, save_checkpoint


def _manifest() -> RunManifest:
    return RunManifest.create(
        "confinement",
        run_id="confined-run",
        created_utc="2026-10-02T00:00:00+00:00",
    )


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    root.mkdir()
    return root


def _sibling(tmp_path: Path) -> Path:
    """A directory sharing a textual prefix with ``root`` but outside it."""
    sibling = tmp_path / "root-backup"
    sibling.mkdir()
    return sibling


# --------------------------------------------------------------------------
# Primitive: resolve_within_root
# --------------------------------------------------------------------------


def test_relative_and_absolute_paths_inside_root_are_accepted(tmp_path: Path) -> None:
    root = _root(tmp_path)
    (root / "lock.txt").write_text("data", encoding="utf-8")

    relative = resolve_within_root("lock.txt", root)
    assert relative == (root / "lock.txt").resolve()

    absolute = resolve_within_root(root / "lock.txt", root)
    assert absolute == relative


def test_traversal_that_lands_back_inside_root_is_accepted(tmp_path: Path) -> None:
    root = _root(tmp_path)
    (root / "nested").mkdir()
    (root / "lock.txt").write_text("data", encoding="utf-8")

    collapsed = resolve_within_root("nested/../lock.txt", root)
    assert collapsed == (root / "lock.txt").resolve()


def test_traversal_escaping_root_is_rejected(tmp_path: Path) -> None:
    root = _root(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    with pytest.raises(ValueError, match="escapes confined root"):
        resolve_within_root("../outside.txt", root)

    with pytest.raises(ValueError, match="escapes confined root"):
        resolve_within_root("nested/../../outside.txt", root)


def test_sibling_sharing_a_textual_prefix_is_rejected(tmp_path: Path) -> None:
    """``root``/``root-backup`` is the bypass a ``str.startswith`` check misses."""
    root = _root(tmp_path)
    sibling = _sibling(tmp_path)

    with pytest.raises(ValueError, match="escapes confined root"):
        resolve_within_root("../root-backup/loot.txt", root)

    with pytest.raises(ValueError, match="escapes confined root"):
        resolve_within_root(sibling / "loot.txt", root)


def test_symlink_pointing_outside_root_is_rejected(tmp_path: Path) -> None:
    root = _root(tmp_path)
    target = tmp_path / "outside.txt"
    target.write_text("secret", encoding="utf-8")
    link = root / "link.txt"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        pytest.skip("symlink creation unavailable on this platform/privilege level")

    with pytest.raises(ValueError, match="escapes confined root"):
        resolve_within_root("link.txt", root)


def test_root_itself_is_not_treated_as_an_escape(tmp_path: Path) -> None:
    root = _root(tmp_path)
    assert resolve_within_root(root, root) == root.resolve()


# --------------------------------------------------------------------------
# Opt-in adapter: resolve_target
# --------------------------------------------------------------------------


def test_resolve_target_without_root_preserves_historical_behaviour(tmp_path: Path) -> None:
    assert resolve_target("some/relative/path") == Path("some/relative/path")
    absolute = tmp_path / "file.txt"
    assert resolve_target(absolute) == absolute

    root = _root(tmp_path)
    # Without a root there is no confinement: traversal is the caller's call.
    assert resolve_target("../outside.txt") == Path("../outside.txt")
    assert resolve_target("../outside.txt", None) == Path("../outside.txt")

    # Supplying the same root flips it to a hard rejection.
    with pytest.raises(ValueError, match="escapes confined root"):
        resolve_target("../outside.txt", root)


# --------------------------------------------------------------------------
# Wiring: RunManifest.record_environment_lock
# --------------------------------------------------------------------------


def test_record_environment_lock_accepts_confined_relative_path(tmp_path: Path) -> None:
    root = _root(tmp_path)
    (root / "solver.lock").write_text("pins\n", encoding="utf-8")
    manifest = _manifest()

    digest = manifest.record_environment_lock("solver", "solver.lock", root=root)

    from cds.provenance import sha256_file

    assert digest == sha256_file(root / "solver.lock")
    assert manifest.metadata["lock.solver.sha256"] == digest


def test_record_environment_lock_rejects_traversal(tmp_path: Path) -> None:
    root = _root(tmp_path)
    outside = tmp_path / "outside.lock"
    outside.write_text("secret\n", encoding="utf-8")
    manifest = _manifest()

    with pytest.raises(ValueError, match="escapes confined root"):
        manifest.record_environment_lock("solver", "../outside.lock", root=root)

    assert "lock.solver.sha256" not in manifest.metadata


def test_record_environment_lock_rejects_sibling_prefix_escape(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _sibling(tmp_path)
    manifest = _manifest()

    with pytest.raises(ValueError, match="escapes confined root"):
        manifest.record_environment_lock("solver", "../root-backup/loot.lock", root=root)


def test_record_environment_lock_still_validates_name_before_resolving(
    tmp_path: Path,
) -> None:
    """An empty name is still a name error, not a confinement error."""
    root = _root(tmp_path)
    manifest = _manifest()

    with pytest.raises(ValueError, match="lock name"):
        manifest.record_environment_lock("  ", "solver.lock", root=root)


# --------------------------------------------------------------------------
# Wiring: save_checkpoint / load_checkpoint
# --------------------------------------------------------------------------


def test_checkpoint_round_trips_inside_confined_root(tmp_path: Path) -> None:
    root = _root(tmp_path)
    manifest = _manifest()

    save_checkpoint("run.json", manifest, {"step": 3}, root=root)

    restored_manifest, state = load_checkpoint("run.json", root=root)
    assert state == {"step": 3}
    assert restored_manifest.run_id == manifest.run_id
    assert (root / "run.json").is_file()


def test_save_checkpoint_rejects_escape_before_touching_the_filesystem(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    manifest = _manifest()

    with pytest.raises(ValueError, match="escapes confined root"):
        save_checkpoint("../escaped.json", manifest, {"step": 1}, root=root)

    # No sibling directory and no partial temp file were created.
    assert not (tmp_path / "escaped.json").exists()
    assert list(tmp_path.iterdir()) == [root]


def test_load_checkpoint_rejects_escape_and_reads_nothing(tmp_path: Path) -> None:
    root = _root(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="escapes confined root"):
        load_checkpoint("../outside.json", root=root)

    with pytest.raises(ValueError, match="escapes confined root"):
        load_checkpoint(outside, root=root)


def test_checkpoint_apis_remain_backward_compatible_without_root(tmp_path: Path) -> None:
    """Omitting ``root`` keeps the pre-existing unconfined behaviour."""
    manifest = _manifest()
    target = tmp_path / "legacy.json"

    save_checkpoint(target, manifest, {"legacy": True})

    restored_manifest, state = load_checkpoint(target)
    assert state == {"legacy": True}
    assert restored_manifest.run_id == manifest.run_id
