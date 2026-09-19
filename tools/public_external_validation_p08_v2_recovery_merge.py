#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

EXPECTED_SESSIONS = {
    "sub-Bubbles": ("ses-01", "ses-04"),
    "sub-Buttercup": ("ses-04", "ses-05"),
}
SESSION_STATUS = "P08_OPEN_SOURCE_SECONDARY_PREPROCESSING_V2_SESSION_SHARD_COMPLETE"
MERGED_STATUS = "P08_OPEN_SOURCE_SECONDARY_PREPROCESSING_V2_COMPLETE"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def find_single(root: Path, pattern: str) -> Path:
    found = sorted(root.rglob(pattern))
    if len(found) != 1:
        raise RuntimeError(f"Expected exactly one {pattern!r} under {root}, got {found}")
    return found[0]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def copy_exact(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if src.stat().st_size != dst.stat().st_size or sha256_file(src) != sha256_file(dst):
            raise RuntimeError(f"Conflicting duplicate while merging: {src} -> {dst}")
        return
    shutil.copy2(src, dst)


def verify_shard(pre_root: Path, prov_root: Path, participant: str, session: str) -> dict:
    result_path = find_single(pre_root, "opensource_preprocessing_result.json")
    recon_path = find_single(prov_root, "input_reconstruction.json")
    result = load_json(result_path)
    recon = load_json(recon_path)

    if result.get("status") != SESSION_STATUS:
        raise RuntimeError(f"{participant} {session}: ineligible shard status {result.get('status')}")
    if result.get("participant") != participant or result.get("session_filter") != session:
        raise RuntimeError(f"{participant} {session}: shard identity mismatch")
    if result.get("expected_run_count") != 2 or result.get("mni_optcom_run_count") != 2:
        raise RuntimeError(f"{participant} {session}: expected exactly two frozen runs")
    if not result.get("preprocessing_complete", False):
        raise RuntimeError(f"{participant} {session}: preprocessing_complete=false")
    if result.get("self_other_glm_computed") or result.get("roi_effect_computed") or result.get("neural_effect_computed"):
        raise RuntimeError(f"{participant} {session}: target neural outcome unexpectedly present")

    if recon.get("participant") != participant or recon.get("session_filter") != session or not recon.get("all_exact_verified"):
        raise RuntimeError(f"{participant} {session}: exact input provenance not eligible")

    mni = sorted(pre_root.rglob(
        f"{participant}_{session}_task-SORPF_run-*_space-MNI152NLin2009cAsym_res-02_desc-optcom_bold.nii.gz"
    ))
    if len(mni) != 2:
        raise RuntimeError(f"{participant} {session}: expected 2 MNI optcom files, got {mni}")

    recorded = {
        Path(x["path"]).name: x
        for x in result.get("selected_output_files", [])
        if "res-02_desc-optcom_bold.nii.gz" in str(x.get("path", ""))
    }
    for p in mni:
        rec = recorded.get(p.name)
        if rec is None:
            raise RuntimeError(f"{participant} {session}: missing recorded output hash for {p.name}")
        got = sha256_file(p)
        if got != rec.get("sha256") or p.stat().st_size != int(rec.get("bytes", -1)):
            raise RuntimeError(f"{participant} {session}: output hash/size mismatch for {p.name}")

    events = sorted(pre_root.rglob(f"{participant}_{session}_task-SORPF_run-*_events.tsv"))
    if len(events) != 2:
        raise RuntimeError(f"{participant} {session}: expected 2 event files, got {events}")

    motion = sorted(pre_root.rglob(f"afni/{session}/results/dfile*.1D"))
    if len(motion) != 1:
        raise RuntimeError(f"{participant} {session}: expected one AFNI motion file, got {motion}")

    affine = find_single(prov_root, "T1_to_MNI_0GenericAffine.mat")
    warp = find_single(prov_root, "T1_to_MNI_1Warp.nii.gz")
    return {
        "result": result,
        "recon": recon,
        "mni": mni,
        "events": events,
        "motion": motion[0],
        "affine": affine,
        "warp": warp,
        "affine_sha256": sha256_file(affine),
        "warp_sha256": sha256_file(warp),
    }


def merge_reconstruction(shards: list[dict], participant: str) -> dict:
    by_path = {}
    for s in shards:
        for rec in s["recon"].get("files", []):
            p = rec.get("path")
            if not p:
                raise RuntimeError("Reconstruction record without path")
            if p in by_path:
                a = by_path[p]
                keys = ("bytes", "hash_algorithm", "hash", "storage", "role")
                if any(a.get(k) != rec.get(k) for k in keys):
                    raise RuntimeError(f"Conflicting exact-input provenance for {p}")
            else:
                by_path[p] = rec
    files = [by_path[p] for p in sorted(by_path)]
    return {
        "status": "P08_RECOVERY_MERGED_EXACT_INPUT_PROVENANCE",
        "participant": participant,
        "frozen_dataset": "OpenNeuro ds002278 v2.0.0",
        "frozen_commit": "ed03c47c368000e511401021c1d13c3fb916b470",
        "recovery_sessions": list(EXPECTED_SESSIONS[participant]),
        "files": files,
        "selected_files": len(files),
        "selected_bytes_unique": sum(int(x.get("bytes", 0)) for x in files),
        "all_exact_verified": True,
        "neural_effect_computed": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", required=True, choices=sorted(EXPECTED_SESSIONS))
    ap.add_argument("--session1-preproc", required=True)
    ap.add_argument("--session1-prov", required=True)
    ap.add_argument("--session2-preproc", required=True)
    ap.add_argument("--session2-prov", required=True)
    ap.add_argument("--output-preproc", required=True)
    ap.add_argument("--output-prov", required=True)
    args = ap.parse_args()

    participant = args.participant
    sessions = EXPECTED_SESSIONS[participant]
    roots = [
        (Path(args.session1_preproc), Path(args.session1_prov), sessions[0]),
        (Path(args.session2_preproc), Path(args.session2_prov), sessions[1]),
    ]
    shards = [verify_shard(pre, prov, participant, ses) for pre, prov, ses in roots]

    if shards[0]["affine_sha256"] != shards[1]["affine_sha256"]:
        raise RuntimeError(f"{participant}: ANTs affine differs across session shards")
    if shards[0]["warp_sha256"] != shards[1]["warp_sha256"]:
        raise RuntimeError(f"{participant}: ANTs nonlinear warp differs across session shards")

    out_pre = Path(args.output_preproc)
    out_prov = Path(args.output_prov)
    out_pre.mkdir(parents=True, exist_ok=True)
    out_prov.mkdir(parents=True, exist_ok=True)

    selected = []
    sessions_record = {}
    for shard, session in zip(shards, sessions):
        for p in shard["mni"]:
            dst = out_pre / "mni" / p.name
            copy_exact(p, dst)
            selected.append({
                "path": f"mni/{p.name}",
                "bytes": dst.stat().st_size,
                "sha256": sha256_file(dst),
            })
        for p in shard["events"]:
            dst = out_pre / "bids" / participant / session / "func" / p.name
            copy_exact(p, dst)
        motion_dst = out_pre / "afni" / session / "results" / shard["motion"].name
        copy_exact(shard["motion"], motion_dst)
        sessions_record[session] = {
            "input_runs": [1, 2],
            "mni_optcom": {
                str(i): str(next(x for x in selected if f"_{session}_task-SORPF_run-{i}_" in x["path"])["path"])
                for i in (1, 2)
            },
            "motion_sha256": sha256_file(motion_dst),
        }

    merged_recon = merge_reconstruction(shards, participant)
    (out_prov / "input_reconstruction.json").write_text(
        json.dumps(merged_recon, indent=2) + "\n", encoding="utf-8"
    )
    copy_exact(shards[0]["affine"], out_prov / "ants" / "T1_to_MNI_0GenericAffine.mat")
    copy_exact(shards[0]["warp"], out_prov / "ants" / "T1_to_MNI_1Warp.nii.gz")

    # Verify every merged event against the exact reconstruction records.
    recon_by_path = {x["path"]: x for x in merged_recon["files"]}
    for p in sorted(out_pre.rglob("*_events.tsv")):
        rel_suffix = f"{participant}/" + p.as_posix().split(f"{participant}/", 1)[1]
        rec = recon_by_path.get(rel_suffix)
        if rec is None:
            raise RuntimeError(f"Merged event missing exact provenance: {rel_suffix}")
        if sha256_file(p) != rec.get("hash"):
            raise RuntimeError(f"Merged event hash mismatch: {rel_suffix}")

    merged = {
        "status": MERGED_STATUS,
        "participant": participant,
        "lane": "open_source_secondary_sensitivity_only",
        "recovery_mode": "session_sharded_operational_recovery",
        "recovery_sessions": list(sessions),
        "preprocessing_complete": True,
        "expected_run_count": 4,
        "mni_optcom_run_count": 4,
        "sessions_expected": {s: [1, 2] for s in sessions},
        "sessions": sessions_record,
        "selected_output_files": sorted(selected, key=lambda x: x["path"]),
        "ants_transform_identity_across_shards": {
            "affine_sha256": shards[0]["affine_sha256"],
            "warp_sha256": shards[0]["warp_sha256"],
            "identical": True,
        },
        "exact_input_provenance_merged": True,
        "self_other_glm_computed": False,
        "roi_effect_computed": False,
        "neural_effect_computed": False,
        "primary_lane_modified": False,
    }
    (out_pre / "opensource_preprocessing_result.json").write_text(
        json.dumps(merged, indent=2) + "\n", encoding="utf-8"
    )

    summary = {
        "status": "P08_V2_SESSION_RECOVERY_PARTICIPANT_MERGE_COMPLETE",
        "participant": participant,
        "sessions": list(sessions),
        "mni_run_count": len(selected),
        "ants_affine_sha256": shards[0]["affine_sha256"],
        "ants_warp_sha256": shards[0]["warp_sha256"],
        "exact_input_provenance_merged": True,
        "self_other_glm_computed": False,
        "roi_effect_computed": False,
        "neural_effect_computed": False,
    }
    (out_prov / "recovery_merge_result.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
