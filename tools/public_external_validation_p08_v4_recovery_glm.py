#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm.first_level import FirstLevelModel
from nipype.algorithms.confounds import compute_dvars
from scipy.stats import binomtest

TR = 1.5
GLM_LOCK_FILE = "OPEN_SOURCE_V2_GLM_LOCK_V4_RECOVERY_2026-09-19.json"
HIGH_PASS = 1.0 / 128.0
FD_THRESHOLD = 0.5
STDDVARS_THRESHOLD = 1.5
HEAD_RADIUS_MM = 50.0
EXPECTED_PARTICIPANTS = ("sub-Bubbles", "sub-Buttercup", "sub-PILOT02")
EXPECTED_STATUS = "P08_OPEN_SOURCE_SECONDARY_PREPROCESSING_V2_COMPLETE"
ROI_SPECS = {
    "Frontal Medial Cortex": {
        "filename": "roi-FrontalMedialCortex_mask.nii.gz",
        "sha256": "67b69213e1501b9a8be69622370d1367ebfbe0c809d72ef31e5fc9563c82dd1a",
        "index": 25,
        "total_voxels": 1059,
    },
    "Cingulate Gyrus, posterior division": {
        "filename": "roi-CingulateGyrusPosterior_mask.nii.gz",
        "sha256": "706ff59c121370ab2d8e589eea16657ed051eedd5073da6fbe80591c21f15f45",
        "index": 30,
        "total_voxels": 2506,
    },
}
TRIAL_TYPES = (
    "Self",
    "Other",
    "Malleable",
    "Flanker-Congruent",
    "Flanker-Incongruent",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_entities(name: str) -> tuple[str, str, int]:
    pm = re.search(r"(sub-[A-Za-z0-9]+)", name)
    sm = re.search(r"(ses-[A-Za-z0-9]+)", name)
    rm = re.search(r"_run-(\d+)_", name)
    if not (pm and sm and rm):
        raise RuntimeError(f"Cannot parse participant/session/run from {name}")
    return pm.group(1), sm.group(1), int(rm.group(1))


def find_single(root: Path, pattern: str) -> Path:
    found = sorted(root.rglob(pattern))
    if len(found) != 1:
        raise RuntimeError(f"Expected one {pattern!r} under {root}, got {found}")
    return found[0]


def artifact_root_for(path: Path, participant: str) -> Path:
    marker = f"public-p08-opensource-v2-preprocessed-{participant}"
    for parent in [path.parent, *path.parents]:
        if parent.name == marker:
            return parent
    # Some download layouts place the artifact contents directly at the requested path.
    # Fall back to the highest input root containing the participant BIDS subtree.
    cur = path.parent
    while cur.parent != cur and not (cur / "bids" / participant).exists():
        cur = cur.parent
    if (cur / "bids" / participant).exists():
        return cur
    raise RuntimeError(f"Cannot identify artifact root for {path}")


def load_and_verify_roi_masks(roi_root: Path) -> dict[str, Path]:
    out = {}
    for name, spec in ROI_SPECS.items():
        p = find_single(roi_root, spec["filename"])
        got = sha256_file(p)
        if got != spec["sha256"]:
            raise RuntimeError(f"ROI hash mismatch for {name}: {got} != {spec['sha256']}")
        img = nib.load(p)
        n = int((np.asanyarray(img.dataobj) > 0).sum())
        if n != spec["total_voxels"]:
            raise RuntimeError(f"ROI voxel count mismatch for {name}: {n} != {spec['total_voxels']}")
        out[name] = p
    return out


def verify_preprocessing_root(artifact_root: Path, prov_root: Path, participant: str) -> dict:
    result_path = find_single(artifact_root, "opensource_preprocessing_result.json")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("participant") != participant:
        raise RuntimeError(f"Preprocessing participant mismatch: {result.get('participant')} != {participant}")
    if result.get("status") != EXPECTED_STATUS or not result.get("preprocessing_complete", False):
        raise RuntimeError(f"Preprocessing not eligible: {result.get('status')}")
    if result.get("self_other_glm_computed") or result.get("roi_effect_computed") or result.get("neural_effect_computed"):
        raise RuntimeError("Preprocessing artifact unexpectedly contains target neural outcome")

    recon_path = find_single(prov_root, "input_reconstruction.json")
    recon = json.loads(recon_path.read_text(encoding="utf-8"))
    if recon.get("participant") != participant or not recon.get("all_exact_verified"):
        raise RuntimeError("Exact frozen input reconstruction provenance failed eligibility")

    selected = {Path(x["path"]).name: x for x in result.get("selected_output_files", [])}
    mni_files = sorted(artifact_root.rglob(f"{participant}_ses-*_task-SORPF_run-*_space-MNI152NLin2009cAsym_res-02_desc-optcom_bold.nii.gz"))
    expected_n = int(result.get("expected_run_count", -1))
    if len(mni_files) != expected_n or expected_n <= 0:
        raise RuntimeError(f"MNI run count mismatch: files={len(mni_files)}, expected={expected_n}")
    for p in mni_files:
        rec = selected.get(p.name)
        if rec is None:
            raise RuntimeError(f"No preprocessing hash record for {p.name}")
        got = sha256_file(p)
        if got != rec.get("sha256"):
            raise RuntimeError(f"Preprocessed BOLD hash mismatch for {p.name}")
    return {"result": result, "reconstruction": recon, "mni_files": mni_files}


def load_motion_session(artifact_root: Path, session: str) -> np.ndarray:
    candidates = sorted(artifact_root.rglob(f"afni/{session}/results/dfile_rall.1D"))
    if len(candidates) != 1:
        # tolerate AFNI naming variants only if there is exactly one dfile* under session/results
        candidates = sorted(artifact_root.rglob(f"afni/{session}/results/dfile*.1D"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one AFNI motion dfile for {session}, got {candidates}")
    arr = np.loadtxt(candidates[0], ndmin=2)
    if arr.shape[1] == 6:
        mot = arr
    elif arr.shape[1] == 9:
        mot = arr[:, 1:7]
    else:
        raise RuntimeError(f"Unexpected motion dfile shape {arr.shape} in {candidates[0]}")
    return np.asarray(mot, dtype=float)


def power_fd(motion: np.ndarray) -> np.ndarray:
    if motion.ndim != 2 or motion.shape[1] != 6:
        raise RuntimeError(f"Expected Nx6 AFNI motion, got {motion.shape}")
    diff = np.vstack([np.zeros((1, 6)), np.diff(motion, axis=0)])
    rot_mm = np.abs(diff[:, :3]) * (math.pi / 180.0) * HEAD_RADIUS_MM
    trans_mm = np.abs(diff[:, 3:6])
    return rot_mm.sum(axis=1) + trans_mm.sum(axis=1)


def make_nonzero_mask(bold_path: Path, out_path: Path) -> tuple[Path, np.ndarray]:
    img = nib.load(bold_path)
    data = np.asanyarray(img.dataobj)
    if data.ndim != 4:
        raise RuntimeError(f"BOLD is not 4D: {bold_path} {data.shape}")
    mean = np.nanmean(data, axis=3)
    mask = np.isfinite(mean) & (mean != 0)
    if int(mask.sum()) == 0:
        raise RuntimeError(f"Zero valid voxels in {bold_path}")
    hdr = img.header.copy()
    hdr.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), img.affine, hdr), out_path)
    return out_path, mask


def load_events(artifact_root: Path, participant: str, session: str, run: int, recon: dict) -> tuple[pd.DataFrame, str]:
    pattern = f"{participant}_{session}_task-SORPF_run-{run}_events.tsv"
    p = find_single(artifact_root, pattern)
    rel_suffix = f"{participant}/{session}/func/{pattern}"
    matches = [x for x in recon.get("files", []) if str(x.get("path", "")).endswith(rel_suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"Missing exact reconstruction record for {rel_suffix}")
    got = sha256_file(p)
    if got != matches[0].get("hash"):
        raise RuntimeError(f"Event hash mismatch for {pattern}")
    df = pd.read_csv(p, sep="\t")
    for col in ("onset", "duration", "trial_type"):
        if col not in df:
            raise RuntimeError(f"Missing {col} in {p}")
    unknown = sorted(set(df["trial_type"].astype(str)) - set(TRIAL_TYPES))
    if unknown:
        raise RuntimeError(f"Unexpected trial types in {p}: {unknown}")
    return df[["onset", "duration", "trial_type"]].copy(), got


def holm_adjust(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pvals[idx])
        running = max(running, val)
        adjusted[idx] = running
    return adjusted


def participant_mode(args) -> int:
    participant = args.participant
    if participant not in EXPECTED_PARTICIPANTS:
        raise RuntimeError(f"Unexpected participant {participant}")
    preproc_root = Path(args.preproc_root)
    prov_root = Path(args.prov_root)
    roi_root = Path(args.roi_root)
    outroot = Path(args.output_root) / participant
    outroot.mkdir(parents=True, exist_ok=True)

    roi_paths = load_and_verify_roi_masks(roi_root)
    pattern = f"{participant}_ses-*_task-SORPF_run-*_space-MNI152NLin2009cAsym_res-02_desc-optcom_bold.nii.gz"
    mni_candidates = sorted(preproc_root.rglob(pattern))
    if not mni_candidates:
        raise RuntimeError(f"No V2 MNI optcom files found for {participant} under {preproc_root}")
    any_mni = mni_candidates[0]
    artifact_root = artifact_root_for(any_mni, participant)
    prep = verify_preprocessing_root(artifact_root, prov_root, participant)
    recon = prep["reconstruction"]
    mni_files = prep["mni_files"]

    by_session: dict[str, list[tuple[int, Path]]] = {}
    for p in mni_files:
        pp, ses, run = parse_entities(p.name)
        if pp != participant:
            raise RuntimeError("Participant entity mismatch")
        by_session.setdefault(ses, []).append((run, p))

    run_results = []
    run_effect_paths = []
    run_mask_arrays = []
    motion_provenance = {}

    for ses, runs in sorted(by_session.items()):
        runs = sorted(runs)
        motion_all = load_motion_session(artifact_root, ses)
        nvols = [int(nib.load(p).shape[3]) for _, p in runs]
        if motion_all.shape[0] != sum(nvols):
            raise RuntimeError(f"{participant} {ses}: motion rows {motion_all.shape[0]} != total volumes {sum(nvols)}")
        start = 0
        for (run_num, bold_path), nt in zip(runs, nvols):
            motion = motion_all[start:start+nt]
            start += nt
            run_id = f"{participant}_{ses}_run-{run_num}"
            run_dir = outroot / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            mask_path, mask_arr = make_nonzero_mask(bold_path, run_dir / "analysis_mask.nii.gz")
            fd = power_fd(motion)
            std_dvars, _, _ = compute_dvars(
                str(bold_path),
                str(mask_path),
                remove_zerovariance=True,
                intensity_normalization=1000,
                variance_tol=1e-7,
            )
            std_dvars = np.concatenate([[0.0], np.asarray(std_dvars, dtype=float)])
            if len(std_dvars) != nt:
                raise RuntimeError(f"{run_id}: standardized DVARS length {len(std_dvars)} != {nt}")
            censor = (fd > FD_THRESHOLD) | (std_dvars > STDDVARS_THRESHOLD)

            conf = pd.DataFrame(
                motion,
                columns=["roll_deg","pitch_deg","yaw_deg","dS_mm","dL_mm","dP_mm"],
            )
            for idx in np.flatnonzero(censor):
                spike = np.zeros(nt, dtype=float)
                spike[idx] = 1.0
                conf[f"outlier_{idx:04d}"] = spike
            conf.to_csv(run_dir / "confounds.tsv", sep="\t", index=False)
            pd.DataFrame({"fd_mm":fd, "std_dvars":std_dvars, "censor":censor.astype(int)}).to_csv(
                run_dir / "motion_qc.tsv", sep="\t", index=False
            )

            events, events_sha = load_events(artifact_root, participant, ses, run_num, recon)
            counts = {t:int((events["trial_type"] == t).sum()) for t in TRIAL_TYPES}
            if counts["Self"] <= 0 or counts["Other"] <= 0:
                raise RuntimeError(f"{run_id}: missing target events")
            events.to_csv(run_dir / "events_model.tsv", sep="\t", index=False)

            model = FirstLevelModel(
                t_r=TR,
                slice_time_ref=0.0,
                hrf_model="spm",
                drift_model="cosine",
                high_pass=HIGH_PASS,
                mask_img=str(mask_path),
                smoothing_fwhm=None,
                standardize="psc",
                signal_scaling=False,
                noise_model="ar1",
                n_jobs=2,
                minimize_memory=True,
            )
            model = model.fit(str(bold_path), events=events, confounds=conf)
            design = model.design_matrices_[0]
            design.to_csv(run_dir / "design_matrix.tsv", sep="\t", index=False)
            if "Self" not in design.columns or "Other" not in design.columns:
                raise RuntimeError(f"{run_id}: Self/Other absent from design columns {list(design.columns)}")
            contrast = np.zeros(design.shape[1], dtype=float)
            contrast[design.columns.get_loc("Self")] = 1.0
            contrast[design.columns.get_loc("Other")] = -1.0
            effect_img = model.compute_contrast(contrast, output_type="effect_size")
            effect_path = run_dir / "contrast-SelfMinusOther_effect_size.nii.gz"
            effect_img.to_filename(effect_path)

            effect = np.asanyarray(effect_img.dataobj)
            roi_run = {}
            for roi_name, roi_path in roi_paths.items():
                roi = np.asanyarray(nib.load(roi_path).dataobj) > 0
                valid = roi & mask_arr & np.isfinite(effect)
                nvalid = int(valid.sum())
                roi_run[roi_name] = {
                    "effect_mean_psc": float(np.mean(effect[valid])) if nvalid else None,
                    "valid_voxels": nvalid,
                    "total_roi_voxels": int(ROI_SPECS[roi_name]["total_voxels"]),
                    "coverage_fraction": nvalid / float(ROI_SPECS[roi_name]["total_voxels"]),
                }

            rank = int(np.linalg.matrix_rank(design.to_numpy(dtype=float)))
            run_results.append({
                "participant":participant,
                "session":ses,
                "run":run_num,
                "bold_file":bold_path.name,
                "bold_sha256":sha256_file(bold_path),
                "events_sha256":events_sha,
                "n_volumes":nt,
                "event_counts":counts,
                "fd_gt_0p5_count":int((fd > FD_THRESHOLD).sum()),
                "stddvars_gt_1p5_count":int((std_dvars > STDDVARS_THRESHOLD).sum()),
                "censored_union_count":int(censor.sum()),
                "design_columns":list(design.columns),
                "design_rank":rank,
                "design_n_columns":int(design.shape[1]),
                "roi_run_effects":roi_run,
                "effect_map_sha256":sha256_file(effect_path),
            })
            run_effect_paths.append(effect_path)
            run_mask_arrays.append(mask_arr)

        motion_file = sorted(artifact_root.rglob(f"afni/{ses}/results/dfile*.1D"))
        if len(motion_file) == 1:
            motion_provenance[ses] = {
                "path":str(motion_file[0].relative_to(artifact_root)),
                "sha256":sha256_file(motion_file[0]),
                "rows":int(motion_all.shape[0]),
                "columns":int(motion_all.shape[1]),
            }

    if len(run_effect_paths) != len(mni_files):
        raise RuntimeError("Run effect count mismatch")
    ref = nib.load(run_effect_paths[0])
    effects = [np.asanyarray(nib.load(p).dataobj, dtype=np.float32) for p in run_effect_paths]
    common_mask = np.logical_and.reduce(run_mask_arrays)
    participant_effect = np.full(effects[0].shape, np.nan, dtype=np.float32)
    participant_effect[common_mask] = np.mean(np.stack([x[common_mask] for x in effects], axis=0), axis=0)
    part_effect_path = outroot / f"{participant}_contrast-SelfMinusOther_desc-equalRunMean_effect_size.nii.gz"
    nib.save(nib.Nifti1Image(participant_effect, ref.affine, ref.header), part_effect_path)

    part_roi = {}
    for roi_name, roi_path in roi_paths.items():
        roi = np.asanyarray(nib.load(roi_path).dataobj) > 0
        valid = roi & common_mask & np.isfinite(participant_effect)
        nvalid = int(valid.sum())
        part_roi[roi_name] = {
            "effect_mean_psc": float(np.mean(participant_effect[valid])) if nvalid else None,
            "valid_voxels":nvalid,
            "total_roi_voxels":int(ROI_SPECS[roi_name]["total_voxels"]),
            "coverage_fraction":nvalid / float(ROI_SPECS[roi_name]["total_voxels"]),
        }

    result = {
        "status":"P08_OPEN_SOURCE_V2_PARTICIPANT_SELF_OTHER_GLM_COMPLETE",
        "participant":participant,
        "lane":"open_source_secondary_sensitivity_only",
        "glm_lock":GLM_LOCK_FILE,
        "n_runs":len(run_results),
        "run_results":run_results,
        "motion_provenance":motion_provenance,
        "participant_combination":"equal-weight arithmetic mean of all frozen run effect-size maps",
        "participant_roi_effects":part_roi,
        "participant_effect_map":part_effect_path.name,
        "participant_effect_map_sha256":sha256_file(part_effect_path),
        "primary_lane_modified":False,
    }
    (outroot / "participant_glm_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


def aggregate_mode(args) -> int:
    root = Path(args.participant_results_root)
    outroot = Path(args.output_root)
    outroot.mkdir(parents=True, exist_ok=True)
    files = sorted(root.rglob("participant_glm_result.json"))
    results = [json.loads(p.read_text(encoding="utf-8")) for p in files]
    byp = {r["participant"]:r for r in results}
    if set(byp) != set(EXPECTED_PARTICIPANTS):
        raise RuntimeError(f"Expected exactly {EXPECTED_PARTICIPANTS}, got {sorted(byp)}")
    group = {}
    raw_ps = []
    names = list(ROI_SPECS)
    for roi_name in names:
        vals = [byp[p]["participant_roi_effects"][roi_name]["effect_mean_psc"] for p in EXPECTED_PARTICIPANTS]
        if any(v is None for v in vals):
            pval = None
        else:
            nonzero = [v for v in vals if v != 0]
            if nonzero:
                positives = sum(v > 0 for v in nonzero)
                pval = float(binomtest(positives, len(nonzero), 0.5, alternative="two-sided").pvalue)
            else:
                pval = 1.0
        raw_ps.append(1.0 if pval is None else pval)
        arr = np.array([np.nan if v is None else v for v in vals], dtype=float)
        group[roi_name] = {
            "participant_order":list(EXPECTED_PARTICIPANTS),
            "participant_effects_psc":vals,
            "mean_psc":float(np.nanmean(arr)),
            "median_psc":float(np.nanmedian(arr)),
            "sample_sd_psc":float(np.nanstd(arr, ddof=1)) if np.isfinite(arr).sum() > 1 else None,
            "exact_two_sided_sign_test_p":pval,
        }
    adj = holm_adjust(raw_ps)
    for roi_name, p_adj in zip(names, adj):
        group[roi_name]["holm_adjusted_p_two_rois"] = p_adj

    result = {
        "status":"P08_OPEN_SOURCE_V2_SELF_OTHER_GLM_ROI_COMPLETE",
        "lane":"open_source_secondary_sensitivity_only",
        "independent_unit":"participant",
        "n_participants":3,
        "group_results":group,
        "minimum_attainable_two_sided_sign_p_with_n3":0.25,
        "interpretation_limit":"Secondary low-power sensitivity analysis; not the frozen primary source-consistent result and not decisive confirmation or refutation.",
        "primary_lane_modified":False,
    }
    (outroot / "group_glm_roi_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    rows=[]
    for p in EXPECTED_PARTICIPANTS:
        for roi_name in names:
            rec=byp[p]["participant_roi_effects"][roi_name]
            rows.append({
                "participant":p,
                "roi":roi_name,
                "effect_mean_psc":rec["effect_mean_psc"],
                "valid_voxels":rec["valid_voxels"],
                "total_roi_voxels":rec["total_roi_voxels"],
                "coverage_fraction":rec["coverage_fraction"],
            })
    pd.DataFrame(rows).to_csv(outroot/"participant_roi_effects.tsv",sep="\t",index=False)
    print(json.dumps(result, indent=2))
    return 0


def main() -> int:
    ap=argparse.ArgumentParser()
    sub=ap.add_subparsers(dest="mode",required=True)
    p=sub.add_parser("participant")
    p.add_argument("--participant",required=True,choices=EXPECTED_PARTICIPANTS)
    p.add_argument("--preproc-root",required=True)
    p.add_argument("--prov-root",required=True)
    p.add_argument("--roi-root",required=True)
    p.add_argument("--output-root",required=True)
    a=sub.add_parser("aggregate")
    a.add_argument("--participant-results-root",required=True)
    a.add_argument("--output-root",required=True)
    args=ap.parse_args()
    return participant_mode(args) if args.mode=="participant" else aggregate_mode(args)


if __name__=="__main__":
    raise SystemExit(main())
