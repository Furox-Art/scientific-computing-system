#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm.first_level import FirstLevelModel
from nipype.algorithms.confounds import FramewiseDisplacement, compute_dvars

SOURCE_COMMIT = "2e273d8466162208bbccd8591337e55b7a8b5721"
AFNI_IMAGE = "afni/afni_make_build@sha256:c8365868751116f1ef2d0811e8a4dd2df58a689da457c74ea07fedb0e11dc0d6"
TR = 0.8
FD_THRESHOLD = 0.5
STDDVARS_THRESHOLD = 1.5
MIN_RETAINED_FRACTION = 0.8
MNI_SPACE = "MNI152NLin6Asym"
MNI_RESOLUTION = 2
EXPECTED_NETWORK_SHA = {
    "network-affective_HOSPA-bilateralAmygdala_mask.nii.gz": "01d1db9d7de763aea6f0e004e7d230653febcf6192c25367b2974c4cd84d61fd",
    "network-control_HOCPA-MFG-ACC_mask.nii.gz": "77286594c141dbdbca02a34100704a1c935a111c9d6b84419acbfda24b128b3d",
    "tpl-MNI152NLin6Asym_res-02_T1w.nii.gz": "2a814da50173599a857d96246dc057d548072bd6dffa499f75724dbad20792b1",
}

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def hash_file(path: Path, alg: str) -> str:
    h = hashlib.new(alg)
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def run(cmd, *, cwd=None, env=None, log: Path | None = None):
    cmd = [str(x) for x in cmd]
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(
        cmd, cwd=str(cwd) if cwd else None, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, check=False,
    )
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as f:
            f.write("+ " + " ".join(cmd) + "\n")
            f.write(proc.stdout or "")
            if not (proc.stdout or "").endswith("\n"):
                f.write("\n")
    else:
        print(proc.stdout or "", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc.stdout or ""

def cpath(part_root: Path, path: Path) -> str:
    return "/work/" + path.resolve().relative_to(part_root.resolve()).as_posix()

def docker_afni(part_root: Path, args, *, log: Path):
    home = part_root / "docker_home"
    home.mkdir(parents=True, exist_ok=True)
    cmd = [
        "docker", "run", "--rm",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/work/docker_home",
        "-v", f"{part_root.resolve()}:/work",
        AFNI_IMAGE,
    ] + [str(x) for x in args]
    return run(cmd, log=log)

def read_manifest(path: Path, participant: str) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["participant_id"] == participant]
    if len(rows) != 18:
        raise RuntimeError(f"{participant}: expected 18 frozen manifest rows, got {len(rows)}")
    return rows

def reconstruct_exact_input(source_root: Path, manifest_path: Path, participant: str, part_root: Path, log: Path):
    rows = read_manifest(manifest_path, participant)
    bids = part_root / "bids"
    details = []
    for rec in rows:
        rel = Path(rec["path"])
        dest = bids / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        storage = rec["storage"]
        expected_bytes = int(rec["bytes"])
        alg = rec["hash_algorithm"].strip().lower()
        expected_hash = rec["hash"].strip().lower()
        if storage == "git-blob":
            src = source_root / rel
            if not src.is_file() or src.is_symlink():
                raise RuntimeError(f"Git-blob source is not a regular file: {src}")
            shutil.copy2(src, dest)
        elif storage == "git-annex":
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            s3 = rec["s3_url"]
            run(["aws", "s3", "cp", "--no-sign-request", "--only-show-errors", s3, str(dest)], log=log)
        else:
            raise RuntimeError(f"Unsupported storage {storage}: {rel}")

        if not dest.is_file():
            raise RuntimeError(f"Missing reconstructed input {dest}")
        if dest.stat().st_size != expected_bytes:
            raise RuntimeError(f"Byte mismatch {rel}: {dest.stat().st_size} != {expected_bytes}")
        got = hash_file(dest, alg)
        if got.lower() != expected_hash:
            raise RuntimeError(f"Hash mismatch {rel}: {got} != {expected_hash}")
        details.append({
            "path": rel.as_posix(),
            "storage": storage,
            "bytes": expected_bytes,
            "hash_algorithm": alg,
            "hash": expected_hash,
        })
    out = {
        "status": "DS005237_EXACT_PARTICIPANT_INPUT_RECONSTRUCTED",
        "participant": participant,
        "source_commit": SOURCE_COMMIT,
        "files": details,
        "file_count": len(details),
        "bytes": sum(x["bytes"] for x in details),
        "all_exact_verified": True,
        "neural_group_contrast_computed": False,
    }
    (part_root / "input_reconstruction.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return bids, out

def find_single(root: Path, pattern: str) -> Path:
    xs = sorted(root.glob(pattern))
    if len(xs) != 1:
        raise RuntimeError(f"Expected one {pattern} under {root}, got {xs}")
    return xs[0]

def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def verify_network_resources(network_root: Path):
    found = {}
    for name, expected in EXPECTED_NETWORK_SHA.items():
        xs = sorted(network_root.rglob(name))
        if len(xs) != 1:
            raise RuntimeError(f"Expected one frozen resource {name}, got {xs}")
        p = xs[0]
        got = sha256_file(p)
        if got != expected:
            raise RuntimeError(f"Frozen resource hash mismatch {name}: {got} != {expected}")
        found[name] = p
    return found

def normalize_anatomy(bids: Path, participant: str, part_root: Path, template: Path, log: Path):
    t1 = find_single(bids / participant / "anat", "*_T1w.nii.gz")
    anat = part_root / "ants"
    anat.mkdir(parents=True, exist_ok=True)
    t1_n4 = anat / "T1w_N4.nii.gz"
    run(["N4BiasFieldCorrection", "-d", "3", "-i", str(t1), "-o", str(t1_n4)], log=log)
    prefix = anat / "T1_to_MNI_"
    env = os.environ.copy()
    env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "2"
    run([
        "antsRegistrationSyNQuick.sh", "-d", "3",
        "-f", str(template), "-m", str(t1_n4),
        "-o", str(prefix), "-t", "s",
    ], env=env, log=log)
    affine = Path(str(prefix) + "0GenericAffine.mat")
    warp = Path(str(prefix) + "1Warp.nii.gz")
    warped = Path(str(prefix) + "Warped.nii.gz")
    for p in (t1_n4, affine, warp, warped):
        if not p.is_file():
            raise RuntimeError(f"Missing ANTs anatomy output {p}")
    return t1_n4, affine, warp

def task_paths(bids: Path, participant: str, task: str):
    func = bids / participant / "func"
    stem = func / f"{participant}_task-{task}_run-01"
    bold = Path(str(stem) + "_bold.nii.gz")
    js = Path(str(stem) + "_bold.json")
    events = Path(str(stem) + "_events.tsv")
    for p in (bold, js, events):
        if not p.is_file():
            raise RuntimeError(f"Missing task input {p}")
    return bold, js, events

def fmap_paths(bids: Path, participant: str, task_pe: str):
    fmap = bids / participant / "fmap"
    ap = fmap / f"{participant}_dir-ap_epi.nii.gz"
    pa = fmap / f"{participant}_dir-pa_epi.nii.gz"
    for p in (ap, pa):
        if not p.is_file():
            raise RuntimeError(f"Missing fmap {p}")
    if task_pe == "j-":
        return ap, pa
    if task_pe == "j":
        return pa, ap
    raise RuntimeError(f"Unsupported PhaseEncodingDirection {task_pe}")

def slice_timing_correct(part_root: Path, bold: Path, sidecar: dict, task: str, log: Path):
    timing = sidecar.get("SliceTiming")
    if not isinstance(timing, list) or not timing:
        raise RuntimeError(f"{task}: missing SliceTiming")
    if abs(float(sidecar.get("RepetitionTime")) - TR) > 1e-7:
        raise RuntimeError(f"{task}: TR mismatch")
    outdir = part_root / "tshift" / task
    outdir.mkdir(parents=True, exist_ok=True)
    timing_file = outdir / "slice_timing.1D"
    timing_file.write_text(" ".join(f"{float(x):.8f}" for x in timing) + "\n", encoding="utf-8")
    shifted = outdir / "bold_tshift.nii.gz"
    docker_afni(part_root, [
        "3dTshift", "-overwrite", "-wsinc9", "-tzero", "0",
        "-tpattern", "@" + cpath(part_root, timing_file),
        "-prefix", cpath(part_root, shifted),
        cpath(part_root, bold),
    ], log=log)
    if not shifted.is_file():
        raise RuntimeError(f"{task}: missing slice-time corrected output")
    return shifted

def run_afni_task(part_root: Path, participant: str, task: str, shifted: Path, t1_n4: Path, fwd: Path, rev: Path, log: Path):
    task_root = part_root / "afni" / task
    results = task_root / "results"
    task_root.mkdir(parents=True, exist_ok=True)
    subj = f"{participant.replace('sub-','')}_{task}"
    proc = task_root / f"proc.{subj}"
    cmd = [
        "afni_proc.py",
        "-subj_id", subj,
        "-script", cpath(part_root, proc),
        "-scr_overwrite",
        "-out_dir", cpath(part_root, results),
        "-copy_anat", cpath(part_root, t1_n4),
        "-anat_has_skull", "yes",
        "-dsets", cpath(part_root, shifted),
        "-blocks", "blip", "align", "volreg", "mask",
        "-tcat_remove_first_trs", "0",
        "-blip_forward_dset", cpath(part_root, fwd),
        "-blip_reverse_dset", cpath(part_root, rev),
        "-align_unifize_epi", "local",
        "-align_opts_aea", "-cost", "lpc+ZZ", "-rigid_body", "-giant_move", "-check_flip",
        "-volreg_align_to", "MIN_OUTLIER",
        "-volreg_align_e2a",
        "-volreg_warp_final_interp", "wsinc5",
        "-mask_epi_anat", "yes",
        "-html_review_style", "pythonic",
        "-execute",
    ]
    docker_afni(part_root, cmd, log=log)

    heads = sorted(results.glob("pb*.r01.volreg+orig.HEAD"))
    if len(heads) != 1:
        raise RuntimeError(f"{task}: expected one single-echo volreg dataset, got {heads}")
    mask_heads = sorted(results.glob("full_mask.*+orig.HEAD"))
    if len(mask_heads) != 1:
        raise RuntimeError(f"{task}: expected one full mask, got {mask_heads}")
    motion = results / "dfile_rall.1D"
    if not motion.is_file():
        raise RuntimeError(f"{task}: missing dfile_rall.1D")

    nifti_dir = part_root / "native" / task
    nifti_dir.mkdir(parents=True, exist_ok=True)
    volreg = nifti_dir / "bold_desc-afniPreproc.nii.gz"
    mask = nifti_dir / "mask.nii.gz"
    docker_afni(part_root, [
        "3dAFNItoNIFTI", "-overwrite", "-prefix", cpath(part_root, volreg), cpath(part_root, heads[0])
    ], log=log)
    docker_afni(part_root, [
        "3dAFNItoNIFTI", "-overwrite", "-prefix", cpath(part_root, mask), cpath(part_root, mask_heads[0])
    ], log=log)
    if not volreg.is_file() or not mask.is_file():
        raise RuntimeError(f"{task}: failed to convert AFNI outputs")
    return volreg, mask, motion, results

def compute_qc(volreg: Path, mask: Path, motion: Path, task: str, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    # Exact Nipype implementation of Power FD with AFNI parameter normalization.
    fd_iface = FramewiseDisplacement(
        in_file=str(motion),
        parameter_source="AFNI",
        radius=50,
        save_plot=False,
        out_file=str((outdir / "fd_power_2012.txt").resolve()),
    )
    fd_res = fd_iface.run()
    fd_path = Path(fd_res.outputs.out_file)
    with fd_path.open("r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    fd_skiprows = 1 if first_line == "FramewiseDisplacement" else 0
    fd_raw = np.atleast_1d(np.loadtxt(fd_path, dtype=float, skiprows=fd_skiprows))
    # FD is defined on temporal differences, so prepend 0 for the first volume.
    fd = np.concatenate([[0.0], fd_raw])

    std_dvars, _, _ = compute_dvars(
        str(volreg), str(mask),
        remove_zerovariance=True,
        variance_tol=1e-7,
        intensity_normalization=1000.0,
    )
    std_dvars = np.asarray(std_dvars, dtype=float)
    dvars = np.concatenate([[0.0], std_dvars])

    nvol = int(nib.load(volreg).shape[3])
    if len(fd) != nvol or len(dvars) != nvol:
        raise RuntimeError(f"{task}: QC length mismatch nvol={nvol} fd={len(fd)} dvars={len(dvars)}")
    censor = (fd > FD_THRESHOLD) | (dvars > STDDVARS_THRESHOLD)
    retained_fraction = float((~censor).mean())

    qcdf = pd.DataFrame({
        "volume": np.arange(nvol, dtype=int),
        "fd_mm": fd,
        "std_dvars": dvars,
        "censored": censor.astype(int),
    })
    qcdf.to_csv(outdir / "motion_qc.tsv", sep="\t", index=False)

    mot = np.loadtxt(motion, dtype=float)
    if mot.ndim == 1:
        mot = mot[None, :]
    if mot.shape != (nvol, 6):
        raise RuntimeError(f"{task}: AFNI motion shape {mot.shape}, expected {(nvol,6)}")
    derivatives = np.vstack([np.zeros((1, 6)), np.diff(mot, axis=0)])
    conf = pd.DataFrame(
        np.column_stack([mot, derivatives]),
        columns=[
            "roll","pitch","yaw","dS","dL","dP",
            "roll_derivative","pitch_derivative","yaw_derivative",
            "dS_derivative","dL_derivative","dP_derivative",
        ],
    )
    for idx in np.flatnonzero(censor):
        v = np.zeros(nvol, dtype=float)
        v[idx] = 1.0
        conf[f"spike_{idx:04d}"] = v
    conf.to_csv(outdir / "confounds.tsv", sep="\t", index=False)

    return {
        "n_volumes": nvol,
        "n_censored": int(censor.sum()),
        "retained_fraction": retained_fraction,
        "mean_fd_mm": float(fd.mean()),
        "max_fd_mm": float(fd.max()),
        "mean_std_dvars": float(dvars.mean()),
        "max_std_dvars": float(dvars.max()),
        "passed_retained_fraction": retained_fraction >= MIN_RETAINED_FRACTION,
    }, conf

def prepare_events(events_file: Path, task: str, outdir: Path):
    d = pd.read_csv(events_file, sep="\t", na_values=["n/a","NA","NaN"])
    if task == "hammerAP":
        types = d["trial_type"].astype(str)
        if not set(types.unique()) <= {"face","shape"}:
            raise RuntimeError(f"Hammer unexpected trial types: {sorted(types.unique())}")
        ev = pd.DataFrame({
            "onset": pd.to_numeric(d["onset"], errors="raise"),
            "duration": pd.to_numeric(d["duration"], errors="raise"),
            "trial_type": types,
        })
        contrast = "face - shape"
    elif task in ("stroopAP","stroopPA"):
        types = d["trial_type"].astype(str)
        ok = types.str.startswith("con") | types.str.startswith("inc")
        if not bool(ok.all()):
            raise RuntimeError(f"{task}: unexpected trial types {sorted(types[~ok].unique())}")
        labels = np.where(types.str.startswith("inc"), "incongruent", "congruent")
        ev = pd.DataFrame({
            "onset": pd.to_numeric(d["onset"], errors="raise"),
            "duration": np.zeros(len(d), dtype=float),
            "trial_type": labels,
        })
        contrast = "incongruent - congruent"
    else:
        raise RuntimeError(task)
    ev.to_csv(outdir / "events_model.tsv", sep="\t", index=False)
    return ev, contrast

def fit_task_glm(volreg: Path, mask: Path, events_file: Path, task: str, confounds: pd.DataFrame, outdir: Path):
    events, contrast = prepare_events(events_file, task, outdir)
    model = FirstLevelModel(
        t_r=TR,
        slice_time_ref=0.0,
        hrf_model="spm",
        drift_model="cosine",
        high_pass=0.0078125,
        noise_model="ar1",
        mask_img=str(mask),
        smoothing_fwhm=None,
        signal_scaling=0,
        minimize_memory=True,
        standardize=False,
    )
    model = model.fit(run_imgs=str(volreg), events=events, confounds=confounds)
    dm = model.design_matrices_[0]
    dm.to_csv(outdir / "design_matrix.tsv", sep="\t", index=False)
    effect = model.compute_contrast(contrast, output_type="effect_size")
    native = outdir / "contrast_effect_size_native.nii.gz"
    effect.to_filename(native)
    return native, contrast, list(dm.columns)

def warp_contrast(native: Path, task: str, part_root: Path, template: Path, affine: Path, warp: Path, log: Path):
    out = part_root / "mni" / f"{task}_space-{MNI_SPACE}_res-02_contrast-effect_size.nii.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "2"
    run([
        "antsApplyTransforms", "-d", "3",
        "-i", str(native), "-r", str(template), "-o", str(out),
        "-n", "Linear", "-t", str(warp), "-t", str(affine),
    ], env=env, log=log)
    if not out.is_file():
        raise RuntimeError(f"{task}: missing MNI contrast")
    return out

def warp_coverage_mask(native_mask: Path, task: str, part_root: Path, template: Path, affine: Path, warp: Path, log: Path):
    out = part_root / "mni" / f"{task}_space-{MNI_SPACE}_res-02_coverage-mask.nii.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "2"
    run([
        "antsApplyTransforms", "-d", "3",
        "-i", str(native_mask), "-r", str(template), "-o", str(out),
        "-n", "NearestNeighbor", "-t", str(warp), "-t", str(affine),
    ], env=env, log=log)
    if not out.is_file():
        raise RuntimeError(f"{task}: missing MNI coverage mask")
    return out

def network_mean(image: Path, mask: Path, coverage_mask: Path):
    img = nib.load(image)
    m = nib.load(mask)
    c = nib.load(coverage_mask)
    for other in (m, c):
        if img.shape[:3] != other.shape[:3] or not np.allclose(img.affine, other.affine, atol=1e-5):
            raise RuntimeError(f"Grid mismatch {image} vs {other.get_filename()}")
    x = np.asanyarray(img.dataobj, dtype=np.float64)
    mm = np.asanyarray(m.dataobj) > 0
    covered = np.asanyarray(c.dataobj) > 0.5
    use = mm & covered & np.isfinite(x)
    vals = x[use]
    total = int(mm.sum())
    n_covered = int(use.sum())
    fraction = float(n_covered / total) if total else 0.0
    if vals.size == 0:
        return math.nan, n_covered, total, fraction
    return float(vals.mean()), n_covered, total, fraction

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", required=True)
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--network-root", required=True)
    ap.add_argument("--output-root", required=True)
    args = ap.parse_args()

    participant = args.participant
    source_root = Path(args.source_root)
    manifest = Path(args.manifest)
    network_root = Path(args.network_root)
    root = Path(args.output_root) / participant
    root.mkdir(parents=True, exist_ok=True)
    log = root / "participant_neural.log"
    result = {
        "status": "STARTED",
        "participant": participant,
        "source_commit": SOURCE_COMMIT,
        "design_lock": "NEURAL_DOUBLE_DISSOCIATION_LOCK_2026-09-19.json",
        "afni_image": AFNI_IMAGE,
        "neural_group_contrast_computed": False,
    }

    try:
        resources = verify_network_resources(network_root)
        template = resources["tpl-MNI152NLin6Asym_res-02_T1w.nii.gz"]
        affective_mask = resources["network-affective_HOSPA-bilateralAmygdala_mask.nii.gz"]
        control_mask = resources["network-control_HOCPA-MFG-ACC_mask.nii.gz"]

        bids, recon = reconstruct_exact_input(source_root, manifest, participant, root, log)
        result["exact_input"] = {
            "file_count": recon["file_count"],
            "bytes": recon["bytes"],
            "all_exact_verified": True,
        }

        t1_n4, affine, warp = normalize_anatomy(bids, participant, root, template, log)
        result["anatomy"] = {
            "affine_sha256": sha256_file(affine),
            "warp_sha256": sha256_file(warp),
        }

        task_results = {}
        for task in ("hammerAP", "stroopAP", "stroopPA"):
            bold, js, events_file = task_paths(bids, participant, task)
            sidecar = load_json(js)
            pe = sidecar.get("PhaseEncodingDirection")
            expected_pe = {"hammerAP": "j-", "stroopAP": "j-", "stroopPA": "j"}[task]
            if pe != expected_pe:
                raise RuntimeError(f"{task}: PE {pe} != frozen {expected_pe}")
            if abs(float(sidecar.get("EchoTime")) - 0.037) > 1e-7:
                raise RuntimeError(f"{task}: TE mismatch")
            fwd, rev = fmap_paths(bids, participant, pe)
            shifted = slice_timing_correct(root, bold, sidecar, task, log)
            volreg, mask, motion, _ = run_afni_task(root, participant, task, shifted, t1_n4, fwd, rev, log)

            task_out = root / "task_results" / task
            task_out.mkdir(parents=True, exist_ok=True)
            qc, confounds = compute_qc(volreg, mask, motion, task, task_out)
            if not qc["passed_retained_fraction"]:
                task_results[task] = {"qc": qc, "eligible": False}
                continue

            native, contrast, design_cols = fit_task_glm(volreg, mask, events_file, task, confounds, task_out)
            mni = warp_contrast(native, task, root, template, affine, warp, log)
            coverage_mni = warp_coverage_mask(mask, task, root, template, affine, warp, log)
            aff_mean, aff_valid, aff_total, aff_fraction = network_mean(mni, affective_mask, coverage_mni)
            ctl_mean, ctl_valid, ctl_total, ctl_fraction = network_mean(mni, control_mask, coverage_mni)
            eligible = (
                aff_valid > 0 and ctl_valid > 0 and
                math.isfinite(aff_mean) and math.isfinite(ctl_mean)
            )
            task_results[task] = {
                "qc": qc,
                "eligible": eligible,
                "contrast": contrast,
                "design_columns": design_cols,
                "mni_contrast": mni.relative_to(root).as_posix(),
                "mni_contrast_sha256": sha256_file(mni),
                "mni_coverage_mask": coverage_mni.relative_to(root).as_posix(),
                "mni_coverage_mask_sha256": sha256_file(coverage_mni),
                "affective_mean_psc": aff_mean,
                "affective_valid_voxels": aff_valid,
                "affective_total_voxels": aff_total,
                "affective_coverage_fraction": aff_fraction,
                "control_mean_psc": ctl_mean,
                "control_valid_voxels": ctl_valid,
                "control_total_voxels": ctl_total,
                "control_coverage_fraction": ctl_fraction,
            }

            # Drop the multi-GB reconstructed raw BOLD and heavy AFNI 4D datasets after
            # the locked participant endpoint is materialized.
            try:
                bold.unlink()
            except FileNotFoundError:
                pass
            shutil.rmtree(root / "tshift" / task, ignore_errors=True)
            shutil.rmtree(root / "native" / task, ignore_errors=True)
            shutil.rmtree(root / "afni" / task, ignore_errors=True)

        result["tasks"] = task_results
        required = ("hammerAP", "stroopAP", "stroopPA")
        all_eligible = all(task_results.get(x, {}).get("eligible", False) for x in required)
        result["participant_endpoint_eligible"] = all_eligible

        if all_eligible:
            hammer_aff = task_results["hammerAP"]["affective_mean_psc"]
            hammer_ctl = task_results["hammerAP"]["control_mean_psc"]
            stroop_aff = float(np.mean([
                task_results["stroopAP"]["affective_mean_psc"],
                task_results["stroopPA"]["affective_mean_psc"],
            ]))
            stroop_ctl = float(np.mean([
                task_results["stroopAP"]["control_mean_psc"],
                task_results["stroopPA"]["control_mean_psc"],
            ]))
            result["endpoints"] = {
                "hammer_affective": hammer_aff,
                "hammer_control": hammer_ctl,
                "stroop_affective": stroop_aff,
                "stroop_control": stroop_ctl,
                "hammer_selectivity": hammer_aff - hammer_ctl,
                "stroop_selectivity": stroop_ctl - stroop_aff,
            }
            result["status"] = "DS005237_NEURAL_PARTICIPANT_ENDPOINT_COMPLETE"
        else:
            result["endpoints"] = None
            result["status"] = "DS005237_NEURAL_PARTICIPANT_QC_INELIGIBLE"

        result["neural_group_contrast_computed"] = False
        out = root / "participant_neural_result.json"
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0
    except Exception as e:
        result.update({
            "status": "DS005237_NEURAL_PARTICIPANT_TECHNICAL_FAILURE",
            "error": {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            },
            "neural_group_contrast_computed": False,
        })
        (root / "participant_neural_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
