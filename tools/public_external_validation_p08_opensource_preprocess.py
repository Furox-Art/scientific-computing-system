#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

AFNI_IMAGE = "afni/afni_make_build@sha256:c8365868751116f1ef2d0811e8a4dd2df58a689da457c74ea07fedb0e11dc0d6"
OPEN_ROOT = Path(".public-runner/p08/opensource-v2")
PARTICIPANTS = {"sub-Bubbles", "sub-Buttercup", "sub-PILOT02"}
MNI_SPACE = "MNI152NLin2009cAsym"
MNI_RESOLUTION = 2


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(cmd, *, cwd=None, env=None, log=None):
    cmd = [str(x) for x in cmd]
    print("+", " ".join(cmd), flush=True)
    if log is None:
        subprocess.run(cmd, cwd=cwd, env=env, check=True)
        return
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write("+ " + " ".join(cmd) + "\n")
        f.flush()
        p = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)
        code = p.wait()
        if code:
            raise subprocess.CalledProcessError(code, cmd)


def cpath(part_root: Path, path: Path) -> str:
    return "/work/" + path.resolve().relative_to(part_root.resolve()).as_posix()


def docker_afni(part_root: Path, args, *, log=None):
    home = part_root / "docker_home"
    home.mkdir(parents=True, exist_ok=True)
    cmd = [
        "docker", "run", "--rm",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/work/docker_home",
        "-v", f"{part_root.resolve()}:/work",
        AFNI_IMAGE,
    ] + [str(x) for x in args]
    run(cmd, log=log)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_number(path: Path) -> int:
    m = re.search(r"_run-(\d+)_", path.name)
    if not m:
        raise RuntimeError(f"run entity missing: {path}")
    return int(m.group(1))


def echo_number(path: Path) -> int:
    m = re.search(r"_echo-(\d+)_", path.name)
    if not m:
        raise RuntimeError(f"echo entity missing: {path}")
    return int(m.group(1))


def find_task_runs(bids: Path, participant: str):
    events = sorted((bids / participant).glob("ses-*/func/*task-SORPF*_events.tsv"))
    if not events:
        raise RuntimeError(f"No SORPF events for {participant}")
    sessions = {}
    for ev in events:
        ses = ev.parent.parent.name
        rn = run_number(ev)
        sessions.setdefault(ses, {})[rn] = {"events": ev}
    for ses, runs in sessions.items():
        funcdir = bids / participant / ses / "func"
        for rn in sorted(runs):
            echoes = sorted(
                funcdir.glob(f"{participant}_{ses}_task-SORPF_run-{rn}_echo-*_part-mag_bold.nii.gz"),
                key=echo_number,
            )
            if len(echoes) != 4:
                raise RuntimeError(f"{participant} {ses} run {rn}: expected 4 magnitude echoes, got {len(echoes)}")
            rows = []
            for p in echoes:
                jp = Path(str(p)[:-7] + ".json")
                meta = load_json(jp)
                if "EchoTime" not in meta or "SliceTiming" not in meta:
                    raise RuntimeError(f"Missing EchoTime/SliceTiming in {jp}")
                rows.append({
                    "echo": echo_number(p),
                    "nii": p,
                    "json": jp,
                    "echo_time_s": float(meta["EchoTime"]),
                    "slice_timing_s": [float(x) for x in meta["SliceTiming"]],
                })
            runs[rn]["echoes"] = rows
    return sessions


def select_fieldmaps(bids: Path, participant: str, session: str, task_pe: str):
    fmapdir = bids / participant / session / "fmap"
    candidates = []
    for jp in sorted(fmapdir.glob("*_epi.json")):
        meta = load_json(jp)
        intended = meta.get("IntendedFor", [])
        if isinstance(intended, str):
            intended = [intended]
        if not any("task-SORPF" in str(x) for x in intended):
            continue
        nii = Path(str(jp)[:-5] + ".nii.gz")
        if not nii.exists():
            continue
        candidates.append((meta.get("PhaseEncodingDirection"), nii, jp))
    same = [x for x in candidates if x[0] == task_pe]
    opposite_pe = task_pe[:-1] if task_pe.endswith("-") else task_pe + "-"
    opp = [x for x in candidates if x[0] == opposite_pe]
    if len(same) != 1 or len(opp) != 1:
        raise RuntimeError(
            f"{participant} {session}: expected one same-PE and one opposite-PE SORPF fieldmap; "
            f"same={[(x[0], x[1].name) for x in same]}, opp={[(x[0], x[1].name) for x in opp]}"
        )
    return same[0][1], opp[0][1]


def prepare_tshift(part_root: Path, participant: str, sessions: dict, log: Path):
    out = part_root / "tshift"
    for ses, runs in sessions.items():
        for rn, info in sorted(runs.items()):
            for e in info["echoes"]:
                destdir = out / ses / f"run-{rn}"
                destdir.mkdir(parents=True, exist_ok=True)
                timing = destdir / f"echo-{e['echo']}_slice_timing.1D"
                timing.write_text(" ".join(f"{x:.8f}" for x in e["slice_timing_s"]) + "\n", encoding="utf-8")
                shifted = destdir / f"echo-{e['echo']}_tshift.nii.gz"
                docker_afni(
                    part_root,
                    [
                        "3dTshift", "-overwrite", "-wsinc9", "-tzero", "0",
                        "-tpattern", "@" + cpath(part_root, timing),
                        "-prefix", cpath(part_root, shifted),
                        cpath(part_root, e["nii"]),
                    ],
                    log=log,
                )
                e["shifted"] = shifted


def afni_session(part_root: Path, participant: str, session: str, runs: dict, t1_n4: Path, task_pe: str, log: Path):
    sessroot = part_root / "afni" / session
    results = sessroot / "results"
    sessroot.mkdir(parents=True, exist_ok=True)
    fwd, rev = select_fieldmaps(part_root / "bids", participant, session, task_pe)

    first = runs[sorted(runs)[0]]["echoes"]
    echo_times = [e["echo_time_s"] for e in first]
    for rn, info in sorted(runs.items()):
        these = [e["echo_time_s"] for e in info["echoes"]]
        if len(these) != len(echo_times) or any(abs(a - b) > 1e-7 for a, b in zip(these, echo_times)):
            raise RuntimeError(f"Echo times differ across runs in {participant} {session}: {these} vs {echo_times}")

    subj = f"{participant.replace('sub-','')}_{session.replace('ses-','ses')}"
    proc = sessroot / f"proc.{subj}"
    cmd = [
        "afni_proc.py",
        "-subj_id", subj,
        "-script", cpath(part_root, proc),
        "-scr_overwrite",
        "-out_dir", cpath(part_root, results),
        "-copy_anat", cpath(part_root, t1_n4),
        "-anat_has_skull", "yes",
        "-blocks", "blip", "align", "volreg", "mask",
    ]
    for rn, info in sorted(runs.items()):
        cmd += ["-dsets_me_run"] + [cpath(part_root, e["shifted"]) for e in info["echoes"]]
    cmd += [
        "-echo_times",
        *[f"{1000.0 * x:.5f}" for x in echo_times],
        "-reg_echo", "2",
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
    return results, echo_times, sorted(runs)


def parse_volreg_echoes(results: Path):
    rx = re.compile(r"\.r(\d+)\.e(\d+)\.volreg\+orig\.HEAD$")
    found = {}
    for p in results.glob("pb*.volreg+orig.HEAD"):
        m = rx.search(p.name)
        if not m:
            continue
        rn, en = map(int, m.groups())
        found.setdefault(rn, {})[en] = p
    return found


def convert_for_tedana(part_root: Path, results: Path, tedroot: Path, expected_runs: list[int], log: Path):
    found = parse_volreg_echoes(results)
    if sorted(found) != list(range(1, len(expected_runs) + 1)):
        raise RuntimeError(f"AFNI volreg run index mismatch: found={sorted(found)}, expected output indices={list(range(1,len(expected_runs)+1))}")
    mask_head = next(iter(sorted(results.glob("full_mask.*+orig.HEAD"))), None)
    if mask_head is None:
        raise RuntimeError(f"No AFNI full_mask in {results}")
    mask_nii = tedroot / "full_mask.nii.gz"
    mask_nii.parent.mkdir(parents=True, exist_ok=True)
    docker_afni(
        part_root,
        ["3dAFNItoNIFTI", "-overwrite", "-prefix", cpath(part_root, mask_nii), cpath(part_root, mask_head)],
        log=log,
    )

    converted = {}
    for afni_run_idx, original_run in enumerate(expected_runs, start=1):
        echoes = found.get(afni_run_idx, {})
        if sorted(echoes) != [1, 2, 3, 4]:
            raise RuntimeError(f"Run {afni_run_idx}: missing AFNI echo volreg outputs: {sorted(echoes)}")
        run_dir = tedroot / f"run-{original_run}"
        run_dir.mkdir(parents=True, exist_ok=True)
        files = []
        for en in [1, 2, 3, 4]:
            out = run_dir / f"echo-{en}_desc-afniPreproc_bold.nii.gz"
            docker_afni(
                part_root,
                ["3dAFNItoNIFTI", "-overwrite", "-prefix", cpath(part_root, out), cpath(part_root, echoes[en])],
                log=log,
            )
            files.append(out)
        converted[original_run] = files
    return converted, mask_nii


def run_tedana(session: str, converted: dict, mask_nii: Path, echo_times_s: list[float], part_root: Path, log: Path):
    outputs = {}
    for original_run, files in sorted(converted.items()):
        outdir = part_root / "tedana" / session / f"run-{original_run}" / "t2smap"
        outdir.mkdir(parents=True, exist_ok=True)
        prefix = f"{session}_run-{original_run}_"
        cmd = [
            "t2smap",
            "-d", *[str(x) for x in files],
            "-e", *[f"{x:.6f}" for x in echo_times_s],
            "--out-dir", str(outdir),
            "--prefix", prefix,
            "--convention", "bids",
            "--mask", str(mask_nii),
            "--fittype", "curvefit",
            "--fitmode", "all",
            "--combmode", "t2s",
            "--n-threads", "2",
            "--overwrite",
        ]
        run(cmd, log=log)
        opt = sorted(outdir.glob("*desc-optcom_bold.nii.gz"))
        if len(opt) != 1:
            raise RuntimeError(f"{session} run {original_run}: expected one tedana optcom, got {opt}")
        t2s = sorted(outdir.glob("*T2starmap.nii.gz")) + sorted(outdir.glob("*desc-t2star_statmap.nii.gz"))
        outputs[original_run] = {
            "optcom": opt[0],
            "t2star_candidates": t2s,
        }
        for p in files:
            try:
                p.unlink()
            except FileNotFoundError:
                pass
    try:
        mask_nii.unlink()
    except FileNotFoundError:
        pass
    return outputs


def normalize_anatomy(part_root: Path, t1: Path, log: Path):
    from templateflow.api import get

    anatdir = part_root / "ants"
    anatdir.mkdir(parents=True, exist_ok=True)
    t1_n4 = anatdir / "T1w_N4.nii.gz"
    run(["N4BiasFieldCorrection", "-d", "3", "-i", str(t1), "-o", str(t1_n4)], log=log)

    template = get(MNI_SPACE, resolution=MNI_RESOLUTION, desc=None, suffix="T1w")
    if isinstance(template, (list, tuple)):
        if len(template) != 1:
            raise RuntimeError(f"TemplateFlow returned multiple T1w templates: {template}")
        template = template[0]
    template = Path(template)

    prefix = anatdir / "T1_to_MNI_"
    env = os.environ.copy()
    env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "2"
    run([
        "antsRegistrationSyNQuick.sh",
        "-d", "3",
        "-f", str(template),
        "-m", str(t1_n4),
        "-o", str(prefix),
        "-t", "s",
    ], env=env, log=log)

    affine = Path(str(prefix) + "0GenericAffine.mat")
    warp = Path(str(prefix) + "1Warp.nii.gz")
    warped = Path(str(prefix) + "Warped.nii.gz")
    for p in [affine, warp, warped]:
        if not p.exists():
            raise RuntimeError(f"Missing ANTs registration output {p}")
    return t1_n4, template, affine, warp, warped

def reuse_anatomy(part_root: Path, participant: str, anatomy_root: Path, log: Path):
    from templateflow.api import get

    result_files = sorted(anatomy_root.rglob("anatomy_result.json"))
    if len(result_files) != 1:
        raise RuntimeError(f"Expected one anatomy_result.json under {anatomy_root}, got {result_files}")
    result_path = result_files[0]
    rec = load_json(result_path)
    if rec.get("status") != "P08_OPEN_SOURCE_V2_ANATOMY_NORMALIZATION_COMPLETE":
        raise RuntimeError(f"Ineligible anatomy status: {rec.get('status')}")
    if rec.get("participant") != participant or rec.get("anatomy_complete") is not True:
        raise RuntimeError(f"Anatomy artifact identity/completion mismatch for {participant}")
    if rec.get("self_other_glm_computed") or rec.get("roi_effect_computed") or rec.get("neural_effect_computed"):
        raise RuntimeError("Anatomy artifact unexpectedly contains target neural outcome")

    source_part = result_path.parent
    recorded = {x["path"]: x for x in rec.get("selected_output_files", [])}
    required = [
        "ants/T1w_N4.nii.gz",
        "ants/T1_to_MNI_0GenericAffine.mat",
        "ants/T1_to_MNI_1Warp.nii.gz",
        "ants/T1_to_MNI_Warped.nii.gz",
    ]
    anatdir = part_root / "ants"
    anatdir.mkdir(parents=True, exist_ok=True)
    copied = {}
    for rel in required:
        src = source_part / rel
        info = recorded.get(rel)
        if info is None or not src.is_file():
            raise RuntimeError(f"Missing recorded anatomy output {rel}")
        if src.stat().st_size != int(info["bytes"]) or sha256_file(src) != info["sha256"]:
            raise RuntimeError(f"Anatomy artifact hash/size mismatch for {rel}")
        dst = anatdir / Path(rel).name
        shutil.copy2(src, dst)
        copied[rel] = dst

    template = get(MNI_SPACE, resolution=MNI_RESOLUTION, desc=None, suffix="T1w")
    if isinstance(template, (list, tuple)):
        if len(template) != 1:
            raise RuntimeError(f"TemplateFlow returned multiple T1w templates: {template}")
        template = template[0]
    template = Path(template)
    print(f"reusing verified anatomy artifact for {participant}: {result_path}", flush=True)
    return (
        copied["ants/T1w_N4.nii.gz"],
        template,
        copied["ants/T1_to_MNI_0GenericAffine.mat"],
        copied["ants/T1_to_MNI_1Warp.nii.gz"],
        copied["ants/T1_to_MNI_Warped.nii.gz"],
    )



def warp_optcom(participant: str, session: str, ted_outputs: dict, part_root: Path, template: Path, affine: Path, warp: Path, log: Path):
    out = {}
    env = os.environ.copy()
    env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "2"
    mniroot = part_root / "mni"
    mniroot.mkdir(parents=True, exist_ok=True)
    for run_num, rec in sorted(ted_outputs.items()):
        dest = mniroot / f"{participant}_{session}_task-SORPF_run-{run_num}_space-{MNI_SPACE}_res-02_desc-optcom_bold.nii.gz"
        run([
            "antsApplyTransforms",
            "-d", "3", "-e", "3",
            "-i", str(rec["optcom"]),
            "-r", str(template),
            "-o", str(dest),
            "-n", "LanczosWindowedSinc",
            "-t", str(warp),
            "-t", str(affine),
        ], env=env, log=log)
        if not dest.exists():
            raise RuntimeError(f"Missing MNI optcom output {dest}")
        out[run_num] = dest
    return out


def prune_raw_bold_after_tshift(bids: Path, participant: str):
    removed = 0
    freed = 0
    for pattern in (
        "*task-SORPF*_part-mag_bold.nii.gz",
        "*task-SORPF*_part-mag_sbref.nii.gz",
    ):
        for p in (bids / participant).glob(f"ses-*/func/{pattern}"):
            try:
                freed += p.stat().st_size
                p.unlink()
                removed += 1
            except FileNotFoundError:
                pass
    print(f"resource cleanup: removed {removed} consumed raw BOLD/SBRef files, freed {freed} bytes", flush=True)


def cleanup_afni_heavy(results: Path):
    removed = 0
    freed = 0
    for p in list(results.rglob("*")):
        if not p.is_file():
            continue
        name = p.name
        if (
            ".BRIK" in name
            or name.endswith(".HEAD")
            or name.endswith(".nii")
            or name.endswith(".nii.gz")
        ):
            try:
                freed += p.stat().st_size
                p.unlink()
                removed += 1
            except FileNotFoundError:
                pass
    print(f"resource cleanup: removed {removed} AFNI heavy image files, freed {freed} bytes", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", required=True, choices=sorted(PARTICIPANTS))
    ap.add_argument("--session", default=None, help="Optional BIDS session for outcome-blind recovery sharding.")
    ap.add_argument("--run", type=int, default=None, help="Optional SORPF run number; requires --session.")
    ap.add_argument("--anatomy-root", default=None, help="Optional verified precomputed anatomy artifact root.")
    args = ap.parse_args()

    participant = args.participant
    session_filter = args.session
    run_filter = args.run
    anatomy_root = Path(args.anatomy_root) if args.anatomy_root else None
    if session_filter is not None and not session_filter.startswith("ses-"):
        raise SystemExit("--session must be a BIDS session label such as ses-01")
    if run_filter is not None and session_filter is None:
        raise SystemExit("--run requires --session")
    if run_filter is not None and run_filter < 1:
        raise SystemExit("--run must be >= 1")
    part_root = OPEN_ROOT / participant
    bids = part_root / "bids"
    outjson = part_root / "opensource_preprocessing_result.json"
    log = part_root / "opensource_preprocessing.log"
    result = {
        "status": "STARTED",
        "participant": participant,
        "session_filter": session_filter,
        "run_filter": run_filter,
        "precomputed_anatomy_requested": anatomy_root is not None,
        "lane": "open_source_secondary_sensitivity_only",
        "primary_lane_modified": False,
        "frozen_source_commit": "ed03c47c368000e511401021c1d13c3fb916b470",
        "toolchain": {
            "afni_image": AFNI_IMAGE,
            "afni_version": "AFNI_26.2.05",
            "ants_version": "2.6.5",
            "tedana_version": "26.0.3",
            "templateflow_version": "25.1.2",
            "mni_output_resolution_mm": MNI_RESOLUTION,
            "lock_version": "OPEN_SOURCE_SENSITIVITY_LOCK_V2_2026-09-19",
        },
        "self_other_glm_computed": False,
        "roi_effect_computed": False,
        "neural_effect_computed": False,
    }
    try:
        if not bids.exists():
            raise RuntimeError(f"Exact reconstructed BIDS directory missing: {bids}")

        task_meta = load_json(bids / "task-SORPF_bold.json")
        task_pe = str(task_meta["PhaseEncodingDirection"])

        t1s = sorted((bids / participant).glob("ses-*/anat/*T1w.nii.gz"))
        if len(t1s) != 1:
            raise RuntimeError(f"Expected exactly one frozen T1w for {participant}, got {t1s}")
        t1 = t1s[0]

        sessions = find_task_runs(bids, participant)
        if session_filter is not None:
            if session_filter not in sessions:
                raise RuntimeError(f"Requested session {session_filter} has no frozen SORPF runs for {participant}; found {sorted(sessions)}")
            sessions = {session_filter: sessions[session_filter]}
        if run_filter is not None:
            runs_for_session = sessions[session_filter]
            if run_filter not in runs_for_session:
                raise RuntimeError(f"Requested run {run_filter} missing in {participant} {session_filter}; found {sorted(runs_for_session)}")
            sessions = {session_filter: {run_filter: runs_for_session[run_filter]}}
        result["sessions_expected"] = {ses: sorted(runs) for ses, runs in sessions.items()}

        if anatomy_root is not None:
            t1_n4, template, affine, warp, warped_t1 = reuse_anatomy(part_root, participant, anatomy_root, log)
            result["precomputed_anatomy_reused"] = True
        else:
            t1_n4, template, affine, warp, warped_t1 = normalize_anatomy(part_root, t1, log)
            result["precomputed_anatomy_reused"] = False
        prepare_tshift(part_root, participant, sessions, log)
        prune_raw_bold_after_tshift(bids, participant)

        session_results = {}
        all_mni = []
        for ses, runs in sorted(sessions.items()):
            afni_results, echo_times_s, original_runs = afni_session(
                part_root, participant, ses, runs, t1_n4, task_pe, log
            )
            shutil.rmtree(part_root / "tshift" / ses, ignore_errors=True)
            tedroot = part_root / "tedana" / ses
            converted, mask = convert_for_tedana(part_root, afni_results, tedroot, original_runs, log)
            ted = run_tedana(ses, converted, mask, echo_times_s, part_root, log)
            cleanup_afni_heavy(afni_results)
            mni = warp_optcom(participant, ses, ted, part_root, template, affine, warp, log)
            for rec in ted.values():
                try:
                    rec["optcom"].unlink()
                except FileNotFoundError:
                    pass
            all_mni.extend(mni.values())
            session_results[ses] = {
                "input_runs": original_runs,
                "echo_times_seconds": echo_times_s,
                "afni_results_dir": str(afni_results),
                "tedana_optcom": {str(k): str(v["optcom"]) for k, v in ted.items()},
                "mni_optcom": {str(k): str(v) for k, v in mni.items()},
            }

        expected_run_count = sum(len(v) for v in sessions.values())
        if len(all_mni) != expected_run_count:
            raise RuntimeError(f"MNI output count mismatch: {len(all_mni)} != {expected_run_count}")

        files = []
        for p in sorted(all_mni + [warped_t1, affine, warp]):
            if p.is_file():
                files.append({
                    "path": p.relative_to(part_root).as_posix(),
                    "bytes": p.stat().st_size,
                    "sha256": sha256_file(p),
                })

        result.update({
            "status": (
                "P08_OPEN_SOURCE_SECONDARY_PREPROCESSING_V2_RUN_SHARD_COMPLETE"
                if run_filter is not None
                else (
                    "P08_OPEN_SOURCE_SECONDARY_PREPROCESSING_V2_SESSION_SHARD_COMPLETE"
                    if session_filter is not None
                    else "P08_OPEN_SOURCE_SECONDARY_PREPROCESSING_V2_COMPLETE"
                )
            ),
            "sessions": session_results,
            "mni_optcom_run_count": len(all_mni),
            "expected_run_count": expected_run_count,
            "selected_output_files": files,
            "selected_output_file_count": len(files),
            "license_secret_required": False,
            "freesurfer_used": False,
            "preprocessing_complete": True,
            "self_other_glm_computed": False,
            "roi_effect_computed": False,
            "neural_effect_computed": False,
            "scientific_note": (
                "This is a pre-outcome open-source sensitivity preprocessing lane. "
                "It does not replace the frozen primary fMRIPrep/FreeSurfer lane."
            ),
        })
        outjson.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0
    except Exception as e:
        result.update({
            "status": (
                "P08_OPEN_SOURCE_SECONDARY_PREPROCESSING_V2_RUN_SHARD_TECHNICAL_FAILURE"
                if run_filter is not None
                else (
                    "P08_OPEN_SOURCE_SECONDARY_PREPROCESSING_V2_SESSION_SHARD_TECHNICAL_FAILURE"
                    if session_filter is not None
                    else "P08_OPEN_SOURCE_SECONDARY_PREPROCESSING_V2_TECHNICAL_FAILURE"
                )
            ),
            "preprocessing_complete": False,
            "error": {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            },
            "self_other_glm_computed": False,
            "roi_effect_computed": False,
            "neural_effect_computed": False,
        })
        outjson.parent.mkdir(parents=True, exist_ok=True)
        outjson.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
