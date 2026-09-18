#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, json, os, re, subprocess, sys, traceback
from pathlib import Path

REMOTE="https://github.com/OpenNeuroDatasets/ds002278.git"
COMMIT="ed03c47c368000e511401021c1d13c3fb916b470"
SUBS=["sub-Bubbles","sub-Buttercup","sub-PILOT02"]
WORK=Path(".public-runner/p08/manifest_repo")
OUT=Path(".public-runner/p08")
CSVOUT=OUT/"raw_preprocessing_input_manifest_2026-09-18.csv"
JSONOUT=OUT/"raw_preprocessing_input_manifest_summary_2026-09-18.json"

ANNEX_RE=re.compile(r"(MD5E|SHA256E)-s(\d+)--([0-9a-f]+)(\.[^/]+(?:\.gz)?)$")

def run(*args,cwd=None):
    return subprocess.check_output(args,cwd=cwd,text=True,stderr=subprocess.STDOUT).strip()

def sha256_bytes(b:bytes)->str:
    return hashlib.sha256(b).hexdigest()

def rel(p:Path,root:Path)->str:
    return p.relative_to(root).as_posix()

def annex_info(p:Path):
    if not p.is_symlink():
        return None
    target=os.readlink(p)
    key=Path(target).name
    m=ANNEX_RE.match(key)
    if not m:
        raise RuntimeError(f"unrecognized annex key for {p}: {key}")
    algo,size,digest,suffix=m.groups()
    return {"annex_key":key,"hash_algorithm":"md5" if algo=="MD5E" else "sha256","expected_hash":digest,"expected_bytes":int(size)}

def normalize_intended(x):
    if isinstance(x,str): return [x]
    if isinstance(x,list): return [str(v) for v in x]
    return []

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    summary={"status":"STARTED","frozen_commit":COMMIT,"participants":SUBS,"selection_rules":[]}
    try:
        if WORK.exists():
            subprocess.run(["rm","-rf",str(WORK)],check=True)
        WORK.mkdir(parents=True)
        run("git","init",cwd=WORK)
        run("git","remote","add","origin",REMOTE,cwd=WORK)
        run("git","fetch","--depth","1","origin",COMMIT,cwd=WORK)
        run("git","checkout","--detach","FETCH_HEAD",cwd=WORK)
        got=run("git","rev-parse","HEAD",cwd=WORK)
        if got!=COMMIT: raise RuntimeError(f"commit mismatch {got}")

        selected={}
        sorpf_sessions=set()

        # Root metadata required to interpret the BIDS subset.
        for rp in ["dataset_description.json","participants.tsv","task-SORPF_bold.json","task-SORPF_events.json"]:
            p=WORK/rp
            if p.exists(): selected[rp]="root_metadata"

        # Detect sessions from frozen SORPF events, then include SORPF magnitude BOLD/SBRef and sidecars.
        for sub in SUBS:
            for ev in (WORK/sub).glob("ses-*/func/*task-SORPF*_events.tsv"):
                session=ev.parts[-3]
                sorpf_sessions.add((sub,session))
                selected[rel(ev,WORK)]="sorpf_events"
                funcdir=ev.parent
                for p in funcdir.glob("*task-SORPF*"):
                    name=p.name
                    if name.endswith("_part-mag_bold.nii.gz") or name.endswith("_part-mag_sbref.nii.gz"):
                        selected[rel(p,WORK)]="sorpf_magnitude_signal"
                    elif name.endswith("_part-mag_bold.json") or name.endswith("_part-mag_sbref.json"):
                        selected[rel(p,WORK)]="sorpf_magnitude_sidecar"
                # session bookkeeping if present
                sesdir=funcdir.parent
                for p in sesdir.glob("*_scans.tsv"):
                    selected[rel(p,WORK)]="session_scans_metadata"

        # All T1w anatomical inputs for each frozen participant, independent of outcome.
        for sub in SUBS:
            for p in (WORK/sub).glob("ses-*/anat/*T1w.nii.gz"):
                selected[rel(p,WORK)]="t1w_anatomical"
                jp=Path(str(p)[:-7]+".json")
                if jp.exists(): selected[rel(jp,WORK)]="t1w_sidecar"

        # Fieldmaps: in SORPF sessions, prefer explicit IntendedFor linkage.
        # If a session has no explicit SORPF-linked func fmap, include all acq-func EPI fmap pairs in that SORPF session conservatively.
        fmap_selection={}
        for sub,session in sorted(sorpf_sessions):
            fmapdir=WORK/sub/session/"fmap"
            if not fmapdir.exists(): continue
            explicit=[]
            func_jsons=sorted(fmapdir.glob("*acq-func*_epi.json"))
            for jp in func_jsons:
                try:
                    meta=json.loads(jp.read_text())
                except Exception:
                    meta={}
                intended=normalize_intended(meta.get("IntendedFor"))
                if any("task-SORPF" in v for v in intended):
                    explicit.append(jp)
            use=explicit if explicit else func_jsons
            mode="explicit_intendedfor" if explicit else "conservative_all_acq_func_in_sorpf_session"
            fmap_selection[f"{sub}/{session}"]={"mode":mode,"json_count":len(use)}
            for jp in use:
                selected[rel(jp,WORK)]="func_fmap_sidecar"
                np=Path(str(jp)[:-5]+".nii.gz")
                if np.exists(): selected[rel(np,WORK)]="func_fmap_signal"

        rows=[]
        annex_total=0
        regular_total=0
        by_role={}
        by_sub={}
        for rp,role in sorted(selected.items()):
            p=WORK/rp
            ai=annex_info(p)
            if ai:
                row={"path":rp,"role":role,"storage":"git_annex","expected_bytes":ai["expected_bytes"],
                     "hash_algorithm":ai["hash_algorithm"],"expected_hash":ai["expected_hash"],"annex_key":ai["annex_key"]}
                annex_total+=ai["expected_bytes"]
            else:
                b=p.read_bytes()
                row={"path":rp,"role":role,"storage":"git_blob","expected_bytes":len(b),
                     "hash_algorithm":"sha256","expected_hash":sha256_bytes(b),"annex_key":""}
                regular_total+=len(b)
            rows.append(row)
            by_role[role]=by_role.get(role,0)+1
            sub=rp.split("/",1)[0] if rp.startswith("sub-") else "root"
            by_sub[sub]=by_sub.get(sub,0)+1

        with CSVOUT.open("w",newline="",encoding="utf-8") as f:
            fields=["path","role","storage","expected_bytes","hash_algorithm","expected_hash","annex_key"]
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

        summary.update({
            "status":"P08_FROZEN_PREPROCESSING_INPUT_MANIFEST_CREATED",
            "frozen_commit_verified":True,
            "sorpf_sessions":[{"participant":s,"session":se} for s,se in sorted(sorpf_sessions)],
            "selection_rules":[
                "SORPF sessions are detected only from frozen task-SORPF events.tsv paths.",
                "Include every magnitude BOLD echo and magnitude SBRef in those SORPF functional directories, plus their JSON sidecars.",
                "Exclude phase-BOLD/phase-SBRef, dMRI, rest and unrelated tasks from the preprocessing subset.",
                "Include all T1w anatomical inputs for each frozen participant.",
                "For each SORPF session, include acq-func EPI fieldmaps explicitly linked to SORPF by IntendedFor; if no explicit linkage exists, conservatively include all acq-func EPI fieldmaps in that SORPF session.",
                "Include dataset/task/session metadata required to interpret the BIDS subset."
            ],
            "fieldmap_selection":fmap_selection,
            "files_total":len(rows),
            "annex_files":sum(1 for r in rows if r["storage"]=="git_annex"),
            "git_blob_files":sum(1 for r in rows if r["storage"]=="git_blob"),
            "annex_payload_bytes":annex_total,
            "git_blob_bytes":regular_total,
            "total_expected_bytes":annex_total+regular_total,
            "counts_by_role":by_role,
            "counts_by_subject_or_root":by_sub,
            "manifest_csv":str(CSVOUT),
            "neural_effect_computed":False
        })
        JSONOUT.write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(summary,indent=2))
        return 0
    except Exception as e:
        summary["status"]="P08_PREPROCESSING_INPUT_MANIFEST_FAILED"
        summary["error"]={"type":type(e).__name__,"message":str(e),"traceback":traceback.format_exc()}
        JSONOUT.write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(summary,indent=2)); return 2

if __name__=="__main__": sys.exit(main())
