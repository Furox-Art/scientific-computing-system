#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json, shutil
from pathlib import Path
import nibabel as nib
import numpy as np

EXPECTED={
  "sub-Bubbles":[("ses-01",1),("ses-01",2),("ses-04",1),("ses-04",2)],
  "sub-Buttercup":[("ses-04",1),("ses-04",2),("ses-05",1),("ses-05",2)],
}
RUN_STATUS="P08_OPEN_SOURCE_SECONDARY_PREPROCESSING_V2_RUN_SHARD_COMPLETE"
MERGED_STATUS="P08_OPEN_SOURCE_SECONDARY_PREPROCESSING_V2_COMPLETE"

def sha256_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()

def find_single(root:Path,pattern:str)->Path:
    xs=sorted(root.rglob(pattern))
    if len(xs)!=1: raise RuntimeError(f"Expected one {pattern} under {root}, got {xs}")
    return xs[0]

def load(path:Path):
    return json.loads(path.read_text(encoding="utf-8"))

def copy_exact(src:Path,dst:Path):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():
        if dst.stat().st_size!=src.stat().st_size or sha256_file(dst)!=sha256_file(src):
            raise RuntimeError(f"Conflicting duplicate {src} -> {dst}")
        return
    shutil.copy2(src,dst)

def roots(input_root:Path,p:str,s:str,r:int):
    pre=input_root/f"public-p08-v2-run-recovery-preprocessed-{p}-{s}-run-{r}"
    prov=input_root/f"public-p08-v2-run-recovery-provenance-{p}-{s}-run-{r}"
    if not pre.is_dir() or not prov.is_dir():
        raise RuntimeError(f"Missing run-shard artifact directories for {p} {s} run {r}")
    return pre,prov

def verify_shard(input_root:Path,p:str,s:str,r:int):
    pre,prov=roots(input_root,p,s,r)
    result=load(find_single(pre,"opensource_preprocessing_result.json"))
    recon=load(find_single(prov,"input_reconstruction.json"))
    if result.get("status")!=RUN_STATUS: raise RuntimeError(f"{p} {s} run {r}: bad status {result.get('status')}")
    if result.get("participant")!=p or result.get("session_filter")!=s or int(result.get("run_filter",-1))!=r:
        raise RuntimeError(f"{p} {s} run {r}: identity mismatch")
    if result.get("expected_run_count")!=1 or result.get("mni_optcom_run_count")!=1:
        raise RuntimeError(f"{p} {s} run {r}: expected one run")
    if result.get("precomputed_anatomy_reused") is not True:
        raise RuntimeError(f"{p} {s} run {r}: anatomy not reused")
    if not result.get("preprocessing_complete",False):
        raise RuntimeError(f"{p} {s} run {r}: preprocessing incomplete")
    if result.get("self_other_glm_computed") or result.get("roi_effect_computed") or result.get("neural_effect_computed"):
        raise RuntimeError(f"{p} {s} run {r}: neural outcome unexpectedly present")
    if recon.get("participant")!=p or recon.get("session_filter")!=s or not recon.get("all_exact_verified"):
        raise RuntimeError(f"{p} {s} run {r}: exact input provenance ineligible")

    bold=find_single(pre,f"{p}_{s}_task-SORPF_run-{r}_space-MNI152NLin2009cAsym_res-02_desc-optcom_bold.nii.gz")
    selected={Path(x["path"]).name:x for x in result.get("selected_output_files",[])}
    rec=selected.get(bold.name)
    if rec is None or int(rec.get("bytes",-1))!=bold.stat().st_size or rec.get("sha256")!=sha256_file(bold):
        raise RuntimeError(f"{p} {s} run {r}: BOLD hash/size mismatch")

    event=find_single(pre,f"{p}_{s}_task-SORPF_run-{r}_events.tsv")
    rel=f"{p}/{s}/func/{event.name}"
    er={x["path"]:x for x in recon.get("files",[])}.get(rel)
    if er is None or er.get("hash_algorithm")!="sha256" or er.get("hash")!=sha256_file(event):
        raise RuntimeError(f"{p} {s} run {r}: event exact provenance mismatch")

    motion=find_single(pre,"dfile*.1D")
    marr=np.loadtxt(motion,ndmin=2)
    nt=int(nib.load(bold).shape[3])
    if marr.shape[0]!=nt:
        raise RuntimeError(f"{p} {s} run {r}: motion rows {marr.shape[0]} != BOLD volumes {nt}")
    if marr.shape[1] not in (6,9):
        raise RuntimeError(f"{p} {s} run {r}: unexpected motion columns {marr.shape[1]}")

    anatomy_result=find_single(prov,"anatomy_result.json")
    ar=load(anatomy_result)
    if ar.get("status")!="P08_OPEN_SOURCE_V2_ANATOMY_NORMALIZATION_COMPLETE" or ar.get("participant")!=p:
        raise RuntimeError(f"{p} {s} run {r}: anatomy result ineligible")
    amap={x["path"]:x for x in ar.get("selected_output_files",[])}
    copied={}
    for relname,filename in (
        ("ants/T1w_N4.nii.gz","T1w_N4.nii.gz"),
        ("ants/T1_to_MNI_0GenericAffine.mat","T1_to_MNI_0GenericAffine.mat"),
        ("ants/T1_to_MNI_1Warp.nii.gz","T1_to_MNI_1Warp.nii.gz"),
    ):
        pth=find_single(prov,filename)
        info=amap.get(relname)
        if info is None or pth.stat().st_size!=int(info["bytes"]) or sha256_file(pth)!=info["sha256"]:
            raise RuntimeError(f"{p} {s} run {r}: anatomy copy mismatch {filename}")
        copied[filename]=sha256_file(pth)

    return {
      "pre":pre,"prov":prov,"result":result,"recon":recon,
      "bold":bold,"event":event,"motion":motion,
      "motion_shape":tuple(marr.shape),"nvols":nt,
      "anatomy_result_sha256":sha256_file(anatomy_result),
      "anatomy_hashes":copied,
    }

def merge_recon(shards,p):
    by={}
    for sh in shards:
        for rec in sh["recon"].get("files",[]):
            path=rec["path"]
            if path in by:
                for k in ("bytes","hash_algorithm","hash","storage","role"):
                    if by[path].get(k)!=rec.get(k):
                        raise RuntimeError(f"Conflicting exact provenance for {path}")
            else:
                by[path]=rec
    files=[by[k] for k in sorted(by)]
    return {
      "status":"P08_RUN_RECOVERY_MERGED_EXACT_INPUT_PROVENANCE",
      "participant":p,
      "frozen_dataset":"OpenNeuro ds002278 v2.0.0",
      "frozen_commit":"ed03c47c368000e511401021c1d13c3fb916b470",
      "files":files,
      "selected_files":len(files),
      "selected_bytes_unique":sum(int(x.get("bytes",0)) for x in files),
      "all_exact_verified":True,
      "neural_effect_computed":False,
    }

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--participant",required=True,choices=sorted(EXPECTED))
    ap.add_argument("--input-root",required=True)
    ap.add_argument("--output-preproc",required=True)
    ap.add_argument("--output-prov",required=True)
    args=ap.parse_args()

    p=args.participant
    input_root=Path(args.input_root)
    out_pre=Path(args.output_preproc); out_prov=Path(args.output_prov)
    out_pre.mkdir(parents=True,exist_ok=True); out_prov.mkdir(parents=True,exist_ok=True)

    shards=[]
    for s,r in EXPECTED[p]:
        sh=verify_shard(input_root,p,s,r)
        sh["session"]=s; sh["run"]=r
        shards.append(sh)

    anatomy_result_hashes={sh["anatomy_result_sha256"] for sh in shards}
    anatomy_hash_sets={json.dumps(sh["anatomy_hashes"],sort_keys=True) for sh in shards}
    if len(anatomy_result_hashes)!=1 or len(anatomy_hash_sets)!=1:
        raise RuntimeError(f"{p}: anatomy artifact/transforms differ across run shards")

    selected=[]
    sessions={}
    for sh in shards:
        s,r=sh["session"],sh["run"]
        dst=out_pre/"mni"/sh["bold"].name
        copy_exact(sh["bold"],dst)
        selected.append({"path":f"mni/{dst.name}","bytes":dst.stat().st_size,"sha256":sha256_file(dst)})
        evdst=out_pre/"bids"/p/s/"func"/sh["event"].name
        copy_exact(sh["event"],evdst)
        sessions.setdefault(s,{"input_runs":[],"motion_parts":[],"mni_optcom":{}})
        sessions[s]["input_runs"].append(r)
        sessions[s]["motion_parts"].append((r,sh["motion"]))
        sessions[s]["mni_optcom"][str(r)]=f"mni/{dst.name}"

    for s,rec in sessions.items():
        rec["input_runs"]=sorted(rec["input_runs"])
        if rec["input_runs"]!=[1,2]:
            raise RuntimeError(f"{p} {s}: expected run 1 and 2, got {rec['input_runs']}")
        parts=sorted(rec.pop("motion_parts"),key=lambda x:x[0])
        cols=[]
        texts=[]
        rows=0
        for r,m in parts:
            arr=np.loadtxt(m,ndmin=2)
            cols.append(arr.shape[1]); rows+=arr.shape[0]
            texts.append(m.read_text(encoding="utf-8").rstrip()+"\n")
        if len(set(cols))!=1:
            raise RuntimeError(f"{p} {s}: motion column mismatch across runs {cols}")
        mdst=out_pre/"afni"/s/"results"/"dfile_rall.1D"
        mdst.parent.mkdir(parents=True,exist_ok=True)
        mdst.write_text("".join(texts),encoding="utf-8")
        merged_arr=np.loadtxt(mdst,ndmin=2)
        expected_rows=sum(sh["nvols"] for sh in shards if sh["session"]==s)
        if merged_arr.shape[0]!=expected_rows or merged_arr.shape[0]!=rows:
            raise RuntimeError(f"{p} {s}: merged motion row mismatch")
        rec["motion_sha256"]=sha256_file(mdst)
        rec["motion_rows"]=int(merged_arr.shape[0])
        rec["motion_columns"]=int(merged_arr.shape[1])

    merged_recon=merge_recon(shards,p)
    recon_path=out_prov/"input_reconstruction.json"
    recon_path.write_text(json.dumps(merged_recon,indent=2)+"\n",encoding="utf-8")

    recon_by={x["path"]:x for x in merged_recon["files"]}
    for ev in sorted(out_pre.rglob("*_events.tsv")):
        rel=ev.relative_to(out_pre/"bids").as_posix()
        info=recon_by.get(rel)
        if info is None or info.get("hash_algorithm")!="sha256" or info.get("hash")!=sha256_file(ev):
            raise RuntimeError(f"Merged event provenance mismatch: {rel}")

    anatomy_identity={
      "anatomy_result_sha256":next(iter(anatomy_result_hashes)),
      "transform_hashes":shards[0]["anatomy_hashes"],
      "identical_across_four_run_shards":True,
    }
    (out_prov/"anatomy_identity.json").write_text(json.dumps(anatomy_identity,indent=2)+"\n")

    result={
      "status":MERGED_STATUS,
      "participant":p,
      "lane":"open_source_secondary_sensitivity_only",
      "recovery_mode":"run_sharded_operational_recovery_with_reused_anatomy",
      "preprocessing_complete":True,
      "expected_run_count":4,
      "mni_optcom_run_count":4,
      "sessions_expected":{s:[1,2] for s in sorted(sessions)},
      "sessions":sessions,
      "selected_output_files":sorted(selected,key=lambda x:x["path"]),
      "exact_input_provenance_merged":True,
      "anatomy_identity":anatomy_identity,
      "self_other_glm_computed":False,
      "roi_effect_computed":False,
      "neural_effect_computed":False,
      "primary_lane_modified":False,
    }
    (out_pre/"opensource_preprocessing_result.json").write_text(json.dumps(result,indent=2)+"\n")
    summary={
      "status":"P08_V2_RUN_RECOVERY_PARTICIPANT_MERGE_COMPLETE",
      "participant":p,
      "run_count":4,
      "sessions":sorted(sessions),
      "anatomy_identity":anatomy_identity,
      "neural_effect_computed":False,
    }
    (out_prov/"run_recovery_merge_result.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
