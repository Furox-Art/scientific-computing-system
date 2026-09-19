#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, json, os, re, subprocess
from pathlib import Path

SOURCE_COMMIT="2e273d8466162208bbccd8591337e55b7a8b5721"
DATASET="ds005237"

def sha256_file(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def parse_annex_key(link:str):
    parts=Path(link).parts
    keys=[x for x in parts if re.match(r'^(MD5E|SHA256E|SHA1E|MD5|SHA256|SHA1)-',x)]
    unique=sorted(set(keys))
    if len(unique)!=1:
        raise RuntimeError(f"Could not identify one unique annex key in symlink target: {link}; keys={keys}")
    key=unique[0]
    m=re.match(r'^(MD5E|SHA256E|SHA1E|MD5|SHA256|SHA1)-s(\d+)--(.+)$',key)
    if not m:
        raise RuntimeError(f"Unsupported annex key format: {key}")
    alg=m.group(1).replace("E","").lower()
    size=int(m.group(2))
    tail=m.group(3)
    digest=tail.split(".",1)[0]
    return key,alg,size,digest

def git_blob_sha(path:str)->str:
    return subprocess.check_output(["git","rev-parse",f"HEAD:{path}"],text=True).strip()

def main():
    root=Path(".")
    participants=list(csv.DictReader((root/"participants.tsv").open(),delimiter="\t"))
    all_paths={p.as_posix() for p in root.rglob("*") if p.is_file() or p.is_symlink()}
    selected=[]
    eligible=[]
    missing_counts={k:0 for k in ["t1","fmap_ap","fmap_pa","hammer_bold","hammer_events","stroop_ap_bold","stroop_ap_events","stroop_pa_bold","stroop_pa_events"]}

    for row in participants:
        s=row["participant_id"]
        t1=sorted((root/s/"anat").glob("*_T1w.nii.gz"))
        req={
          "t1":len(t1)==1,
          "fmap_ap":f"{s}/fmap/{s}_dir-ap_epi.nii.gz" in all_paths,
          "fmap_pa":f"{s}/fmap/{s}_dir-pa_epi.nii.gz" in all_paths,
          "hammer_bold":f"{s}/func/{s}_task-hammerAP_run-01_bold.nii.gz" in all_paths,
          "hammer_events":f"{s}/func/{s}_task-hammerAP_run-01_events.tsv" in all_paths,
          "stroop_ap_bold":f"{s}/func/{s}_task-stroopAP_run-01_bold.nii.gz" in all_paths,
          "stroop_ap_events":f"{s}/func/{s}_task-stroopAP_run-01_events.tsv" in all_paths,
          "stroop_pa_bold":f"{s}/func/{s}_task-stroopPA_run-01_bold.nii.gz" in all_paths,
          "stroop_pa_events":f"{s}/func/{s}_task-stroopPA_run-01_events.tsv" in all_paths,
        }
        for k,v in req.items():
            if not v: missing_counts[k]+=1
        if not all(req.values()): continue
        eligible.append(row)

        required=[
          t1[0],
          t1[0].with_suffix("").with_suffix(".json"),
          root/s/"fmap"/f"{s}_dir-ap_epi.nii.gz",
          root/s/"fmap"/f"{s}_dir-ap_epi.json",
          root/s/"fmap"/f"{s}_dir-pa_epi.nii.gz",
          root/s/"fmap"/f"{s}_dir-pa_epi.json",
        ]
        for task in ("hammerAP","stroopAP","stroopPA"):
            stem=root/s/"func"/f"{s}_task-{task}_run-01"
            required += [
              Path(str(stem)+"_bold.nii.gz"),
              Path(str(stem)+"_bold.json"),
              Path(str(stem)+"_events.tsv"),
              Path(str(stem)+"_events.json"),
            ]
        for p in required:
            if not (p.exists() or p.is_symlink()):
                raise RuntimeError(f"eligible participant missing required sidecar/input: {p}")
            rel=p.as_posix()
            if p.is_symlink():
                target=os.readlink(p)
                key,alg,size,digest=parse_annex_key(target)
                selected.append({
                  "participant_id":s,
                  "group":row["Group"],
                  "site":row["Site"],
                  "path":rel,
                  "storage":"git-annex",
                  "annex_key":key,
                  "hash_algorithm":alg,
                  "hash":digest,
                  "bytes":size,
                  "s3_url":f"s3://openneuro.org/{DATASET}/{rel}",
                })
            else:
                selected.append({
                  "participant_id":s,
                  "group":row["Group"],
                  "site":row["Site"],
                  "path":rel,
                  "storage":"git-blob",
                  "git_blob_sha1":git_blob_sha(rel),
                  "hash_algorithm":"sha256",
                  "hash":sha256_file(p),
                  "bytes":p.stat().st_size,
                })

    groups={}
    for r in eligible: groups[r["Group"]]=groups.get(r["Group"],0)+1
    if len(eligible)!=212 or groups!={"Patient":126,"GenPop":86}:
        raise RuntimeError(f"eligibility mismatch: N={len(eligible)} groups={groups}")
    if any(sum(1 for x in selected if x["participant_id"]==r["participant_id"])!=18 for r in eligible):
        raise RuntimeError("expected exactly 18 frozen files per eligible participant")

    selected=sorted(selected,key=lambda x:(x["participant_id"],x["path"]))
    outdir=Path(".public-runner/ds005237/neural-manifest"); outdir.mkdir(parents=True,exist_ok=True)
    fields=["participant_id","group","site","path","storage","annex_key","git_blob_sha1","hash_algorithm","hash","bytes","s3_url"]
    with (outdir/"raw_input_manifest.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for x in selected: w.writerow({k:x.get(k,"") for k in fields})
    with (outdir/"eligible_participants.tsv").open("w",newline="") as f:
        fields2=["participant_id","Group","Site","age","sex"]
        w=csv.DictWriter(f,fieldnames=fields2,delimiter="\t"); w.writeheader()
        for x in eligible: w.writerow({k:x[k] for k in fields2})

    summary={
      "status":"DS005237_NEURAL_RAW_MANIFEST_COMPLETE",
      "dataset":DATASET,
      "source_commit":SOURCE_COMMIT,
      "participants_source_total":len(participants),
      "eligible_participants":len(eligible),
      "eligible_group_counts":groups,
      "files_per_participant":18,
      "selected_files":len(selected),
      "selected_annex_files":sum(x["storage"]=="git-annex" for x in selected),
      "selected_git_blob_files":sum(x["storage"]=="git-blob" for x in selected),
      "selected_bytes":sum(int(x["bytes"]) for x in selected),
      "missing_counts":missing_counts,
      "neural_group_contrast_computed":False,
      "manifest_sha256":sha256_file(outdir/"raw_input_manifest.csv"),
      "eligible_tsv_sha256":sha256_file(outdir/"eligible_participants.tsv"),
    }
    (outdir/"manifest_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
