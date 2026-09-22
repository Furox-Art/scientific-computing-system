#!/usr/bin/env python3
import hashlib, json, shutil, subprocess, tempfile
from pathlib import Path

SOURCE_URL="https://gitlab.com/michael.pereira/evidenceaccumulation_ppc.git"
SOURCE_COMMIT="c9794592e5c97f8a2c79b0f46e2c8f295735df5b"
OUT=Path("pereira_audit/SOURCE_MODEL_SPEC_AUDIT_2026-09-22.json")
TARGETS=[
 "README",
 "eeg/conf/config_model.m",
 "eeg/conf/paramset_1storder.m",
 "eeg/conf/paramset_2ndorder.m",
 "eeg/model/README",
 "eeg/model/paramset.m",
 "eeg/model/paramset2.m",
 "eeg/model/functions/evacc.m",
 "eeg/model/functions/fitconf.m",
 "eeg/model/functions/fitconf_wrapper.m",
 "eeg/model/functions/fiterp.m",
 "eeg/model/functions/fiterp_wrapper.m",
 "eeg/model/functions/readout.m",
 "eeg/model/functions/simconf.m",
 "eeg/model/step1_fitresp.m",
 "eeg/model/step2_fitresp_check.m",
 "eeg/model/step3_fitconf_max.m",
 "eeg/model/step3_fitconf_resp.m",
 "eeg/model/step4_evaluate.m"
]
KEYS=["bic","aic","loglik","logl","likelihood","wilcoxon","signrank","ranksum",
      "npar","numel","alpha","beta","omega","t_ro","theta","ndt","lambda","gamma",
      "10000","1000","500","756","66","save(","load(","fminsearch","confidence",
      "mean(","std(","median(","ks","kstest"]

def run(cmd,cwd=None,capture=False):
    return subprocess.run(cmd,cwd=cwd,check=True,text=True,
      stdout=subprocess.PIPE if capture else None,
      stderr=subprocess.PIPE if capture else None)

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

root=Path(tempfile.mkdtemp(prefix="pereira-source-audit-"))
try:
    src=root/"src"
    run(["git","clone","--filter=blob:none","--no-tags",SOURCE_URL,str(src)])
    run(["git","checkout","--detach",SOURCE_COMMIT],cwd=src)
    got=run(["git","rev-parse","HEAD"],cwd=src,capture=True).stdout.strip()
    if got!=SOURCE_COMMIT: raise RuntimeError(f"source commit mismatch {got}")

    audit={
      "status":"PEREIRA2021_EXACT_SOURCE_MODEL_SPEC_AUDITED",
      "source_repo":SOURCE_URL,
      "source_commit":got,
      "files":{},
      "model_tree":[],
      "supdata_files":[]
    }

    for p in sorted(x for x in (src/"eeg/model").rglob("*") if x.is_file()):
        audit["model_tree"].append({
          "path":str(p.relative_to(src)).replace("\\","/"),
          "bytes":p.stat().st_size,
          "sha256":sha256(p)
        })
    sup=src/"supdata"
    if sup.exists():
        for p in sorted(x for x in sup.rglob("*") if x.is_file()):
            audit["supdata_files"].append({
              "path":str(p.relative_to(src)).replace("\\","/"),
              "bytes":p.stat().st_size,
              "sha256":sha256(p)
            })

    for path in TARGETS:
        p=src/path
        if not p.exists():
            audit["files"][path]={"missing":True}
            continue
        txt=p.read_text(encoding="utf-8",errors="replace")
        lines=txt.splitlines()
        hits=[]
        for i,line in enumerate(lines,1):
            low=line.lower()
            if any(k in low for k in KEYS):
                hits.append({"line":i,"text":line.strip()})
        audit["files"][path]={
          "bytes":p.stat().st_size,
          "sha256":sha256(p),
          "line_count":len(lines),
          "function_signatures":[{"line":i,"text":ln.strip()} for i,ln in enumerate(lines,1) if ln.strip().lower().startswith("function")],
          "relevant_lines":hits[:1200]
        }

    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({
      "source_commit":got,
      "model_tree":audit["model_tree"],
      "supdata_files":audit["supdata_files"],
      "step4":audit["files"].get("eeg/model/step4_evaluate.m",{}).get("relevant_lines",[]),
      "step3_max":audit["files"].get("eeg/model/step3_fitconf_max.m",{}).get("relevant_lines",[]),
      "step3_resp":audit["files"].get("eeg/model/step3_fitconf_resp.m",{}).get("relevant_lines",[]),
      "paramset2":audit["files"].get("eeg/model/paramset2.m",{}).get("relevant_lines",[])
    },indent=2))
finally:
    shutil.rmtree(root,ignore_errors=True)
