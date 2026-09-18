#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, json, shutil, time
from pathlib import Path
import numpy as np
import requests
from scipy.io import loadmat

BASE="https://data.nemar.org/on004563/v1.0.0/"
MANIFEST=Path("tools/public_p10_source_bf_manifest.csv")
OUT=Path(".public-runner/p10/bfqa")
CHUNK=1048576
GRID=[0,93,186,279,372,465,558,651,743]

def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def download(url, expected, want, out):
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists() and out.stat().st_size==expected and sha256(out)==want: return
    out.unlink(missing_ok=True)
    parts=out.parent/(out.name+".parts")
    shutil.rmtree(parts,ignore_errors=True); parts.mkdir()
    s=requests.Session(); start=0; i=0
    while start<expected:
        end=min(start+CHUNK-1,expected-1)
        for attempt in range(1,5):
            try:
                r=s.get(url,headers={"Range":f"bytes={start}-{end}","Accept-Encoding":"identity","User-Agent":"github-actions-public-validation/1.0"},timeout=(30,180),allow_redirects=True)
                if r.status_code not in (200,206): raise RuntimeError(f"HTTP {r.status_code}")
                break
            except Exception:
                if attempt==4: raise
                time.sleep(min(2**(attempt-1),8))
        b=r.content
        if r.status_code==206:
            if len(b)!=(end-start+1): raise RuntimeError("short range")
            (parts/f"part-{i:05d}.bin").write_bytes(b); start=end+1; i+=1
        elif r.status_code==200 and start==0 and len(b)==expected:
            out.write_bytes(b); break
        else: raise RuntimeError("unexpected non-range response")
    if not out.exists():
        with out.open("wb") as dst:
            for p in sorted(parts.glob("part-*.bin")):
                with p.open("rb") as src: shutil.copyfileobj(src,dst)
    shutil.rmtree(parts,ignore_errors=True)
    got=sha256(out)
    if out.stat().st_size!=expected or got!=want: raise RuntimeError(f"verify failed {out.name}")

def write_one(path, arr, bf):
    n=arr.shape[2]
    fields=["train_i","test_i","source_bf"]+[f"x{i+1}" for i in range(n)]
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for ti in GRID:
            for tj in GRID:
                row={"train_i":ti,"test_i":tj,"source_bf":repr(float(bf[ti,tj]))}
                for k,v in enumerate(arr[ti,tj,:]): row[f"x{k+1}"]=repr(float(v))
                w.writerow(row)

def write_two(path, x, y, bf):
    fields=["train_i","test_i","source_bf"]+[f"x{i+1}" for i in range(x.shape[2])]+[f"y{i+1}" for i in range(y.shape[2])]
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for ti in GRID:
            for tj in GRID:
                row={"train_i":ti,"test_i":tj,"source_bf":repr(float(bf[ti,tj]))}
                for k,v in enumerate(x[ti,tj,:]): row[f"x{k+1}"]=repr(float(v))
                for k,v in enumerate(y[ti,tj,:]): row[f"y{k+1}"]=repr(float(v))
                w.writerow(row)

OUT.mkdir(parents=True,exist_ok=True)
with MANIFEST.open(newline="") as f:
    rows={Path(r["path"]).name:r for r in csv.DictReader(f)}
for fn,row in rows.items():
    p=OUT/fn
    download(BASE+row["path"],int(row["bytes"]),row["sha256"].lower(),p)

vt=loadmat(OUT/"bayes_factors_VT_group.mat")
nvt=loadmat(OUT/"bayes_factors_NoVT_group.mat")
grp=loadmat(OUT/"bayes_factors_groups.mat")

write_one(OUT/"vt_touch2vis_qa.csv",np.asarray(vt["timegen_data_touch2vis"],float),np.asarray(vt["bf_touch2vis"],float))
write_one(OUT/"nvt_touch2vis_qa.csv",np.asarray(nvt["timegen_data_touch2vis"],float),np.asarray(nvt["bf_touch2vis"],float))
write_two(OUT/"between_touch2vis_qa.csv",np.asarray(vt["timegen_data_touch2vis"],float),np.asarray(nvt["timegen_data_touch2vis"],float),np.asarray(grp["bf_BetweenGroups"],float))

meta={"status":"QA_INPUTS_READY","grid_indices_zero_based":GRID,"cells_per_test":len(GRID)**2,"selection_rule":"fixed 9x9 index grid chosen independently of outcome values","source_files":[{"name":fn,"bytes":int(row["bytes"]),"sha256":row["sha256"]} for fn,row in rows.items()]}
(OUT/"qa_input_manifest.json").write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps(meta,indent=2))
