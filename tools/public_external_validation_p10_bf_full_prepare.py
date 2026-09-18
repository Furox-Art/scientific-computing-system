#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
import requests
from scipy.io import loadmat

BASE="https://data.nemar.org/on004563/v1.0.0/"
MANIFEST=Path("tools/public_p10_source_bf_manifest.csv")
CHUNK=1048576

def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def download(url,expected,want,out):
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists() and out.stat().st_size==expected and sha256(out)==want: return
    out.unlink(missing_ok=True)
    parts=out.parent/(out.name+".parts"); shutil.rmtree(parts,ignore_errors=True); parts.mkdir()
    s=requests.Session(); start=0; i=0
    while start<expected:
        end=min(start+CHUNK-1,expected-1)
        for attempt in range(1,5):
            try:
                r=s.get(url,headers={"Range":f"bytes={start}-{end}","Accept-Encoding":"identity","User-Agent":"github-actions-p10-full-bf/1.0"},timeout=(30,180),allow_redirects=True)
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
        else: raise RuntimeError("unexpected response")
    if not out.exists():
        with out.open("wb") as dst:
            for p in sorted(parts.glob("part-*.bin")):
                with p.open("rb") as src: shutil.copyfileobj(src,dst)
    shutil.rmtree(parts,ignore_errors=True)
    if out.stat().st_size!=expected or sha256(out)!=want: raise RuntimeError(f"verify failed {out.name}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--shard",type=int,required=True)
    ap.add_argument("--n-shards",type=int,default=12)
    args=ap.parse_args()
    if 744%args.n_shards!=0: raise SystemExit("n_shards must divide 744 exactly")
    width=744//args.n_shards
    start=args.shard*width; stop=start+width
    outdir=Path(".public-runner/p10/bf-full")/f"shard-{args.shard:02d}"
    outdir.mkdir(parents=True,exist_ok=True)
    with MANIFEST.open(newline="") as f:
        rows={Path(r["path"]).name:r for r in csv.DictReader(f)}
    mats={}
    for fn in ["bayes_factors_VT_group.mat","bayes_factors_NoVT_group.mat","bayes_factors_groups.mat"]:
        row=rows[fn]; p=outdir/fn
        download(BASE+row["path"],int(row["bytes"]),row["sha256"].lower(),p)
        mats[fn]=loadmat(p)
    vt=mats["bayes_factors_VT_group.mat"]; nvt=mats["bayes_factors_NoVT_group.mat"]; bg=mats["bayes_factors_groups.mat"]
    X=np.asarray(vt["timegen_data_touch2vis"],dtype=float)
    Y=np.asarray(nvt["timegen_data_touch2vis"],dtype=float)
    sv=np.asarray(vt["bf_touch2vis"],dtype=float)
    sn=np.asarray(nvt["bf_touch2vis"],dtype=float)
    sb=np.asarray(bg["bf_BetweenGroups"],dtype=float)
    fields=["train_i","test_i","source_vt","source_nvt","source_between"]+[f"vt{i+1}" for i in range(X.shape[2])]+[f"nvt{i+1}" for i in range(Y.shape[2])]
    csvpath=outdir/f"shard-{args.shard:02d}-input.csv"
    with csvpath.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for test_i in range(start,stop):
            for train_i in range(744):
                row={"train_i":train_i,"test_i":test_i,"source_vt":repr(float(sv[train_i,test_i])),
                     "source_nvt":repr(float(sn[train_i,test_i])),"source_between":repr(float(sb[train_i,test_i]))}
                for k,v in enumerate(X[train_i,test_i,:]): row[f"vt{k+1}"]=repr(float(v))
                for k,v in enumerate(Y[train_i,test_i,:]): row[f"nvt{k+1}"]=repr(float(v))
                w.writerow(row)
    meta={"status":"SHARD_INPUT_READY","shard":args.shard,"n_shards":args.n_shards,"test_time_start":start,"test_time_stop_exclusive":stop,
          "cells":(stop-start)*744,"VT_n":X.shape[2],"NVT_n":Y.shape[2],
          "source_files":[{"name":fn,"bytes":int(rows[fn]["bytes"]),"sha256":rows[fn]["sha256"]} for fn in rows]}
    (outdir/f"shard-{args.shard:02d}-input-manifest.json").write_text(json.dumps(meta,indent=2)+"\n")
    for fn in mats: (outdir/fn).unlink(missing_ok=True)
    print(json.dumps(meta,indent=2))

if __name__=="__main__": main()
