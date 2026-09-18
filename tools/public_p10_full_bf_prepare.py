#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,hashlib,shutil,time
from pathlib import Path
import numpy as np, requests
from scipy.io import loadmat

BASE="https://data.nemar.org/on004563/v1.0.0/"
MAN=Path("tools/public_p10_source_bf_manifest.csv")
CHUNK=1048576

def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def dl(row,out):
    exp=int(row["bytes"]); want=row["sha256"].lower()
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists() and out.stat().st_size==exp and sha256(out)==want:return
    out.unlink(missing_ok=True); parts=out.parent/(out.name+".parts"); shutil.rmtree(parts,ignore_errors=True); parts.mkdir()
    s=requests.Session(); start=0;i=0
    while start<exp:
        end=min(start+CHUNK-1,exp-1)
        for a in range(1,5):
            try:
                r=s.get(BASE+row["path"],headers={"Range":f"bytes={start}-{end}","Accept-Encoding":"identity","User-Agent":"github-actions-public-validation/1.0"},timeout=(30,180),allow_redirects=True)
                if r.status_code not in (200,206): raise RuntimeError(f"HTTP {r.status_code}")
                break
            except Exception:
                if a==4: raise
                time.sleep(min(2**(a-1),8))
        b=r.content
        if r.status_code==206:
            if len(b)!=end-start+1: raise RuntimeError("short range")
            (parts/f"p-{i:04d}").write_bytes(b);start=end+1;i+=1
        elif r.status_code==200 and start==0 and len(b)==exp:
            out.write_bytes(b);break
        else: raise RuntimeError("unexpected response")
    if not out.exists():
        with out.open("wb") as dst:
            for p in sorted(parts.glob("p-*")):
                with p.open("rb") as src: shutil.copyfileobj(src,dst)
    shutil.rmtree(parts,ignore_errors=True)
    if out.stat().st_size!=exp or sha256(out)!=want: raise RuntimeError("hash/size verification failed")

ap=argparse.ArgumentParser();ap.add_argument("--shard",type=int,required=True);ap.add_argument("--n-shards",type=int,required=True);a=ap.parse_args()
out=Path(".public-runner/p10/bf_full")/f"shard_{a.shard:02d}";out.mkdir(parents=True,exist_ok=True)
with MAN.open(newline="") as f: rows={Path(r["path"]).name:r for r in csv.DictReader(f)}
for fn in ["bayes_factors_VT_group.mat","bayes_factors_NoVT_group.mat"]:
    dl(rows[fn],out/fn)
vt=np.asarray(loadmat(out/"bayes_factors_VT_group.mat")["timegen_data_touch2vis"],float)
nvt=np.asarray(loadmat(out/"bayes_factors_NoVT_group.mat")["timegen_data_touch2vis"],float)
if vt.shape!=(744,744,16) or nvt.shape!=(744,744,18): raise RuntimeError(f"bad shapes {vt.shape} {nvt.shape}")
fields=["train_i","test_i"]+[f"x{i+1}" for i in range(16)]+[f"y{i+1}" for i in range(18)]
p=out/f"bf_input_shard_{a.shard:02d}.csv"
with p.open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
    for ti in range(744):
        if ti%a.n_shards!=a.shard: continue
        for tj in range(744):
            row={"train_i":ti,"test_i":tj}
            for k,v in enumerate(vt[ti,tj,:]): row[f"x{k+1}"]=repr(float(v))
            for k,v in enumerate(nvt[ti,tj,:]): row[f"y{k+1}"]=repr(float(v))
            w.writerow(row)
(out/"bayes_factors_VT_group.mat").unlink();(out/"bayes_factors_NoVT_group.mat").unlink()
print(p)
