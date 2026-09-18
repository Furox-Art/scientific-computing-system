#!/usr/bin/env python3
from __future__ import annotations
import csv,hashlib,json,math,shutil,time
from pathlib import Path
import numpy as np, requests
from scipy.io import loadmat
BASE="https://data.nemar.org/on004563/v1.0.0/"; MAN=Path("tools/public_p10_source_bf_manifest.csv"); CHUNK=1048576
ROOT=Path(".public-runner/p10/bf_full"); SH=ROOT/"shards"; OUT=ROOT/"aggregate"; OUT.mkdir(parents=True,exist_ok=True)

def sha256(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()

def dl(row,out):
 exp=int(row["bytes"]);want=row["sha256"].lower()
 if out.exists() and out.stat().st_size==exp and sha256(out)==want:return
 out.unlink(missing_ok=True);parts=out.parent/(out.name+".parts");shutil.rmtree(parts,ignore_errors=True);parts.mkdir();s=requests.Session();start=0;i=0
 while start<exp:
  end=min(start+CHUNK-1,exp-1)
  for a in range(1,5):
   try:
    r=s.get(BASE+row["path"],headers={"Range":f"bytes={start}-{end}","Accept-Encoding":"identity","User-Agent":"github-actions-public-validation/1.0"},timeout=(30,180),allow_redirects=True)
    if r.status_code not in (200,206):raise RuntimeError(f"HTTP {r.status_code}")
    break
   except Exception:
    if a==4:raise
    time.sleep(min(2**(a-1),8))
  b=r.content
  if r.status_code==206:
   if len(b)!=end-start+1:raise RuntimeError("short")
   (parts/f"p-{i:04d}").write_bytes(b);start=end+1;i+=1
  elif r.status_code==200 and start==0 and len(b)==exp:out.write_bytes(b);break
  else:raise RuntimeError("unexpected")
 if not out.exists():
  with out.open("wb") as dst:
   for p in sorted(parts.glob("p-*")):
    with p.open("rb") as src:shutil.copyfileobj(src,dst)
 shutil.rmtree(parts,ignore_errors=True)
 if out.stat().st_size!=exp or sha256(out)!=want:raise RuntimeError("verify failed")

maps={k:np.full((744,744),np.nan,float) for k in ["VT","NVT","Between"]}
files=sorted(SH.glob("bf_output_shard_*.csv"))
if len(files)!=16:raise RuntimeError(f"expected 16 shard outputs, got {len(files)}")
seen=set()
for p in files:
 with p.open(newline="") as f:
  for r in csv.DictReader(f):
   i=int(r["train_i"]);j=int(r["test_i"]);key=(i,j)
   if key in seen:raise RuntimeError(f"duplicate {key}")
   seen.add(key);maps["VT"][i,j]=float(r["VT_BF"]);maps["NVT"][i,j]=float(r["NVT_BF"]);maps["Between"][i,j]=float(r["Between_BF"])
if len(seen)!=744*744 or any(np.isnan(x).any() for x in maps.values()):raise RuntimeError("incomplete map")

with MAN.open(newline="") as f:rows={Path(r["path"]).name:r for r in csv.DictReader(f)}
for fn in ["bayes_factors_VT_group.mat","bayes_factors_NoVT_group.mat","bayes_factors_groups.mat"]:dl(rows[fn],OUT/fn)
src={
 "VT":np.asarray(loadmat(OUT/"bayes_factors_VT_group.mat")["bf_touch2vis"],float),
 "NVT":np.asarray(loadmat(OUT/"bayes_factors_NoVT_group.mat")["bf_touch2vis"],float),
 "Between":np.asarray(loadmat(OUT/"bayes_factors_groups.mat")["bf_BetweenGroups"],float)
}
start=-1101.5625;step=3.90625
def summ(name,a,s):
 d=np.abs(a-s);rd=d/np.maximum(np.abs(s),np.finfo(float).eps);idx=np.unravel_index(np.argmax(a),a.shape)
 return {
  "cells":int(a.size),"BF_gt_6_cells":int(np.sum(a>6)),"BF_gt_6_fraction":float(np.mean(a>6)),
  "BF_lt_1_over_6_cells":int(np.sum(a<(1/6))),"BF_lt_1_over_6_fraction":float(np.mean(a<(1/6))),
  "max_BF":float(a[idx]),"max_coordinate":{"train_index":int(idx[0]),"test_index":int(idx[1]),"train_time_ms":start+idx[0]*step,"test_time_ms":start+idx[1]*step},
  "source_reproduction":{"max_abs_diff":float(np.max(d)),"median_abs_diff":float(np.median(d)),"max_rel_diff":float(np.max(rd)),"median_rel_diff":float(np.median(rd)),"pearson":float(np.corrcoef(a.ravel(),s.ravel())[0,1]),"pass":bool(np.max(d)<=1e-8 and np.max(rd)<=1e-6)}
 }
result={"status":"FULL_PRIMARY_TOUCH2VIS_BAYESFACTOR_RECOMPUTATION_COMPLETE","software":{"R":"4.3.3","BayesFactor":"0.9.12-4.2"},"maps":{k:summ(k,maps[k],src[k]) for k in maps},"all_source_reproduction_checks_pass":True}
result["all_source_reproduction_checks_pass"]=all(v["source_reproduction"]["pass"] for v in result["maps"].values())
if not result["all_source_reproduction_checks_pass"]:raise RuntimeError(json.dumps(result,indent=2))
np.savez_compressed(OUT/"full_primary_touch2vis_bf_maps.npz",VT=maps["VT"],NVT=maps["NVT"],Between=maps["Between"])
(OUT/"full_primary_bf_recomputation_result.json").write_text(json.dumps(result,indent=2)+"\n")
for fn in ["bayes_factors_VT_group.mat","bayes_factors_NoVT_group.mat","bayes_factors_groups.mat"]:(OUT/fn).unlink(missing_ok=True)
print(json.dumps(result,indent=2))
