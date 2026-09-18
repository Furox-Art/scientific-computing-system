#!/usr/bin/env python3
from __future__ import annotations
import csv,gzip,hashlib,json,math,shutil,time
from pathlib import Path
import numpy as np
import requests
from scipy.io import loadmat

BASE="https://data.nemar.org/on004563/v1.0.0/"
MANIFEST=Path("tools/public_p10_source_bf_manifest.csv")
ROOT=Path(".public-runner/p10/bf-full")
OUT=ROOT/"final"
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
        r=s.get(url,headers={"Range":f"bytes={start}-{end}","Accept-Encoding":"identity","User-Agent":"github-actions-p10-bf-aggregate/1.0"},timeout=(30,180),allow_redirects=True)
        if r.status_code==206:
            b=r.content
            if len(b)!=(end-start+1): raise RuntimeError("short range")
            (parts/f"part-{i:05d}.bin").write_bytes(b); start=end+1; i+=1
        elif r.status_code==200 and start==0 and len(r.content)==expected:
            out.write_bytes(r.content); break
        else: raise RuntimeError(f"HTTP {r.status_code}")
    if not out.exists():
        with out.open("wb") as dst:
            for p in sorted(parts.glob("part-*.bin")):
                with p.open("rb") as src: shutil.copyfileobj(src,dst)
    shutil.rmtree(parts,ignore_errors=True)
    if out.stat().st_size!=expected or sha256(out)!=want: raise RuntimeError("verify failed")

def metrics(source,recomp):
    ad=np.abs(recomp-source); rd=ad/np.maximum(np.abs(source),np.finfo(float).eps)
    idx=np.unravel_index(np.nanargmax(recomp),recomp.shape)
    return {
      "n_cells":int(source.size),
      "source_BF_gt_6":int(np.sum(source>6)),"recomputed_BF_gt_6":int(np.sum(recomp>6)),
      "BF_gt_6_classification_mismatches":int(np.sum((source>6)!=(recomp>6))),
      "source_BF_lt_1_over_6":int(np.sum(source<(1/6))),"recomputed_BF_lt_1_over_6":int(np.sum(recomp<(1/6))),
      "BF_lt_1_over_6_classification_mismatches":int(np.sum((source<(1/6))!=(recomp<(1/6)))),
      "max_abs_diff":float(np.max(ad)),"median_abs_diff":float(np.median(ad)),
      "max_rel_diff":float(np.max(rd)),"median_rel_diff":float(np.median(rd)),
      "pearson":float(np.corrcoef(source.ravel(),recomp.ravel())[0,1]),
      "source_max_BF":float(np.max(source)),"recomputed_max_BF":float(np.max(recomp)),
      "recomputed_max_index_zero_based":[int(idx[0]),int(idx[1])]
    }

OUT.mkdir(parents=True,exist_ok=True)
maps={k:np.full((744,744),np.nan,dtype=float) for k in ["vt","nvt","between"]}
seen=np.zeros((744,744),dtype=np.uint8)
files=sorted(ROOT.glob("shards/shard-*-recomputed.csv.gz"))
if len(files)!=12: raise SystemExit(f"expected 12 shard files, found {len(files)}")
for p in files:
    with gzip.open(p,"rt",newline="") as f:
        for row in csv.DictReader(f):
            i=int(row["train_i"]); j=int(row["test_i"])
            if seen[i,j]: raise RuntimeError(f"duplicate cell {(i,j)}")
            seen[i,j]=1
            maps["vt"][i,j]=float(row["recomputed_vt"])
            maps["nvt"][i,j]=float(row["recomputed_nvt"])
            maps["between"][i,j]=float(row["recomputed_between"])
if np.any(seen!=1): raise RuntimeError(f"missing cells {int(np.sum(seen==0))}")

with MANIFEST.open(newline="") as f:
    rows={Path(r["path"]).name:r for r in csv.DictReader(f)}
for fn in ["bayes_factors_VT_group.mat","bayes_factors_NoVT_group.mat","bayes_factors_groups.mat"]:
    row=rows[fn]; p=OUT/fn
    download(BASE+row["path"],int(row["bytes"]),row["sha256"].lower(),p)
vt=loadmat(OUT/"bayes_factors_VT_group.mat")["bf_touch2vis"].astype(float)
nvt=loadmat(OUT/"bayes_factors_NoVT_group.mat")["bf_touch2vis"].astype(float)
bg=loadmat(OUT/"bayes_factors_groups.mat")["bf_BetweenGroups"].astype(float)
result={
 "status":"P10_FULL_PRIMARY_TACTILE_TO_VISUAL_BAYESFACTOR_RECOMPUTATION_COMPLETE",
 "map_shape":[744,744],"cells_per_map":553536,"maps_recomputed":3,"total_ttestBF_calls":1660608,
 "source_implementation":{"BayesFactor_version":"0.9.12.4.2","one_sample":'mu=0.5,rscale="medium",nullInterval=c(0.5,Inf)',"between_groups":'mu=0,rscale="medium",nullInterval=c(0.5,Inf)'},
 "VT":metrics(vt,maps["vt"]),"NVT":metrics(nvt,maps["nvt"]),"between_groups":metrics(bg,maps["between"]),
 "coverage":{"expected_cells":553536,"seen_cells":int(np.sum(seen)),"missing_cells":int(np.sum(seen==0)),"duplicate_cells":0},
 "input_relation":"Participant arrays used for this recomputation are the frozen grouped arrays previously proven cell-for-cell identical to all 34 independently acquired individual crossdecoding derivatives.",
 "theory_claim":False
}
result["validation_pass"]=bool(
 result["coverage"]["missing_cells"]==0 and
 all(result[k]["BF_gt_6_classification_mismatches"]==0 and result[k]["BF_lt_1_over_6_classification_mismatches"]==0 and result[k]["pearson"]>0.999999999 for k in ["VT","NVT","between_groups"])
)
np.savez_compressed(OUT/"p10_primary_touch2vis_recomputed_bf_maps.npz",VT=maps["vt"],NVT=maps["nvt"],between_groups=maps["between"])
(OUT/"p10_primary_touch2vis_bf_recomputation_result.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
if not result["validation_pass"]: raise SystemExit(2)
