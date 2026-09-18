#!/usr/bin/env python3
from __future__ import annotations

import csv, hashlib, json, math, shutil, sys, time, traceback
from pathlib import Path
import numpy as np
import requests
from scipy.io import loadmat

MANIFEST=Path("tools/public_p10_crossdecoding_manifest.csv")
BASE="https://data.nemar.org/on004563/v1.0.0/"
CHUNK=262144
OUTDIR=Path(".public-runner/p10/bulk")
FILES=OUTDIR/"files"
RESULT=OUTDIR/"bulk_descriptive_result.json"
CSVOUT=OUTDIR/"participant_descriptives.csv"

VT={8,9,12,14,20,21,23,28,29,30,31,35,36,38,39,40}
NVT={1,2,3,4,5,6,7,11,15,17,19,22,24,25,26,27,34,37}
INCLUDED=VT|NVT

def shasum(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def download(url, expected_bytes, out):
    if out.exists() and out.stat().st_size==expected_bytes:
        return {"reused_existing":True,"chunks":0}
    out.parent.mkdir(parents=True,exist_ok=True)
    parts=out.parent/(out.name+".parts")
    shutil.rmtree(parts,ignore_errors=True); parts.mkdir()
    s=requests.Session(); rec=[]; start=0; idx=0
    while start<expected_bytes:
        end=min(start+CHUNK-1, expected_bytes-1)
        last=None
        for attempt in range(1,5):
            try:
                r=s.get(url,headers={"Range":f"bytes={start}-{end}","Accept-Encoding":"identity","User-Agent":"github-actions-public-validation/1.0"},timeout=(30,180),allow_redirects=True)
                if r.status_code not in (200,206): raise RuntimeError(f"HTTP {r.status_code}; final={r.url}")
                break
            except Exception as e:
                last=e
                if attempt==4: raise
                time.sleep(min(2**(attempt-1),8))
        body=r.content
        if r.status_code==206:
            need=end-start+1
            if len(body)!=need: raise RuntimeError(f"short chunk {len(body)} != {need}")
            cr=r.headers.get("Content-Range","")
            if not cr.lower().startswith(f"bytes {start}-{end}/".lower()): raise RuntimeError(f"bad Content-Range {cr!r}")
            p=parts/f"part-{idx:05d}.bin"; p.write_bytes(body)
            rec.append({"i":idx,"start":start,"end":end,"bytes":len(body),"attempt":attempt,"host":requests.utils.urlparse(r.url).hostname})
            start=end+1; idx+=1
        elif r.status_code==200 and start==0 and len(body)==expected_bytes:
            out.write_bytes(body); rec.append({"full_body":True,"bytes":len(body),"attempt":attempt}); break
        else:
            raise RuntimeError(f"range ignored unexpectedly status={r.status_code} received={len(body)}")
    if not out.exists():
        with out.open("wb") as dst:
            for p in sorted(parts.glob("part-*.bin")):
                with p.open("rb") as src: shutil.copyfileobj(src,dst)
    shutil.rmtree(parts,ignore_errors=True)
    return {"reused_existing":False,"chunks":len(rec),"chunk_records":rec}

def mat_struct_summary(x):
    d={"type":type(x).__name__}
    if hasattr(x,"_fieldnames"): d["fields"]=list(x._fieldnames)
    for name in getattr(x,"_fieldnames",[]) or []:
        v=getattr(x,name)
        if isinstance(v,np.ndarray):
            d[name]={"shape":list(v.shape),"dtype":str(v.dtype),"size":int(v.size)}
        elif hasattr(v,"_fieldnames"):
            d[name]={"type":type(v).__name__,"fields":list(v._fieldnames)}
        else:
            try:
                a=np.asarray(v)
                d[name]={"shape":list(a.shape),"dtype":str(a.dtype),"size":int(a.size)}
            except Exception:
                d[name]={"type":type(v).__name__}
    return d

def sample_vector(obj):
    if not hasattr(obj,"samples"): raise RuntimeError("CoSMo struct has no samples field")
    a=np.asarray(obj.samples,dtype=float)
    return a.reshape(-1)

def stats(v):
    finite=v[np.isfinite(v)]
    return {
        "n_cells":int(v.size),
        "n_finite":int(finite.size),
        "mean":float(np.mean(finite)),
        "median":float(np.median(finite)),
        "sd":float(np.std(finite,ddof=1)),
        "min":float(np.min(finite)),
        "max":float(np.max(finite)),
        "fraction_gt_chance":float(np.mean(finite>0.5)),
        "fraction_eq_chance":float(np.mean(finite==0.5)),
        "q025":float(np.quantile(finite,0.025)),
        "q975":float(np.quantile(finite,0.975)),
    }

def main():
    OUTDIR.mkdir(parents=True,exist_ok=True); FILES.mkdir(parents=True,exist_ok=True)
    result={"status":"STARTED","chunk_bytes":CHUNK,"included_ids":sorted(INCLUDED),"VT_ids":sorted(VT),"NVT_ids":sorted(NVT),"participants":[],"all_hash_verified":False,"effect_inference_computed":False}
    rows_out=[]
    try:
        with MANIFEST.open(newline="",encoding="utf-8") as f:
            rows=list(csv.DictReader(f))
        by_sub={}
        for row in rows:
            s=int(row["participant_id"].split("-")[1])
            by_sub[s]=row
        missing=sorted(INCLUDED-set(by_sub))
        if missing: raise RuntimeError(f"included participants missing from manifest: {missing}")

        vectors_t2v=[]; vectors_v2t=[]; groups=[]; ids=[]
        first_structure=None
        for n,sid in enumerate(sorted(INCLUDED),1):
            row=by_sub[sid]
            rel=row["derivative_path"]; expected=int(row["bytes"]); want=row["sha256"].lower()
            out=FILES/Path(rel).name
            dl=download(BASE+rel,expected,out)
            got=shasum(out)
            if out.stat().st_size!=expected or got!=want:
                raise RuntimeError(f"sub-{sid:02d} verification failed")
            mat=loadmat(out,squeeze_me=True,struct_as_record=False)
            t2v=mat["res_timegen_touch2vis"]; v2t=mat["res_timegen_vis2touch"]
            if first_structure is None:
                first_structure={"touch2vis":mat_struct_summary(t2v),"vis2touch":mat_struct_summary(v2t)}
            a=sample_vector(t2v); b=sample_vector(v2t)
            if a.size!=744*744 or b.size!=744*744:
                raise RuntimeError(f"sub-{sid:02d}: unexpected cell count t2v={a.size}, v2t={b.size}")
            st=stats(a); sv=stats(b); grp="VT" if sid in VT else "NVT"
            vectors_t2v.append(a.astype(np.float32)); vectors_v2t.append(b.astype(np.float32)); groups.append(grp); ids.append(sid)
            rec={"participant_id":f"sub-{sid:02d}","group":grp,"path":rel,"bytes":expected,"sha256":want,"hash_verified":True,"download_chunks":dl.get("chunks",0),"touch2vis":st,"vis2touch":sv}
            result["participants"].append(rec)
            rows_out.append({
                "participant_id":f"sub-{sid:02d}","group":grp,
                "touch2vis_mean":st["mean"],"touch2vis_median":st["median"],"touch2vis_sd":st["sd"],"touch2vis_min":st["min"],"touch2vis_max":st["max"],"touch2vis_fraction_gt_0_5":st["fraction_gt_chance"],
                "vis2touch_mean":sv["mean"],"vis2touch_median":sv["median"],"vis2touch_sd":sv["sd"],"vis2touch_min":sv["min"],"vis2touch_max":sv["max"],"vis2touch_fraction_gt_0_5":sv["fraction_gt_chance"]
            })
            print(f"[{n}/{len(INCLUDED)}] sub-{sid:02d} {grp} verified; t2v mean={st['mean']:.6f}; v2t mean={sv['mean']:.6f}")

        A=np.stack(vectors_t2v,axis=0); B=np.stack(vectors_v2t,axis=0)
        g=np.array(groups); idsarr=np.array(ids)
        def group_summary(X,mask):
            gm=np.mean(X[mask],axis=0)
            return {
                "participants":int(np.sum(mask)),
                "participant_global_mean_mean":float(np.mean(np.mean(X[mask],axis=1))),
                "participant_global_mean_sd":float(np.std(np.mean(X[mask],axis=1),ddof=1)),
                "group_mean_matrix_global_mean":float(np.mean(gm)),
                "group_mean_matrix_min":float(np.min(gm)),
                "group_mean_matrix_max":float(np.max(gm)),
                "group_mean_matrix_fraction_gt_chance":float(np.mean(gm>0.5)),
                "group_mean_matrix_fraction_ge_0_52":float(np.mean(gm>=0.52)),
            }
        result["first_struct_structure"]=first_structure
        result["descriptive_groups"]={
            "touch2vis":{"VT":group_summary(A,g=="VT"),"NVT":group_summary(A,g=="NVT")},
            "vis2touch":{"VT":group_summary(B,g=="VT"),"NVT":group_summary(B,g=="NVT")}
        }
        result["descriptive_between_group_global_mean_difference"]={
            "touch2vis_VT_minus_NVT":float(np.mean(np.mean(A[g=="VT"],axis=1))-np.mean(np.mean(A[g=="NVT"],axis=1))),
            "vis2touch_VT_minus_NVT":float(np.mean(np.mean(B[g=="VT"],axis=1))-np.mean(np.mean(B[g=="NVT"],axis=1)))
        }
        result["all_hash_verified"]=True
        result["status"]="BULK_34_INCLUDED_EXACT_HASH_VERIFIED_DESCRIPTIVES_COMPLETE"

        with CSVOUT.open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(rows_out[0].keys())); w.writeheader(); w.writerows(rows_out)
        RESULT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(result["descriptive_groups"],indent=2))
        return 0
    except Exception as e:
        result["status"]="BULK_ACQUISITION_OR_PARSE_FAILED"
        result["error"]={"type":type(e).__name__,"message":str(e),"traceback":traceback.format_exc()}
        RESULT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(result,indent=2))
        return 2

if __name__=="__main__":
    sys.exit(main())
