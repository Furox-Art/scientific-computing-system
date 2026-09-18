#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, json, shutil, sys, time, traceback
from pathlib import Path
import numpy as np
import requests
from scipy.io import loadmat, whosmat

MANIFEST=Path("tools/public_p10_source_bf_manifest.csv")
BASE="https://data.nemar.org/on004563/v1.0.0/"
CHUNK=262144
OUTDIR=Path(".public-runner/p10/source_bf")
RESULT=OUTDIR/"source_bf_result.json"

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def download(url, expected, out):
    out.parent.mkdir(parents=True,exist_ok=True)
    parts=out.parent/(out.name+".parts")
    shutil.rmtree(parts,ignore_errors=True); parts.mkdir()
    if out.exists(): out.unlink()
    s=requests.Session(); start=0; idx=0; rec=[]
    while start<expected:
        end=min(start+CHUNK-1,expected-1)
        for attempt in range(1,5):
            try:
                r=s.get(url,headers={"Range":f"bytes={start}-{end}","Accept-Encoding":"identity","User-Agent":"github-actions-public-validation/1.0"},timeout=(30,180),allow_redirects=True)
                if r.status_code not in (200,206): raise RuntimeError(f"HTTP {r.status_code}; final={r.url}")
                break
            except Exception:
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
        elif r.status_code==200 and start==0 and len(body)==expected:
            out.write_bytes(body); rec.append({"full_body":True,"bytes":len(body),"attempt":attempt}); break
        else:
            raise RuntimeError(f"unexpected non-range response status={r.status_code} received={len(body)}")
    if not out.exists():
        with out.open("wb") as dst:
            for p in sorted(parts.glob("part-*.bin")):
                with p.open("rb") as src: shutil.copyfileobj(src,dst)
    shutil.rmtree(parts,ignore_errors=True)
    return rec

def num_summary(a):
    x=np.asarray(a,dtype=float)
    finite=x[np.isfinite(x)]
    out={"shape":list(x.shape),"size":int(x.size),"finite":int(finite.size)}
    if finite.size:
        out.update({
            "min":float(np.min(finite)),
            "max":float(np.max(finite)),
            "mean":float(np.mean(finite)),
            "median":float(np.median(finite)),
            "q025":float(np.quantile(finite,0.025)),
            "q975":float(np.quantile(finite,0.975)),
        })
        if "bf" in CURRENT_VAR.lower():
            out.update({
                "count_gt_6":int(np.sum(finite>6)),
                "fraction_gt_6":float(np.mean(finite>6)),
                "count_lt_1_over_6":int(np.sum(finite<(1/6))),
                "fraction_lt_1_over_6":float(np.mean(finite<(1/6))),
                "count_gt_10":int(np.sum(finite>10)),
            })
        if "timegen_data" in CURRENT_VAR.lower():
            out.update({
                "fraction_gt_0_5":float(np.mean(finite>0.5)),
                "fraction_ge_0_52":float(np.mean(finite>=0.52)),
            })
    return out

CURRENT_VAR=""

def main():
    global CURRENT_VAR
    OUTDIR.mkdir(parents=True,exist_ok=True)
    result={"status":"STARTED","chunk_bytes":CHUNK,"files":[],"all_hash_verified":False}
    try:
        with MANIFEST.open(newline="",encoding="utf-8") as f:
            rows=list(csv.DictReader(f))
        for row in rows:
            rel=row["path"]; expected=int(row["bytes"]); want=row["sha256"].lower()
            out=OUTDIR/Path(rel).name
            chunks=download(BASE+rel,expected,out)
            got=sha256(out)
            if out.stat().st_size!=expected or got!=want:
                raise RuntimeError(f"{rel}: hash/size mismatch")
            vars_meta=[{"name":n,"shape":list(sh),"class":cl} for n,sh,cl in whosmat(out)]
            mat=loadmat(out,squeeze_me=False,struct_as_record=False)
            summaries={}
            for meta in vars_meta:
                n=meta["name"]
                if n.startswith("__"): continue
                CURRENT_VAR=n
                try:
                    summaries[n]=num_summary(mat[n])
                except Exception as e:
                    summaries[n]={"summary_error":repr(e),"shape":list(np.asarray(mat[n]).shape),"dtype":str(np.asarray(mat[n]).dtype)}
            result["files"].append({
                "path":rel,"expected_bytes":expected,"actual_bytes":out.stat().st_size,
                "expected_sha256":want,"actual_sha256":got,"size_match":True,"sha256_match":True,
                "chunk_count":len(chunks),"variables":vars_meta,"summaries":summaries
            })
            print(rel, vars_meta)
        result["all_hash_verified"]=True
        result["status"]="SOURCE_BF_DERIVATIVES_EXACT_HASH_VERIFIED_AND_PARSED"
        RESULT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(result,indent=2))
        return 0
    except Exception as e:
        result["status"]="SOURCE_BF_DERIVATIVE_RECOVERY_FAILED"
        result["error"]={"type":type(e).__name__,"message":str(e),"traceback":traceback.format_exc()}
        RESULT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(result,indent=2)); return 2

if __name__=="__main__":
    sys.exit(main())
