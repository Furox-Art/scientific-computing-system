#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, shutil, sys, time, traceback
from pathlib import Path
import requests

URLS=[
 "https://data.nemar.org/on004563/v1.0.0/derivatives/crossdecoding/sub-01_task-touchdecoding_crossdecoding.mat",
 "https://s3.amazonaws.com/openneuro.org/ds004563/derivatives/crossdecoding/sub-01_task-touchdecoding_crossdecoding.mat?versionId=dmmyVIKm0LMWq3b6lsN50OIfUTroTzuZ",
]
EXPECTED_BYTES=2731199
EXPECTED_SHA256="f99f426a2115b5b882febf080937cf7cc1676abde16729ff19e2890fe40a4385"
CHUNK=262144
OUT=Path(".public-runner/p10/sub-01_task-touchdecoding_crossdecoding.mat")
RESULT=Path(".public-runner/p10/result.json")

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def download(url):
    OUT.parent.mkdir(parents=True,exist_ok=True)
    parts=OUT.parent/"parts"
    shutil.rmtree(parts,ignore_errors=True); parts.mkdir()
    if OUT.exists(): OUT.unlink()
    s=requests.Session()
    rec=[]
    start=0; i=0
    while start<EXPECTED_BYTES:
        end=min(start+CHUNK-1,EXPECTED_BYTES-1)
        last=None
        for attempt in range(1,5):
            try:
                r=s.get(url,headers={"Range":f"bytes={start}-{end}","Accept-Encoding":"identity","User-Agent":"github-actions-public-validation/1.0"},timeout=(30,180),allow_redirects=True)
                if r.status_code not in (200,206): raise RuntimeError(f"HTTP {r.status_code}; final_url={r.url}")
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
            if not cr.lower().startswith(f"bytes {start}-{end}/".lower()):
                raise RuntimeError(f"unexpected Content-Range {cr!r}")
            p=parts/f"part-{i:05d}.bin"; p.write_bytes(body)
            rec.append({"i":i,"start":start,"end":end,"bytes":len(body),"status":206,"attempt":attempt,"final_host":requests.utils.urlparse(r.url).hostname})
            start=end+1; i+=1
        elif r.status_code==200 and start==0 and len(body)==EXPECTED_BYTES:
            OUT.write_bytes(body)
            rec.append({"i":0,"start":0,"end":EXPECTED_BYTES-1,"bytes":len(body),"status":200,"attempt":attempt,"note":"range ignored; exact full body returned","final_host":requests.utils.urlparse(r.url).hostname})
            break
        else:
            raise RuntimeError(f"range ignored unexpectedly status={r.status_code} received={len(body)}")
    if not OUT.exists():
        with OUT.open("wb") as dst:
            for p in sorted(parts.glob("part-*.bin")):
                with p.open("rb") as src: shutil.copyfileobj(src,dst)
    shutil.rmtree(parts,ignore_errors=True)
    return rec

def inspect_mat(path):
    out={}
    with path.open("rb") as f: out["header_ascii"]=f.read(128).decode("latin-1","replace").rstrip("\x00")
    try:
        import scipy.io
        out["scipy_whosmat"]=[{"name":n,"shape":list(sh),"class":cl} for n,sh,cl in scipy.io.whosmat(path)]
        out["format"]="scipy_mat"
        return out
    except Exception as e: out["scipy_error"]=repr(e)
    try:
        import h5py
        rows=[]
        with h5py.File(path,"r") as h5:
            def visit(name,obj):
                d={"name":name,"type":type(obj).__name__}
                if hasattr(obj,"shape"): d["shape"]=list(obj.shape)
                if hasattr(obj,"dtype"): d["dtype"]=str(obj.dtype)
                rows.append(d)
            h5.visititems(visit)
        out["hdf5_inventory"]=rows; out["format"]="hdf5_mat_v7_3"
    except Exception as e:
        out["hdf5_error"]=repr(e)
    return out

def main():
    result={"status":"STARTED","expected_bytes":EXPECTED_BYTES,"expected_sha256":EXPECTED_SHA256,"chunk_bytes":CHUNK,"routes":[],"scientific_effect_computed":False}
    try:
        for url in URLS:
            try:
                chunks=download(url)
                actual_bytes=OUT.stat().st_size; actual_sha=sha256(OUT)
                rr={"url":url,"actual_bytes":actual_bytes,"actual_sha256":actual_sha,"size_match":actual_bytes==EXPECTED_BYTES,"sha256_match":actual_sha==EXPECTED_SHA256,"chunks":chunks}
                result["routes"].append(rr)
                if actual_bytes==EXPECTED_BYTES and actual_sha==EXPECTED_SHA256:
                    result["status"]="EXACT_BINARY_ACQUIRED_AND_HASH_VERIFIED"
                    result["verified_route"]=url
                    result["mat_inventory"]=inspect_mat(OUT)
                    RESULT.parent.mkdir(parents=True,exist_ok=True)
                    RESULT.write_text(json.dumps(result,indent=2)+"\n")
                    print(json.dumps(result,indent=2))
                    return 0
            except Exception as e:
                result["routes"].append({"url":url,"error_type":type(e).__name__,"error":str(e)})
        raise RuntimeError("all official routes failed or verification did not pass")
    except Exception as e:
        result["status"]="ACQUISITION_FAILED"; result["error"]={"type":type(e).__name__,"message":str(e),"traceback":traceback.format_exc()}
        RESULT.parent.mkdir(parents=True,exist_ok=True); RESULT.write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(result,indent=2)); return 2

if __name__=="__main__": sys.exit(main())
