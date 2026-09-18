#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, shutil, sys, time, traceback
from pathlib import Path
import requests

PATH="sub-Bubbles/ses-01/func/sub-Bubbles_ses-01_task-SORPF_run-1_echo-1_part-mag_bold.nii.gz"
EXPECTED_BYTES=172087381
EXPECTED_MD5="f252ffb803aa079f0378c3ce0f527976"
VERSION_ID="2vSFHKVfu45OF7u_9uOGKkAQ8HBGeQBO"
URLS=[
 f"https://s3.amazonaws.com/openneuro.org/ds002278/{PATH}?versionId={VERSION_ID}",
 f"https://openneuro.org/crn/datasets/ds002278/snapshots/2.0.0/files/{PATH.replace('/',':')}",
]
CHUNK=1048576
OUTDIR=Path(".public-runner/p08")
OUT=OUTDIR/Path(PATH).name
RESULT=OUTDIR/"first_bold_exact_acquisition.json"

def hashes(p):
    md5=hashlib.md5(); sha=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):
            md5.update(b); sha.update(b)
    return md5.hexdigest(),sha.hexdigest()

def download(url):
    OUTDIR.mkdir(parents=True,exist_ok=True)
    OUT.unlink(missing_ok=True)
    parts=OUTDIR/"parts"; shutil.rmtree(parts,ignore_errors=True); parts.mkdir()
    s=requests.Session(); start=0; i=0; rec=[]
    while start<EXPECTED_BYTES:
        end=min(start+CHUNK-1,EXPECTED_BYTES-1)
        for attempt in range(1,5):
            try:
                r=s.get(url,headers={"Range":f"bytes={start}-{end}","Accept-Encoding":"identity","User-Agent":"github-actions-openneuro-validation/1.0"},timeout=(30,180),allow_redirects=True)
                if r.status_code not in (200,206): raise RuntimeError(f"HTTP {r.status_code}; final={r.url}")
                break
            except Exception:
                if attempt==4: raise
                time.sleep(min(2**(attempt-1),8))
        b=r.content
        if r.status_code==206:
            need=end-start+1
            if len(b)!=need: raise RuntimeError(f"short chunk {len(b)} != {need}")
            cr=r.headers.get("Content-Range","")
            if not cr.lower().startswith(f"bytes {start}-{end}/".lower()): raise RuntimeError(f"bad Content-Range {cr!r}")
            (parts/f"part-{i:05d}.bin").write_bytes(b)
            rec.append({"i":i,"start":start,"end":end,"bytes":len(b),"attempt":attempt,"host":requests.utils.urlparse(r.url).hostname})
            start=end+1; i+=1
        elif r.status_code==200 and start==0 and len(b)==EXPECTED_BYTES:
            OUT.write_bytes(b); rec.append({"full_body":True,"bytes":len(b),"attempt":attempt,"host":requests.utils.urlparse(r.url).hostname}); break
        else: raise RuntimeError(f"unexpected response status={r.status_code}, bytes={len(b)}")
    if not OUT.exists():
        with OUT.open("wb") as dst:
            for p in sorted(parts.glob("part-*.bin")):
                with p.open("rb") as src: shutil.copyfileobj(src,dst)
    shutil.rmtree(parts,ignore_errors=True)
    return rec

def main():
    result={"status":"STARTED","dataset":"ds002278 v2.0.0","path":PATH,"expected_bytes":EXPECTED_BYTES,"expected_md5":EXPECTED_MD5,"version_id":VERSION_ID,"chunk_bytes":CHUNK,"routes":[],"neural_effect_computed":False}
    try:
        for url in URLS:
            try:
                chunks=download(url)
                md5,sha=hashes(OUT)
                rr={"url":url,"actual_bytes":OUT.stat().st_size,"actual_md5":md5,"actual_sha256":sha,
                    "size_match":OUT.stat().st_size==EXPECTED_BYTES,"md5_match":md5==EXPECTED_MD5,
                    "chunk_count":len(chunks),"first_chunk":chunks[0] if chunks else None,"last_chunk":chunks[-1] if chunks else None}
                result["routes"].append(rr)
                if rr["size_match"] and rr["md5_match"]:
                    result["status"]="P08_FIRST_RAW_BOLD_EXACT_BYTES_ACQUIRED_AND_MD5_VERIFIED"
                    result["verified_route"]=url
                    RESULT.write_text(json.dumps(result,indent=2)+"\n")
                    print(json.dumps(result,indent=2))
                    return 0
            except Exception as e:
                result["routes"].append({"url":url,"error_type":type(e).__name__,"error":str(e)})
        raise RuntimeError("all exact routes failed verification")
    except Exception as e:
        result["status"]="P08_FIRST_RAW_BOLD_ACQUISITION_FAILED"
        result["error"]={"type":type(e).__name__,"message":str(e),"traceback":traceback.format_exc()}
        RESULT.write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(result,indent=2)); return 2

if __name__=="__main__": sys.exit(main())
