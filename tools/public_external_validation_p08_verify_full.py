#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,hashlib,json,time,traceback
from pathlib import Path
import requests

BASE="https://s3.amazonaws.com/openneuro.org/ds002278/"
MANIFEST=Path(".public-runner/p08/raw_preprocessing_input_manifest_2026-09-18.csv")
OUTROOT=Path(".public-runner/p08/full-acquisition")
CHUNK=16*1024*1024

def verify_one(session,url,expected_bytes,algo,want):
    h=hashlib.md5() if algo=="md5" else hashlib.sha256()
    start=0; requests_n=0; max_attempt=0; hosts=set()
    while start<expected_bytes:
        end=min(start+CHUNK-1,expected_bytes-1)
        for attempt in range(1,5):
            try:
                r=session.get(url,headers={"Range":f"bytes={start}-{end}","Accept-Encoding":"identity","User-Agent":"github-actions-openneuro-full-validation/1.0"},timeout=(30,240),allow_redirects=True)
                if r.status_code not in (200,206):
                    raise RuntimeError(f"HTTP {r.status_code}; final={r.url}")
                break
            except Exception:
                if attempt==4: raise
                time.sleep(min(2**(attempt-1),8))
        max_attempt=max(max_attempt,attempt); requests_n+=1
        hosts.add(requests.utils.urlparse(r.url).hostname)
        body=r.content
        if r.status_code==206:
            need=end-start+1
            if len(body)!=need: raise RuntimeError(f"short chunk {len(body)} != {need}")
            cr=r.headers.get("Content-Range","")
            if not cr.lower().startswith(f"bytes {start}-{end}/".lower()):
                raise RuntimeError(f"bad Content-Range {cr!r}")
            h.update(body); start=end+1
        elif r.status_code==200 and start==0 and len(body)==expected_bytes:
            h.update(body); start=expected_bytes
        else:
            raise RuntimeError(f"range ignored unexpectedly status={r.status_code} received={len(body)}")
    got=h.hexdigest()
    return {
        "actual_bytes":start,
        "hash_algorithm":algo,
        "expected_hash":want,
        "actual_hash":got,
        "hash_match":got.lower()==want.lower(),
        "size_match":start==expected_bytes,
        "http_requests":requests_n,
        "max_attempt":max_attempt,
        "hosts":sorted(x for x in hosts if x)
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--participant",required=True,choices=["sub-Bubbles","sub-Buttercup","sub-PILOT02"])
    args=ap.parse_args()
    outdir=OUTROOT/args.participant
    outdir.mkdir(parents=True,exist_ok=True)
    result={
      "status":"STARTED","participant":args.participant,
      "chunk_bytes":CHUNK,"files":[],"all_verified":False,
      "neural_effect_computed":False
    }
    try:
        with MANIFEST.open(newline="",encoding="utf-8") as f:
            rows=[r for r in csv.DictReader(f) if r["storage"]=="git_annex" and r["path"].startswith(args.participant+"/")]
        if not rows: raise RuntimeError("no annex rows for participant")
        expected_total=sum(int(r["expected_bytes"]) for r in rows)
        result["expected_files"]=len(rows); result["expected_bytes"]=expected_total
        s=requests.Session(); verified_bytes=0
        for i,row in enumerate(rows,1):
            path=row["path"]; url=BASE+path
            t0=time.time()
            rec=verify_one(s,url,int(row["expected_bytes"]),row["hash_algorithm"],row["expected_hash"])
            rec.update({
              "path":path,"role":row["role"],"expected_bytes":int(row["expected_bytes"]),
              "annex_key":row["annex_key"],"url":url,"elapsed_seconds":time.time()-t0
            })
            result["files"].append(rec)
            if not rec["hash_match"] or not rec["size_match"]:
                raise RuntimeError(f"verification failed for {path}")
            verified_bytes+=rec["actual_bytes"]
            print(f"[{i}/{len(rows)}] {path} verified {rec['actual_bytes']} B {rec['actual_hash']}",flush=True)
        result["verified_files"]=len(result["files"])
        result["verified_bytes"]=verified_bytes
        result["all_verified"]=result["verified_files"]==len(rows) and verified_bytes==expected_total
        result["status"]="ALL_PARTICIPANT_PREPROCESSING_ANNEX_BYTES_EXACT_HASH_VERIFIED"
        (outdir/"acquisition_result.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        with (outdir/"file_verification.csv").open("w",newline="",encoding="utf-8") as f:
            fields=["path","role","expected_bytes","actual_bytes","hash_algorithm","expected_hash","actual_hash","hash_match","size_match","http_requests","max_attempt","elapsed_seconds","url"]
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
            for x in result["files"]:
                w.writerow({k:x.get(k) for k in fields})
        print(json.dumps({k:result[k] for k in ["status","participant","expected_files","verified_files","expected_bytes","verified_bytes","all_verified"]},indent=2))
        return 0
    except Exception as e:
        result["status"]="PARTICIPANT_PREPROCESSING_ANNEX_VERIFICATION_FAILED"
        result["error"]={"type":type(e).__name__,"message":str(e),"traceback":traceback.format_exc()}
        (outdir/"acquisition_result.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(result,indent=2))
        return 2

if __name__=="__main__":
    raise SystemExit(main())
