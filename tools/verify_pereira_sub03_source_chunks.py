#!/usr/bin/env python3
import hashlib, json, tempfile, urllib.request
from pathlib import Path

URL="https://data.nemar.org/on001785/v1.0.0/derivatives/eegprep/sub-03/ses-01/eeg/sub-03_preproc_01hz.fdt"
SIZE=121605120
EXPECTED_MD5="213db2fb392ecbc06f18b5a2dda1b853"
CHUNK=48*1024*1024
OUT=Path("pereira_audit/SUB03_SOURCE_CHUNK_VERIFICATION_2026-09-22.json")

def git_blob_sha(raw):
    h=hashlib.sha1()
    h.update(f"blob {len(raw)}\0".encode())
    h.update(raw)
    return h.hexdigest()

whole_md5=hashlib.md5()
whole_sha=hashlib.sha256()
parts=[]
start=0; idx=1
while start<SIZE:
    end=min(start+CHUNK-1,SIZE-1)
    req=urllib.request.Request(URL,headers={"User-Agent":"GitHub-Hosted-Pereira-Verify/1.0","Range":f"bytes={start}-{end}"})
    with urllib.request.urlopen(req,timeout=900) as r:
        if getattr(r,"status",None)!=206:
            raise RuntimeError(f"Range not honored: {getattr(r,'status',None)}")
        raw=r.read()
    if len(raw)!=end-start+1:
        raise RuntimeError(f"size mismatch part {idx}: {len(raw)}")
    whole_md5.update(raw); whole_sha.update(raw)
    parts.append({
      "index":idx,"start":start,"end":end,"bytes":len(raw),
      "sha256":hashlib.sha256(raw).hexdigest(),
      "git_blob_sha1":git_blob_sha(raw)
    })
    del raw
    start=end+1; idx+=1
md5=whole_md5.hexdigest()
if md5!=EXPECTED_MD5: raise RuntimeError(f"MD5 mismatch {md5}")
out={
  "status":"PEREIRA_SUB03_SOURCE_CHUNKS_VERIFIED",
  "source_url":URL,"bytes":SIZE,"expected_md5":EXPECTED_MD5,
  "verified_md5":md5,"whole_sha256":whole_sha.hexdigest(),
  "chunk_bytes":CHUNK,"parts":parts,
  "compute_location":"GitHub-hosted ephemeral runner; no user PC/WSL raw persistence"
}
OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
