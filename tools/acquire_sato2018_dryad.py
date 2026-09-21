#!/usr/bin/env python3
import hashlib, json, urllib.request
from pathlib import Path
OUT=Path("sato2018_raw_acquisition"); OUT.mkdir(exist_ok=True)
url="https://datadryad.org/api/v2/files/61164/download"
p=OUT/"data_able-bodied.csv"
req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0","Accept":"text/csv,*/*"})
try:
    with urllib.request.urlopen(req,timeout=120) as r, open(p,"wb") as f:
        while True:
            b=r.read(1024*1024)
            if not b: break
            f.write(b)
    h=hashlib.sha256(p.read_bytes()).hexdigest()
    text=p.read_text(encoding="utf-8-sig",errors="replace").splitlines()
    header=text[0] if text else ""
    out={"status":"success","url":url,"bytes":p.stat().st_size,"sha256":h,"header":header,"data_rows":max(0,len(text)-1)}
except Exception as e:
    out={"status":"failure","url":url,"error":repr(e)}
(OUT/"acquisition.json").write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
if out["status"]!="success": raise SystemExit(2)
