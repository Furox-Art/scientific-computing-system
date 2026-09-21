#!/usr/bin/env python3
import hashlib, json, requests
from pathlib import Path
OUT=Path("sato2018_raw_acquisition_session"); OUT.mkdir(exist_ok=True)
landing="https://datadryad.org/dataset/doi:10.5061/dryad.6675p"
legacy="https://datadryad.org/downloads/file_stream/61164"
headers={
 "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36",
 "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
 "Accept-Language":"en-US,en;q=0.9"
}
s=requests.Session()
log={}
r=s.get(landing,headers=headers,timeout=60,allow_redirects=True)
log["landing"]={"status":r.status_code,"final_url":r.url,"cookies":list(s.cookies.keys()),"content_type":r.headers.get("content-type"),"bytes":len(r.content)}
fh=headers.copy()
fh.update({
 "Accept":"text/csv,text/plain,*/*;q=0.8",
 "Referer":r.url,
 "Sec-Fetch-Dest":"document",
 "Sec-Fetch-Mode":"navigate",
 "Sec-Fetch-Site":"same-origin",
 "Sec-Fetch-User":"?1"
})
d=s.get(legacy,headers=fh,timeout=60,allow_redirects=True)
log["legacy"]={"status":d.status_code,"final_url":d.url,"content_type":d.headers.get("content-type"),"content_disposition":d.headers.get("content-disposition"),"bytes":len(d.content),"history":[{"status":x.status_code,"url":x.url,"location":x.headers.get("location")} for x in d.history]}
p=OUT/"data_able-bodied.csv"
if d.status_code==200 and len(d.content)>100 and not (d.headers.get("content-type") or "").startswith("text/html"):
    p.write_bytes(d.content)
    log["success"]=True
    log["sha256"]=hashlib.sha256(d.content).hexdigest()
    log["head"]=d.text[:500]
else:
    log["success"]=False
    log["body_prefix"]=d.text[:1000]
(OUT/"probe.json").write_text(json.dumps(log,indent=2))
print(json.dumps(log,indent=2))
raise SystemExit(0 if log["success"] else 2)
