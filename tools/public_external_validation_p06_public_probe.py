#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,traceback
from pathlib import Path
import requests

NODE="vf9hw"
ENDPOINTS=[
 f"https://api.osf.io/v2/nodes/{NODE}/",
 f"https://api.osf.io/v2/nodes/{NODE}/files/",
 f"https://api.osf.io/v2/nodes/{NODE}/files/osfstorage/",
]
SUPP="https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41598-024-65862-z/MediaObjects/41598_2024_65862_MOESM1_ESM.docx"
OUT=Path(".public-runner/p06/public_access_probe.json")

def compact(j):
    if not isinstance(j,dict): return None
    x={"keys":list(j.keys())}
    d=j.get("data")
    if isinstance(d,list):
        x["data_count"]=len(d)
        x["data"]=[{
            "id":i.get("id"),"type":i.get("type"),
            "attributes":{k:(i.get("attributes") or {}).get(k) for k in ["name","kind","size","path","materialized_path"]},
            "links":{k:(i.get("links") or {}).get(k) for k in ["download"]}
        } for i in d[:100] if isinstance(i,dict)]
    elif isinstance(d,dict):
        x["data"]={"id":d.get("id"),"type":d.get("type"),"attributes":{k:(d.get("attributes") or {}).get(k) for k in ["title","public","current_user_permissions"]}}
    return x

def main():
    s=requests.Session()
    out={"status":"STARTED","node":NODE,"view_only_token_used":False,"endpoints":[],"supplement":{}}
    try:
        for url in ENDPOINTS:
            try:
                r=s.get(url,headers={"User-Agent":"github-actions-osf-public-probe/1.0"},timeout=(30,120),allow_redirects=True)
                rec={"url":url,"http_status":r.status_code,"final_url":r.url,"content_type":r.headers.get("content-type")}
                if "json" in (r.headers.get("content-type") or ""):
                    try: rec["json"]=compact(r.json())
                    except Exception as e: rec["json_error"]=repr(e)
                else:
                    rec["body_prefix"]=r.text[:300]
                out["endpoints"].append(rec)
            except Exception as e:
                out["endpoints"].append({"url":url,"error":repr(e)})
        try:
            r=s.get(SUPP,headers={"User-Agent":"github-actions-osf-public-probe/1.0"},timeout=(30,120),allow_redirects=True)
            out["supplement"]={"url":SUPP,"http_status":r.status_code,"final_url":r.url,"bytes":len(r.content),
                "sha256":hashlib.sha256(r.content).hexdigest() if r.ok else None,"content_type":r.headers.get("content-type")}
        except Exception as e:
            out["supplement"]={"url":SUPP,"error":repr(e)}
        provider=next((x for x in out["endpoints"] if x.get("url","").endswith("/files/osfstorage/")),{})
        data=((provider.get("json") or {}).get("data") if isinstance(provider.get("json"),dict) else None)
        out["public_file_index_acquired"]=bool(isinstance(data,list) and len(data)>0)
        out["status"]="P06_PUBLIC_OSF_PROBE_COMPLETE"
        OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(out,indent=2)+"\n")
        print(json.dumps(out,indent=2)); return 0
    except Exception as e:
        out["status"]="P06_PUBLIC_OSF_PROBE_FAILED"; out["error"]={"type":type(e).__name__,"message":str(e),"traceback":traceback.format_exc()}
        OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out,indent=2)); return 2

if __name__=="__main__": raise SystemExit(main())
