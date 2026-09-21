#!/usr/bin/env python3
import csv, hashlib, json, os, re, sys, urllib.request, urllib.parse, zipfile
from pathlib import Path

OUT=Path("direct_raw_acquisition")
RAW=OUT/"raw"; RAW.mkdir(parents=True, exist_ok=True)
manifest=[]; schema=[]; errors=[]

def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def get(url, dest, headers=None):
    req=urllib.request.Request(url,headers=headers or {"User-Agent":"Mozilla/5.0"})
    with urllib.request.urlopen(req,timeout=120) as r, open(dest,"wb") as f:
        while True:
            b=r.read(1024*1024)
            if not b: break
            f.write(b)
    manifest.append({"path":str(dest),"source_url":url,"bytes":dest.stat().st_size,"sha256":sha256(dest)})
    return dest

def text_json(url):
    req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0","Accept":"application/json"})
    with urllib.request.urlopen(req,timeout=120) as r:
        return json.load(r)

def csv_schema(p):
    try:
        with open(p,"r",encoding="utf-8-sig",errors="replace",newline="") as f:
            rd=csv.reader(f); header=next(rd,[]); n=sum(1 for _ in rd)
        return {"path":str(p),"kind":"csv","columns":header,"data_rows":n}
    except Exception as e:
        return {"path":str(p),"kind":"csv","schema_error":repr(e)}

# Braun / PLOS Supporting Information
try:
    u="https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0111967.s001&type=supplementary"
    z=get(u,RAW/"braun2014_s001.zip")
    with zipfile.ZipFile(z) as zz:
        for info in zz.infolist():
            if info.is_dir(): continue
            target=RAW/"braun2014"/info.filename
            target.parent.mkdir(parents=True,exist_ok=True)
            with zz.open(info) as src, open(target,"wb") as dst: dst.write(src.read())
            manifest.append({"path":str(target),"container":"braun2014_s001.zip","bytes":target.stat().st_size,"sha256":sha256(target)})
            if target.suffix.lower() in [".csv",".tsv",".txt"]:
                schema.append(csv_schema(target))
            else:
                schema.append({"path":str(target),"kind":target.suffix.lower(),"bytes":target.stat().st_size})
except Exception as e:
    errors.append({"dataset":"braun2014","error":repr(e)})

# OSF vf9hw recursive file acquisition
try:
    token="513a8c23ad4145418ea7f5dffad526f1"
    queue=[f"https://api.osf.io/v2/nodes/vf9hw/files/osfstorage/?view_only={token}"]
    seen=set()
    while queue:
        url=queue.pop(0)
        if url in seen: continue
        seen.add(url)
        obj=text_json(url)
        for item in obj.get("data",[]):
            attrs=item.get("attributes",{}); links=item.get("links",{})
            kind=attrs.get("kind"); name=attrs.get("name") or item.get("id","file")
            if kind=="folder":
                rel=item.get("relationships",{}).get("files",{}).get("links",{}).get("related",{}).get("href")
                if rel:
                    sep="&" if "?" in rel else "?"
                    queue.append(rel+sep+"view_only="+token)
            elif kind=="file":
                dl=links.get("download")
                if dl:
                    sep="&" if "?" in dl else "?"
                    dl2=dl+sep+"view_only="+token
                    safe=re.sub(r"[^A-Za-z0-9._-]+","_",name)
                    p=get(dl2,RAW/"ciaunica_vf9hw"/safe)
                    if p.suffix.lower() in [".csv",".tsv",".txt"]:
                        schema.append(csv_schema(p))
                    else:
                        schema.append({"path":str(p),"kind":p.suffix.lower(),"bytes":p.stat().st_size})
        nxt=obj.get("links",{}).get("next")
        if nxt:
            sep="&" if "?" in nxt else "?"
            queue.append(nxt+sep+"view_only="+token)
except Exception as e:
    errors.append({"dataset":"ciaunica_vf9hw","error":repr(e)})

# Dryad dataset page -> locate data_able-bodied.csv download href
try:
    page="https://datadryad.org/dataset/doi%3A10.5061%2Fdryad.6675p"
    req=urllib.request.Request(page,headers={"User-Agent":"Mozilla/5.0"})
    html=urllib.request.urlopen(req,timeout=120).read().decode("utf-8","replace")
    candidates=re.findall(r'href=["\']([^"\']+)["\']',html)
    chosen=None
    for href in candidates:
        if "data_able-bodied.csv" in href or "file_stream" in href:
            chosen=urllib.parse.urljoin(page,href)
            if "data_able-bodied.csv" in href: break
    if not chosen:
        # fallback API discovery
        api="https://datadryad.org/api/v2/datasets/doi%3A10.5061%2Fdryad.6675p"
        meta=text_json(api)
        manifest.append({"dryad_api_metadata":meta})
        raise RuntimeError("Dryad file link not found on landing page")
    p=get(chosen,RAW/"sato2018_data_able-bodied.csv")
    schema.append(csv_schema(p))
except Exception as e:
    errors.append({"dataset":"sato2018","error":repr(e)})

OUT.mkdir(exist_ok=True)
with open(OUT/"manifest.json","w") as f: json.dump(manifest,f,indent=2)
with open(OUT/"schema_report.json","w") as f: json.dump(schema,f,indent=2)
with open(OUT/"errors.json","w") as f: json.dump(errors,f,indent=2)

summary={"manifest_entries":len(manifest),"schema_entries":len(schema),"errors":errors}
print(json.dumps(summary,indent=2))
if not any("braun2014" in x.get("path","") for x in manifest):
    print("BRAUN acquisition failed",file=sys.stderr)
    sys.exit(2)
