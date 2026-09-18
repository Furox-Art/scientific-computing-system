#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, sys, traceback, zipfile
from pathlib import Path
from io import BytesIO
import requests
from PIL import Image

URL="https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41598-025-12695-z/MediaObjects/41598_2025_12695_MOESM1_ESM.docx"
OUT=Path(".public-runner/p07/figures")
RESULT=Path(".public-runner/p07/figure_manifest.json")

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    result={"status":"STARTED","url":URL,"images":[]}
    try:
        r=requests.get(URL,timeout=(30,180),allow_redirects=True,headers={"User-Agent":"github-actions-public-validation/1.0"})
        if r.status_code!=200: raise RuntimeError(f"HTTP {r.status_code}")
        result["docx_bytes"]=len(r.content)
        result["docx_sha256"]=hashlib.sha256(r.content).hexdigest()
        with zipfile.ZipFile(BytesIO(r.content)) as z:
            media=sorted([n for n in z.namelist() if n.startswith("word/media/") and not n.endswith("/")])
            for n in media:
                b=z.read(n)
                rec={"zip_path":n,"bytes":len(b),"sha256":hashlib.sha256(b).hexdigest()}
                suffix=Path(n).suffix.lower()
                dest=OUT/Path(n).name
                dest.write_bytes(b)
                try:
                    with Image.open(BytesIO(b)) as im:
                        rec.update({"format":im.format,"width":im.width,"height":im.height,"mode":im.mode})
                except Exception as e:
                    rec["image_probe_error"]=repr(e)
                rec["artifact_path"]=str(dest)
                result["images"].append(rec)
        result["status"]="SUPPLEMENT_MEDIA_EXTRACTED"
        result["image_count"]=len(result["images"])
        RESULT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(result,indent=2))
        return 0
    except Exception as e:
        result["status"]="SUPPLEMENT_MEDIA_EXTRACTION_FAILED"
        result["error"]={"type":type(e).__name__,"message":str(e),"traceback":traceback.format_exc()}
        RESULT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(result,indent=2)); return 2

if __name__=="__main__":
    sys.exit(main())
