#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, sys, traceback
from pathlib import Path
import requests

BASE="https://data.nemar.org/on004563/v1.0.0/"
FILES=[
("code/Experiment_presentation_script.m",30037,"75bb4bfeff2453a04074ed598262d0e80bc177e64045455e9193930d109ddfae"),
("code/Step2_within_mod_decoding.m",5835,"0345025d62cb47e7030ad5fd664b02d4de45101b94254a5b1930231ceaf029c7"),
("code/Step3a_between_mod_decoding.m",3731,"4b1d44bf70a6663af8dc6e5e4419caaaf6ecc77c390d59644423c742f4ec7c2c"),
("code/Step3b_between_mod_searchlight.m",4201,"a999943080b5976ea910026a3e40674e8add8e4b182541555801d6f5d8ae2c83"),
("code/Step4a_plot_within_mod_decoding_with_searchlight.m",12155,"c1216716b7ffeec18a007305deedfe182f38af91c79727f89c9d53d190a6fc9f"),
("code/Step4b_plot_between_mod_decoding.m",14196,"906fe5bff1eb151c008ee293cfb2f24b524ede2dccb6761f21d66b0d81953ab3"),
("code/Step4c_plot_between_mod_decoding_difference.m",7154,"ac0bffacc96ddb480d5cdb1bdbdcb68e6cba9474ff9c7664b232dea508194eda"),
("code/Step4d_plot_between_mod_decoding_searchlight.m",6419,"5b739bd3b31fa8bf492b199a839fc327bfb6a810e6a8e49b9b7c29cbd827291d"),
("code/bayesfactor_R_wrapper.m",4800,"a8a7f97f9bb56dc637018e164b75816645a3aa91fe7188b8322d818c6513cf7a"),
("code/bayesfactor_R_wrapper_2sample.m",5062,"4ba7f144e48e92024f1f81b9cf7d6942f6f742cd4fc06668e2ad24a2d6414191"),
]
OUT=Path(".public-runner/p10/source_code")
RESULT=Path(".public-runner/p10/source_code_result.json")

def sha256(b:bytes)->str:
    return hashlib.sha256(b).hexdigest()

def main()->int:
    OUT.mkdir(parents=True,exist_ok=True)
    result={"status":"STARTED","files":[],"all_verified":False}
    try:
        for rel,size,want in FILES:
            url=BASE+rel
            r=requests.get(url,timeout=(30,120),allow_redirects=True,headers={"User-Agent":"github-actions-public-validation/1.0"})
            rec={"path":rel,"url":url,"http_status":r.status_code,"final_url":r.url}
            if r.status_code!=200:
                raise RuntimeError(f"{rel}: HTTP {r.status_code}")
            body=r.content
            got=sha256(body)
            rec.update({"bytes":len(body),"expected_bytes":size,"sha256":got,"expected_sha256":want,
                        "size_match":len(body)==size,"sha256_match":got==want})
            if len(body)!=size or got!=want:
                raise RuntimeError(f"{rel}: verification failed")
            dest=OUT/Path(rel).name
            dest.write_bytes(body)
            result["files"].append(rec)
        result["status"]="ALL_SOURCE_CODE_EXACT_AND_HASH_VERIFIED"
        result["all_verified"]=True
        RESULT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(result,indent=2))
        return 0
    except Exception as e:
        result["status"]="SOURCE_CODE_ACQUISITION_FAILED"
        result["error"]={"type":type(e).__name__,"message":str(e),"traceback":traceback.format_exc()}
        RESULT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(result,indent=2))
        return 2

if __name__=="__main__":
    sys.exit(main())
