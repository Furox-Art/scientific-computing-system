#!/usr/bin/env python3
from __future__ import annotations

import csv, hashlib, json, shutil, sys, time, traceback
from pathlib import Path
import numpy as np
import requests
from scipy.io import loadmat

BASE="https://data.nemar.org/on004563/v1.0.0/"
IND_MANIFEST=Path("tools/public_p10_crossdecoding_manifest.csv")
BF_MANIFEST=Path("tools/public_p10_source_bf_manifest.csv")
OUT=Path(".public-runner/p10/cellwise")
RESULT=OUT/"cellwise_source_array_crosscheck.json"
CHUNK=1048576

VT=[8,9,12,14,20,21,23,28,29,30,31,35,36,38,39,40]
NVT=[1,2,3,4,5,6,7,11,15,17,19,22,24,25,26,27,34,37]
GROUPS={"VT":VT,"NVT":NVT}

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def download(url:str, expected:int, want:str, out:Path)->dict:
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists() and out.stat().st_size==expected and sha256(out)==want:
        return {"reused":True,"bytes":expected,"sha256":want}
    if out.exists(): out.unlink()
    parts=out.parent/(out.name+".parts")
    shutil.rmtree(parts,ignore_errors=True); parts.mkdir()
    s=requests.Session(); start=0; idx=0; attempts=[]
    while start<expected:
        end=min(start+CHUNK-1,expected-1)
        last=None
        for attempt in range(1,5):
            try:
                r=s.get(url,headers={"Range":f"bytes={start}-{end}","Accept-Encoding":"identity","User-Agent":"github-actions-public-validation/1.0"},timeout=(30,180),allow_redirects=True)
                if r.status_code not in (200,206): raise RuntimeError(f"HTTP {r.status_code}; final={r.url}")
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
                raise RuntimeError(f"bad Content-Range {cr!r}")
            p=parts/f"part-{idx:05d}.bin"; p.write_bytes(body)
            attempts.append(attempt); start=end+1; idx+=1
        elif r.status_code==200 and start==0 and len(body)==expected:
            out.write_bytes(body); attempts.append(attempt); break
        else:
            raise RuntimeError(f"unexpected response status={r.status_code} received={len(body)}")
    if not out.exists():
        with out.open("wb") as dst:
            for p in sorted(parts.glob("part-*.bin")):
                with p.open("rb") as src: shutil.copyfileobj(src,dst)
    shutil.rmtree(parts,ignore_errors=True)
    got=sha256(out)
    if out.stat().st_size!=expected or got!=want:
        raise RuntimeError(f"verification failed for {url}: {out.stat().st_size}/{expected}, {got}/{want}")
    return {"reused":False,"chunks":idx if idx else 1,"max_attempt":max(attempts) if attempts else 1,"bytes":expected,"sha256":got}

def read_manifests():
    with IND_MANIFEST.open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    ind={int(r["participant_id"].split("-")[1]):r for r in rows}
    with BF_MANIFEST.open(newline="",encoding="utf-8") as f:
        bfs={Path(r["path"]).name:r for r in csv.DictReader(f)}
    return ind,bfs

def samples(path:Path,name:str)->np.ndarray:
    m=loadmat(path,squeeze_me=True,struct_as_record=False)
    obj=m[name]
    a=np.asarray(obj.samples,dtype=np.float64).reshape(-1)
    if a.size!=744*744: raise RuntimeError(f"{path.name} {name} size {a.size}")
    return a

def compare_matrix(vec:np.ndarray, target:np.ndarray)->dict:
    c=vec.reshape((744,744),order="C")
    f=vec.reshape((744,744),order="F")
    def s(a):
        d=np.abs(a-target)
        return {
            "exact_equal":bool(np.array_equal(a,target)),
            "mismatched_cells":int(np.count_nonzero(a!=target)),
            "max_abs_diff":float(np.max(d)),
            "mean_abs_diff":float(np.mean(d))
        }
    sc, sf=s(c),s(f)
    if sc["exact_equal"] and not sf["exact_equal"]: chosen="C"
    elif sf["exact_equal"] and not sc["exact_equal"]: chosen="F"
    elif sc["exact_equal"] and sf["exact_equal"]: chosen="BOTH"
    else: chosen="NONE"
    return {"C":sc,"F":sf,"chosen_order":chosen}

def main()->int:
    OUT.mkdir(parents=True,exist_ok=True)
    result={
        "status":"STARTED",
        "dataset":"on004563 v1.0.0",
        "chunk_bytes":CHUNK,
        "groups":{"VT":VT,"NVT":NVT},
        "directions":["touch2vis","vis2touch"],
        "participants":[],
        "aggregate":{},
        "scientific_effect_computed":False
    }
    try:
        ind,bfs=read_manifests()

        # Source grouped arrays
        source={}
        for group,fn in [("VT","bayes_factors_VT_group.mat"),("NVT","bayes_factors_NoVT_group.mat")]:
            row=bfs[fn]
            p=OUT/fn
            download(BASE+row["path"],int(row["bytes"]),row["sha256"].lower(),p)
            m=loadmat(p,squeeze_me=False,struct_as_record=False)
            source[(group,"touch2vis")]=np.asarray(m["timegen_data_touch2vis"],dtype=np.float64)
            source[(group,"vis2touch")]=np.asarray(m["timegen_data_vis2touch"],dtype=np.float64)
            expn=len(GROUPS[group])
            for direction in ["touch2vis","vis2touch"]:
                if source[(group,direction)].shape!=(744,744,expn):
                    raise RuntimeError(f"unexpected source shape {group} {direction}: {source[(group,direction)].shape}")

        # Individual files, compare exact values to corresponding frozen grouped source slice.
        total_exact=0; total_checks=0; total_mismatch_cells=0; max_diff=0.0
        chosen_orders=set()
        for group,ids in GROUPS.items():
            for idx,sid in enumerate(ids):
                row=ind[sid]
                p=OUT/Path(row["derivative_path"]).name
                dl=download(BASE+row["derivative_path"],int(row["bytes"]),row["sha256"].lower(),p)
                rec={"participant_id":f"sub-{sid:02d}","group":group,"source_group_index_zero_based":idx,
                     "individual_sha256":row["sha256"].lower(),"download":dl,"directions":{}}
                for direction,var in [("touch2vis","res_timegen_touch2vis"),("vis2touch","res_timegen_vis2touch")]:
                    vec=samples(p,var)
                    target=source[(group,direction)][:,:,idx]
                    cmp=compare_matrix(vec,target)
                    rec["directions"][direction]=cmp
                    total_checks+=1
                    if cmp["chosen_order"]=="NONE":
                        raise RuntimeError(f"no exact reshape match: sub-{sid:02d} {direction}; C={cmp['C']}; F={cmp['F']}")
                    chosen_orders.add(cmp["chosen_order"])
                    chosen=cmp["C"] if cmp["chosen_order"] in ("C","BOTH") else cmp["F"]
                    total_exact+=int(chosen["exact_equal"])
                    total_mismatch_cells+=chosen["mismatched_cells"]
                    max_diff=max(max_diff,chosen["max_abs_diff"])
                result["participants"].append(rec)
                p.unlink(missing_ok=True)
                print(f"{group} sub-{sid:02d}: t2v={rec['directions']['touch2vis']['chosen_order']} v2t={rec['directions']['vis2touch']['chosen_order']}")
        for fn in ["bayes_factors_VT_group.mat","bayes_factors_NoVT_group.mat"]:
            (OUT/fn).unlink(missing_ok=True)

        result["aggregate"]={
            "participants":34,
            "direction_checks":total_checks,
            "exact_direction_checks":total_exact,
            "total_participant_direction_cells":34*2*744*744,
            "mismatched_cells_after_source_exact_reshape":total_mismatch_cells,
            "maximum_absolute_cell_difference":max_diff,
            "chosen_orders":sorted(chosen_orders),
            "all_participant_direction_arrays_exact":bool(total_exact==total_checks and total_mismatch_cells==0 and max_diff==0.0)
        }
        if not result["aggregate"]["all_participant_direction_arrays_exact"]:
            raise RuntimeError("aggregate exactness invariant failed")
        result["status"]="ALL_34_PARTICIPANT_BOTH_DIRECTIONS_CELLWISE_EXACT_MATCH_SOURCE_GROUPED_ARRAYS"
        RESULT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(result["aggregate"],indent=2))
        return 0
    except Exception as e:
        result["status"]="CELLWISE_CROSSCHECK_FAILED"
        result["error"]={"type":type(e).__name__,"message":str(e),"traceback":traceback.format_exc()}
        RESULT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(result,indent=2))
        return 2

if __name__=="__main__":
    sys.exit(main())
