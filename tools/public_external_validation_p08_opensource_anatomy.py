#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json, os, subprocess, traceback
from pathlib import Path

MNI_SPACE="MNI152NLin2009cAsym"
MNI_RESOLUTION=2
PARTICIPANTS={"sub-Bubbles","sub-Buttercup"}

def sha256_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()

def run(cmd:list[str], env=None):
    print("+"," ".join(cmd),flush=True)
    subprocess.run(cmd,check=True,env=env)

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--participant",required=True,choices=sorted(PARTICIPANTS))
    ap.add_argument("--root",default=".public-runner/p08/opensource-anatomy")
    args=ap.parse_args()

    participant=args.participant
    part_root=Path(args.root)/participant
    bids=part_root/"bids"
    out=part_root/"anatomy_result.json"
    result={
      "status":"STARTED",
      "participant":participant,
      "lane":"open_source_secondary_sensitivity_only",
      "frozen_source_commit":"ed03c47c368000e511401021c1d13c3fb916b470",
      "ants_version":"2.6.5",
      "templateflow_version":"25.1.2",
      "space":MNI_SPACE,
      "resolution_mm":MNI_RESOLUTION,
      "self_other_glm_computed":False,
      "roi_effect_computed":False,
      "neural_effect_computed":False,
    }
    try:
        t1s=sorted((bids/participant).glob("ses-*/anat/*T1w.nii.gz"))
        if len(t1s)!=1:
            raise RuntimeError(f"Expected exactly one frozen T1w, got {t1s}")
        t1=t1s[0]

        from templateflow.api import get
        template=get(MNI_SPACE,resolution=MNI_RESOLUTION,desc=None,suffix="T1w")
        if isinstance(template,(list,tuple)):
            if len(template)!=1: raise RuntimeError(f"TemplateFlow returned multiple templates: {template}")
            template=template[0]
        template=Path(template)

        ants=part_root/"ants"; ants.mkdir(parents=True,exist_ok=True)
        t1_n4=ants/"T1w_N4.nii.gz"
        run(["N4BiasFieldCorrection","-d","3","-i",str(t1),"-o",str(t1_n4)])

        prefix=ants/"T1_to_MNI_"
        env=os.environ.copy(); env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"]="2"
        run(["antsRegistrationSyNQuick.sh","-d","3","-f",str(template),"-m",str(t1_n4),"-o",str(prefix),"-t","s"],env=env)

        affine=Path(str(prefix)+"0GenericAffine.mat")
        warp=Path(str(prefix)+"1Warp.nii.gz")
        warped=Path(str(prefix)+"Warped.nii.gz")
        for p in (t1_n4,affine,warp,warped):
            if not p.exists(): raise RuntimeError(f"Missing anatomy output {p}")

        files=[]
        for p in (t1_n4,affine,warp,warped):
            files.append({"path":p.relative_to(part_root).as_posix(),"bytes":p.stat().st_size,"sha256":sha256_file(p)})
        result.update({
          "status":"P08_OPEN_SOURCE_V2_ANATOMY_NORMALIZATION_COMPLETE",
          "anatomy_complete":True,
          "template":str(template),
          "selected_output_files":files,
        })
        out.write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(result,indent=2))
        return 0
    except Exception as e:
        result.update({
          "status":"P08_OPEN_SOURCE_V2_ANATOMY_NORMALIZATION_TECHNICAL_FAILURE",
          "anatomy_complete":False,
          "error":{"type":type(e).__name__,"message":str(e),"traceback":traceback.format_exc()},
        })
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(result,indent=2))
        return 2

if __name__=="__main__":
    raise SystemExit(main())
