#!/usr/bin/env python3
from __future__ import annotations
import csv,json
from pathlib import Path

ROOT=Path(".public-runner/p08/full-acquisition")
participants=["sub-Bubbles","sub-Buttercup","sub-PILOT02"]
results=[]
for p in participants:
    fp=ROOT/p/"acquisition_result.json"
    if not fp.exists(): raise SystemExit(f"missing {fp}")
    results.append(json.loads(fp.read_text()))
all_files=sum(r.get("verified_files",0) for r in results)
all_bytes=sum(r.get("verified_bytes",0) for r in results)
expected_files=sum(r.get("expected_files",0) for r in results)
expected_bytes=sum(r.get("expected_bytes",0) for r in results)
out={
  "status":"P08_ALL_85_PREPROCESSING_ANNEX_OBJECTS_EXACT_HASH_VERIFIED",
  "participants":participants,
  "participant_results":[
    {k:r.get(k) for k in ["participant","status","expected_files","verified_files","expected_bytes","verified_bytes","all_verified"]}
    for r in results
  ],
  "expected_files":expected_files,
  "verified_files":all_files,
  "expected_bytes":expected_bytes,
  "verified_bytes":all_bytes,
  "all_verified":all(r.get("all_verified") for r in results) and all_files==expected_files and all_bytes==expected_bytes,
  "scope_note":"Verification-only acquisition: every frozen annex payload in the corrected preprocessing manifest was transferred and cryptographically checked; binaries were not retained as a final artifact.",
  "neural_effect_computed":False
}
if not out["all_verified"]: raise SystemExit(json.dumps(out,indent=2))
(ROOT/"full_acquisition_summary.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
