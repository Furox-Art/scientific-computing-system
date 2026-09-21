#!/usr/bin/env python3
import csv, hashlib, itertools, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

RAW=Path("external_data/sato2018/data_able-bodied.csv")
OUT=Path("sato2018_direct_reanalysis")
OUT.mkdir(exist_ok=True)

EXPECTED_SHA="873336974dad77ea5b94cc8c9b17fcb0a2bd1c5a54ab05a970da649350b23f94"
LOCK_COMMIT="191e58bf5e95de830de9f62bdfbafbdad3e66c8a"
LOCK_BLOB="03a740c5022076f82d1b5fe5c4d1c5585e91314b"

def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def exact_signflip(x):
    x=np.asarray(x,float)
    if np.isnan(x).any(): raise RuntimeError("NaN in exact sign-flip vector")
    obs=abs(float(np.mean(x)))
    n=len(x)
    ge=0
    total=1<<n
    # n=15 => 32768 exact assignments
    for bits in range(total):
        s=np.ones(n)
        for i in range(n):
            if (bits>>i)&1: s[i]=-1.0
        if abs(float(np.mean(s*x))) >= obs - 1e-15:
            ge += 1
    return ge/total

def holm(ps):
    ps=np.asarray(ps,float); m=len(ps)
    order=np.argsort(ps); adj=np.empty(m,float); running=0.0
    for rank,idx in enumerate(order):
        val=min(1.0,(m-rank)*ps[idx])
        running=max(running,val)
        adj[idx]=running
    return adj.tolist()

def summarize(x):
    x=np.asarray(x,float)
    n=len(x); mean=float(np.mean(x)); sd=float(np.std(x,ddof=1))
    se=sd/math.sqrt(n)
    crit=float(stats.t.ppf(.975,n-1))
    try:
        w=stats.wilcoxon(x,alternative="two-sided",zero_method="wilcox")
        wp=float(w.pvalue)
    except Exception:
        wp=None
    return {
      "n":n,"mean":mean,"sd":sd,"se":se,
      "median":float(np.median(x)),
      "ci95":[float(mean-crit*se),float(mean+crit*se)],
      "cohen_dz":float(mean/sd) if sd>0 else None,
      "exact_signflip_p_two_sided":float(exact_signflip(x)),
      "wilcoxon_p_two_sided":wp
    }

got=sha256(RAW)
if got!=EXPECTED_SHA:
    raise RuntimeError(f"raw SHA mismatch: {got}")

df=pd.read_csv(RAW)
expected_cols=["paintbrush","movement","participant","SO","SA","SO (control)","SA (control)","Proprioceptive drift (mm)"]
if list(df.columns)!=expected_cols:
    raise RuntimeError(f"column mismatch: {list(df.columns)}")
if len(df)!=60: raise RuntimeError(f"expected 60 rows, got {len(df)}")
if df["participant"].nunique()!=15: raise RuntimeError("expected 15 participants")
if set(df["paintbrush"])!={"with","without"}: raise RuntimeError("paintbrush levels invalid")
if set(df["movement"])!={"sync","async"}: raise RuntimeError("movement levels invalid")
if df[expected_cols[3:]].isna().any().any(): raise RuntimeError("missing numeric outcome/control values")

# Every participant must have exactly one row in each 2x2 cell.
cell_counts=df.groupby(["participant","paintbrush","movement"]).size()
if len(cell_counts)!=60 or not (cell_counts==1).all():
    raise RuntimeError("2x2 repeated-measures cell gate failed")

df["ownership_specific"]=df["SO"]-df["SO (control)"]
df["agency_specific"]=df["SA"]-df["SA (control)"]

def participant_means(value):
    p=df.pivot(index="participant",columns=["paintbrush","movement"],values=value)
    required=[("with","sync"),("with","async"),("without","sync"),("without","async")]
    if list(p.index)!=list(range(1,16)):
        p=p.sort_index()
    for c in required:
        if c not in p.columns: raise RuntimeError(f"missing cell {c} for {value}")
    return p

O=participant_means("ownership_specific")
A=participant_means("agency_specific")
SO=participant_means("SO")
SA=participant_means("SA")
D=participant_means("Proprioceptive drift (mm)")

def avg_move(p, movement):
    return (p[("with",movement)].to_numpy(float)+p[("without",movement)].to_numpy(float))/2
def avg_brush(p, brush):
    return (p[(brush,"sync")].to_numpy(float)+p[(brush,"async")].to_numpy(float))/2

o_sync,o_async=avg_move(O,"sync"),avg_move(O,"async")
a_sync,a_async=avg_move(A,"sync"),avg_move(A,"async")
movement_selectivity=(o_sync-o_async)-(a_sync-a_async)
outphase_separation=a_async-o_async

p1=summarize(movement_selectivity)
p2=summarize(outphase_separation)
adj=holm([p1["exact_signflip_p_two_sided"],p2["exact_signflip_p_two_sided"]])
p1["holm_p"]=adj[0]; p2["holm_p"]=adj[1]
signs=[p1["mean"]>0,p2["mean"]>0]
passes=[signs[0] and p1["holm_p"]<.05,signs[1] and p2["holm_p"]<.05]
if all(passes):
    classification="STRICT_OWNERSHIP_AGENCY_DISSOCIATION_CRITERION_MET_DIRECT_RETROSPECTIVE"
elif any(passes):
    classification="PARTIAL_SELECTIVE_EVIDENCE"
else:
    classification="NO_EVIDENCE_FOR_LOCKED_OWNERSHIP_AGENCY_DISSOCIATION"

# Secondary tactile selectivity.
o_with,o_without=avg_brush(O,"with"),avg_brush(O,"without")
a_with,a_without=avg_brush(A,"with"),avg_brush(A,"without")
tactile_selectivity=(o_with-o_without)-(a_with-a_without)

# Secondary ownership-convergent drift movement effect.
d_sync,d_async=avg_move(D,"sync"),avg_move(D,"async")
drift_movement=d_sync-d_async

# Raw uncorrected sensitivity.
so_sync,so_async=avg_move(SO,"sync"),avg_move(SO,"async")
sa_sync,sa_async=avg_move(SA,"sync"),avg_move(SA,"async")
raw_movement_selectivity=(so_sync-so_async)-(sa_sync-sa_async)
raw_outphase_separation=sa_async-so_async

def cells(value):
    p=participant_means(value)
    out={}
    for brush in ["with","without"]:
        for mov in ["sync","async"]:
            x=p[(brush,mov)].to_numpy(float)
            out[f"{brush}_{mov}"]={"mean":float(np.mean(x)),"sd":float(np.std(x,ddof=1)),"median":float(np.median(x))}
    return out

result={
 "status":"SATO2018_DIRECT_RETROSPECTIVE_OWNERSHIP_AGENCY_REANALYSIS_COMPLETE",
 "evidence_status":"direct retrospective small-sample ownership-agency dissociation/sensitivity test; not prospective confirmatory",
 "lock":{"commit":LOCK_COMMIT,"blob_sha":LOCK_BLOB},
 "source":{
   "article_doi":"10.1098/rsos.172170",
   "dryad_doi":"10.5061/dryad.6675p",
   "path":str(RAW),
   "sha256":got,
   "bytes":RAW.stat().st_size
 },
 "sample":{"n_participants":15,"rows":60,"participant_ids":list(range(1,16)),"additional_exclusions":0},
 "primary":{
   "movement_selectivity":{
      "definition":"(ownership-specific sync-async) - (agency-specific sync-async), averaged across paintbrush",
      "predicted_sign":"positive",
      "stats":p1
   },
   "out_of_phase_component_separation":{
      "definition":"agency-specific async - ownership-specific async, averaged across paintbrush",
      "predicted_sign":"positive",
      "stats":p2
   },
   "strict_rule_components":{"predicted_signs":signs,"holm_pass":passes},
   "classification":classification
 },
 "secondary":{
   "tactile_selectivity":{
      "definition":"(ownership-specific with-without) - (agency-specific with-without), averaged across movement",
      "predicted_sign":"positive",
      "stats":summarize(tactile_selectivity),
      "role":"secondary; cannot rescue primary"
   },
   "proprioceptive_drift_movement":{
      "definition":"drift sync-async averaged across paintbrush",
      "predicted_sign":"positive",
      "stats":summarize(drift_movement),
      "role":"ownership-convergent secondary"
   }
 },
 "sensitivity":{
   "raw_uncorrected_movement_selectivity":summarize(raw_movement_selectivity),
   "raw_uncorrected_out_of_phase_separation":summarize(raw_outphase_separation),
   "role":"raw SO/SA without control subtraction; sensitivity only"
 },
 "cell_descriptives":{
   "SO":cells("SO"),"SA":cells("SA"),
   "SO_control":cells("SO (control)"),"SA_control":cells("SA (control)"),
   "ownership_specific":cells("ownership_specific"),
   "agency_specific":cells("agency_specific"),
   "proprioceptive_drift_mm":cells("Proprioceptive drift (mm)")
 },
 "interpretation_guardrails":[
   "Primary classification concerns the locked ownership-versus-agency separation under movement conflict only.",
   "This small retrospective dataset cannot establish global Internalization Theory or Stage-4 causality.",
   "Secondary tactile/drift results cannot rescue a failed primary.",
   "Published outcomes and raw values were visible before the theory-specific reanalysis; prospective/outcome-blind language is prohibited."
 ],
 "outcome_dependent_tuning":False
}

with open(OUT/"result.json","w") as f: json.dump(result,f,indent=2)
with open(OUT/"participant_primary_contrasts.csv","w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["participant","movement_selectivity","out_of_phase_component_separation","tactile_selectivity","drift_sync_minus_async"])
    for i,(x,y,t,d) in enumerate(zip(movement_selectivity,outphase_separation,tactile_selectivity,drift_movement),start=1):
        w.writerow([i,f"{x:.12g}",f"{y:.12g}",f"{t:.12g}",f"{d:.12g}"])
with open(OUT/"provenance.json","w") as f:
    json.dump({
      "analysis_script_sha256":sha256(Path(__file__)),
      "raw_sha256":got,
      "lock_commit":LOCK_COMMIT,
      "lock_blob_sha":LOCK_BLOB,
      "python_exact_signflip_assignments":32768,
      "dependencies":{"numpy":np.__version__,"pandas":pd.__version__,"scipy":stats.__version__ if hasattr(stats,"__version__") else "scipy"},
      "outcome_dependent_tuning":False
    },f,indent=2)

print(json.dumps({
  "classification":classification,
  "movement_selectivity":p1,
  "out_of_phase_component_separation":p2,
  "tactile_selectivity":result["secondary"]["tactile_selectivity"]["stats"],
  "drift_movement":result["secondary"]["proprioceptive_drift_movement"]["stats"]
},indent=2))
