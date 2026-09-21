#!/usr/bin/env python3
import hashlib, json, math, urllib.request, zipfile
from pathlib import Path
import numpy as np
import pyreadstat
from scipy import stats

OUT=Path("braun2014_direct_reanalysis")
RAW=OUT/"raw"; RAW.mkdir(parents=True,exist_ok=True)
SOURCE_URL="https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0111967.s001&type=supplementary"
EXPECTED_ZIP_SHA="68a048dfcf0faca32bcea4dade99ca079c7d21c49c33766fbd6b46e61c441351"
LOCK_BLOB_SHA="1759619063616ab2399f1f7d90f28caa15ffb4d5"
LOCK_COMMIT="14a9beb9f607a463c772f642d3c70cd2619b4ed0"
SUBJECTS=["12","13","14","15","16","17","18","19","20","21","22","23","26","27","28","29","30","31","32","33","34","35","36","37","38"]

def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def download():
    z=RAW/"braun2014_s001.zip"
    req=urllib.request.Request(SOURCE_URL,headers={"User-Agent":"Mozilla/5.0"})
    with urllib.request.urlopen(req,timeout=120) as r, open(z,"wb") as f:
        while True:
            b=r.read(1024*1024)
            if not b: break
            f.write(b)
    got=sha256(z)
    if got != EXPECTED_ZIP_SHA:
        raise RuntimeError(f"source ZIP SHA mismatch: {got}")
    with zipfile.ZipFile(z) as zz:
        zz.extractall(RAW/"extracted")
    return z

def holm(ps):
    ps=np.asarray(ps,float); m=len(ps)
    order=np.argsort(ps); adj=np.empty(m,float); prev=0.0
    for rank,idx in enumerate(order):
        val=min(1.0,(m-rank)*ps[idx])
        prev=max(prev,val); adj[idx]=prev
    return adj.tolist()

def onesample(x):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    n=len(x); mean=float(np.mean(x)); sd=float(np.std(x,ddof=1))
    se=sd/math.sqrt(n); t=mean/se if se>0 else float("nan")
    p=float(2*stats.t.sf(abs(t),df=n-1)) if np.isfinite(t) else float("nan")
    crit=float(stats.t.ppf(.975,df=n-1))
    ci=[float(mean-crit*se),float(mean+crit*se)]
    dz=float(mean/sd) if sd>0 else float("nan")
    try: w=stats.wilcoxon(x,alternative="two-sided",zero_method="wilcox")
    except Exception: w=None
    return {"n":n,"mean":mean,"sd":sd,"se":se,"t":float(t),"df":n-1,"p_two_sided":p,"ci95":ci,"cohen_dz":dz,
            "median":float(np.median(x)),"wilcoxon_p_two_sided":None if w is None else float(w.pvalue)}

def signflip(x,draws=100000,seed=20260921):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    rng=np.random.default_rng(seed)
    obs=abs(float(np.mean(x))); ge=0; batch=5000
    done=0
    while done<draws:
        b=min(batch,draws-done)
        signs=rng.choice(np.array([-1.0,1.0]),size=(b,len(x)))
        means=np.mean(signs*x,axis=1)
        ge += int(np.sum(np.abs(means)>=obs-1e-15))
        done += b
    return {"draws":draws,"seed":seed,"two_sided":True,"p":float((ge+1)/(draws+1))}

def factor_effects(mat):
    # cols: con_self, con_other, incon_self, incon_other
    mat=np.asarray(mat,float)
    agent=(mat[:,0]+mat[:,2])/2-(mat[:,1]+mat[:,3])/2
    pos=(mat[:,0]+mat[:,1])/2-(mat[:,2]+mat[:,3])/2
    inter=(mat[:,0]-mat[:,1])-(mat[:,2]-mat[:,3])
    return agent,pos,inter

def cells(mat,names=("con_self","con_other","incon_self","incon_other")):
    out={}
    for j,nm in enumerate(names):
        x=np.asarray(mat[:,j],float)
        out[nm]={"mean":float(np.mean(x)),"sd":float(np.std(x,ddof=1)),"n":int(np.isfinite(x).sum())}
    return out

z=download()
sav=RAW/"extracted"/"all.sav"
if not sav.exists():
    hits=list((RAW/"extracted").rglob("all.sav"))
    if len(hits)!=1: raise RuntimeError("all.sav not uniquely found")
    sav=hits[0]
df,meta=pyreadstat.read_sav(str(sav))
if len(df)!=25: raise RuntimeError(f"expected 25 rows, got {len(df)}")

cols={
 "A":["A_questionnaire_con_self","A_questionnaire_con_other","A_questionnaire_incon_self","A_questionnaire_incon_other"],
 "O":["O_questionnaire_con_self","O_questionnaire_con_other","O_questionnaire_incon_self","O_questionnaire_incon_other"],
 "IB":["IB_con_self","IB_con_other","IB_incon_self","IB_incon_other"],
 "DRIFT":["drift_con_self","drift_con_other","drift_incon_self","drift_incon_other"],
 "AC":["AC_questionnaire_con_self","AC_questionnaire_con_other","AC_questionnaire_incon_self","AC_questionnaire_incon_other"],
 "OC":["OC_questionnaire_con_self","OC_questionnaire_con_other","OC_questionnaire_incon_self","OC_questionnaire_incon_other"]
}
required=sum(cols.values(),[])
missing=[c for c in required if c not in df.columns]
if missing: raise RuntimeError("missing columns: "+repr(missing))
if df[cols["A"]+cols["O"]].isna().any().any(): raise RuntimeError("missing primary explicit cells")

A=df[cols["A"]].to_numpy(float); O=df[cols["O"]].to_numpy(float)
IB=df[cols["IB"]].to_numpy(float); DR=df[cols["DRIFT"]].to_numpy(float)
AC=df[cols["AC"]].to_numpy(float); OC=df[cols["OC"]].to_numpy(float)
a_agent,a_pos,a_int=factor_effects(A)
o_agent,o_pos,o_int=factor_effects(O)
primary_agent=a_agent-o_agent
primary_pos=o_pos-a_pos
pstats=[onesample(primary_agent),onesample(primary_pos)]
padj=holm([pstats[0]["p_two_sided"],pstats[1]["p_two_sided"]])
for s,a in zip(pstats,padj): s["holm_p"]=a
perm=[signflip(primary_agent),signflip(primary_pos)]
pred=[pstats[0]["mean"]>0,pstats[1]["mean"]>0]
passed=[pred[i] and pstats[i]["holm_p"]<.05 for i in range(2)]
if all(passed): cls="STRICT_CROSSOVER_CRITERION_MET_DIRECT_RETROSPECTIVE"
elif any(passed): cls="PARTIAL_SELECTIVE_EVIDENCE"
else: cls="NO_STRICT_CROSSOVER_EVIDENCE"

# Secondary implicit: pooled z separately by measure
IBz=(IB-np.nanmean(IB))/np.nanstd(IB,ddof=1)
DRz=(DR-np.nanmean(DR))/np.nanstd(DR,ddof=1)
ib_agent,ib_pos,ib_int=factor_effects(IBz)
dr_agent,dr_pos,dr_int=factor_effects(DRz)
imp_agent=ib_agent-dr_agent
imp_pos=dr_pos-ib_pos
istats=[onesample(imp_agent),onesample(imp_pos)]
iadj=holm([istats[0]["p_two_sided"],istats[1]["p_two_sided"]])
for s,a in zip(istats,iadj): s["holm_p"]=a

result={
 "status":"BRAUN2014_DIRECT_RETROSPECTIVE_REANALYSIS_COMPLETE",
 "evidence_status":"direct retrospective discriminative reanalysis; not prospective confirmatory",
 "lock":{"private_repo_commit":LOCK_COMMIT,"lock_blob_sha":LOCK_BLOB_SHA},
 "source":{"url":SOURCE_URL,"zip_sha256":sha256(z),"all_sav_sha256":sha256(sav)},
 "sample":{"n":25,"subject_ids":SUBJECTS,"source_exclusions":{"11":"task instructions","24":"sticking key","25":"corrupt IB file"}},
 "primary_explicit":{
   "agent_selectivity":{"definition":"agent effect on explicit agency minus agent effect on explicit ownership","predicted_sign":"positive","stats":pstats[0],"signflip":perm[0]},
   "position_selectivity":{"definition":"position effect on explicit ownership minus position effect on explicit agency","predicted_sign":"positive","stats":pstats[1],"signflip":perm[1]},
   "strict_rule_components":{"predicted_signs":pred,"holm_pass":passed},
   "classification":cls,
   "raw_cell_summaries":{"agency":cells(A),"ownership":cells(O)},
   "component_effects":{
      "agency_outcome":{"agent":onesample(a_agent),"position":onesample(a_pos),"agent_x_position":onesample(a_int)},
      "ownership_outcome":{"agent":onesample(o_agent),"position":onesample(o_pos),"agent_x_position":onesample(o_int)}
   }
 },
 "secondary_implicit":{
   "standardization":"pooled z separately within intentional binding and proprioceptive drift across 25x4 cells",
   "agent_selectivity":{"definition":"agent effect on z intentional binding minus agent effect on z proprioceptive drift","predicted_sign":"positive","stats":istats[0]},
   "position_selectivity":{"definition":"position effect on z proprioceptive drift minus position effect on z intentional binding","predicted_sign":"positive","stats":istats[1]},
   "raw_cell_summaries":{"intentional_binding":cells(IB),"proprioceptive_drift":cells(DR)},
   "component_effects":{
      "intentional_binding_z":{"agent":onesample(ib_agent),"position":onesample(ib_pos),"agent_x_position":onesample(ib_int)},
      "proprioceptive_drift_z":{"agent":onesample(dr_agent),"position":onesample(dr_pos),"agent_x_position":onesample(dr_int)}
   },
   "role":"secondary; cannot rescue failed primary"
 },
 "control_scale_descriptives":{"agency_control":cells(AC),"ownership_control":cells(OC)},
 "interpretation_guardrails":[
   "Classification concerns the prespecified agency-versus-bodily-ownership crossover only.",
   "This retrospective dataset cannot establish Stage 4 causality or prospective confirmation.",
   "Secondary implicit results cannot rescue a failed primary explicit crossover.",
   "No aggregate theory-support score is computed."
 ],
 "outcome_dependent_tuning":False
}
OUT.mkdir(exist_ok=True)
with open(OUT/"result.json","w") as f: json.dump(result,f,indent=2)
with open(OUT/"participant_primary_contrasts.csv","w") as f:
    f.write("subject,agent_selectivity,position_selectivity\n")
    for sid,x,y in zip(SUBJECTS,primary_agent,primary_pos): f.write(f"{sid},{x:.12g},{y:.12g}\n")
with open(OUT/"provenance.json","w") as f:
    json.dump({
      "analysis_script_sha256":sha256(Path(__file__)),
      "source_zip_sha256":sha256(z),
      "all_sav_sha256":sha256(sav),
      "lock_blob_sha":LOCK_BLOB_SHA,
      "lock_commit":LOCK_COMMIT,
      "numpy":np.__version__
    },f,indent=2)
print(json.dumps({"classification":cls,"primary_agent":pstats[0],"primary_position":pstats[1],"implicit_agent":istats[0],"implicit_position":istats[1]},indent=2))
