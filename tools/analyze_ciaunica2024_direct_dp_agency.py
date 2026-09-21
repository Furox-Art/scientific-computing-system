#!/usr/bin/env python3
import hashlib, json, math, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

OUT=Path("ciaunica2024_direct_reanalysis")
RAW=OUT/"raw"; RAW.mkdir(parents=True,exist_ok=True)
LOCK_COMMIT="5f1ca28adc025c90ed21f19ac28828bbd4a81655"
LOCK_BLOB="0cfabe21d21fb46c4888d58ca8ddd208aff3d3b9"

SOURCES={
 "Extraction_IB_NEW.csv":{
  "url":"https://osf.io/download/8zuae/?view_only=513a8c23ad4145418ea7f5dffad526f1",
  "sha":"bd1c16cd414ab265907371d72ab66daa78f957f9b0e11d97b4a2fd74517fb10f"},
 "GROUP.xlsx":{
  "url":"https://osf.io/download/r4w8b/?view_only=513a8c23ad4145418ea7f5dffad526f1",
  "sha":"3070ed9bf3eba9c8111ca4558a9c56119687b0a108ac65faacb61a89a83e7f74"},
 "STATS_IB.R":{
  "url":"https://osf.io/download/kpa5h/?view_only=513a8c23ad4145418ea7f5dffad526f1",
  "sha":"503a218903f2c83a495884c356faa944f6526c22fcee1bab8a8e9ec6b367752b"}
}

def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def download_all():
    out={}
    for name,s in SOURCES.items():
        p=RAW/name
        req=urllib.request.Request(s["url"],headers={"User-Agent":"Mozilla/5.0"})
        with urllib.request.urlopen(req,timeout=120) as r, open(p,"wb") as f:
            while True:
                b=r.read(1024*1024)
                if not b: break
                f.write(b)
        got=sha256(p)
        if got != s["sha"]: raise RuntimeError(f"{name} SHA mismatch: {got}")
        out[name]={"path":str(p),"sha256":got,"bytes":p.stat().st_size,"url":s["url"]}
    return out

def holm(ps):
    ps=np.asarray(ps,float); m=len(ps); order=np.argsort(ps); adj=np.empty(m); prev=0.0
    for rank,idx in enumerate(order):
        val=min(1.0,(m-rank)*ps[idx]); prev=max(prev,val); adj[idx]=prev
    return adj.tolist()

def fit_coef(df, formula, term):
    m=smf.ols(formula,data=df).fit(cov_type="HC3")
    if term not in m.params.index:
        raise RuntimeError(f"term {term} absent; terms={list(m.params.index)}")
    return {
      "n":int(m.nobs),"formula":formula,"term":term,
      "estimate":float(m.params[term]),"se_hc3":float(m.bse[term]),
      "t":float(m.tvalues[term]),"p_two_sided":float(m.pvalues[term]),
      "ci95":[float(x) for x in m.conf_int().loc[term].tolist()],
      "r_squared":float(m.rsquared),"adj_r_squared":float(m.rsquared_adj)
    }

def desc(df,var):
    out={}
    for g in ["LOW","HIGH"]:
        x=pd.to_numeric(df.loc[df.GROUP==g,var],errors="coerce").dropna().to_numpy(float)
        out[g]={"n":int(len(x)),"mean":float(np.mean(x)),"sd":float(np.std(x,ddof=1)),"median":float(np.median(x))}
    return out

files=download_all()
av=pd.read_csv(RAW/"Extraction_IB_NEW.csv", dtype={"ID":str})
grp=pd.read_excel(RAW/"GROUP.xlsx")
# Mirror R's type inference/join semantics: CSV character IDs retain exact
# whitespace; only numeric Excel ID cells are rendered as integer strings.
def excel_id_to_r_character(v):
    if pd.isna(v): return None
    if isinstance(v,(int,np.integer)): return str(int(v))
    if isinstance(v,(float,np.floating)) and float(v).is_integer(): return str(int(v))
    return str(v)
grp["ID"]=grp["ID"].map(excel_id_to_r_character)
av["ID"]=av["ID"].map(lambda v: None if pd.isna(v) else str(v))
df=grp.merge(av,on="ID",how="outer")
audit=[{"stage":"full_join","n":int(len(df))}]
df=df.dropna(subset=["GROUP"]).copy(); audit.append({"stage":"drop_missing_GROUP","n":int(len(df))})
df=df[df["Gender"]!="Non-binary"].copy(); audit.append({"stage":"exclude_nonbinary","n":int(len(df))})
for i in range(1,15): df[f"STT_{i}"]=pd.to_numeric(df[f"STT_{i}"],errors="coerce")
df=df.dropna(subset=["STT_2","STT_7","STT_8"]).copy(); audit.append({"stage":"drop_missing_STT_2_7_8","n":int(len(df))})
df["Ratio_Missed"]=pd.to_numeric(df["Missed"],errors="coerce")/150.0
df=df[df["Ratio_Missed"]<=0.15].copy(); audit.append({"stage":"missed_le_15pct","n":int(len(df))})
base900=pd.to_numeric(df["Estimate_Baseline_900"],errors="coerce")
z=(base900-base900.mean())/base900.std(ddof=1)
df=df[z.abs()<=2].copy(); audit.append({"stage":"baseline900_abs_z_le_2","n":int(len(df))})

df["GROUP"]=df["GROUP"].astype(str).str.upper().str.strip()
df["Gender"]=df["Gender"].astype(str).str.strip()
for c in ["Age","SCORE_CDS","Slope_Agency","Slope_Binding"]:
    df[c]=pd.to_numeric(df[c],errors="coerce")

counts=df["GROUP"].value_counts().to_dict()
if len(df)!=93 or counts.get("HIGH")!=46 or counts.get("LOW")!=47:
    raise RuntimeError(f"validity gate failed final N/groups: N={len(df)} counts={counts}")

groupterm="C(GROUP, Treatment(reference='LOW'))[T.HIGH]"
agency=fit_coef(df,"Slope_Agency ~ C(GROUP, Treatment(reference='LOW')) + Age + C(Gender)",groupterm)
binding=fit_coef(df,"Slope_Binding ~ C(GROUP, Treatment(reference='LOW')) + Age + C(Gender)",groupterm)
primary_adj=holm([agency["p_two_sided"],binding["p_two_sided"]])
agency["holm_p"]=primary_adj[0]; binding["holm_p"]=primary_adj[1]
passes=[agency["holm_p"]<0.05,binding["holm_p"]<0.05]
if all(passes): classification="BROAD_DP_RELATED_EXPLICIT_AND_IMPLICIT_AGENCY_DIFFERENCE"
elif any(passes): classification="SELECTIVE_DP_RELATED_AGENCY_COMPONENT_DIFFERENCE"
else: classification="NO_EVIDENCE_FOR_OVERALL_DP_GROUP_DIFFERENCE_IN_AGENCY_SLOPES"

# Nearest rival: participant-level mean absolute temporal distortion.
delays=[100,300,500,700,900]
for condition in ["Baseline","Operant"]:
    vals=[]
    for d in delays:
        v=pd.to_numeric(df[f"Estimate_{condition}_{d}"],errors="coerce")
        vals.append((d-v).abs())
    df[f"{condition.lower()}_abs_distortion"]=pd.concat(vals,axis=1).mean(axis=1)
base=fit_coef(df,"baseline_abs_distortion ~ C(GROUP, Treatment(reference='LOW')) + Age + C(Gender)",groupterm)
oper=fit_coef(df,"operant_abs_distortion ~ C(GROUP, Treatment(reference='LOW')) + Age + C(Gender)",groupterm)
rival_adj=holm([base["p_two_sided"],oper["p_two_sided"]])
base["holm_p"]=rival_adj[0]; oper["holm_p"]=rival_adj[1]

# Continuous DP sensitivity.
scoreterm="SCORE_CDS"
agency_c=fit_coef(df,"Slope_Agency ~ SCORE_CDS + Age + C(Gender)",scoreterm)
binding_c=fit_coef(df,"Slope_Binding ~ SCORE_CDS + Age + C(Gender)",scoreterm)
sens_adj=holm([agency_c["p_two_sided"],binding_c["p_two_sided"]])
agency_c["holm_p"]=sens_adj[0]; binding_c["holm_p"]=sens_adj[1]

result={
 "status":"CIAUNICA2024_DIRECT_RETROSPECTIVE_DP_AGENCY_REANALYSIS_COMPLETE",
 "evidence_status":"direct retrospective depersonalisation-agency component test; not prospective confirmatory",
 "lock":{"commit":LOCK_COMMIT,"blob_sha":LOCK_BLOB},
 "source_files":files,
 "sample":{"final_n":int(len(df)),"group_counts":{k:int(v) for k,v in counts.items()},"source_filter_audit":audit},
 "primary_overall_agency":{
   "explicit_agency_slope_group_effect":agency,
   "implicit_binding_slope_group_effect":binding,
   "classification":classification,
   "holm_pass":passes,
   "descriptives":{"Slope_Agency":desc(df,"Slope_Agency"),"Slope_Binding":desc(df,"Slope_Binding")}
 },
 "nearest_rival_temporal_estimation":{
   "baseline_abs_distortion_group_effect":base,
   "operant_abs_distortion_group_effect":oper,
   "role":"nearest-rival temporal-estimation family; cannot rescue null agency primary"
 },
 "continuous_dp_sensitivity":{
   "SCORE_CDS_to_explicit_agency_slope":agency_c,
   "SCORE_CDS_to_implicit_binding_slope":binding_c,
   "role":"secondary sensitivity only"
 },
 "interpretation_guardrails":[
   "This lane directly tests a depersonalisation-agency component relation, not global Internalization Theory.",
   "No ownership-neutral conscious-presence or access outcome is available, so H17/Stage-4 causal claims are not tested.",
   "Temporal-estimation differences are reported as a rival explanation rather than agency evidence.",
   "Published outcomes were known before this retrospective reanalysis."
 ],
 "outcome_dependent_tuning":False
}
OUT.mkdir(exist_ok=True)
with open(OUT/"result.json","w") as f: json.dump(result,f,indent=2)
with open(OUT/"analysis_sample_ids.csv","w") as f:
    f.write("ID,GROUP\n")
    for _,r in df[["ID","GROUP"]].sort_values("ID",key=lambda s:s.astype(str)).iterrows(): f.write(f"{r.ID},{r.GROUP}\n")
with open(OUT/"provenance.json","w") as f:
    json.dump({
      "analysis_script_sha256":sha256(Path(__file__)),
      "lock_commit":LOCK_COMMIT,"lock_blob_sha":LOCK_BLOB,
      "source_hashes":{k:v["sha256"] for k,v in files.items()},
      "dependencies":{"numpy":np.__version__,"pandas":pd.__version__},
      "outcome_dependent_tuning":False
    },f,indent=2)
print(json.dumps({
 "final_n":len(df),"counts":counts,"classification":classification,
 "primary":{"agency":agency,"binding":binding},
 "rival":{"baseline":base,"operant":oper},
 "sensitivity":{"agency":agency_c,"binding":binding_c}
},indent=2))
