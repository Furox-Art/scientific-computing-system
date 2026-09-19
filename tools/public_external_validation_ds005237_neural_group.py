#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

SOURCE_COMMIT = "2e273d8466162208bbccd8591337e55b7a8b5721"
EXPECTED_SOURCE_N = 212

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()

def find_result_files(root: Path):
    return sorted(root.rglob("participant_neural_result.json"))

def load_results(root: Path):
    files=find_result_files(root)
    rows=[]
    seen=set()
    raw=[]
    for p in files:
        d=json.loads(p.read_text())
        pid=d["participant"]
        if pid in seen:
            raise RuntimeError(f"Duplicate participant result: {pid}")
        seen.add(pid)
        raw.append((p,d))
    return raw

def stats(x):
    x=pd.to_numeric(pd.Series(x),errors="coerce")
    x=x[np.isfinite(x)]
    return {
        "n":int(len(x)),
        "mean":float(x.mean()) if len(x) else math.nan,
        "median":float(x.median()) if len(x) else math.nan,
        "sd":float(x.std(ddof=1)) if len(x)>1 else math.nan,
        "min":float(x.min()) if len(x) else math.nan,
        "max":float(x.max()) if len(x) else math.nan,
    }

def ttest_record(fit, expr: str):
    t=fit.t_test(expr)
    effect=float(np.asarray(t.effect).ravel()[0])
    se=float(np.asarray(t.sd).ravel()[0])
    p=float(np.asarray(t.pvalue).ravel()[0])
    ci=np.asarray(t.conf_int(alpha=0.05)).reshape(-1,2)[0]
    return {"estimate":effect,"cluster_robust_se":se,"p_two_sided":p,"ci95":[float(ci[0]),float(ci[1])]}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--participant-results-root",required=True)
    ap.add_argument("--eligible-tsv",required=True)
    ap.add_argument("--output-root",required=True)
    args=ap.parse_args()

    result_root=Path(args.participant_results_root)
    eligible=pd.read_csv(args.eligible_tsv,sep="\t",dtype={"Site":str,"Group":str,"sex":str})
    if len(eligible)!=EXPECTED_SOURCE_N:
        raise RuntimeError(f"Eligible TSV N={len(eligible)} != {EXPECTED_SOURCE_N}")
    if eligible["participant_id"].duplicated().any():
        raise RuntimeError("Duplicate participant in eligible TSV")
    if set(eligible["Group"]) != {"Patient","GenPop"}:
        raise RuntimeError(f"Unexpected groups {set(eligible['Group'])}")

    loaded=load_results(result_root)
    by_id={d["participant"]:(p,d) for p,d in loaded}
    expected=set(eligible["participant_id"])
    missing=sorted(expected-set(by_id))
    extra=sorted(set(by_id)-expected)
    if missing or extra:
        raise RuntimeError(f"Participant result set mismatch missing={missing[:20]} extra={extra[:20]}")

    records=[]
    exclusions=[]
    for _,m in eligible.iterrows():
        pid=m["participant_id"]
        p,d=by_id[pid]
        base={
            "participant_id":pid,
            "Group":m["Group"],
            "Patient":1 if m["Group"]=="Patient" else 0,
            "age":float(m["age"]),
            "sex":str(m["sex"]),
            "Site":str(m["Site"]),
            "result_status":d.get("status"),
            "result_json_sha256":sha256_file(p),
        }
        if not d.get("participant_endpoint_eligible",False) or not d.get("endpoints"):
            reason=[]
            for task,t in d.get("tasks",{}).items():
                if not t.get("eligible",False):
                    q=t.get("qc",{})
                    if q and not q.get("passed_retained_fraction",False):
                        reason.append(f"{task}:retained_fraction_lt_0.8")
                    else:
                        reason.append(f"{task}:endpoint_or_coverage_ineligible")
            if not reason:
                reason=[d.get("status","unknown")]
            exclusions.append({**base,"reason":";".join(reason)})
            continue
        ep=d["endpoints"]
        records.append({
            **base,
            "hammer_affective":float(ep["hammer_affective"]),
            "hammer_control":float(ep["hammer_control"]),
            "stroop_affective":float(ep["stroop_affective"]),
            "stroop_control":float(ep["stroop_control"]),
            "hammer_selectivity":float(ep["hammer_selectivity"]),
            "stroop_selectivity":float(ep["stroop_selectivity"]),
        })

    df=pd.DataFrame(records)
    exc=pd.DataFrame(exclusions)
    if len(df)<2 or df["Patient"].nunique()!=2:
        raise RuntimeError("Final QC-complete sample does not contain both groups")
    for col in ["hammer_selectivity","stroop_selectivity"]:
        if not np.isfinite(df[col]).all():
            raise RuntimeError(f"Nonfinite selectivity {col}")
        sd=float(df[col].std(ddof=1))
        if not math.isfinite(sd) or sd<=0:
            raise RuntimeError(f"Cannot pooled-z standardize {col}; SD={sd}")
        df["z_"+col]=(df[col]-float(df[col].mean()))/sd

    age_mean=float(df["age"].mean())
    df["age_centered"]=df["age"]-age_mean

    long=pd.concat([
        df.assign(
            Domain="Hammer",
            DomainStroop=0,
            z_selectivity=df["z_hammer_selectivity"],
        ),
        df.assign(
            Domain="Stroop",
            DomainStroop=1,
            z_selectivity=df["z_stroop_selectivity"],
        ),
    ],ignore_index=True)

    formula="z_selectivity ~ Patient * DomainStroop + age_centered + C(sex) + C(Site)"
    fit=smf.ols(formula,data=long).fit(
        cov_type="cluster",
        cov_kwds={"groups":long["participant_id"],"use_correction":True},
    )
    interaction=ttest_record(fit,"Patient:DomainStroop = 0")
    hammer_effect=ttest_record(fit,"Patient = 0")
    stroop_effect=ttest_record(fit,"Patient + Patient:DomainStroop = 0")
    raw_simple=[hammer_effect["p_two_sided"],stroop_effect["p_two_sided"]]
    _,holm,_,_=multipletests(raw_simple,alpha=0.05,method="holm")
    hammer_effect["holm_adjusted_p_two_simple_effects"]=float(holm[0])
    stroop_effect["holm_adjusted_p_two_simple_effects"]=float(holm[1])

    strict=(
        interaction["p_two_sided"]<0.05
        and np.sign(hammer_effect["estimate"])!=0
        and np.sign(stroop_effect["estimate"])!=0
        and np.sign(hammer_effect["estimate"])!=np.sign(stroop_effect["estimate"])
        and hammer_effect["holm_adjusted_p_two_simple_effects"]<0.05
        and stroop_effect["holm_adjusted_p_two_simple_effects"]<0.05
    )

    raw_cells={}
    for group in ["GenPop","Patient"]:
        g=df[df["Group"]==group]
        raw_cells[group]={
            col:stats(g[col])
            for col in [
                "hammer_affective","hammer_control",
                "stroop_affective","stroop_control",
                "hammer_selectivity","stroop_selectivity",
                "z_hammer_selectivity","z_stroop_selectivity",
            ]
        }

    qc_summary={
        "source_eligible_n":int(len(eligible)),
        "source_group_counts":eligible["Group"].value_counts().to_dict(),
        "final_endpoint_n":int(len(df)),
        "final_group_counts":df["Group"].value_counts().to_dict(),
        "excluded_n":int(len(exc)),
        "excluded_group_counts":exc["Group"].value_counts().to_dict() if len(exc) else {},
        "excluded_reasons":exc["reason"].value_counts().to_dict() if len(exc) else {},
        "final_site_group_counts":(
            df.groupby(["Group","Site"]).size().rename("n").reset_index().to_dict("records")
        ),
    }

    out=Path(args.output_root)
    out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/"participant_neural_endpoints.tsv",sep="\t",index=False)
    long.to_csv(out/"group_model_long.tsv",sep="\t",index=False)
    exc.to_csv(out/"qc_exclusions.tsv",sep="\t",index=False)

    result={
        "status":"DS005237_NEURAL_DOUBLE_DISSOCIATION_GROUP_COMPLETE",
        "source_commit":SOURCE_COMMIT,
        "lock":"NEURAL_DOUBLE_DISSOCIATION_LOCK_2026-09-19.json",
        "formula":formula,
        "covariance":"cluster-robust by participant",
        "age_center_mean":age_mean,
        "qc":qc_summary,
        "primary_patient_by_domain_interaction":interaction,
        "simple_group_effects":{
            "hammer_selectivity_patient_minus_genpop":hammer_effect,
            "stroop_selectivity_patient_minus_genpop":stroop_effect,
        },
        "strict_double_dissociation_criterion_met":bool(strict),
        "raw_group_summaries":raw_cells,
        "interpretation_class":(
            "STRICT_NEURAL_DOUBLE_DISSOCIATION_CRITERION_MET"
            if strict else
            "NO_STRICT_NEURAL_DOUBLE_DISSOCIATION"
        ),
        "behavioral_result_is_separate":True,
        "outcome_dependent_tuning":False,
    }
    files={}
    for p in [out/"participant_neural_endpoints.tsv",out/"group_model_long.tsv",out/"qc_exclusions.tsv"]:
        files[p.name]={"bytes":p.stat().st_size,"sha256":sha256_file(p)}
    result["files"]=files
    (out/"neural_double_dissociation_result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
