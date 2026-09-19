#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import urllib.parse
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

DATASET = "ds005237"
SOURCE_REPO = "OpenNeuroDatasets/ds005237"
SOURCE_COMMIT = "2e273d8466162208bbccd8591337e55b7a8b5721"
VERSION = "1.1.3"
PARTICIPANTS_BLOB = "3dd58d4514c467c1192f05df5a6ea89c3bb20a07"
S3_BASE = "https://s3.amazonaws.com/openneuro.org/ds005237/"
GROUPS = {"Patient", "GenPop"}


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def request_bytes(url: str, headers=None, retries: int = 4) -> bytes:
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=headers or {}, timeout=90)
            r.raise_for_status()
            return r.content
        except Exception as e:
            last = e
    raise RuntimeError(f"Failed after {retries} attempts: {url}: {last}")


def github_headers() -> dict:
    h = {"Accept": "application/vnd.github+json", "User-Agent": "p08-ds005237-validator"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def get_tree() -> list[dict]:
    url = f"https://api.github.com/repos/{SOURCE_REPO}/git/trees/{SOURCE_COMMIT}?recursive=1"
    data = json.loads(request_bytes(url, github_headers()))
    if data.get("truncated"):
        raise RuntimeError("GitHub recursive tree is truncated")
    return data["tree"]


def download_exact_inputs(root: pathlib.Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    tree = get_tree()
    blobs = {x["path"]: x["sha"] for x in tree if x.get("type") == "blob"}

    if blobs.get("participants.tsv") != PARTICIPANTS_BLOB:
        raise RuntimeError(f"participants.tsv blob mismatch in frozen tree: {blobs.get('participants.tsv')}")

    part_url = f"https://raw.githubusercontent.com/{SOURCE_REPO}/{SOURCE_COMMIT}/participants.tsv"
    part_bytes = request_bytes(part_url, github_headers())
    if git_blob_sha(part_bytes) != PARTICIPANTS_BLOB:
        raise RuntimeError("Downloaded participants.tsv does not match frozen Git blob")
    (root / "participants.tsv").write_bytes(part_bytes)

    wanted = []
    for path, sha in blobs.items():
        if not path.endswith("_events.tsv"):
            continue
        if "/func/" not in path:
            continue
        name = pathlib.PurePosixPath(path).name
        if "task-hammer" in name or "task-stroop" in name:
            wanted.append((path, sha))
    wanted.sort()
    if not wanted:
        raise RuntimeError("No Hammer/Stroop event files found in frozen tree")

    manifest = []
    failures = []
    for idx, (path, expected_blob) in enumerate(wanted, 1):
        url = S3_BASE + urllib.parse.quote(path, safe="/")
        data = request_bytes(url)
        got_blob = git_blob_sha(data)
        if got_blob != expected_blob:
            failures.append({"path": path, "expected_git_blob": expected_blob, "got_git_blob": got_blob})
            continue
        dest = root / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        manifest.append({
            "path": path,
            "bytes": len(data),
            "git_blob_sha1": got_blob,
            "sha256": sha256_bytes(data),
        })
        if idx % 100 == 0:
            print(f"verified {idx}/{len(wanted)} event files", flush=True)

    if failures:
        raise RuntimeError("Frozen event verification failed: " + json.dumps(failures[:10]))

    by_task = defaultdict(int)
    for x in manifest:
        n = pathlib.PurePosixPath(x["path"]).name
        by_task["hammer" if "task-hammer" in n else "stroop"] += 1

    source = {
        "dataset": DATASET,
        "version": VERSION,
        "source_commit": SOURCE_COMMIT,
        "participants_blob_sha1": PARTICIPANTS_BLOB,
        "participants_sha256": sha256_bytes(part_bytes),
        "event_files_verified": len(manifest),
        "event_files_by_task": dict(by_task),
        "all_event_git_blobs_exact": True,
        "manifest": manifest,
    }
    (root / "source_verification.json").write_text(json.dumps(source, indent=2) + "\n")
    return source


def load_events(root: pathlib.Path, participants: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required_part = {"participant_id", "age", "sex", "Site", "Group"}
    missing = required_part - set(participants.columns)
    if missing:
        raise RuntimeError(f"participants.tsv missing columns: {sorted(missing)}")
    participants = participants.loc[participants["Group"].isin(GROUPS)].copy()
    participants["participant_id"] = participants["participant_id"].astype(str)
    participants["age"] = pd.to_numeric(participants["age"], errors="coerce")
    participants["Site"] = participants["Site"].astype(str)
    participants["sex"] = participants["sex"].astype(str)

    rows = []
    file_qc = []
    for path in sorted(root.glob("sub-*/func/*_events.tsv")):
        pid = path.parts[-3]
        if pid not in set(participants["participant_id"]):
            continue
        name = path.name
        domain = "hammer" if "task-hammer" in name else "stroop" if "task-stroop" in name else None
        if domain is None:
            continue
        d = pd.read_csv(path, sep="\t", na_values=["n/a", "NA", "NaN"])
        need = {"trial_type", "accuracy_binarized", "response_time"}
        if not need.issubset(d.columns):
            raise RuntimeError(f"{path} missing required columns {sorted(need - set(d.columns))}")
        d["trial_type"] = d["trial_type"].astype(str)
        d["accuracy_binarized"] = pd.to_numeric(d["accuracy_binarized"], errors="coerce")
        d["response_time"] = pd.to_numeric(d["response_time"], errors="coerce")
        if domain == "hammer":
            cond = d["trial_type"].map({"face": "target", "shape": "baseline"})
        else:
            cond = pd.Series(np.where(d["trial_type"].str.startswith("inc"), "target",
                           np.where(d["trial_type"].str.startswith("con"), "baseline", None)),
                           index=d.index)
        keep = cond.notna()
        d = d.loc[keep].copy()
        cond = cond.loc[keep]
        if d.empty:
            continue
        d["condition"] = cond.values
        d["participant_id"] = pid
        d["domain"] = domain
        d["source_file"] = path.relative_to(root).as_posix()
        rows.append(d[["participant_id","domain","condition","accuracy_binarized","response_time","source_file"]])
        file_qc.append({"path": path.relative_to(root).as_posix(), "participant_id": pid, "domain": domain, "rows_used": int(len(d))})
    if not rows:
        raise RuntimeError("No usable event rows")
    ev = pd.concat(rows, ignore_index=True)
    return ev, {"files": file_qc, "n_files": len(file_qc), "n_event_rows": int(len(ev))}


def participant_costs(events: pd.DataFrame, participants: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    records = []
    condition_summary = []
    for (pid, domain), g in events.groupby(["participant_id","domain"], sort=True):
        cells = {}
        for cond in ["baseline","target"]:
            x = g.loc[g["condition"] == cond].copy()
            n_trials = int(len(x))
            acc_valid = x["accuracy_binarized"].isin([0,1])
            n_acc = int(acc_valid.sum())
            accuracy = float(x.loc[acc_valid,"accuracy_binarized"].mean()) if n_acc else math.nan
            rtmask = (x["accuracy_binarized"] == 1) & np.isfinite(x["response_time"]) & (x["response_time"] > 0)
            n_rt = int(rtmask.sum())
            rt = float(x.loc[rtmask,"response_time"].mean()) if n_rt else math.nan
            cells[cond] = {"n_trials":n_trials,"n_acc":n_acc,"accuracy":accuracy,"n_correct_rt":n_rt,"correct_rt_s":rt}
            condition_summary.append({
                "participant_id":pid,"domain":domain,"condition":cond,
                "n_trials":n_trials,"n_accuracy":n_acc,"accuracy":accuracy,
                "n_correct_rt":n_rt,"correct_rt_s":rt,
            })

        acc_eligible = all(cells[c]["n_trials"] >= 10 and np.isfinite(cells[c]["accuracy"]) for c in ["baseline","target"])
        rt_eligible = all(cells[c]["n_correct_rt"] >= 5 and np.isfinite(cells[c]["correct_rt_s"]) for c in ["baseline","target"])
        acc_cost = cells["baseline"]["accuracy"] - cells["target"]["accuracy"] if acc_eligible else math.nan
        rt_cost = cells["target"]["correct_rt_s"] - cells["baseline"]["correct_rt_s"] if rt_eligible else math.nan
        records.append({
            "participant_id":pid,"domain":domain,
            "accuracy_cost":acc_cost,"rt_cost_s":rt_cost,
            "accuracy_eligible":bool(acc_eligible),"rt_eligible":bool(rt_eligible),
            "baseline_trials":cells["baseline"]["n_trials"],"target_trials":cells["target"]["n_trials"],
            "baseline_correct_rt_n":cells["baseline"]["n_correct_rt"],"target_correct_rt_n":cells["target"]["n_correct_rt"],
        })

    cost = pd.DataFrame(records)
    p = participants[["participant_id","Group","age","sex","Site"]].copy()
    cost = cost.merge(p,on="participant_id",how="left",validate="many_to_one")
    cond = pd.DataFrame(condition_summary)
    cond = cond.merge(p,on="participant_id",how="left",validate="many_to_one")
    return cost, {"condition_summary":cond}


def linear_combo(model, weights: dict[str,float]) -> dict:
    names = list(model.params.index)
    v = np.array([weights.get(n,0.0) for n in names], dtype=float)
    est = float(v @ model.params.values)
    cov = np.asarray(model.cov_params())
    se = float(np.sqrt(max(0.0, v @ cov @ v)))
    z = est / se if se > 0 else math.nan
    from scipy.stats import norm
    p = float(2 * norm.sf(abs(z))) if np.isfinite(z) else math.nan
    return {"estimate":est,"se":se,"z":z,"p":p,"ci95":[est-1.959963984540054*se, est+1.959963984540054*se]}


def fit_endpoint(cost: pd.DataFrame, field: str, endpoint: str) -> tuple[dict,pd.DataFrame]:
    # Complete case for both domains by endpoint, plus frozen covariates.
    wide_ok = (
        cost.loc[np.isfinite(cost[field]), ["participant_id","domain"]]
        .drop_duplicates()
        .groupby("participant_id")["domain"].nunique()
    )
    pids = set(wide_ok[wide_ok == 2].index)
    d = cost.loc[cost["participant_id"].isin(pids) & np.isfinite(cost[field])].copy()
    d = d.dropna(subset=["Group","age","sex","Site"])
    # Recheck after covariate completeness.
    complete = d.groupby("participant_id")["domain"].nunique()
    d = d[d["participant_id"].isin(complete[complete==2].index)].copy()
    d["group_patient"] = (d["Group"] == "Patient").astype(int)
    d["domain_stroop"] = (d["domain"] == "stroop").astype(int)
    d["age_centered"] = d["age"] - d.drop_duplicates("participant_id")["age"].mean()
    d["Site"] = d["Site"].astype("category")
    d["sex"] = d["sex"].astype("category")
    if d["participant_id"].nunique() < 20:
        raise RuntimeError(f"Too few complete participants for {endpoint}: {d['participant_id'].nunique()}")

    formula = f"{field} ~ group_patient * domain_stroop + age_centered + C(sex) + C(Site)"
    model = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["participant_id"]})

    interaction_name = "group_patient:domain_stroop"
    if interaction_name not in model.params:
        raise RuntimeError("Interaction coefficient missing")
    hammer = linear_combo(model, {"group_patient":1.0})
    stroop = linear_combo(model, {"group_patient":1.0, interaction_name:1.0})

    desc = []
    for (group,domain), g in d.groupby(["Group","domain"], observed=True):
        desc.append({
            "group":str(group),"domain":str(domain),"n_participants":int(g["participant_id"].nunique()),
            "mean_cost":float(g[field].mean()),"sd_cost":float(g[field].std(ddof=1)),
            "median_cost":float(g[field].median()),
        })

    out = {
        "endpoint":endpoint,
        "field":field,
        "formula":formula,
        "n_participants":int(d["participant_id"].nunique()),
        "group_counts":d.drop_duplicates("participant_id")["Group"].value_counts().to_dict(),
        "site_counts":d.drop_duplicates("participant_id")["Site"].astype(str).value_counts().to_dict(),
        "sex_counts":d.drop_duplicates("participant_id")["sex"].astype(str).value_counts().to_dict(),
        "interaction":{
            "coefficient_name":interaction_name,
            "estimate_stroop_minus_hammer_group_effect":float(model.params[interaction_name]),
            "cluster_robust_se":float(model.bse[interaction_name]),
            "t_or_z":float(model.tvalues[interaction_name]),
            "raw_p":float(model.pvalues[interaction_name]),
            "ci95":[float(model.conf_int().loc[interaction_name,0]),float(model.conf_int().loc[interaction_name,1])],
        },
        "adjusted_group_effects":{
            "hammer_patient_minus_genpop":hammer,
            "stroop_patient_minus_genpop":stroop,
        },
        "descriptive_costs":desc,
        "coefficients":{k:float(v) for k,v in model.params.items()},
    }
    return out, d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-root", default=".public-runner/ds005237/behavior")
    args = ap.parse_args()
    root = pathlib.Path(args.work_root)
    inputs = root / "inputs"
    outdir = root / "output"
    outdir.mkdir(parents=True, exist_ok=True)

    source = download_exact_inputs(inputs)
    participants = pd.read_csv(inputs / "participants.tsv", sep="\t")
    events, event_qc = load_events(inputs, participants)
    costs, extra = participant_costs(events, participants)

    # Save participant/cell tables before model fit for audit.
    costs.to_csv(outdir / "participant_domain_costs.tsv", sep="\t", index=False, na_rep="n/a")
    extra["condition_summary"].to_csv(outdir / "participant_condition_summary.tsv", sep="\t", index=False, na_rep="n/a")

    rt, rtdata = fit_endpoint(costs, "rt_cost_s", "correct_rt_cost_seconds")
    acc, accdata = fit_endpoint(costs, "accuracy_cost", "accuracy_cost")

    raw_ps = [rt["interaction"]["raw_p"], acc["interaction"]["raw_p"]]
    reject, adj, _, _ = multipletests(raw_ps, alpha=0.05, method="holm")
    rt["interaction"]["holm_adjusted_p_across_two_endpoints"] = float(adj[0])
    rt["interaction"]["holm_reject_0_05"] = bool(reject[0])
    acc["interaction"]["holm_adjusted_p_across_two_endpoints"] = float(adj[1])
    acc["interaction"]["holm_reject_0_05"] = bool(reject[1])

    def classify(model_result):
        inter_sig = model_result["interaction"]["holm_reject_0_05"]
        h = model_result["adjusted_group_effects"]["hammer_patient_minus_genpop"]["estimate"]
        s = model_result["adjusted_group_effects"]["stroop_patient_minus_genpop"]["estimate"]
        if not inter_sig:
            return "NO_MULTIPLICITY_CORRECTED_GROUP_BY_DOMAIN_INTERACTION"
        if h == 0 or s == 0:
            return "INTERACTION_WITH_ONE_ZERO_POINT_ESTIMATE"
        if np.sign(h) != np.sign(s):
            return "CROSSOVER_DOUBLE_DISSOCIATION_STYLE_PATTERN"
        return "DIFFERENTIAL_MAGNITUDE_NOT_STRICT_CROSSOVER"

    rt["classification"] = classify(rt)
    acc["classification"] = classify(acc)

    result = {
        "status":"DS005237_HEALTHY_CLINICAL_BEHAVIORAL_DOUBLE_DISSOCIATION_COMPLETE",
        "lane":"healthy_vs_clinical_double_dissociation_behavior",
        "dataset":{
            "id":DATASET,"version":VERSION,"source_commit":SOURCE_COMMIT,
            "license":"CC0",
        },
        "pre_outcome_lock":"external_validation/ds005237/BEHAVIORAL_DOUBLE_DISSOCIATION_LOCK_2026-09-19.json",
        "source_verification":{
            "participants_blob_sha1":source["participants_blob_sha1"],
            "participants_sha256":source["participants_sha256"],
            "event_files_verified":source["event_files_verified"],
            "event_files_by_task":source["event_files_by_task"],
            "all_event_git_blobs_exact":True,
        },
        "event_qc":event_qc,
        "primary_rt":rt,
        "secondary_accuracy":acc,
        "interpretation_rules_applied":{
            "strict_double_dissociation_not_inferred_from_separate_main_effects":True,
            "primary_test":"Group(Patient/GenPop) x Domain(Hammer/Stroop) interaction on participant-level correct-RT costs",
            "multiplicity":"Holm across RT and accuracy interaction p-values",
        },
    }
    result_path = outdir / "behavioral_double_dissociation_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")

    # File hashes.
    hashes={}
    for p in sorted(outdir.iterdir()):
        if p.is_file():
            b=p.read_bytes()
            hashes[p.name]={"bytes":len(b),"sha256":hashlib.sha256(b).hexdigest()}
    (outdir/"output_hashes.json").write_text(json.dumps(hashes,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
