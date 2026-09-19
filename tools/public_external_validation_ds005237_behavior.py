#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

SOURCE_COMMIT = "2e273d8466162208bbccd8591337e55b7a8b5721"
EXPECTED = {
    "participants_total": 245,
    "patient_total": 149,
    "genpop_total": 96,
    "complete_total": 212,
    "complete_patient": 126,
    "complete_genpop": 86,
    "site": {
        ("Patient", "1"): 69,
        ("Patient", "2"): 57,
        ("GenPop", "1"): 54,
        ("GenPop", "2"): 32,
    },
}

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def lexists(path: Path) -> bool:
    return os.path.lexists(path)

def required_paths(root: Path, pid: str) -> list[Path]:
    func = root / pid / "func"
    return [
        func / f"{pid}_task-hammerAP_run-01_bold.nii.gz",
        func / f"{pid}_task-hammerAP_run-01_events.tsv",
        func / f"{pid}_task-stroopAP_run-01_bold.nii.gz",
        func / f"{pid}_task-stroopAP_run-01_events.tsv",
        func / f"{pid}_task-stroopPA_run-01_bold.nii.gz",
        func / f"{pid}_task-stroopPA_run-01_events.tsv",
    ]

def build_cohort(root: Path) -> tuple[pd.DataFrame, dict]:
    p = root / "participants.tsv"
    df = pd.read_csv(p, sep="\t", dtype={"Site": str, "Group": str, "sex": str})
    if df["participant_id"].duplicated().any():
        raise RuntimeError("Duplicate participant_id in pinned participants.tsv")
    counts = {
        "participants_total": int(len(df)),
        "patient_total": int((df["Group"] == "Patient").sum()),
        "genpop_total": int((df["Group"] == "GenPop").sum()),
    }
    for k in ("participants_total","patient_total","genpop_total"):
        if counts[k] != EXPECTED[k]:
            raise RuntimeError(f"Pinned metadata count mismatch {k}: {counts[k]} != {EXPECTED[k]}")

    df = df[df["Group"].isin(["Patient","GenPop"])].copy()
    df["has_all_required"] = [
        all(lexists(x) for x in required_paths(root, pid))
        for pid in df["participant_id"]
    ]
    cohort = df[df["has_all_required"]].copy().reset_index(drop=True)
    ccounts = {
        "complete_total": int(len(cohort)),
        "complete_patient": int((cohort["Group"] == "Patient").sum()),
        "complete_genpop": int((cohort["Group"] == "GenPop").sum()),
    }
    for k in ("complete_total","complete_patient","complete_genpop"):
        if ccounts[k] != EXPECTED[k]:
            raise RuntimeError(f"Complete-case count mismatch {k}: {ccounts[k]} != {EXPECTED[k]}")
    for (g,s), n in EXPECTED["site"].items():
        got = int(((cohort["Group"] == g) & (cohort["Site"] == s)).sum())
        if got != n:
            raise RuntimeError(f"Site count mismatch {g} site {s}: {got} != {n}")

    cohort["age"] = pd.to_numeric(cohort["age"], errors="coerce")
    if cohort["age"].isna().any():
        raise RuntimeError("Missing/non-numeric age in locked complete cohort")
    cohort["patient"] = (cohort["Group"] == "Patient").astype(int)
    return cohort, {**counts, **ccounts}

def load_events(root: Path, pid: str, task: str) -> pd.DataFrame:
    p = root / pid / "func" / f"{pid}_task-{task}_run-01_events.tsv"
    df = pd.read_csv(p, sep="\t", na_values=["n/a","NA","NaN"])
    required = {"trial_type","response_time","accuracy_binarized"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{pid} {task}: missing event columns {sorted(missing)}")
    return df

def finite_correct_rt(df: pd.DataFrame, mask: pd.Series) -> pd.Series:
    rt = pd.to_numeric(df.loc[mask & (df["accuracy_binarized"] == 1), "response_time"], errors="coerce")
    rt = rt[np.isfinite(rt) & (rt > 0)]
    return rt

def accuracy(df: pd.DataFrame, mask: pd.Series) -> float:
    a = pd.to_numeric(df.loc[mask, "accuracy_binarized"], errors="coerce")
    a = a[a.isin([0,1])]
    return float(a.mean()) if len(a) else math.nan

def summarize_participant(root: Path, row: pd.Series) -> dict:
    pid = row["participant_id"]
    hammer = load_events(root, pid, "hammerAP")
    htypes = set(hammer["trial_type"].dropna().astype(str).unique())
    if not htypes <= {"face","shape"}:
        raise RuntimeError(f"{pid}: unexpected hammer trial types {sorted(htypes)}")
    if not {"face","shape"} <= htypes:
        raise RuntimeError(f"{pid}: hammer missing required condition; got {sorted(htypes)}")

    face = hammer["trial_type"].astype(str) == "face"
    shape = hammer["trial_type"].astype(str) == "shape"
    rt_face = finite_correct_rt(hammer, face)
    rt_shape = finite_correct_rt(hammer, shape)
    acc_face = accuracy(hammer, face)
    acc_shape = accuracy(hammer, shape)

    stroops = []
    for task in ("stroopAP","stroopPA"):
        d = load_events(root, pid, task)
        types = d["trial_type"].dropna().astype(str)
        if not types.map(lambda x: x.startswith("con") or x.startswith("inc")).all():
            bad = sorted(set(types[~types.map(lambda x: x.startswith("con") or x.startswith("inc"))]))
            raise RuntimeError(f"{pid} {task}: unexpected Stroop trial types {bad}")
        stroops.append(d)
    stroop = pd.concat(stroops, ignore_index=True)
    stype = stroop["trial_type"].astype(str)
    con = stype.str.startswith("con")
    inc = stype.str.startswith("inc")
    rt_con = finite_correct_rt(stroop, con)
    rt_inc = finite_correct_rt(stroop, inc)
    acc_con = accuracy(stroop, con)
    acc_inc = accuracy(stroop, inc)

    def med(x: pd.Series) -> float:
        return float(np.median(x)) if len(x) else math.nan

    out = {
        "participant_id": pid,
        "Group": row["Group"],
        "patient": int(row["patient"]),
        "age": float(row["age"]),
        "sex": str(row["sex"]),
        "Site": str(row["Site"]),
        "faces_n_face": int(face.sum()),
        "faces_n_shape": int(shape.sum()),
        "faces_n_rt_face": int(len(rt_face)),
        "faces_n_rt_shape": int(len(rt_shape)),
        "faces_median_rt_face": med(rt_face),
        "faces_median_rt_shape": med(rt_shape),
        "faces_accuracy_face": acc_face,
        "faces_accuracy_shape": acc_shape,
        "stroop_n_con": int(con.sum()),
        "stroop_n_inc": int(inc.sum()),
        "stroop_n_rt_con": int(len(rt_con)),
        "stroop_n_rt_inc": int(len(rt_inc)),
        "stroop_median_rt_con": med(rt_con),
        "stroop_median_rt_inc": med(rt_inc),
        "stroop_accuracy_con": acc_con,
        "stroop_accuracy_inc": acc_inc,
    }
    out["faces_rt_cost"] = out["faces_median_rt_face"] - out["faces_median_rt_shape"]
    out["stroop_rt_cost"] = out["stroop_median_rt_inc"] - out["stroop_median_rt_con"]
    out["faces_accuracy_cost"] = out["faces_accuracy_shape"] - out["faces_accuracy_face"]
    out["stroop_accuracy_cost"] = out["stroop_accuracy_con"] - out["stroop_accuracy_inc"]
    out["double_difference_rt"] = out["faces_rt_cost"] - out["stroop_rt_cost"]
    out["double_difference_accuracy"] = out["faces_accuracy_cost"] - out["stroop_accuracy_cost"]
    return out

def model_record(df: pd.DataFrame, outcome: str) -> dict:
    cols = [outcome,"patient","age","sex","Site"]
    d = df[cols].replace([np.inf,-np.inf], np.nan).dropna().copy()
    if d["patient"].nunique() != 2:
        raise RuntimeError(f"{outcome}: both groups not present after endpoint complete-case filtering")
    formula = f"{outcome} ~ patient + age + C(sex) + C(Site)"
    fit = smf.ols(formula, data=d).fit(cov_type="HC3")
    ci = fit.conf_int().loc["patient"].tolist()
    return {
        "outcome": outcome,
        "formula": formula,
        "estimator": "OLS_HC3",
        "n": int(len(d)),
        "n_patient": int((d["patient"] == 1).sum()),
        "n_genpop": int((d["patient"] == 0).sum()),
        "patient_beta": float(fit.params["patient"]),
        "patient_se_hc3": float(fit.bse["patient"]),
        "patient_t": float(fit.tvalues["patient"]),
        "patient_p_two_sided": float(fit.pvalues["patient"]),
        "patient_ci95": [float(ci[0]), float(ci[1])],
        "r_squared": float(fit.rsquared),
    }

def group_summary(df: pd.DataFrame, outcome: str) -> dict:
    out = {}
    for g in ("GenPop","Patient"):
        x = pd.to_numeric(df.loc[df["Group"] == g, outcome], errors="coerce")
        x = x[np.isfinite(x)]
        out[g] = {
            "n": int(len(x)),
            "mean": float(x.mean()) if len(x) else math.nan,
            "median": float(x.median()) if len(x) else math.nan,
            "sd": float(x.std(ddof=1)) if len(x) > 1 else math.nan,
        }
    return out

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--output-root", required=True)
    args = ap.parse_args()
    root = Path(args.source_root)
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)

    cohort, counts = build_cohort(root)
    rows = [summarize_participant(root, row) for _, row in cohort.iterrows()]
    d = pd.DataFrame(rows)

    endpoints = [
        "double_difference_rt",
        "double_difference_accuracy",
        "faces_rt_cost",
        "stroop_rt_cost",
        "faces_accuracy_cost",
        "stroop_accuracy_cost",
    ]
    models = {x: model_record(d, x) for x in endpoints}

    # Locked multiplicity rule: primary RT double-difference is single primary endpoint.
    # Holm family contains accuracy interaction + four simple effects.
    family = [
        "double_difference_accuracy",
        "faces_rt_cost",
        "stroop_rt_cost",
        "faces_accuracy_cost",
        "stroop_accuracy_cost",
    ]
    raw_p = [models[x]["patient_p_two_sided"] for x in family]
    _, holm, _, _ = multipletests(raw_p, alpha=0.05, method="holm")
    for name, adj in zip(family, holm):
        models[name]["holm_adjusted_p_confirmatory_family"] = float(adj)

    faces_beta = models["faces_rt_cost"]["patient_beta"]
    stroop_beta = models["stroop_rt_cost"]["patient_beta"]
    primary_ci = models["double_difference_rt"]["patient_ci95"]
    strict = (
        np.sign(faces_beta) != 0
        and np.sign(stroop_beta) != 0
        and np.sign(faces_beta) != np.sign(stroop_beta)
        and (primary_ci[0] > 0 or primary_ci[1] < 0)
    )

    d.to_csv(out / "participant_task_costs.tsv", sep="\t", index=False)
    cohort[["participant_id","Group","age","sex","Site"]].to_csv(
        out / "locked_complete_case_cohort.tsv", sep="\t", index=False
    )

    result = {
        "status": "DS005237_CLINICAL_HEALTHY_DOUBLE_DISSOCIATION_BEHAVIOR_COMPLETE",
        "source_repository": "OpenNeuroDatasets/ds005237",
        "source_commit": SOURCE_COMMIT,
        "analysis_lock": "CLINICAL_HEALTHY_DOUBLE_DISSOCIATION_BEHAVIOR_LOCK_2026-09-19.json",
        "outcome_tuning": False,
        "cohort_counts": counts,
        "models": models,
        "group_summaries": {x: group_summary(d, x) for x in endpoints},
        "strict_opposite_direction_double_dissociation_rt_criterion_met": bool(strict),
        "primary_endpoint": "double_difference_rt",
        "primary_interpretation_rule": "patient beta tests whether the Patient-vs-GenPop task cost difference differs between Faces and Stroop.",
        "neural_outcome_computed": False,
    }

    files = {}
    for p in [out / "participant_task_costs.tsv", out / "locked_complete_case_cohort.tsv"]:
        files[p.name] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
    result["files"] = files
    (out / "behavior_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["files"]["behavior_result.json"] = {
        "bytes": (out / "behavior_result.json").stat().st_size,
        "sha256_before_self_entry": sha256_file(out / "behavior_result.json"),
    }
    # Rewrite once with the auxiliary self-hash field explicitly labeled as pre-rewrite.
    (out / "behavior_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
