#!/usr/bin/env python3
import json, math, tempfile, urllib.request, zipfile, io, hashlib
from pathlib import Path
import xlrd
import numpy as np
from scipy import stats

URL="https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-021-23540-y/MediaObjects/41467_2021_23540_MOESM4_ESM.zip"
MEMBER="Supplementary Data/Fig.4h.xls"
OUT=Path("pereira_audit/FIG4H_BIC_REPRODUCTION_2026-09-22.json")

req=urllib.request.Request(URL,headers={"User-Agent":"Mozilla/5.0"})
with urllib.request.urlopen(req,timeout=300) as r:
    rawzip=r.read()
zip_sha=hashlib.sha256(rawzip).hexdigest()
with zipfile.ZipFile(io.BytesIO(rawzip)) as z:
    raw=z.read(MEMBER)
xls_sha=hashlib.sha256(raw).hexdigest()
book=xlrd.open_workbook(file_contents=raw)
sheets=[]
for ws in book.sheets():
    rows=[[ws.cell_value(r,c) for c in range(ws.ncols)] for r in range(ws.nrows)]
    sheets.append({"title":ws.name,"rows":rows})

# Robustly find two numeric columns with 18 participant rows.
numeric_candidates=[]
for si,ws in enumerate(book.sheets()):
    for c in range(ws.ncols):
        vals=[]
        ridx=[]
        for r in range(ws.nrows):
            v=ws.cell_value(r,c)
            if isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(float(v)):
                vals.append(float(v)); ridx.append(r)
        if len(vals)>=18:
            numeric_candidates.append({"sheet_index":si,"col":c,"values":vals,"rows":ridx})

# Fig.4h source data are expected to contain exactly the paired model BIC vectors.
# Prefer two columns sharing the same 18-row numeric support; if more exist, retain all
# and report structure for an explicit mapping audit.
pairs=[]
for i,a in enumerate(numeric_candidates):
    for b in numeric_candidates[i+1:]:
        common=sorted(set(a["rows"]) & set(b["rows"]))
        if len(common)==18 and a["sheet_index"]==b["sheet_index"]:
            va=[]; vb=[]
            ws=book.sheet_by_index(a["sheet_index"])
            for r in common:
                xa=ws.cell_value(r,a["col"]); xb=ws.cell_value(r,b["col"])
                if isinstance(xa,(int,float)) and isinstance(xb,(int,float)):
                    va.append(float(xa)); vb.append(float(xb))
            if len(va)==18:
                pairs.append({"sheet_index":a["sheet_index"],"col_a":a["col"],"col_b":b["col"],"rows":common,"a":va,"b":vb})

result={
 "status":"PEREIRA2021_FIG4H_SOURCE_DATA_PARSED",
 "zip_sha256":zip_sha,
 "fig4h_xls_sha256":xls_sha,
 "sheets":sheets,
 "numeric_candidates":numeric_candidates,
 "candidate_pairs":pairs
}

# Infer mapping from source-reported means: maximal≈178.82, resp≈229.31.
best=None
for p in pairs:
    ma=float(np.mean(p["a"])); mb=float(np.mean(p["b"]))
    options=[
      (abs(ma-178.82)+abs(mb-229.31),"a=max,b=resp",p["a"],p["b"]),
      (abs(mb-178.82)+abs(ma-229.31),"b=max,a=resp",p["b"],p["a"])
    ]
    cand=min(options,key=lambda x:x[0])
    if best is None or cand[0]<best[0]:
        best=(cand[0],cand[1],cand[2],cand[3],p)
if best is not None:
    _,mapping,maxv,respv,pair=best
    maxv=np.asarray(maxv,float); respv=np.asarray(respv,float)
    delta=maxv-respv
    wil_auto=stats.wilcoxon(maxv,respv,alternative="two-sided",zero_method="wilcox",method="auto")
    try:
        wil_exact=stats.wilcoxon(maxv,respv,alternative="two-sided",zero_method="wilcox",method="exact")
        exact={"statistic":float(wil_exact.statistic),"p":float(wil_exact.pvalue)}
    except Exception as e:
        exact={"error":repr(e)}
    try:
        wil_approx=stats.wilcoxon(maxv,respv,alternative="two-sided",zero_method="wilcox",method="approx")
        approx={"statistic":float(wil_approx.statistic),"p":float(wil_approx.pvalue)}
    except Exception as e:
        approx={"error":repr(e)}
    nneg=int(np.sum(delta<0)); npos=int(np.sum(delta>0)); ntie=int(np.sum(delta==0))
    # Two-sided exact sign test sensitivity ignoring ties.
    signp=float(stats.binomtest(min(nneg,npos),n=nneg+npos,p=0.5,alternative="two-sided").pvalue) if nneg+npos else None
    result["locked_bic_reproduction"]={
      "mapping":mapping,
      "mapping_error_to_published_means":float(best[0]),
      "rows":[int(r)+1 for r in pair["rows"]],
      "bic_maximal_evidence":maxv.tolist(),
      "bic_fixed_timing_random":respv.tolist(),
      "delta_max_minus_resp":delta.tolist(),
      "summary":{
        "n":int(len(delta)),
        "mean_max":float(np.mean(maxv)),
        "sd_max":float(np.std(maxv,ddof=1)),
        "sem_max":float(stats.sem(maxv)),
        "mean_resp":float(np.mean(respv)),
        "sd_resp":float(np.std(respv,ddof=1)),
        "sem_resp":float(stats.sem(respv)),
        "mean_delta":float(np.mean(delta)),
        "median_delta":float(np.median(delta)),
        "sd_delta":float(np.std(delta,ddof=1)),
        "maximal_lower_bic_wins":nneg,
        "resp_lower_bic_wins":npos,
        "ties":ntie
      },
      "wilcoxon_auto":{"statistic":float(wil_auto.statistic),"p":float(wil_auto.pvalue)},
      "wilcoxon_exact":exact,
      "wilcoxon_approx":approx,
      "sign_test_sensitivity_p":signp,
      "locked_direction_met":bool(np.median(delta)<0),
      "locked_alpha_met":bool(float(wil_auto.pvalue)<0.05),
      "classification":"MAXIMAL_EVIDENCE_FAVORED" if np.median(delta)<0 and float(wil_auto.pvalue)<0.05 else "NO_CLEAR_RIVAL_ADVANTAGE"
    }
OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
print(json.dumps(result,indent=2))
