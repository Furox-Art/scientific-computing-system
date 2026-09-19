#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, json, re, sys, zipfile
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

NS="{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
RNS="{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
PKG="{http://schemas.openxmlformats.org/package/2006/relationships}"

def md5(path:Path)->str:
    h=hashlib.md5()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def col_index(ref:str)->int:
    letters=re.match(r"([A-Z]+)",ref).group(1)
    n=0
    for ch in letters:
        n=n*26+(ord(ch)-64)
    return n-1

def read_xlsx(path:Path):
    with zipfile.ZipFile(path) as z:
        shared=[]
        if "xl/sharedStrings.xml" in z.namelist():
            root=ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall(NS+"si"):
                shared.append("".join(t.text or "" for t in si.iter(NS+"t")))
        wb=ET.fromstring(z.read("xl/workbook.xml"))
        relroot=ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        rels={r.attrib["Id"]:r.attrib["Target"] for r in relroot.findall(PKG+"Relationship")}
        out={}
        for s in wb.find(NS+"sheets"):
            name=s.attrib["name"]
            target=rels[s.attrib[RNS+"id"]]
            target=("xl/"+target.lstrip("/")) if not target.startswith("xl/") else target
            root=ET.fromstring(z.read(target))
            rows=[]
            for row in root.iter(NS+"row"):
                vals={}
                for c in row.findall(NS+"c"):
                    idx=col_index(c.attrib["r"])
                    typ=c.attrib.get("t")
                    v=c.find(NS+"v")
                    if typ=="inlineStr":
                        val="".join(t.text or "" for t in c.iter(NS+"t"))
                    elif v is None:
                        val=""
                    elif typ=="s":
                        val=shared[int(v.text)]
                    elif typ=="b":
                        val="TRUE" if v.text=="1" else "FALSE"
                    else:
                        val=v.text or ""
                    vals[idx]=val
                if vals:
                    width=max(vals)+1
                    rows.append([vals.get(i,"") for i in range(width)])
            width=max((len(r) for r in rows),default=0)
            rows=[r+[""]*(width-len(r)) for r in rows]
            out[name]=rows
        return out

def norm(x):
    return re.sub(r"[^a-z0-9]+","",str(x).strip().lower())

def main():
    source=Path(sys.argv[1])
    xlsx=Path(sys.argv[2])
    out=Path(sys.argv[3]); out.mkdir(parents=True,exist_ok=True)
    expected_md5="66eb1b6d6b6e2587dbafa449e63b471b"
    got=md5(xlsx)
    if got!=expected_md5:
        raise RuntimeError(f"Clinical xlsx MD5 mismatch {got} != {expected_md5}")

    wb=read_xlsx(xlsx)
    if "data_66" not in wb:
        raise RuntimeError(f"Expected data_66 sheet, found {list(wb)}")
    data=wb["data_66"]
    if len(data)<2:
        raise RuntimeError("data_66 has no data rows")
    headers=[str(x).strip() for x in data[0]]
    body=data[1:]
    while headers and not headers[-1]:
        headers.pop()
    body=[r[:len(headers)] for r in body if any(str(x).strip() for x in r[:len(headers)])]
    if len(body)!=66:
        raise RuntimeError(f"Expected 66 clinical rows, got {len(body)}")

    # Save exact extracted table without altering the source workbook.
    with (out/"clinical_data_66_extracted.tsv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f,delimiter="\t"); w.writerow(headers); w.writerows(body)

    hnorm=[norm(h) for h in headers]
    candidates={
      "id_columns":[headers[i] for i,h in enumerate(hnorm) if h in {"id","subject","subjectid","participant","participantid","sub","subid","number","num","no"} or "subject" in h or "participant" in h],
      "group_columns":[headers[i] for i,h in enumerate(hnorm) if h in {"gp","group","grupo","condition","diagnosis","dx"} or "group" in h or "grupo" in h or "diagnos" in h],
    }
    low_card=[]
    for i,h in enumerate(headers):
        vals=[str(r[i]).strip() for r in body if i<len(r) and str(r[i]).strip()!=""]
        uniq=sorted(set(vals))
        if 1<len(uniq)<=6:
            low_card.append({"column":h,"unique_values":uniq,"counts":dict(Counter(vals))})

    subjects=sorted(p.name for p in source.glob("sub-*") if p.is_dir())
    availability=[]
    trial_counts=defaultdict(Counter)
    all_trial_types=Counter()
    for s in subjects:
        func=source/s/"func"
        bold=func/f"{s}_task-epr_bold.nii.gz"
        events=func/f"{s}_task-epr_events.tsv"
        t1=list((source/s/"anat").glob("*_T1w.nii.gz"))
        rec={"participant_id":s,"t1_count":len(t1),"task_bold":bold.exists() or bold.is_symlink(),"events":events.exists()}
        availability.append(rec)
        if events.exists():
            with events.open(encoding="utf-8-sig",newline="") as f:
                rows=list(csv.DictReader(f,delimiter="\t"))
            for r in rows:
                tt=(r.get("trial_type") or "").strip()
                if tt:
                    trial_counts[s][tt]+=1
                    all_trial_types[tt]+=1

    missing_task=[x for x in availability if not(x["t1_count"]==1 and x["task_bold"] and x["events"])]
    with (out/"task_file_availability.tsv").open("w",newline="",encoding="utf-8") as f:
        fields=["participant_id","t1_count","task_bold","events"]
        w=csv.DictWriter(f,fieldnames=fields,delimiter="\t"); w.writeheader(); w.writerows(availability)
    with (out/"task_trial_counts.tsv").open("w",newline="",encoding="utf-8") as f:
        types=sorted(all_trial_types)
        w=csv.writer(f,delimiter="\t"); w.writerow(["participant_id",*types])
        for s in subjects: w.writerow([s,*[trial_counts[s].get(t,0) for t in types]])

    sheet_info={}
    for name,rows in wb.items():
        hdr=rows[0] if rows else []
        sheet_info[name]={"rows":max(0,len(rows)-1),"columns":len(hdr),"headers":hdr[:100]}
    summary={
      "status":"DS004144_HEALTHY_CLINICAL_METADATA_PREFLIGHT_COMPLETE",
      "source_commit":"b6e525adf7b861d5dab7e539bff19f90ce482032",
      "source_tree_sha":"24955a6321fa628d82b862006ecd2ffb992bfab6",
      "openneuro_dataset":"ds004144 v1.0.2",
      "openneuro_license":"CC0",
      "zenodo_record":7032997,
      "clinical_file":"Clinical_fm_66_.xlsx",
      "clinical_file_md5":got,
      "clinical_file_sha256":sha256(xlsx),
      "clinical_rows":len(body),
      "clinical_headers":headers,
      "candidate_columns":candidates,
      "low_cardinality_columns":low_card,
      "sheet_info":sheet_info,
      "bids_subjects":len(subjects),
      "subjects_with_t1_task_bold_and_events":sum(x["t1_count"]==1 and x["task_bold"] and x["events"] for x in availability),
      "missing_task_subjects":missing_task,
      "unique_trial_types":sorted(all_trial_types),
      "global_trial_type_counts":dict(sorted(all_trial_types.items())),
      "clinical_extracted_tsv_sha256":sha256(out/"clinical_data_66_extracted.tsv"),
      "availability_tsv_sha256":sha256(out/"task_file_availability.tsv"),
      "trial_counts_tsv_sha256":sha256(out/"task_trial_counts.tsv"),
      "neural_group_contrast_computed":False,
      "behavioral_group_contrast_computed":False,
    }
    (out/"metadata_preflight.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2,ensure_ascii=False))

if __name__=="__main__":
    main()
