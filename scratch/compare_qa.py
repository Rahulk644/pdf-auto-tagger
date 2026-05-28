"""Compile PREP vs my-pipeline (pre-S8, Stage8 before-fix, Stage8+fix) vs OpenDataLoader."""
import json
from pathlib import Path

MODAL = Path("/Users/rahulkhatri/PREP QA Tool/scratch/qa_results_modal")
S8_PRE = Path("/Users/rahulkhatri/PREP QA Tool/scratch/qa_results_modal_stage8_beforefix")
S8_LIST = Path("/Users/rahulkhatri/PREP QA Tool/scratch/qa_results_modal_stage8_listfix")
S8_TOC = Path("/Users/rahulkhatri/PREP QA Tool/scratch/qa_results_modal_stage8")
ODL = Path("/Users/rahulkhatri/PREP QA Tool/scratch/qa_results_odl")


def accuracy(path: Path):
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    tot = cor = 0
    for r in data.get("results", []):
        md = r.get("metadata", {}) if isinstance(r, dict) else {}
        tot += md.get("element_count", 0)
        cor += md.get("elements_correct", 0)
    if tot == 0 and "total_elements" in data:
        tot, cor = data["total_elements"], data.get("correct", 0)
    return (cor, tot, cor / tot if tot else 0.0)


# name -> (prep%, pre_S8 report, stage8 stem, ODL report)
DOCS = [
    ("Miramar",        38, MODAL / "qa_miramar_untagged.json",                                      "miramar_untagged",                                 ODL / "qa_CITY OF MIRAMAR, FLORIDA_tagged.json"),
    ("Missouri",       43, MODAL / "Missouri State Epidemiological Profile July 2018_qa_report.json", "Missouri State Epidemiological Profile July 2018", ODL / "qa_Missouri State Epidemiological Profile July 2018_tagged.json"),
    ("Osteoarthritis", 52, MODAL / "Osteoarthritis_qa_report.json",                                  "Osteoarthritis",                                   ODL / "qa_Osteoarthritis_tagged.json"),
    ("Summary",        62, MODAL / "Summary of Revenues and Expenditures_qa_report.json",            "Summary of Revenues and Expenditures",             ODL / "qa_Summary of Revenues and Expenditures_tagged.json"),
    ("NYVRA",          26, MODAL / "nyvra-factsheet_qa_report.json",                                 "nyvra-factsheet",                                  ODL / "qa_nyvra-factsheet_tagged.json"),
]


def f(a):
    return f"{a[2]:.1%}" if a else "(pending)"


print(f"{'Document':16}{'PREP':>7}{'pre-S8':>9}{'list-fix':>10}{'TOC-fix':>9}{'ODL':>8}")
print("-" * 60)
for name, prep, pre_p, stem, odl_p in DOCS:
    pre = accuracy(pre_p)
    lst = accuracy(S8_LIST / f"qa_{stem}.json")
    toc = accuracy(S8_TOC / f"qa_{stem}.json")
    odl = accuracy(odl_p)
    print(f"{name:16}{str(prep)+'%':>7}{f(pre):>9}{f(lst):>10}{f(toc):>9}{f(odl):>8}")
