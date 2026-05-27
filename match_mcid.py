import json

with open("/Users/rahulkhatri/Tagger/output_modal/miramar_untagged_report.json") as f:
    pipeline = json.load(f)

pipe_elements = pipeline.get("elements", [])

mcid_map = {}
mcid_counter = 0

for el in pipe_elements:
    if el.get("pdf_tag") != "Artifact":
        mcid_map[str(mcid_counter)] = el
        mcid_counter += 1

with open("/Users/rahulkhatri/PREP QA Tool/scratch/qa_results_modal/miramar_untagged_qa_report.json") as f:
    qa = json.load(f)

for page in qa.get("results", []):
    for row in page.get("data", []):
        if not row.get("is_correct") and row.get("suggested_tag") == "Artifact":
            mcid = str(row.get("mcid"))
            pipe_el = mcid_map.get(mcid)
            if pipe_el:
                print(f"MCID {mcid} -> Pipeline Tag: {pipe_el.get('pdf_tag')}, Text: {repr(pipe_el.get('text'))}")
                print(f"          -> QA Reason: {row.get('corrective_reasoning')[:80]}")
