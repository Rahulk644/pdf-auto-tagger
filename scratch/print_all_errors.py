import json

with open("/Users/rahulkhatri/QA Tool/scratch/qa_results_modal/miramar_untagged_qa_report.json") as f:
    qa = json.load(f)

for page in qa.get("results", []):
    for row in page.get("data", []):
        if not row.get("is_correct"):
            print(f"Error: current={row.get('current_tag')}, suggested={row.get('suggested_tag')}. Text: {repr(row.get('text', ''))[:60]}")
