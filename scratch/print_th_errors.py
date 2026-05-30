import json

with open('/Users/rahulkhatri/QA Tool/scratch/qa_results_modal/miramar_untagged_qa_report.json') as f:
    data = json.load(f)

for page_data in data.get("results", []):
    for item in page_data.get("data", []):
        if "row header" in str(item.get("corrective_reasoning", "")).lower():
            print(f"MCID: {item.get('mcid')} | Text={repr(item.get('extracted_text', item.get('text', '')))}")
