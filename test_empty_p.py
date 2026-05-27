import json

with open("/Users/rahulkhatri/Tagger/output_modal/miramar_untagged_report.json") as f:
    d = json.load(f)
    
for el in d.get("elements", []):
    if el.get("pdf_tag") in ["P", "H1", "H2", "H3"]:
        print(f"ID: {el.get('element_id')} Tag: {el.get('pdf_tag')} Length: {len(el.get('text', ''))}")
