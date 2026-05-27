import json

with open("/Users/rahulkhatri/Tagger/output_modal/miramar_untagged_report.json") as f:
    pipeline = json.load(f)

# Collect pipeline elements
pipe_elements = pipeline.get("elements", [])
print(f"Loaded {len(pipe_elements)} pipeline elements")

# Read QA
with open("/Users/rahulkhatri/PREP QA Tool/scratch/qa_results_modal/miramar_untagged_qa_report.json") as f:
    qa = json.load(f)

# The QA report only has MCIDs.
# Since we know MCIDs are assigned sequentially per page starting from 0 (in V2) or document-wide?
# Let's see how MCIDs are assigned. In struct_tree_writer.py, mcid_counter = 0 is per document!
# Wait: `mcid_counter = 0` is initialized OUTSIDE the page loop!
# So MCIDs are document-wide!

for page in qa.get("results", []):
    for row in page.get("data", []):
        if not row.get("is_correct") and row.get("suggested_tag") == "Artifact":
            print(f"QA Error: MCID {row.get('mcid')} expected Artifact, got {row.get('current_tag')}. Text: {repr(row.get('text', ''))}")

