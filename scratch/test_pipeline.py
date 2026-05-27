import logging
import json
from tagger.pipeline import AutoTaggerPipeline

logging.basicConfig(level=logging.INFO)
pipeline = AutoTaggerPipeline()

print("Running pipeline...")
report = pipeline.run(
    input_pdf="miramar_untagged.pdf",
    output_pdf="scratch/miramar_nested_table_test.pdf",
    report_path="scratch/miramar_nested_table_test_report.json"
)
print("Done. Check scratch/miramar_nested_table_test.pdf")
