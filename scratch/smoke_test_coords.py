from tagger.pipeline import Pipeline
import logging

logging.basicConfig(level=logging.INFO)
pipeline = Pipeline("miramar_untagged.pdf", output_dir="scratch/out")

# We just want to run stages 1 through 5 locally and inspect the cells.
print("Starting smoke test...")
pipeline.run(page_range=(1, 1), use_modal=False, skip_qa=True)
