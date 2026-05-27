"""
Runs the pipeline on miramar_untagged.pdf and prints element counts
at Stage 3 (MinerU regions), Stage 4/5 (tagged elements), and Stage 10 input.
No pipeline code modified — hooks in by monkey-patching the stage methods.
"""
import sys
import logging
sys.path.insert(0, "/Users/rahulkhatri/Tagger")

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)

from tagger.pipeline import AutoTaggerPipeline

INPUT = "/Users/rahulkhatri/Tagger/miramar_untagged.pdf"

# ── Monkey-patch stage methods to print counts ──────────────────────────────

original_stage3 = AutoTaggerPipeline._stage3_layout

def patched_stage3(self, input_pdf, doc_data):
    original_stage3(self, input_pdf, doc_data)
    print("\n=== STAGE 3: MinerU Layout Regions ===")
    for page_num, page_data in sorted(doc_data.pages.items()):
        regions = page_data.layout_regions or []
        from collections import Counter
        cats = Counter(r.category.value if hasattr(r.category, "value") else str(r.category) for r in regions)
        print(f"  Page {page_num}: {len(regions)} regions  {dict(cats)}")

AutoTaggerPipeline._stage3_layout = patched_stage3


original_stage45 = AutoTaggerPipeline._stage4_5_route_extract

def patched_stage45(self, doc_data):
    original_stage45(self, doc_data)
    print("\n=== STAGE 4/5: Tagged Elements (after route + table specialist) ===")
    for page_num, page_data in sorted(doc_data.pages.items()):
        elems = page_data.tagged_elements or []
        from collections import Counter
        tags = Counter(e.pdf_tag.value if hasattr(e.pdf_tag, "value") else str(e.pdf_tag) for e in elems)
        merged_from_counts = [len(e.merged_from) for e in elems]
        empty_mf = sum(1 for c in merged_from_counts if c == 0)
        print(f"  Page {page_num}: {len(elems)} elements  tags={dict(tags)}  empty_merged_from={empty_mf}")

AutoTaggerPipeline._stage4_5_route_extract = patched_stage45


# Patch stage 10 to print what it receives BEFORE writing
original_stage10 = AutoTaggerPipeline._stage10_write

def patched_stage10(self, input_pdf, output_pdf, doc_data):
    print("\n=== STAGE 10 INPUT: tagged_elements per page ===")
    for page_num, page_data in sorted(doc_data.pages.items()):
        elems = page_data.tagged_elements or []
        from collections import Counter
        tags = Counter(e.pdf_tag.value if hasattr(e.pdf_tag, "value") else str(e.pdf_tag) for e in elems)
        empty_mf = sum(1 for e in elems if len(e.merged_from) == 0)
        table_elems = [e for e in elems if "Table" in str(e.pdf_tag)]
        print(f"  Page {page_num}: {len(elems)} elements  tags={dict(tags)}  empty_merged_from={empty_mf}  table_elems={len(table_elems)}")
    original_stage10(self, input_pdf, output_pdf, doc_data)
    print("\n=== STAGE 10 OUTPUT: BDC markers injected ===")
    import pikepdf
    pdf = pikepdf.open(output_pdf)
    for i, page in enumerate(pdf.pages):
        cs = pikepdf.parse_content_stream(page)
        bdc = sum(1 for op in cs if str(op.operator) == "BDC")
        emc = sum(1 for op in cs if str(op.operator) == "EMC")
        print(f"  Page {i+1}: BDC={bdc} EMC={emc}")

AutoTaggerPipeline._stage10_write = patched_stage10


# ── Run the pipeline ────────────────────────────────────────────────────────

OUT = "/tmp/miramar_diag_out.pdf"
pipeline = AutoTaggerPipeline()
print(f"Running pipeline on {INPUT}...")
pipeline.run(INPUT, output_pdf=OUT)
print(f"\nOutput written to {OUT}")
