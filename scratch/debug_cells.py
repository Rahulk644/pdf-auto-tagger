import logging
from tagger.pipeline import AutoTaggerPipeline
from tagger.models.data_types import PDFTag

logging.basicConfig(level=logging.ERROR)
pipeline = AutoTaggerPipeline()
doc_data = pipeline.tracker.init_document("miramar_untagged.pdf")

# Run stages 0 to 5 manually
pipeline._stage0_classify("miramar_untagged.pdf")
pipeline._stage1_extract("miramar_untagged.pdf", pipeline.tracker.get_classifications())
# wait, tracker doesn't return classifications, stage 0 returns them.
classifications = pipeline._stage0_classify("miramar_untagged.pdf")
raw_elements = pipeline._stage1_extract("miramar_untagged.pdf", classifications)
for page_num, els in raw_elements.items():
    doc_data.pages.setdefault(page_num, pipeline.tracker.DocumentData.pages.__class__.__args__[0](page_num=page_num))
    doc_data.pages[page_num].elements = els
    doc_data.pages[page_num].classification = next(c for c in classifications if c.page_num == page_num)

pipeline._stage2_merge(doc_data)
pipeline._stage3_layout("miramar_untagged.pdf", doc_data)
pipeline._stage4_5_route_extract(doc_data)

for page_num, page_data in doc_data.pages.items():
    for el in page_data.tagged_elements:
        if el.pdf_tag == PDFTag.TABLE:
            print(f"Page {page_num} Table ID {el.element_id}")
            cells = el.specialist_data.get("cells", [])
            print(f"Has cells? {len(cells)}")
            if cells:
                print(f"First cell: {cells[0]}")
