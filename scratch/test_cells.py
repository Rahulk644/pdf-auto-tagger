import pdfplumber
from unittest.mock import MagicMock
from tagger.stage5_specialists.table_extractor import extract_table_native
from tagger.models.data_types import LayoutRegion, PageType, PageElement

region = MagicMock()
region.region_id = "t1"
region.page_num = 1
region.bbox = (71.0, 39.0, 530.0, 696.0)

classification = MagicMock()
classification.page_type = PageType.NATIVE

el = MagicMock()
el.element_id = "fake_1"
el.page_num = 1
# el.bbox needs to be mapped to Top-Left 72 DPI using scale
# STANDARD_DPI=150, PDF_NATIVE_DPI=72. scale = 150/72 = 2.0833
# bbox = (x0, top, x1, bottom)
# let's set bbox in 150 DPI such that it lands around x=80, y=50 in PDF coords
# x0_pdf = 80, y0_pdf = 50 -> x0_std = 80 * 150/72 = 166.6
el.bbox = (166.6, 104.16, 200.0, 110.0)
el.merged_from = ["p1_c1", "p1_c2"]

struct = extract_table_native(
    pdf_path="miramar_untagged.pdf",
    page_num=1,
    region=region,
    classification=classification,
    page_elements=[el]
)

if struct and hasattr(struct, "cells"):
    print("Cells count:", len(struct.cells))
    print("First cell merged_from:", struct.cells[0]["merged_from"])
    print("Cells data:", struct.cells[:2])
else:
    print("Struct has no cells")
