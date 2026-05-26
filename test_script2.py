import tagger.pipeline as pipe
file_path = pipe.__file__
with open(file_path, "r") as f:
    lines = f.readlines()

start_idx = -1
end_idx = -1

for i, line in enumerate(lines):
    if "def _stage4_5_route_extract(" in line:
        start_idx = i
    if start_idx != -1 and "def _stage6_validate(" in line:
        end_idx = i
        break

if start_idx != -1 and end_idx != -1:
    new_method = """    def _stage4_5_route_extract(self, doc_data: DocumentData):
        \"\"\"Stage 4+5: Route regions and create initial tagged elements.\"\"\"
        logger.info("[Stage 4+5] Routing and initial tagging...")

        category_to_tag: dict[LayoutCategory, PDFTag] = {
            LayoutCategory.TITLE:          PDFTag.H1,
            LayoutCategory.SECTION_HEADER: PDFTag.H2,
            LayoutCategory.TEXT:           PDFTag.P,
            LayoutCategory.LIST_ITEM:      PDFTag.LI,
            LayoutCategory.TABLE:          PDFTag.TABLE,
            LayoutCategory.FORMULA:        PDFTag.FORMULA,
            LayoutCategory.PICTURE:        PDFTag.FIGURE,
            LayoutCategory.CAPTION:        PDFTag.P,
            LayoutCategory.FOOTNOTE:       PDFTag.NOTE,
            LayoutCategory.PAGE_HEADER:    PDFTag.ARTIFACT,
            LayoutCategory.PAGE_FOOTER:    PDFTag.ARTIFACT,
        }

        total_tagged = 0
        for page_num, page_data in doc_data.pages.items():
            tagged: list[TaggedElement] = []
            
            el_to_cat = {}
            if page_data.layout_regions and not page_data.layout_regions[0].matched_elements:
                for el in page_data.elements:
                    ex0, ey0, ex1, ey1 = el.bbox
                    ecx, ecy = (ex0 + ex1)/2, (ey0 + ey1)/2
                    
                    best_region = None
                    best_dist = float('inf')
                    for region in page_data.layout_regions:
                        rx0, ry0, rx1, ry1 = region.bbox
                        rcx, rcy = (rx0 + rx1)/2, (ry0 + ry1)/2
                        dist = (ecx - rcx)**2 + (ecy - rcy)**2
                        if dist < best_dist:
                            best_dist = dist
                            best_region = region
                            
                    if best_region:
                        el_to_cat[el.element_id] = (best_region.category, best_region.confidence)
            else:
                for region in page_data.layout_regions:
                    for eid in (region.matched_elements or []):
                        el_to_cat[eid] = (region.category, region.confidence)

            for el in page_data.elements:
                cat, conf = el_to_cat.get(el.element_id, (LayoutCategory.TEXT, 1.0))
                pdf_tag = category_to_tag.get(cat, PDFTag.P)
                tagged_el = TaggedElement(
                    element_id=el.element_id,
                    page_num=page_num,
                    pdf_tag=pdf_tag,
                    text=el.text,
                    bbox=el.bbox,
                    confidence=conf,
                    original_mcid=el.mcid,
                    metadata=el.metadata,
                )
                tagged.append(tagged_el)
                total_tagged += 1

            page_data.tagged_elements = tagged

        logger.debug(f"Created {total_tagged} initially tagged elements.")

"""
    lines[start_idx:end_idx] = [new_method]
    with open(file_path, "w") as f:
        f.writelines(lines)
    print("✅ Pipeline robust matching patched!")
else:
    print(f"⚠️ Could not find boundaries. start={start_idx}, end={end_idx}")
