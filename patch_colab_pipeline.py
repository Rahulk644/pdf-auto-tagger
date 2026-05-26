import os
import tagger.pipeline as pipe
import ast

file_path = pipe.__file__
with open(file_path, "r") as f:
    content = f.read()

# I will replace the inner loop of `_stage4_5_route_extract`
old_code = """            for region in page_data.layout_regions:
                # Find elements within this region
                matched_els = []
                if region.matched_elements:
                    matched_els = [
                        element_map[eid] for eid in region.matched_elements
                        if eid in element_map
                    ]
                else:
                    # Fall back to spatial matching (center point containment)
                    for el in page_data.elements:
                        rx0, ry0, rx1, ry1 = region.bbox
                        ex0, ey0, ex1, ey1 = el.bbox
                        cx = (ex0 + ex1) / 2
                        cy = (ey0 + ey1) / 2
                        
                        # Add a generous tolerance since MinerU boxes can be imprecise
                        tol = 15.0
                        if (rx0 - tol <= cx <= rx1 + tol) and (ry0 - tol <= cy <= ry1 + tol):
                            matched_els.append(el)

                pdf_tag = category_to_tag.get(region.category, PDFTag.P)

                for el in matched_els:
                    tagged_el = TaggedElement(
                        element_id=el.element_id,
                        page_num=page_num,
                        text=el.text,
                        bbox=el.bbox,
                        metadata=el.metadata,
                        pdf_tag=pdf_tag,
                    )
                    tagged.append(tagged_el)
                    total_tagged += 1"""

new_code = """            # Global distance-based matching to handle AI layout bounding box shifts
            # 1. If layout regions have matched_elements, use them directly (fallback mode)
            # 2. Otherwise, assign each element to the layout region with the closest center
            
            # Map element ID to region category
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
                        el_to_cat[el.element_id] = best_region.category
            else:
                for region in page_data.layout_regions:
                    for eid in (region.matched_elements or []):
                        el_to_cat[eid] = region.category

            for el in page_data.elements:
                cat = el_to_cat.get(el.element_id, LayoutCategory.TEXT)
                pdf_tag = category_to_tag.get(cat, PDFTag.P)
                tagged_el = TaggedElement(
                    element_id=el.element_id,
                    page_num=page_num,
                    text=el.text,
                    bbox=el.bbox,
                    metadata=el.metadata,
                    pdf_tag=pdf_tag,
                )
                tagged.append(tagged_el)
                total_tagged += 1"""

content = content.replace(old_code, new_code)

with open(file_path, "w") as f:
    f.write(content)
print("Pipeline robust matching patched!")
